"""
Tensor Parallel Linear Example

Demonstrates how to use MACO for tensor parallel linear layers
with optimized communication.
"""

import torch
import torch.nn as nn
import maco
from maco.core import SMConfig, Role


class TPLinear(nn.Module):
    """
    Tensor Parallel Linear Layer with MACO optimization.

    The weight matrix is split column-wise across GPUs.
    After the local matmul, an allreduce combines the results.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size

        # Each GPU holds out_features/world_size columns
        self.local_out_features = out_features // world_size

        self.weight = nn.Parameter(
            torch.randn(self.local_out_features, in_features) / (in_features ** 0.5)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized allreduce.

        Args:
            x: Input tensor of shape (batch, in_features)

        Returns:
            Output tensor of shape (batch, out_features)
        """
        # Local matmul: (batch, in_features) @ (in_features, local_out)
        y = torch.matmul(x, self.weight.T)

        # Gather results from all GPUs
        # In a real TP setup, each GPU computes a different part
        # and allreduce combines them
        y = maco.allreduce(y, overlap_with_next=True)

        if self.bias is not None:
            y = y + self.bias

        return y


class TPTransformerBlock(nn.Module):
    """
    Simplified Transformer block with tensor parallelism.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        world_size: int,
    ):
        super().__init__()
        self.world_size = world_size

        # Attention projection (simplified)
        self.qkv_proj = TPLinear(hidden_size, hidden_size * 3, world_size)
        self.o_proj = TPLinear(hidden_size, hidden_size, world_size)

        # FFN
        self.ffn_up = TPLinear(hidden_size, ffn_size, world_size)
        self.ffn_down = TPLinear(ffn_size, hidden_size, world_size)

        # Norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention (simplified - no actual attention computation)
        residual = x
        x = self.norm1(x)
        qkv = self.qkv_proj(x)
        # ... attention would happen here ...
        attn_out = self.o_proj(qkv[..., :qkv.shape[-1]//3])  # Simplified
        x = residual + attn_out

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn_up(x)
        x = torch.nn.functional.gelu(x)
        x = self.ffn_down(x)
        x = residual + x

        return x


def main():
    # Initialize
    maco.init()
    world_size = maco.get_world_size()
    rank = maco.get_rank()

    print(f"Running TP example on rank {rank}/{world_size}")

    # Configure SM allocation
    sm_config = SMConfig(total_sms=132)
    sm_config.assign_role((0, 100), Role.COMPUTE)
    sm_config.assign_role((100, 124), Role.LOCAL_SCHEDULER)
    sm_config.assign_role((124, 132), Role.REMOTE_SCHEDULER)

    print(f"SM Config: {sm_config.compute_sms} compute, "
          f"{sm_config.local_scheduler_sms} local sched, "
          f"{sm_config.remote_scheduler_sms} remote sched")

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TPTransformerBlock(
        hidden_size=4096,
        ffn_size=11008,
        world_size=max(1, world_size),
    ).to(device)

    # Create input
    batch_size = 4
    seq_len = 512
    x = torch.randn(batch_size, seq_len, 4096, device=device)

    # Forward pass
    with maco.Context(compute_sms=100, comm_sms=32):
        y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output sum: {y.sum().item():.4f}")

    print("\n[MACO] Tensor Parallel example completed!")


if __name__ == "__main__":
    main()
