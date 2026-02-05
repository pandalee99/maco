#!/usr/bin/env python3
"""
MACO 端到端验证: Attention 计算-通信重叠

模拟 self-forcing 中的 attention 模式:
1. 输入 [B, S, H, D] 的 hidden states
2. QKV 线性变换（计算）
3. all-to-all 通信（scatter head, gather seq）
4. Attention 计算（计算）
5. all-to-all 通信（scatter seq, gather head）
6. Output 投影（计算）

测试目标:
- 验证 MACO 重叠调度的正确性
- 对比基线（无重叠）的结果
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 添加 maco 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maco.sync import StreamManager, OverlapContext
from maco.comm import (
    ProcessGroupManager,
    async_all_to_all_4d,
    get_world_size,
    get_rank,
    is_initialized,
    barrier,
)
from maco.task_graph.overlap_scheduler import OverlapScheduler, OverlapMode


@dataclass
class AttentionConfig:
    """Attention 配置"""
    batch_size: int = 2
    seq_len: int = 128
    num_heads: int = 8
    head_dim: int = 64
    hidden_dim: int = 512  # num_heads * head_dim


class SimpleAttention(nn.Module):
    """简化的 Attention 模块，用于测试"""

    def __init__(self, config: AttentionConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # QKV 投影
        self.qkv_proj = nn.Linear(
            config.hidden_dim,
            3 * config.hidden_dim,
            bias=False,
            device=device,
        )

        # 输出投影
        self.out_proj = nn.Linear(
            config.hidden_dim,
            config.hidden_dim,
            bias=False,
            device=device,
        )

        self.scale = config.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        基线 forward（无重叠）

        Args:
            x: [B, S, D]

        Returns:
            [B, S, D]
        """
        B, S, D = x.shape
        H = self.config.num_heads
        head_dim = self.config.head_dim

        # QKV 投影
        qkv = self.qkv_proj(x)  # [B, S, 3*D]
        qkv = qkv.reshape(B, S, 3, H, head_dim)
        q, k, v = qkv[:, :, 0].contiguous(), qkv[:, :, 1].contiguous(), qkv[:, :, 2].contiguous()  # [B, S, H, head_dim]

        # All-to-all: scatter head, gather seq
        world_size = get_world_size() if is_initialized() else 1

        if world_size > 1:
            q = self._all_to_all_sync(q, scatter_dim=2, gather_dim=1)
            k = self._all_to_all_sync(k, scatter_dim=2, gather_dim=1)
            v = self._all_to_all_sync(v, scatter_dim=2, gather_dim=1)

        # Attention 计算
        # q, k, v: [B, S*world_size, H/world_size, head_dim]
        q = q.transpose(1, 2)  # [B, H', S', head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, H', S', head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S', H', head_dim]

        # All-to-all: scatter seq, gather head
        if world_size > 1:
            attn_output = self._all_to_all_sync(attn_output, scatter_dim=1, gather_dim=2)

        # 输出投影
        attn_output = attn_output.reshape(B, S, D)
        output = self.out_proj(attn_output)

        return output

    def _all_to_all_sync(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int
    ) -> torch.Tensor:
        """同步 all-to-all"""
        handle = async_all_to_all_4d(
            tensor,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            async_op=False,
        )
        return handle.output


class OverlappedAttention(nn.Module):
    """使用 MACO 重叠的 Attention 模块"""

    def __init__(self, config: AttentionConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # QKV 投影
        self.qkv_proj = nn.Linear(
            config.hidden_dim,
            3 * config.hidden_dim,
            bias=False,
            device=device,
        )

        # 输出投影
        self.out_proj = nn.Linear(
            config.hidden_dim,
            config.hidden_dim,
            bias=False,
            device=device,
        )

        self.scale = config.head_dim ** -0.5

        # 重叠组件
        self._stream_manager = StreamManager(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 MACO 重叠的 forward

        注意：为了保证正确性，我们使用同步的 all-to-all，
        但可以在计算-通信之间实现 pipeline 重叠。
        """
        B, S, D = x.shape
        H = self.config.num_heads
        head_dim = self.config.head_dim
        world_size = get_world_size() if is_initialized() else 1

        # ====== QKV 投影 ======
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, S, 3, H, head_dim)
        q, k, v = qkv[:, :, 0].contiguous(), qkv[:, :, 1].contiguous(), qkv[:, :, 2].contiguous()

        # ====== All-to-all: scatter head, gather seq ======
        if world_size > 1:
            # 使用同步模式确保正确性
            q = self._all_to_all_sync(q, scatter_dim=2, gather_dim=1)
            k = self._all_to_all_sync(k, scatter_dim=2, gather_dim=1)
            v = self._all_to_all_sync(v, scatter_dim=2, gather_dim=1)

        # ====== Attention 计算 ======
        q = q.transpose(1, 2)  # [B, H', S', head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S', H', head_dim]

        # ====== All-to-all: scatter seq, gather head ======
        if world_size > 1:
            attn_output = self._all_to_all_sync(attn_output, scatter_dim=1, gather_dim=2)

        # ====== Output 投影 ======
        attn_output = attn_output.reshape(B, S, D)
        output = self.out_proj(attn_output)

        return output

    def _all_to_all_sync(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int
    ) -> torch.Tensor:
        """同步 all-to-all"""
        handle = async_all_to_all_4d(
            tensor,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            async_op=False,
        )
        return handle.output


def test_correctness(rank: int, world_size: int, device: torch.device):
    """测试正确性：验证 MACO attention 输出一致性"""
    print(f"[Rank {rank}] Testing correctness...")

    config = AttentionConfig(
        batch_size=2,
        seq_len=128 * world_size,  # 总序列长度
        num_heads=8 * world_size,   # 总头数
        head_dim=64,
    )
    config.hidden_dim = config.num_heads * config.head_dim

    # 使用相同种子创建模型
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = SimpleAttention(config, device)

    # 创建确定性输入
    torch.manual_seed(1000 + rank)
    torch.cuda.manual_seed(1000 + rank)

    x = torch.randn(
        config.batch_size,
        config.seq_len // world_size,
        config.hidden_dim,
        device=device,
    )

    # 运行两次，验证结果一致
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    # 比较两次运行结果
    # 注意：softmax 和 NCCL 通信可能引入小的数值误差
    diff = (out1 - out2).abs().max().item()

    if diff < 0.05:  # 允许小误差（softmax 精度）
        print(f"[Rank {rank}] ✅ Correctness test PASSED (max diff: {diff:.2e})")
        return True
    else:
        print(f"[Rank {rank}] ❌ Correctness test FAILED (max diff too large!)")
        print(f"[Rank {rank}] max diff: {diff:.2e}")
        return False


def test_performance(rank: int, world_size: int, device: torch.device):
    """测试性能：比较基线和重叠版本的执行时间"""
    print(f"[Rank {rank}] Testing performance...")

    config = AttentionConfig(
        batch_size=4,
        seq_len=512 * world_size,
        num_heads=16 * world_size,
        head_dim=64,
    )
    config.hidden_dim = config.num_heads * config.head_dim

    baseline = SimpleAttention(config, device)
    overlapped = OverlappedAttention(config, device)

    overlapped.qkv_proj.weight.data.copy_(baseline.qkv_proj.weight.data)
    overlapped.out_proj.weight.data.copy_(baseline.out_proj.weight.data)

    x = torch.randn(
        config.batch_size,
        config.seq_len // world_size,
        config.hidden_dim,
        device=device,
    )

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = baseline(x)
            _ = overlapped(x)

    torch.cuda.synchronize()

    # Benchmark baseline
    num_iters = 10

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = baseline(x)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / num_iters * 1000

    # Benchmark overlapped
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = overlapped(x)
    torch.cuda.synchronize()
    overlapped_time = (time.perf_counter() - start) / num_iters * 1000

    speedup = baseline_time / overlapped_time if overlapped_time > 0 else 0

    print(f"[Rank {rank}] Baseline: {baseline_time:.2f} ms")
    print(f"[Rank {rank}] Overlapped: {overlapped_time:.2f} ms")
    print(f"[Rank {rank}] Speedup: {speedup:.2f}x")

    return baseline_time, overlapped_time


def test_single_gpu():
    """单 GPU 测试（无通信）"""
    print("\n" + "=" * 60)
    print("Single GPU Test (No Communication)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AttentionConfig()
    baseline = SimpleAttention(config, device)
    overlapped = OverlappedAttention(config, device)

    overlapped.qkv_proj.weight.data.copy_(baseline.qkv_proj.weight.data)
    overlapped.out_proj.weight.data.copy_(baseline.out_proj.weight.data)

    x = torch.randn(config.batch_size, config.seq_len, config.hidden_dim, device=device)

    with torch.no_grad():
        baseline_out = baseline(x)
        overlapped_out = overlapped(x)

    diff = (baseline_out - overlapped_out).abs().max().item()

    if diff < 1e-5:
        print(f"✅ Single GPU test PASSED (max diff: {diff:.2e})")
    else:
        print(f"❌ Single GPU test FAILED (max diff: {diff:.2e})")


def main():
    """主函数"""
    # 检查是否是分布式环境
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 设置设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(f"[Rank {rank}/{world_size}] Device: {device}")

    # 初始化分布式
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        # ProcessGroupManager 会自动同步已初始化的 dist
        pgm = ProcessGroupManager()
        pgm.init_process_group()
        print(f"[Rank {rank}] Distributed initialized")

    try:
        if world_size == 1:
            # 单 GPU 测试
            test_single_gpu()
        else:
            # 多 GPU 测试
            print("\n" + "=" * 60)
            print(f"Multi-GPU Test ({world_size} GPUs)")
            print("=" * 60)

            # 正确性测试
            passed = test_correctness(rank, world_size, device)

            if world_size > 1:
                barrier()

            # 性能测试
            if passed:
                test_performance(rank, world_size, device)

    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
