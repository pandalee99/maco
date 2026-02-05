#!/usr/bin/env python3
"""
Test all-to-all determinism
"""

import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maco.comm import (
    ProcessGroupManager,
    async_all_to_all_4d,
    get_world_size,
    get_rank,
    barrier,
)


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(f"[Rank {rank}] Device: {device}")

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        pgm = ProcessGroupManager()
        pgm.init_process_group()

    try:
        # Create deterministic input
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)

        B, S, H, D = 2, 32, 8 * world_size, 64
        x = torch.randn(B, S, H, D, device=device)

        print(f"[Rank {rank}] Input shape: {x.shape}")
        print(f"[Rank {rank}] Input mean: {x.mean().item():.6f}")

        # Run all-to-all twice
        handle1 = async_all_to_all_4d(x.clone(), scatter_dim=2, gather_dim=1, async_op=False)
        out1 = handle1.output

        handle2 = async_all_to_all_4d(x.clone(), scatter_dim=2, gather_dim=1, async_op=False)
        out2 = handle2.output

        print(f"[Rank {rank}] Output 1 shape: {out1.shape}, mean: {out1.mean().item():.6f}")
        print(f"[Rank {rank}] Output 2 shape: {out2.shape}, mean: {out2.mean().item():.6f}")

        # Check if outputs are the same
        diff = (out1 - out2).abs().max().item()
        if diff < 1e-6:
            print(f"[Rank {rank}] ✅ all-to-all is deterministic (diff: {diff:.2e})")
        else:
            print(f"[Rank {rank}] ❌ all-to-all is NOT deterministic (diff: {diff:.2e})")

        # Also run the reverse all-to-all
        handle3 = async_all_to_all_4d(out1, scatter_dim=1, gather_dim=2, async_op=False)
        out3 = handle3.output

        handle4 = async_all_to_all_4d(out2, scatter_dim=1, gather_dim=2, async_op=False)
        out4 = handle4.output

        diff2 = (out3 - out4).abs().max().item()
        print(f"[Rank {rank}] Reverse all-to-all diff: {diff2:.2e}")

        # Check roundtrip
        diff_roundtrip = (x - out3).abs().max().item()
        print(f"[Rank {rank}] Roundtrip diff: {diff_roundtrip:.2e}")

    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
