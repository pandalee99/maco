#!/usr/bin/env python3
"""
Test raw PyTorch all-to-all determinism
"""

import os
import sys
import torch
import torch.distributed as dist


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

    try:
        # Create deterministic input
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)

        # Simple 2D tensor [total_size, dim]
        x = torch.randn(world_size * 100, 64, device=device)

        print(f"[Rank {rank}] Input shape: {x.shape}, mean: {x.mean().item():.6f}")

        # Prepare for all-to-all
        input_list = list(torch.chunk(x, world_size, dim=0))
        input_list = [t.contiguous() for t in input_list]

        # First all-to-all
        output_list1 = [torch.empty_like(input_list[0]) for _ in range(world_size)]
        dist.all_to_all(output_list1, input_list)
        torch.cuda.synchronize()
        out1 = torch.cat(output_list1, dim=0)

        # Second all-to-all (same input)
        output_list2 = [torch.empty_like(input_list[0]) for _ in range(world_size)]
        dist.all_to_all(output_list2, input_list)
        torch.cuda.synchronize()
        out2 = torch.cat(output_list2, dim=0)

        # Check determinism
        diff = (out1 - out2).abs().max().item()
        if diff < 1e-6:
            print(f"[Rank {rank}] ✅ Raw all-to-all is deterministic (diff: {diff:.2e})")
        else:
            print(f"[Rank {rank}] ❌ Raw all-to-all is NOT deterministic (diff: {diff:.2e})")

        print(f"[Rank {rank}] Out1 mean: {out1.mean().item():.6f}, Out2 mean: {out2.mean().item():.6f}")

    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
