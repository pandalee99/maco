#!/usr/bin/env python3
"""
Test MACO async_all_to_all determinism
"""

import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maco.comm import (
    ProcessGroupManager,
    async_all_to_all,
    async_all_to_all_4d,
    get_world_size,
    get_rank,
)


def test_2d():
    """Test 2D all-to-all"""
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    # 2D tensor
    x = torch.randn(world_size * 100, 64, device=device)

    print(f"[Rank {rank}] 2D Test - Input shape: {x.shape}")

    # Run twice
    handle1 = async_all_to_all(x.clone(), scatter_dim=0, gather_dim=0, async_op=False)
    out1 = handle1.output

    handle2 = async_all_to_all(x.clone(), scatter_dim=0, gather_dim=0, async_op=False)
    out2 = handle2.output

    diff = (out1 - out2).abs().max().item()
    if diff < 1e-6:
        print(f"[Rank {rank}] ✅ 2D all-to-all is deterministic (diff: {diff:.2e})")
    else:
        print(f"[Rank {rank}] ❌ 2D all-to-all is NOT deterministic (diff: {diff:.2e})")

    return diff < 1e-6


def test_4d():
    """Test 4D all-to-all"""
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    # 4D tensor [B, S, H, D]
    B, S, H, D = 2, 32, 8 * world_size, 64
    x = torch.randn(B, S, H, D, device=device)

    print(f"[Rank {rank}] 4D Test - Input shape: {x.shape}")

    # Run twice
    x1 = x.clone()
    x2 = x.clone()

    # Verify inputs are identical
    assert (x1 - x2).abs().max().item() < 1e-10, "Input clones differ!"

    handle1 = async_all_to_all_4d(x1, scatter_dim=2, gather_dim=1, async_op=False)
    out1 = handle1.output

    handle2 = async_all_to_all_4d(x2, scatter_dim=2, gather_dim=1, async_op=False)
    out2 = handle2.output

    diff = (out1 - out2).abs().max().item()
    if diff < 1e-6:
        print(f"[Rank {rank}] ✅ 4D all-to-all is deterministic (diff: {diff:.2e})")
    else:
        print(f"[Rank {rank}] ❌ 4D all-to-all is NOT deterministic (diff: {diff:.2e})")
        print(f"[Rank {rank}] Out1 shape: {out1.shape}, mean: {out1.mean().item():.6f}")
        print(f"[Rank {rank}] Out2 shape: {out2.shape}, mean: {out2.mean().item():.6f}")

    return diff < 1e-6


def test_4d_step_by_step():
    """Test 4D all-to-all step by step"""
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    B, S, H, D = 2, 32, 8 * world_size, 64
    x = torch.randn(B, S, H, D, device=device)

    print(f"[Rank {rank}] Step-by-step Test - Input shape: {x.shape}")

    # First call - manual steps
    x1 = x.clone()

    # Split along scatter_dim=2 (H)
    input_list1 = list(torch.chunk(x1, world_size, dim=2))
    input_list1 = [t.contiguous() for t in input_list1]

    print(f"[Rank {rank}] chunk[0] shape: {input_list1[0].shape}")

    output_list1 = [torch.empty_like(input_list1[0]) for _ in range(world_size)]

    dist.all_to_all(output_list1, input_list1)
    torch.cuda.synchronize()

    out1 = torch.cat(output_list1, dim=1)  # gather_dim=1 (S)

    # Second call - manual steps
    x2 = x.clone()

    input_list2 = list(torch.chunk(x2, world_size, dim=2))
    input_list2 = [t.contiguous() for t in input_list2]

    output_list2 = [torch.empty_like(input_list2[0]) for _ in range(world_size)]

    dist.all_to_all(output_list2, input_list2)
    torch.cuda.synchronize()

    out2 = torch.cat(output_list2, dim=1)

    # Compare
    diff = (out1 - out2).abs().max().item()
    if diff < 1e-6:
        print(f"[Rank {rank}] ✅ Step-by-step 4D all-to-all is deterministic (diff: {diff:.2e})")
    else:
        print(f"[Rank {rank}] ❌ Step-by-step 4D all-to-all is NOT deterministic (diff: {diff:.2e})")

    return diff < 1e-6


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}] Started")

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        pgm = ProcessGroupManager()
        pgm.init_process_group()

    try:
        test_2d()
        dist.barrier()

        test_4d()
        dist.barrier()

        test_4d_step_by_step()
        dist.barrier()

    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
