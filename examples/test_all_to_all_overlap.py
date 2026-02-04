"""
Test all-to-all + compute overlap for Sequence Parallel workloads.

This example demonstrates how MACO can overlap all-to-all communication
with compute operations, which is the communication pattern used in
Sequence Parallel (SP) architectures like Self-Forcing-14B.

Run with 2 GPUs:
    cd /mini_mirage/maco
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 examples/test_all_to_all_overlap.py
"""

import os
import torch
import torch.distributed as dist
import time
import maco


def benchmark_sequential(input_tensor, compute_size, world_size, warmup=5, iters=20):
    """Benchmark sequential all-to-all + compute."""
    rank = dist.get_rank()

    # Prepare output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Compute tensor
    a = torch.randn(compute_size, compute_size, device="cuda")
    b = torch.randn(compute_size, compute_size, device="cuda")

    # Warmup
    for _ in range(warmup):
        dist.all_to_all_single(output_tensor, input_tensor)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_to_all_single(output_tensor, input_tensor)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iters

    return elapsed


def benchmark_overlapped(input_tensor, compute_size, world_size, warmup=5, iters=20):
    """Benchmark overlapped all-to-all + compute using MACO."""
    rank = dist.get_rank()

    # Prepare output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Compute tensor
    a = torch.randn(compute_size, compute_size, device="cuda")
    b = torch.randn(compute_size, compute_size, device="cuda")

    # Warmup
    for _ in range(warmup):
        compute_result, _ = maco.overlap_all_to_all_with_compute(
            all_to_all_fn=lambda: dist.all_to_all_single(output_tensor, input_tensor),
            compute_fn=lambda: torch.matmul(a, b)
        )
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        compute_result, _ = maco.overlap_all_to_all_with_compute(
            all_to_all_fn=lambda: dist.all_to_all_single(output_tensor, input_tensor),
            compute_fn=lambda: torch.matmul(a, b)
        )
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iters

    return elapsed


def benchmark_overlap_region(input_tensor, compute_size, world_size, warmup=5, iters=20):
    """Benchmark overlapped using overlap_region API."""
    rank = dist.get_rank()

    # Prepare output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Compute tensor
    a = torch.randn(compute_size, compute_size, device="cuda")
    b = torch.randn(compute_size, compute_size, device="cuda")

    # Warmup
    for _ in range(warmup):
        with maco.overlap_region() as region:
            region.comm(lambda: dist.all_to_all_single(output_tensor, input_tensor))
            region.compute(lambda: torch.matmul(a, b))
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        with maco.overlap_region() as region:
            region.comm(lambda: dist.all_to_all_single(output_tensor, input_tensor))
            region.compute(lambda: torch.matmul(a, b))
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / iters

    return elapsed


def benchmark_all_to_all_4D(batch, seq_per_rank, heads, head_dim, world_size, warmup=5, iters=20):
    """Benchmark all_to_all_4D (the pattern used in Self-Forcing-14B)."""
    rank = dist.get_rank()

    # Input: [B, S/P, H, D] - sequence parallel layout
    input_tensor = torch.randn(batch, seq_per_rank, heads, head_dim, device="cuda")

    # Compute tensor
    compute_size = heads * head_dim
    a = torch.randn(compute_size, compute_size, device="cuda")
    b = torch.randn(compute_size, compute_size, device="cuda")

    # Warmup
    for _ in range(warmup):
        # Sequential
        output = maco.all_to_all_4D(input_tensor.clone(), scatter_dim=2, gather_dim=1)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

    # Sequential benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        output = maco.all_to_all_4D(input_tensor.clone(), scatter_dim=2, gather_dim=1)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    seq_time = (time.perf_counter() - start) * 1000 / iters

    # Overlapped benchmark
    for _ in range(warmup):
        compute_result, comm_result = maco.overlap_all_to_all_with_compute(
            all_to_all_fn=lambda: maco.all_to_all_4D(input_tensor.clone(), scatter_dim=2, gather_dim=1),
            compute_fn=lambda: torch.matmul(a, b)
        )
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        compute_result, comm_result = maco.overlap_all_to_all_with_compute(
            all_to_all_fn=lambda: maco.all_to_all_4D(input_tensor.clone(), scatter_dim=2, gather_dim=1),
            compute_fn=lambda: torch.matmul(a, b)
        )
        torch.cuda.synchronize()
    overlap_time = (time.perf_counter() - start) * 1000 / iters

    return seq_time, overlap_time


def main():
    # Initialize MACO
    maco.init(backend="nccl")

    rank = maco.get_rank()
    world_size = maco.get_world_size()

    if rank == 0:
        print("=" * 70)
        print("MACO All-to-All + Compute Overlap Benchmark")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name()}")
        print()

    # Test configurations
    configs = [
        # (tensor_size_mb, compute_size)
        (8, 1024),
        (16, 2048),
        (32, 2048),
        (32, 4096),
    ]

    if rank == 0:
        print("Test 1: all_to_all_single + matmul overlap")
        print("-" * 70)
        print(f"{'Config':<30} {'Sequential':<12} {'Overlapped':<12} {'Region':<12} {'Speedup':<10}")
        print("-" * 70)

    for tensor_size_mb, compute_size in configs:
        # Create input tensor
        elements = tensor_size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
        input_tensor = torch.randn(elements, device="cuda")

        # Run benchmarks
        seq_time = benchmark_sequential(input_tensor, compute_size, world_size)
        overlap_time = benchmark_overlapped(input_tensor, compute_size, world_size)
        region_time = benchmark_overlap_region(input_tensor, compute_size, world_size)

        speedup = seq_time / overlap_time

        if rank == 0:
            config_str = f"{tensor_size_mb}MB comm, {compute_size}x{compute_size} matmul"
            print(f"{config_str:<30} {seq_time:>8.2f} ms  {overlap_time:>8.2f} ms  {region_time:>8.2f} ms  {speedup:>6.2f}x")

    # Test all_to_all_4D (the Self-Forcing-14B pattern)
    if rank == 0:
        print()
        print("Test 2: all_to_all_4D (Self-Forcing-14B pattern)")
        print("-" * 70)
        print(f"{'Config':<40} {'Sequential':<12} {'Overlapped':<12} {'Speedup':<10}")
        print("-" * 70)

    # Configurations mimicking Self-Forcing-14B
    # [B, S/P, H, D] where P = world_size
    a2a_configs = [
        # (batch, seq_per_rank, heads, head_dim)
        (1, 256, 16, 128),    # Small
        (1, 512, 32, 128),    # Medium
        (1, 1024, 32, 128),   # Large
        (2, 512, 40, 128),    # Like 14B model
    ]

    for batch, seq_per_rank, heads, head_dim in a2a_configs:
        seq_time, overlap_time = benchmark_all_to_all_4D(
            batch, seq_per_rank, heads, head_dim, world_size
        )
        speedup = seq_time / overlap_time

        if rank == 0:
            tensor_shape = f"[{batch}, {seq_per_rank}, {heads}, {head_dim}]"
            config_str = f"{tensor_shape}"
            print(f"{config_str:<40} {seq_time:>8.2f} ms  {overlap_time:>8.2f} ms  {speedup:>6.2f}x")

    if rank == 0:
        print()
        print("=" * 70)
        print("Benchmark completed!")
        print("=" * 70)

    # Cleanup
    maco.cleanup()


if __name__ == "__main__":
    main()
