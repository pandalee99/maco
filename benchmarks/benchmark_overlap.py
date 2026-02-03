#!/usr/bin/env python3
"""
Benchmark: Pure compute-communication overlap test.

This script directly measures the benefit of overlapping communication
with computation using CUDA streams.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 benchmarks/benchmark_overlap.py
"""

import os
import sys
import time
import torch
import torch.distributed as dist

# Add maco to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sequential_allreduce_and_compute(
    comm_tensor: torch.Tensor,
    compute_fn,
    num_iterations: int = 100,
) -> float:
    """Sequential: allreduce then compute."""
    times = []

    for _ in range(num_iterations):
        t = comm_tensor.clone()
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()

        # Allreduce first
        dist.all_reduce(t)
        torch.cuda.synchronize()

        # Then compute
        result = compute_fn()
        torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def overlapped_allreduce_and_compute(
    comm_tensor: torch.Tensor,
    compute_fn,
    comm_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    num_iterations: int = 100,
) -> float:
    """Overlapped: allreduce and compute in parallel."""
    times = []

    for _ in range(num_iterations):
        t = comm_tensor.clone()
        torch.cuda.synchronize()
        dist.barrier()

        # Record start event
        start_event = torch.cuda.Event(enable_timing=False)
        start_event.record()

        start = time.perf_counter()

        # Launch allreduce on comm stream
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(start_event)
            work = dist.all_reduce(t, async_op=True)

        # Launch compute on compute stream
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(start_event)
            result = compute_fn()

        # Wait for both
        work.wait()
        torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def main():
    # Initialize distributed
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size < 2:
        print("This benchmark requires at least 2 GPUs")
        print("Usage: CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 benchmarks/benchmark_overlap.py")
        return

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("="*70)
        print("Compute-Communication Overlap Benchmark")
        print("="*70)
        print(f"World size: {world_size}, Device: {device}")
        print()

    # Create streams
    comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)

    # Test configurations
    configs = [
        # (comm_size, compute_size, description)
        ((2048, 2048), (1024, 1024), "Small compute, medium comm"),
        ((4096, 4096), (2048, 2048), "Medium compute, large comm"),
        ((4096, 4096), (4096, 4096), "Large compute, large comm"),
        ((8192, 4096), (4096, 4096), "Large compute, very large comm"),
    ]

    results = []

    for comm_size, compute_size, desc in configs:
        # Create tensors
        comm_tensor = torch.randn(comm_size, dtype=torch.bfloat16, device=device)
        compute_a = torch.randn(compute_size, dtype=torch.bfloat16, device=device)
        compute_b = torch.randn(compute_size, dtype=torch.bfloat16, device=device)

        comm_mb = comm_tensor.numel() * 2 / (1024 * 1024)
        compute_flops = 2 * compute_size[0] * compute_size[1] * compute_size[0]

        def compute_fn():
            return torch.matmul(compute_a, compute_b.T)

        # Warmup
        for _ in range(10):
            t = comm_tensor.clone()
            dist.all_reduce(t)
            compute_fn()
        torch.cuda.synchronize()
        dist.barrier()

        # Measure sequential
        seq_time = sequential_allreduce_and_compute(
            comm_tensor, compute_fn, num_iterations=50
        )

        # Measure overlapped
        overlap_time = overlapped_allreduce_and_compute(
            comm_tensor, compute_fn, comm_stream, compute_stream, num_iterations=50
        )

        speedup = seq_time / overlap_time

        if rank == 0:
            print(f"{desc}:")
            print(f"  Comm tensor: {comm_size} ({comm_mb:.1f} MB)")
            print(f"  Compute: {compute_size[0]}x{compute_size[1]} matmul")
            print(f"  Sequential time: {seq_time:.2f} ms")
            print(f"  Overlapped time: {overlap_time:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Time saved: {seq_time - overlap_time:.2f} ms ({(1 - overlap_time/seq_time)*100:.1f}%)")
            print()

            results.append({
                "desc": desc,
                "comm_mb": comm_mb,
                "sequential_ms": seq_time,
                "overlap_ms": overlap_time,
                "speedup": speedup,
            })

    # Summary
    if rank == 0:
        print("="*70)
        print("Summary")
        print("="*70)
        print(f"{'Configuration':<40} {'Speedup':>10} {'Saved':>10}")
        print("-"*60)
        for r in results:
            saved_pct = (1 - r["overlap_ms"]/r["sequential_ms"]) * 100
            print(f"{r['desc']:<40} {r['speedup']:>9.2f}x {saved_pct:>9.1f}%")

        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        print("-"*60)
        print(f"{'Average':<40} {avg_speedup:>9.2f}x")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
