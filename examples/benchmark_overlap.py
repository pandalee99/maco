#!/usr/bin/env python3
"""
MACO 性能基准测试

测试内容:
1. 通信延迟基准
2. 计算-通信重叠效率
3. 不同配置下的 speedup
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass
from typing import List, Tuple, Dict

# 添加 maco 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maco.sync import StreamManager, OverlapContext
from maco.comm import (
    ProcessGroupManager,
    async_all_reduce,
    async_all_gather,
    async_all_to_all,
    async_all_to_all_4d,
    get_world_size,
    get_rank,
    is_initialized,
    barrier,
)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    baseline_ms: float
    overlapped_ms: float
    speedup: float
    overlap_efficiency: float  # 理论加速 vs 实际加速


def benchmark_comm_latency(
    rank: int,
    world_size: int,
    device: torch.device,
    tensor_sizes: List[Tuple[int, ...]],
    num_iters: int = 100,
) -> Dict[str, float]:
    """
    测试通信延迟

    Returns:
        各操作的平均延迟 (ms)
    """
    results = {}

    for size in tensor_sizes:
        size_str = "x".join(map(str, size))
        tensor = torch.randn(*size, device=device)

        # Warmup
        for _ in range(10):
            handle = async_all_reduce(tensor.clone(), async_op=True)
            handle.wait()

        torch.cuda.synchronize()
        if world_size > 1:
            barrier()

        # AllReduce
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            t = tensor.clone()
            handle = async_all_reduce(t, async_op=True)
            handle.wait()
        torch.cuda.synchronize()
        allreduce_time = (time.perf_counter() - start) / num_iters * 1000
        results[f"AllReduce_{size_str}"] = allreduce_time

        # AllGather
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            handle = async_all_gather(tensor, async_op=True)
            handle.wait()
        torch.cuda.synchronize()
        allgather_time = (time.perf_counter() - start) / num_iters * 1000
        results[f"AllGather_{size_str}"] = allgather_time

        # AllToAll (if 4D)
        if len(size) == 4:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_iters):
                handle = async_all_to_all_4d(tensor, scatter_dim=2, gather_dim=1, async_op=True)
                handle.wait()
            torch.cuda.synchronize()
            all_to_all_time = (time.perf_counter() - start) / num_iters * 1000
            results[f"AllToAll4D_{size_str}"] = all_to_all_time

    return results


def benchmark_matmul(
    device: torch.device,
    sizes: List[Tuple[int, int, int]],
    num_iters: int = 100,
) -> Dict[str, float]:
    """
    测试矩阵乘法延迟

    Args:
        sizes: List of (M, K, N)

    Returns:
        各配置的平均延迟 (ms)
    """
    results = {}

    for M, K, N in sizes:
        a = torch.randn(M, K, device=device)
        b = torch.randn(K, N, device=device)

        # Warmup
        for _ in range(10):
            _ = torch.mm(a, b)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters * 1000

        results[f"MatMul_{M}x{K}x{N}"] = elapsed

    return results


def benchmark_overlap_efficiency(
    rank: int,
    world_size: int,
    device: torch.device,
    compute_sizes: List[Tuple[int, int, int]],
    comm_size: Tuple[int, ...],
    num_iters: int = 50,
) -> List[BenchmarkResult]:
    """
    测试计算-通信重叠效率

    对于每个计算大小，测量:
    1. 串行执行时间 (计算 + 通信)
    2. 重叠执行时间
    3. 计算重叠效率
    """
    results = []
    stream_manager = StreamManager(device)
    overlap_context = OverlapContext(device)

    comm_tensor = torch.randn(*comm_size, device=device)

    for M, K, N in compute_sizes:
        a = torch.randn(M, K, device=device)
        b = torch.randn(K, N, device=device)

        # ====== 测量单独的计算时间 ======
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        compute_time = (time.perf_counter() - start) / num_iters * 1000

        # ====== 测量单独的通信时间 ======
        if world_size > 1:
            torch.cuda.synchronize()
            barrier()
            start = time.perf_counter()
            for _ in range(num_iters):
                t = comm_tensor.clone()
                handle = async_all_reduce(t, async_op=True)
                handle.wait()
            torch.cuda.synchronize()
            comm_time = (time.perf_counter() - start) / num_iters * 1000
        else:
            comm_time = 0.1  # 模拟通信时间

        # ====== 串行执行 ======
        baseline_time = compute_time + comm_time

        # ====== 重叠执行 ======
        compute_stream = stream_manager.compute_stream
        comm_stream = stream_manager.comm_stream

        # Warmup
        for _ in range(5):
            with torch.cuda.stream(compute_stream):
                _ = torch.mm(a, b)
            overlap_context.signal_compute_done(wave_id=0)

            if world_size > 1:
                overlap_context.wait_compute_done(wave_id=0)
                with torch.cuda.stream(comm_stream):
                    t = comm_tensor.clone()
                    handle = async_all_reduce(t, async_op=True)
                overlap_context.signal_comm_done(wave_id=0)
                overlap_context.wait_comm_done(wave_id=0)
                handle.wait()

            stream_manager.sync_all()

        torch.cuda.synchronize()
        if world_size > 1:
            barrier()

        start = time.perf_counter()
        for _ in range(num_iters):
            # 计算
            with torch.cuda.stream(compute_stream):
                _ = torch.mm(a, b)
            overlap_context.signal_compute_done(wave_id=0)

            # 通信（重叠）
            if world_size > 1:
                overlap_context.wait_compute_done(wave_id=0)
                with torch.cuda.stream(comm_stream):
                    t = comm_tensor.clone()
                    handle = async_all_reduce(t, async_op=True)
                overlap_context.signal_comm_done(wave_id=0)

            # 同步
            stream_manager.sync_all()
            if world_size > 1:
                handle.wait()

        torch.cuda.synchronize()
        overlapped_time = (time.perf_counter() - start) / num_iters * 1000

        # 计算 speedup 和效率
        speedup = baseline_time / overlapped_time if overlapped_time > 0 else 1.0

        # 理论最佳 speedup = baseline / max(compute, comm)
        ideal_overlapped = max(compute_time, comm_time)
        ideal_speedup = baseline_time / ideal_overlapped if ideal_overlapped > 0 else 1.0
        efficiency = (speedup - 1) / (ideal_speedup - 1) * 100 if ideal_speedup > 1 else 100

        results.append(BenchmarkResult(
            name=f"MatMul_{M}x{K}x{N}",
            baseline_ms=baseline_time,
            overlapped_ms=overlapped_time,
            speedup=speedup,
            overlap_efficiency=efficiency,
        ))

    return results


def print_results(rank: int, title: str, results: Dict[str, float]):
    """打印结果表格"""
    if rank != 0:
        return

    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    print(f"{'Operation':<35} {'Latency (ms)':>15}")
    print(f"{'-' * 50}")
    for name, latency in results.items():
        print(f"{name:<35} {latency:>15.3f}")


def print_overlap_results(rank: int, results: List[BenchmarkResult]):
    """打印重叠效率结果"""
    if rank != 0:
        return

    print(f"\n{'=' * 70}")
    print(f" Overlap Efficiency")
    print(f"{'=' * 70}")
    print(f"{'Operation':<25} {'Baseline':>10} {'Overlapped':>12} {'Speedup':>10} {'Efficiency':>12}")
    print(f"{'-' * 70}")
    for r in results:
        print(f"{r.name:<25} {r.baseline_ms:>10.2f} {r.overlapped_ms:>12.2f} {r.speedup:>10.2f}x {r.overlap_efficiency:>11.1f}%")


def main():
    parser = argparse.ArgumentParser(description="MACO Overlap Benchmark")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    parser.add_argument("--skip-comm", action="store_true", help="Skip communication benchmarks")
    args = parser.parse_args()

    # 分布式设置
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"\n{'#' * 60}")
        print(f"# MACO Performance Benchmark")
        print(f"# World Size: {world_size}")
        print(f"# Device: {device}")
        print(f"# Iterations: {args.iters}")
        print(f"{'#' * 60}")

    # 初始化分布式
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        pgm = ProcessGroupManager()
        pgm.init_process_group()

    try:
        # 1. 矩阵乘法基准
        matmul_sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (512, 4096, 512),   # 类似 MLP 投影
            (1024, 4096, 1024),
        ]
        matmul_results = benchmark_matmul(device, matmul_sizes, num_iters=args.iters)
        print_results(rank, "Matrix Multiplication Latency", matmul_results)

        if not args.skip_comm and world_size > 1:
            # 2. 通信延迟基准
            tensor_sizes = [
                (1024, 1024),
                (4096, 1024),
                (2, 128, 8, 64),     # 小 attention
                (4, 512, 16, 64),    # 中 attention
                (8, 1024, 32, 64),   # 大 attention
            ]
            comm_results = benchmark_comm_latency(
                rank, world_size, device, tensor_sizes, num_iters=args.iters
            )
            print_results(rank, f"Communication Latency ({world_size} GPUs)", comm_results)

            # 3. 重叠效率基准
            compute_sizes = [
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (512, 4096, 512),
            ]
            comm_size = (4096, 1024)

            overlap_results = benchmark_overlap_efficiency(
                rank, world_size, device,
                compute_sizes, comm_size,
                num_iters=args.iters,
            )
            print_overlap_results(rank, overlap_results)

        elif world_size == 1:
            if rank == 0:
                print("\n⚠️  Single GPU mode: skipping communication benchmarks")
                print("Run with torchrun --nproc_per_node=N for multi-GPU benchmarks")

        # Summary
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(" Benchmark Complete")
            print(f"{'=' * 60}")

    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
