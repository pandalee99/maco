#!/usr/bin/env python3
"""
MACO NCCL Benchmark - 2 GPU 测试

测试 NCCL 通信原语的性能，用于评估 MACO 优化效果。

运行方式:
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 examples/benchmark_nccl_2gpu.py

测试内容:
    1. AllReduce 性能
    2. All-to-All 性能
    3. 计算-通信重叠效果
"""

import os
import time
import torch
import torch.distributed as dist
from typing import Callable, Tuple


def init_distributed():
    """初始化分布式环境"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def benchmark_fn(
    fn: Callable,
    warmup: int = 10,
    iterations: int = 100,
    sync: bool = True
) -> float:
    """
    测量函数执行时间

    Returns:
        平均执行时间 (ms)
    """
    # Warmup
    for _ in range(warmup):
        fn()
        if sync:
            torch.cuda.synchronize()

    # Barrier
    if dist.is_initialized():
        dist.barrier()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
        if sync:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000  # ms


def benchmark_allreduce(rank: int, world_size: int):
    """测试 AllReduce 性能"""
    if rank == 0:
        print("\n" + "=" * 60)
        print("AllReduce Benchmark")
        print("=" * 60)
        print(f"{'Size':<15} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20}")
        print("-" * 50)

    sizes = [
        (1024, "1 KB"),
        (1024 * 1024, "4 MB"),
        (4 * 1024 * 1024, "16 MB"),
        (16 * 1024 * 1024, "64 MB"),
        (64 * 1024 * 1024, "256 MB"),
    ]

    for numel, name in sizes:
        tensor = torch.randn(numel, device="cuda", dtype=torch.float32)
        size_bytes = tensor.numel() * tensor.element_size()

        def fn():
            dist.all_reduce(tensor)

        time_ms = benchmark_fn(fn)

        # AllReduce 传输 2 * (world_size - 1) / world_size * size
        effective_bytes = 2 * size_bytes * (world_size - 1) / world_size
        bandwidth = effective_bytes / (time_ms / 1000) / 1e9

        if rank == 0:
            print(f"{name:<15} {time_ms:<15.3f} {bandwidth:<20.2f}")


def benchmark_all_to_all(rank: int, world_size: int):
    """测试 All-to-All 性能"""
    if rank == 0:
        print("\n" + "=" * 60)
        print("All-to-All Benchmark")
        print("=" * 60)
        print(f"{'Size':<15} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20}")
        print("-" * 50)

    # 模拟 Sequence Parallel 的 all_to_all
    configs = [
        # (batch, seq_per_gpu, heads, head_dim, name)
        (1, 256, 8, 128, "Small"),
        (1, 512, 16, 128, "Medium"),
        (1, 1024, 32, 128, "Large"),
        (2, 1024, 32, 128, "XLarge"),
    ]

    for batch, seq, heads, dim, name in configs:
        # 确保可以被 world_size 整除
        heads = (heads // world_size) * world_size

        input_tensor = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.float32)

        # 准备 all_to_all 的输入输出 - 必须是连续的
        input_list = [x.contiguous() for x in input_tensor.chunk(world_size, dim=2)]
        output_list = [torch.empty_like(x) for x in input_list]

        def fn():
            dist.all_to_all(output_list, input_list)

        time_ms = benchmark_fn(fn)

        size_bytes = input_tensor.numel() * input_tensor.element_size()
        # All-to-all 传输 (world_size - 1) / world_size * size
        effective_bytes = size_bytes * (world_size - 1) / world_size
        bandwidth = effective_bytes / (time_ms / 1000) / 1e9

        if rank == 0:
            print(f"{name:<15} {time_ms:<15.3f} {bandwidth:<20.2f}")


def benchmark_overlap(rank: int, world_size: int):
    """测试计算-通信重叠效果"""
    if rank == 0:
        print("\n" + "=" * 60)
        print("Compute-Communication Overlap Benchmark")
        print("=" * 60)

    # 创建两个 CUDA stream
    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    # 测试配置
    comm_size = 4 * 1024 * 1024  # 16 MB
    compute_size = 2048  # matmul size

    comm_tensor = torch.randn(comm_size, device="cuda", dtype=torch.float32)
    a = torch.randn(compute_size, compute_size, device="cuda", dtype=torch.float32)
    b = torch.randn(compute_size, compute_size, device="cuda", dtype=torch.float32)

    # 1. Sequential baseline
    def sequential():
        dist.all_reduce(comm_tensor)
        c = torch.matmul(a, b)
        return c

    seq_time = benchmark_fn(sequential)

    # 2. Overlapped execution
    def overlapped():
        # 通信在 comm_stream
        with torch.cuda.stream(comm_stream):
            dist.all_reduce(comm_tensor)

        # 计算在 compute_stream
        with torch.cuda.stream(compute_stream):
            c = torch.matmul(a, b)

        # 同步
        comm_stream.synchronize()
        compute_stream.synchronize()
        return c

    overlap_time = benchmark_fn(overlapped)

    # 3. 单独测量通信时间
    def comm_only():
        dist.all_reduce(comm_tensor)

    comm_time = benchmark_fn(comm_only)

    # 4. 单独测量计算时间
    def compute_only():
        c = torch.matmul(a, b)
        return c

    compute_time = benchmark_fn(compute_only)

    if rank == 0:
        print(f"\n通信时间 (AllReduce 16MB):     {comm_time:.3f} ms")
        print(f"计算时间 (MatMul {compute_size}x{compute_size}): {compute_time:.3f} ms")
        print(f"\n顺序执行时间:                  {seq_time:.3f} ms")
        print(f"重叠执行时间:                  {overlap_time:.3f} ms")
        print(f"\n理论最优时间:                  {max(comm_time, compute_time):.3f} ms")
        print(f"实际加速比:                    {seq_time / overlap_time:.2f}x")
        print(f"重叠效率:                      {(comm_time + compute_time - overlap_time) / min(comm_time, compute_time) * 100:.1f}%")


def benchmark_matmul(rank: int):
    """测试单 GPU MatMul 性能作为参考"""
    if rank == 0:
        print("\n" + "=" * 60)
        print("MatMul Performance Reference")
        print("=" * 60)
        print(f"{'Size':<15} {'Time (ms)':<15} {'TFLOPS':<15}")
        print("-" * 45)

    sizes = [512, 1024, 2048, 4096]

    for size in sizes:
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)

        def fn():
            return torch.matmul(a, b)

        time_ms = benchmark_fn(fn)

        # FLOPs = 2 * M * N * K
        flops = 2 * size * size * size
        tflops = flops / (time_ms / 1000) / 1e12

        if rank == 0:
            print(f"{size}x{size:<10} {time_ms:<15.3f} {tflops:<15.2f}")


def main():
    rank, world_size, local_rank = init_distributed()

    if rank == 0:
        print("=" * 60)
        print("MACO NCCL Benchmark")
        print("=" * 60)
        print(f"World Size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name(local_rank)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # 运行各项测试
    benchmark_matmul(rank)
    benchmark_allreduce(rank, world_size)
    benchmark_all_to_all(rank, world_size)
    benchmark_overlap(rank, world_size)

    if rank == 0:
        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
