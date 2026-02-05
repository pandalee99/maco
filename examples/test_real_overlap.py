#!/usr/bin/env python3
"""
验证真正的计算-通信重叠是否有性能提升

场景：模拟 self-forcing 中的模式
- 计算：matmul
- 通信：all-reduce（梯度同步）
"""

import os
import sys
import time
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maco.sync import StreamManager
from maco.comm import (
    ProcessGroupManager,
    async_all_reduce,
    get_world_size,
    get_rank,
    barrier,
)


def measure_serial(a, b, comm_tensor, num_iters=20):
    """串行执行：先计算，再通信"""
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iters):
        # 计算
        result = torch.mm(a, b)
        torch.cuda.synchronize()

        # 通信
        t = comm_tensor.clone()
        handle = async_all_reduce(t, async_op=False)
        _ = handle.output
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / num_iters * 1000
    return elapsed


def measure_overlap(a, b, comm_tensor, stream_manager, num_iters=20):
    """重叠执行：计算和通信并行"""
    compute_stream = stream_manager.compute_stream
    comm_stream = stream_manager.comm_stream

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iters):
        # 在 compute stream 启动计算
        with torch.cuda.stream(compute_stream):
            result = torch.mm(a, b)

        # 同时在 comm stream 启动通信
        with torch.cuda.stream(comm_stream):
            t = comm_tensor.clone()
            handle = async_all_reduce(t, async_op=True)

        # 等待两者完成
        stream_manager.sync_all()
        _ = handle.wait()

    elapsed = (time.perf_counter() - start) / num_iters * 1000
    return elapsed


def measure_compute_only(a, b, num_iters=20):
    """仅计算"""
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iters):
        result = torch.mm(a, b)
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / num_iters * 1000
    return elapsed


def measure_comm_only(comm_tensor, num_iters=20):
    """仅通信"""
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iters):
        t = comm_tensor.clone()
        handle = async_all_reduce(t, async_op=False)
        _ = handle.output
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / num_iters * 1000
    return elapsed


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        print("CUDA not available")
        return

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        pgm = ProcessGroupManager()
        pgm.init_process_group()

    try:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f" Real Overlap Performance Test ({world_size} GPUs)")
            print(f"{'='*70}")

        stream_manager = StreamManager(device)

        # 测试不同大小
        test_configs = [
            # (matmul_size, comm_size, description)
            ((2048, 2048, 2048), (2048, 2048), "Medium compute, medium comm"),
            ((4096, 4096, 4096), (1024, 1024), "Large compute, small comm"),
            ((1024, 1024, 1024), (4096, 4096), "Small compute, large comm"),
            ((2048, 4096, 2048), (2048, 4096), "MLP-like pattern"),
        ]

        for (M, K, N), (C1, C2), desc in test_configs:
            a = torch.randn(M, K, device=device)
            b = torch.randn(K, N, device=device)
            comm_tensor = torch.randn(C1, C2, device=device)

            # Warmup
            for _ in range(5):
                _ = torch.mm(a, b)
                if world_size > 1:
                    handle = async_all_reduce(comm_tensor.clone(), async_op=False)

            torch.cuda.synchronize()
            if world_size > 1:
                barrier()

            # 测量
            compute_time = measure_compute_only(a, b)

            if world_size > 1:
                comm_time = measure_comm_only(comm_tensor)
                serial_time = measure_serial(a, b, comm_tensor)
                overlap_time = measure_overlap(a, b, comm_tensor, stream_manager)

                speedup = serial_time / overlap_time if overlap_time > 0 else 1.0
                theoretical_best = max(compute_time, comm_time)
                actual_overlap = serial_time - overlap_time
                expected_overlap = min(compute_time, comm_time)
                efficiency = actual_overlap / expected_overlap * 100 if expected_overlap > 0 else 0
            else:
                comm_time = 0
                serial_time = compute_time
                overlap_time = compute_time
                speedup = 1.0
                efficiency = 0

            if rank == 0:
                print(f"\n{desc}:")
                print(f"  MatMul: {M}x{K}x{N}, Comm: {C1}x{C2}")
                print(f"  Compute only: {compute_time:.2f} ms")
                if world_size > 1:
                    print(f"  Comm only:    {comm_time:.2f} ms")
                    print(f"  Serial:       {serial_time:.2f} ms")
                    print(f"  Overlap:      {overlap_time:.2f} ms")
                    print(f"  Speedup:      {speedup:.2f}x")
                    print(f"  Overlap eff:  {efficiency:.1f}%")

        if rank == 0:
            print(f"\n{'='*70}")
            print(" Test Complete")
            print(f"{'='*70}")

    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
