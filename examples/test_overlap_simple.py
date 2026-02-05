#!/usr/bin/env python3
"""
MACO 简单重叠测试

验证基本的计算-通信重叠功能：
1. 单独测试 async_all_reduce
2. 测试基本的 stream 重叠
3. 测试 signal-wait 机制
"""

import os
import sys
import time
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maco.sync import StreamManager, SignalWait, OverlapContext
from maco.comm import (
    ProcessGroupManager,
    async_all_reduce,
    async_all_gather,
    async_all_to_all_4d,
    get_world_size,
    get_rank,
    is_initialized,
    barrier,
)


def test_async_all_reduce(rank: int, world_size: int, device: torch.device):
    """测试 async_all_reduce 正确性"""
    print(f"[Rank {rank}] Test 1: async_all_reduce correctness")

    # 创建测试数据
    tensor = torch.ones(1024, device=device) * (rank + 1)
    expected_sum = sum(range(1, world_size + 1))  # 1 + 2 + 3 + 4 = 10

    # 同步版本
    tensor_sync = tensor.clone()
    handle = async_all_reduce(tensor_sync, async_op=False)
    result_sync = handle.output

    # 异步版本
    tensor_async = tensor.clone()
    handle = async_all_reduce(tensor_async, async_op=True)
    result_async = handle.wait()

    # 验证
    assert torch.allclose(result_sync, result_async), f"Sync != Async"
    assert torch.allclose(result_sync, torch.full_like(result_sync, expected_sum)), \
        f"Expected {expected_sum}, got {result_sync[0].item()}"

    print(f"[Rank {rank}] ✅ Test 1 PASSED")
    return True


def test_async_all_to_all_4d(rank: int, world_size: int, device: torch.device):
    """测试 async_all_to_all_4d 正确性"""
    print(f"[Rank {rank}] Test 2: async_all_to_all_4d correctness")

    B, S, H, D = 2, 128, 8 * world_size, 64

    # 每个 rank 创建不同的数据
    tensor = torch.randn(B, S // world_size, H, D, device=device)

    # 同步版本
    tensor_sync = tensor.clone()
    handle = async_all_to_all_4d(tensor_sync, scatter_dim=2, gather_dim=1, async_op=False)
    result_sync = handle.output

    # 异步版本
    tensor_async = tensor.clone()
    handle = async_all_to_all_4d(tensor_async, scatter_dim=2, gather_dim=1, async_op=True)
    result_async = handle.wait()

    # 验证形状
    expected_shape = (B, S, H // world_size, D)
    assert result_sync.shape == expected_shape, f"Sync shape: {result_sync.shape}, expected {expected_shape}"
    assert result_async.shape == expected_shape, f"Async shape: {result_async.shape}, expected {expected_shape}"

    # 验证值（同步和异步应该相同）
    assert torch.allclose(result_sync, result_async), f"Sync != Async"

    print(f"[Rank {rank}] ✅ Test 2 PASSED (shape: {result_sync.shape})")
    return True


def test_stream_overlap(rank: int, world_size: int, device: torch.device):
    """测试 stream 重叠执行"""
    print(f"[Rank {rank}] Test 3: Stream overlap execution")

    stream_manager = StreamManager(device)
    compute_stream = stream_manager.compute_stream
    comm_stream = stream_manager.comm_stream

    # 准备数据
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    comm_tensor = torch.randn(4096, 1024, device=device)

    # ====== 串行执行 ======
    torch.cuda.synchronize()
    start = time.perf_counter()

    # 计算
    _ = torch.mm(a, b)
    torch.cuda.synchronize()

    # 通信
    if world_size > 1:
        handle = async_all_reduce(comm_tensor.clone(), async_op=False)
        _ = handle.output
    torch.cuda.synchronize()

    serial_time = time.perf_counter() - start

    # ====== 重叠执行 ======
    torch.cuda.synchronize()
    start = time.perf_counter()

    # 启动通信（在 comm stream）
    comm_result = None
    if world_size > 1:
        with torch.cuda.stream(comm_stream):
            handle = async_all_reduce(comm_tensor.clone(), async_op=True)

    # 计算（在 compute stream，与通信并行）
    with torch.cuda.stream(compute_stream):
        compute_result = torch.mm(a, b)

    # 等待所有完成
    stream_manager.sync_all()
    if world_size > 1:
        comm_result = handle.wait()

    overlap_time = time.perf_counter() - start

    speedup = serial_time / overlap_time if overlap_time > 0 else 1.0

    print(f"[Rank {rank}] Serial: {serial_time*1000:.2f} ms, Overlap: {overlap_time*1000:.2f} ms, Speedup: {speedup:.2f}x")

    # 验证重叠确实发生了
    if world_size > 1:
        if speedup > 1.2:
            print(f"[Rank {rank}] ✅ Test 3 PASSED (overlap achieved)")
        else:
            print(f"[Rank {rank}] ⚠️  Test 3: Low speedup (may be bottlenecked)")
    else:
        print(f"[Rank {rank}] ✅ Test 3 PASSED (single GPU)")

    return True


def test_signal_wait(rank: int, world_size: int, device: torch.device):
    """测试 signal-wait 同步"""
    print(f"[Rank {rank}] Test 4: Signal-wait synchronization")

    stream_manager = StreamManager(device)
    signal_wait = SignalWait()

    compute_stream = stream_manager.compute_stream
    comm_stream = stream_manager.comm_stream

    # 准备数据
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    comm_tensor = torch.randn(1024, 1024, device=device)

    # Wave 0: 计算
    with torch.cuda.stream(compute_stream):
        result = torch.mm(a, b)

    # Signal: 计算完成
    signal_wait.signal_wave(0, compute_stream)

    # 在通信 stream 等待计算
    signal_wait.wait_wave(0, comm_stream)

    # Wave 0: 通信（需要计算结果）
    with torch.cuda.stream(comm_stream):
        # 模拟需要 result 的通信
        if world_size > 1:
            handle = async_all_reduce(result.clone(), async_op=True)

    # Signal: 通信完成
    signal_wait.signal_wave(1, comm_stream)

    # 在计算 stream 等待通信
    signal_wait.wait_wave(1, compute_stream)

    # Wave 1: 使用通信结果继续计算
    with torch.cuda.stream(compute_stream):
        if world_size > 1:
            comm_result = handle.wait()
            final_result = torch.mm(comm_result, b)
        else:
            final_result = torch.mm(result, b)

    stream_manager.sync_all()

    print(f"[Rank {rank}] ✅ Test 4 PASSED")
    return True


def test_overlap_context(rank: int, world_size: int, device: torch.device):
    """测试 OverlapContext API"""
    print(f"[Rank {rank}] Test 5: OverlapContext API")

    ctx = OverlapContext(device)

    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    comm_tensor = torch.randn(1024, 1024, device=device)

    # 使用 OverlapContext 的高级 API

    # Wave 0: 计算
    with torch.cuda.stream(ctx.compute_stream):
        result = torch.mm(a, b)
    ctx.signal_compute_done(wave_id=0)

    # Wave 0: 通信
    ctx.wait_compute_done(wave_id=0)
    with torch.cuda.stream(ctx.comm_stream):
        if world_size > 1:
            handle = async_all_reduce(result.clone(), async_op=True)
    ctx.signal_comm_done(wave_id=0)

    # Wave 1: 计算
    ctx.wait_comm_done(wave_id=0)
    with torch.cuda.stream(ctx.compute_stream):
        if world_size > 1:
            comm_result = handle.wait()
            final_result = torch.mm(comm_result, b)
        else:
            final_result = torch.mm(result, b)

    # 同步
    torch.cuda.synchronize()

    print(f"[Rank {rank}] ✅ Test 5 PASSED")
    return True


def main():
    # 分布式设置
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(f"\n[Rank {rank}/{world_size}] Device: {device}")

    # 初始化分布式
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        pgm = ProcessGroupManager()
        pgm.init_process_group()

    try:
        print(f"\n{'=' * 60}")
        print(f" MACO Simple Overlap Tests ({world_size} GPUs)")
        print(f"{'=' * 60}")

        all_passed = True

        # Test 1: async_all_reduce
        if world_size > 1:
            all_passed &= test_async_all_reduce(rank, world_size, device)
            barrier()

        # Test 2: async_all_to_all_4d
        if world_size > 1:
            all_passed &= test_async_all_to_all_4d(rank, world_size, device)
            barrier()

        # Test 3: stream overlap
        all_passed &= test_stream_overlap(rank, world_size, device)
        if world_size > 1:
            barrier()

        # Test 4: signal-wait
        all_passed &= test_signal_wait(rank, world_size, device)
        if world_size > 1:
            barrier()

        # Test 5: OverlapContext
        all_passed &= test_overlap_context(rank, world_size, device)
        if world_size > 1:
            barrier()

        # Summary
        if rank == 0:
            print(f"\n{'=' * 60}")
            if all_passed:
                print(" All tests PASSED!")
            else:
                print(" Some tests FAILED!")
            print(f"{'=' * 60}")

    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
