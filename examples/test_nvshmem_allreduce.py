#!/usr/bin/env python3
"""
MACO NVSHMEM AllReduce 测试脚本

测试 NVSHMEM 实现的 allreduce 和 all_to_all_4D 操作，
并与 NCCL 进行性能对比。

运行方式 (需要 MPI):
    # 8 GPU 测试
    mpirun -np 8 python examples/test_nvshmem_allreduce.py

    # 4 GPU 测试
    mpirun -np 4 python examples/test_nvshmem_allreduce.py

    # 带 NCCL 对比
    mpirun -np 8 python examples/test_nvshmem_allreduce.py --compare-nccl

环境要求:
    - NVSHMEM extension 已编译安装
    - MPI 环境 (mpirun)
    - CUDA capable GPUs
"""

import argparse
import time
import torch
import os


def test_nvshmem_available():
    """测试 NVSHMEM 扩展是否可用"""
    print("=" * 60)
    print("Testing NVSHMEM availability...")

    try:
        from maco.comm.nvshmem_backend import is_nvshmem_available, init_nvshmem
        available = is_nvshmem_available()
        print(f"  NVSHMEM extension available: {available}")

        if available:
            success = init_nvshmem()
            print(f"  NVSHMEM initialization: {'SUCCESS' if success else 'FAILED'}")
            return success
        else:
            print("  Please build NVSHMEM extension first:")
            print("    cd /mini_mirage/maco && python maco/csrc/setup.py install")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_allreduce_correctness():
    """测试 allreduce 正确性"""
    print("\n" + "=" * 60)
    print("Testing AllReduce correctness...")

    from maco.comm.nvshmem_backend import (
        init_nvshmem, nvshmem_allreduce, nvshmem_my_pe, nvshmem_n_pes
    )

    init_nvshmem()
    pe = nvshmem_my_pe()
    n_pes = nvshmem_n_pes()

    # 每个 PE 创建值为 pe 的张量
    x = torch.full((1024,), float(pe), device="cuda")

    # AllReduce sum
    result = nvshmem_allreduce(x, op="sum")

    # 期望结果: sum(0, 1, ..., n_pes-1) = n_pes * (n_pes - 1) / 2
    expected = n_pes * (n_pes - 1) / 2
    actual = result[0].item()

    if pe == 0:
        print(f"  n_pes: {n_pes}")
        print(f"  Expected sum: {expected}")
        print(f"  Actual sum: {actual}")
        print(f"  Correctness: {'PASS' if abs(actual - expected) < 1e-5 else 'FAIL'}")


def test_allreduce_performance(compare_nccl: bool = False):
    """测试 allreduce 性能"""
    print("\n" + "=" * 60)
    print("Testing AllReduce performance...")

    from maco.comm.nvshmem_backend import (
        init_nvshmem, nvshmem_allreduce, nvshmem_my_pe, nvshmem_n_pes, nvshmem_barrier
    )

    init_nvshmem()
    pe = nvshmem_my_pe()
    n_pes = nvshmem_n_pes()

    # 如果需要对比 NCCL，初始化 torch.distributed
    if compare_nccl:
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ["RANK"] = str(pe)
            os.environ["WORLD_SIZE"] = str(n_pes)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group(backend="nccl")

    # 测试不同大小
    sizes = [
        (1024, "1KB"),
        (1024 * 1024, "4MB"),
        (10 * 1024 * 1024, "40MB"),
        (50 * 1024 * 1024, "200MB"),
    ]

    warmup_iters = 10
    benchmark_iters = 100

    if pe == 0:
        print(f"\n  {'Size':<15} {'NVSHMEM (ms)':<15}", end="")
        if compare_nccl:
            print(f"{'NCCL (ms)':<15} {'Speedup':<10}", end="")
        print()
        print("  " + "-" * 55)

    for numel, size_str in sizes:
        tensor = torch.randn(numel, device="cuda", dtype=torch.bfloat16)

        # Warmup NVSHMEM
        for _ in range(warmup_iters):
            _ = nvshmem_allreduce(tensor.clone())
        nvshmem_barrier()
        torch.cuda.synchronize()

        # Benchmark NVSHMEM
        t0 = time.perf_counter()
        for _ in range(benchmark_iters):
            _ = nvshmem_allreduce(tensor.clone())
        torch.cuda.synchronize()
        nvshmem_time = (time.perf_counter() - t0) / benchmark_iters * 1000

        nccl_time = None
        if compare_nccl:
            import torch.distributed as dist

            # Warmup NCCL
            for _ in range(warmup_iters):
                t = tensor.clone()
                dist.all_reduce(t)
            dist.barrier()
            torch.cuda.synchronize()

            # Benchmark NCCL
            t0 = time.perf_counter()
            for _ in range(benchmark_iters):
                t = tensor.clone()
                dist.all_reduce(t)
            torch.cuda.synchronize()
            nccl_time = (time.perf_counter() - t0) / benchmark_iters * 1000

        if pe == 0:
            print(f"  {size_str:<15} {nvshmem_time:<15.3f}", end="")
            if compare_nccl and nccl_time:
                speedup = nccl_time / nvshmem_time
                print(f"{nccl_time:<15.3f} {speedup:<10.2f}x", end="")
            print()


def test_all_to_all_4D_correctness():
    """测试 all_to_all_4D 正确性"""
    print("\n" + "=" * 60)
    print("Testing All-to-All 4D correctness...")

    from maco.comm.nvshmem_backend import (
        init_nvshmem, nvshmem_all_to_all_4D, nvshmem_my_pe, nvshmem_n_pes
    )

    init_nvshmem()
    pe = nvshmem_my_pe()
    n_pes = nvshmem_n_pes()

    # 创建测试张量
    # Input shape: [B, S/P, H, D] = [1, 256, 32, 128]
    # 假设 P=8, S=2048, H=32
    batch = 1
    seq_per_pe = 256
    heads = 32
    head_dim = 128

    # 每个 PE 的输入用 pe * 1000 填充，便于追踪
    x = torch.full((batch, seq_per_pe, heads, head_dim),
                   float(pe * 1000), device="cuda", dtype=torch.float32)

    # 执行 all_to_all_4D (SP -> HP)
    # scatter_dim=1 (seq), gather_dim=2 (heads)
    y = nvshmem_all_to_all_4D(x, scatter_dim=1, gather_dim=2)

    if pe == 0:
        print(f"  n_pes: {n_pes}")
        print(f"  Input shape: {list(x.shape)}")
        print(f"  Output shape: {list(y.shape)}")
        print(f"  Expected output shape: [{batch}, {seq_per_pe}, {heads // n_pes}, {head_dim}]")

        # 检查形状
        expected_shape = [batch, seq_per_pe, heads // n_pes, head_dim]
        shape_correct = list(y.shape) == expected_shape
        print(f"  Shape correctness: {'PASS' if shape_correct else 'FAIL'}")


def test_all_to_all_4D_performance(compare_nccl: bool = False):
    """测试 all_to_all_4D 性能"""
    print("\n" + "=" * 60)
    print("Testing All-to-All 4D performance...")

    from maco.comm.nvshmem_backend import (
        init_nvshmem, nvshmem_all_to_all_4D, nvshmem_my_pe, nvshmem_n_pes, nvshmem_barrier
    )

    init_nvshmem()
    pe = nvshmem_my_pe()
    n_pes = nvshmem_n_pes()

    # 如果需要对比 NCCL
    if compare_nccl:
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ["RANK"] = str(pe)
            os.environ["WORLD_SIZE"] = str(n_pes)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group(backend="nccl")

    # 测试配置 (模拟 Self-Forcing-14B)
    configs = [
        # (batch, seq_per_pe, heads, head_dim, name)
        (1, 256, 32, 128, "Small (256 seq)"),
        (1, 512, 32, 128, "Medium (512 seq)"),
        (1, 1024, 32, 128, "Large (1024 seq)"),
    ]

    warmup_iters = 10
    benchmark_iters = 100

    if pe == 0:
        print(f"\n  {'Config':<25} {'NVSHMEM (ms)':<15}", end="")
        if compare_nccl:
            print(f"{'NCCL (ms)':<15} {'Speedup':<10}", end="")
        print()
        print("  " + "-" * 65)

    for batch, seq_per_pe, heads, head_dim, name in configs:
        tensor = torch.randn(
            batch, seq_per_pe, heads, head_dim,
            device="cuda", dtype=torch.bfloat16
        )

        # Warmup NVSHMEM
        for _ in range(warmup_iters):
            _ = nvshmem_all_to_all_4D(tensor.clone(), scatter_dim=1, gather_dim=2)
        nvshmem_barrier()
        torch.cuda.synchronize()

        # Benchmark NVSHMEM
        t0 = time.perf_counter()
        for _ in range(benchmark_iters):
            _ = nvshmem_all_to_all_4D(tensor.clone(), scatter_dim=1, gather_dim=2)
        torch.cuda.synchronize()
        nvshmem_time = (time.perf_counter() - t0) / benchmark_iters * 1000

        nccl_time = None
        if compare_nccl:
            import torch.distributed as dist

            def nccl_all_to_all_4D(x, scatter_dim, gather_dim):
                """NCCL fallback implementation"""
                world_size = dist.get_world_size()
                input_list = list(torch.chunk(x, world_size, dim=scatter_dim))
                output_list = [torch.empty_like(chunk) for chunk in input_list]
                dist.all_to_all(output_list, input_list)
                return torch.cat(output_list, dim=gather_dim)

            # Warmup NCCL
            for _ in range(warmup_iters):
                _ = nccl_all_to_all_4D(tensor.clone(), scatter_dim=1, gather_dim=2)
            dist.barrier()
            torch.cuda.synchronize()

            # Benchmark NCCL
            t0 = time.perf_counter()
            for _ in range(benchmark_iters):
                _ = nccl_all_to_all_4D(tensor.clone(), scatter_dim=1, gather_dim=2)
            torch.cuda.synchronize()
            nccl_time = (time.perf_counter() - t0) / benchmark_iters * 1000

        if pe == 0:
            print(f"  {name:<25} {nvshmem_time:<15.3f}", end="")
            if compare_nccl and nccl_time:
                speedup = nccl_time / nvshmem_time
                print(f"{nccl_time:<15.3f} {speedup:<10.2f}x", end="")
            print()


def main():
    parser = argparse.ArgumentParser(description="MACO NVSHMEM Test")
    parser.add_argument(
        "--compare-nccl", action="store_true",
        help="Compare NVSHMEM performance with NCCL"
    )
    parser.add_argument(
        "--skip-correctness", action="store_true",
        help="Skip correctness tests"
    )
    parser.add_argument(
        "--skip-performance", action="store_true",
        help="Skip performance tests"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MACO NVSHMEM Test Suite")
    print("=" * 60)

    # 1. 检查 NVSHMEM 可用性
    if not test_nvshmem_available():
        print("\nNVSHMEM not available. Exiting.")
        return

    # 2. 正确性测试
    if not args.skip_correctness:
        test_allreduce_correctness()
        test_all_to_all_4D_correctness()

    # 3. 性能测试
    if not args.skip_performance:
        test_allreduce_performance(compare_nccl=args.compare_nccl)
        test_all_to_all_4D_performance(compare_nccl=args.compare_nccl)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
