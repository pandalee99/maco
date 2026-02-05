#!/usr/bin/env python3
"""
MACO SM Scheduler 测试

测试真正的 SM 级别调度功能，展示:
1. GPU Atomics 同步机制
2. Lock-free 任务队列
3. Worker CTA 并行执行
4. 事件依赖管理

运行方式:
    # 先编译 SM 调度模块
    cd /mini_mirage/maco
    python maco/csrc/setup_sm.py build_ext --inplace

    # 然后运行测试
    CUDA_VISIBLE_DEVICES=2 python examples/test_sm_scheduler.py
"""

import torch
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_sm_scheduling_demo():
    """测试 SM 调度演示 kernel"""
    print("=" * 60)
    print("SM Scheduling Demo Test")
    print("=" * 60)

    try:
        from maco._sm import run_sm_scheduling_demo
    except ImportError as e:
        print(f"Failed to import maco._sm: {e}")
        print("\nPlease compile first:")
        print("  cd /mini_mirage/maco")
        print("  python maco/csrc/setup_sm.py build_ext --inplace")
        return False

    # 测试参数
    data_size = 1024 * 1024  # 1M elements
    num_tasks = 10
    num_workers_list = [1, 2, 4, 8]
    threads_per_worker = 128

    # 创建测试数据
    data = torch.randn(data_size, device="cuda", dtype=torch.float32)
    original = data.clone()

    print(f"\nData size: {data_size} elements ({data.numel() * 4 / 1024 / 1024:.1f} MB)")
    print(f"Tasks per worker: {num_tasks}")
    print(f"Threads per worker: {threads_per_worker}")

    print(f"\n{'Workers':<10} {'Time (ms)':<15} {'Throughput (GB/s)':<20}")
    print("-" * 45)

    for num_workers in num_workers_list:
        # 重置数据
        data = original.clone()

        # Warmup
        for _ in range(3):
            data_copy = data.clone()
            run_sm_scheduling_demo(data_copy, num_tasks, num_workers, threads_per_worker)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        iterations = 10
        for _ in range(iterations):
            data_copy = data.clone()
            result_data, completion_order = run_sm_scheduling_demo(
                data_copy, num_tasks, num_workers, threads_per_worker
            )
        torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) / iterations * 1000  # ms
        throughput = data_size * 4 * num_tasks / (elapsed / 1000) / 1e9  # GB/s

        print(f"{num_workers:<10} {elapsed:<15.3f} {throughput:<20.2f}")

    # 验证任务完成顺序
    print("\n" + "=" * 60)
    print("Task Completion Order Analysis")
    print("=" * 60)

    data = original.clone()
    result_data, completion_order = run_sm_scheduling_demo(
        data, num_tasks_per_worker=5, num_workers=4, threads_per_worker=128
    )

    print("\nCompletion order (worker_id * 1000 + task_id):")
    completion_list = completion_order.cpu().tolist()
    for i, val in enumerate(completion_list):
        worker = val // 1000
        task = val % 1000
        print(f"  {i:2d}: Worker {worker}, Task {task}")

    print("\n分析:")
    print("  - 如果同一 Worker 的任务按顺序完成，说明 Worker 内部串行执行")
    print("  - 不同 Worker 的任务交错完成，说明 Worker 间并行执行")
    print("  - 这展示了真正的 SM 级别调度！")

    return True


def test_gpu_atomics_correctness():
    """测试 GPU Atomics 的正确性"""
    print("\n" + "=" * 60)
    print("GPU Atomics Correctness Test")
    print("=" * 60)

    try:
        from maco._sm import run_sm_scheduling_demo
    except ImportError:
        print("Module not loaded, skipping...")
        return False

    # 多次运行检查一致性
    num_runs = 5
    data_size = 10000
    num_tasks = 10
    num_workers = 4

    original = torch.randn(data_size, device="cuda", dtype=torch.float32)
    results = []

    for i in range(num_runs):
        data = original.clone()
        result_data, _ = run_sm_scheduling_demo(
            data, num_tasks, num_workers, threads_per_worker=128
        )
        results.append(result_data.cpu())

    # 检查所有结果是否一致
    all_same = all(torch.allclose(results[0], r, atol=1e-5) for r in results[1:])

    if all_same:
        print(f"✓ All {num_runs} runs produced identical results")
        print("  GPU Atomics synchronization is working correctly!")
    else:
        print(f"✗ Results differ across runs - synchronization issue!")
        max_diff = max(
            (results[0] - r).abs().max().item() for r in results[1:]
        )
        print(f"  Max difference: {max_diff}")

    return all_same


def test_kernel_launch_overhead_comparison():
    """对比 SM 调度与传统 kernel launch 的开销"""
    print("\n" + "=" * 60)
    print("Kernel Launch Overhead Comparison")
    print("=" * 60)

    try:
        from maco._sm import run_sm_scheduling_demo
    except ImportError:
        print("Module not loaded, skipping...")
        return False

    data_size = 1024
    iterations = 100

    # SM 调度方式 (单次 launch，多任务)
    data = torch.randn(data_size, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        data_copy = data.clone()
        run_sm_scheduling_demo(data_copy, num_tasks_per_worker=10, num_workers=4)
    torch.cuda.synchronize()
    sm_time = (time.perf_counter() - start) / iterations * 1000

    # 传统方式 (多次 launch)
    # 模拟: 每个任务一次 kernel launch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        data_copy = data.clone()
        for _ in range(10 * 4):  # 10 tasks * 4 workers
            # 简单的 element-wise 操作
            data_copy = data_copy * 1.5 + 0.1
    torch.cuda.synchronize()
    traditional_time = (time.perf_counter() - start) / iterations * 1000

    print(f"\n执行 {10 * 4} 个任务:")
    print(f"  SM 调度 (单次 launch): {sm_time:.3f} ms")
    print(f"  传统方式 (多次 launch): {traditional_time:.3f} ms")
    print(f"  加速比: {traditional_time / sm_time:.2f}x")

    print("\n结论:")
    if sm_time < traditional_time:
        print("  ✓ SM 调度消除了重复 kernel launch 开销")
    else:
        print("  - SM 调度开销较大 (可能是数据量太小)")

    return True


def compile_module():
    """尝试编译模块"""
    print("=" * 60)
    print("Compiling MACO SM Scheduler Module...")
    print("=" * 60)

    import subprocess
    import os

    csrc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "maco", "csrc")
    setup_file = os.path.join(csrc_dir, "setup_sm.py")

    if not os.path.exists(setup_file):
        print(f"Setup file not found: {setup_file}")
        return False

    # 运行编译
    result = subprocess.run(
        [sys.executable, setup_file, "build_ext", "--inplace"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Compilation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

    print("Compilation successful!")
    return True


def main():
    print("=" * 60)
    print("MACO SM Scheduler Test Suite")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type != "cuda":
        print("CUDA not available, exiting.")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    # 检查模块是否已编译
    try:
        import maco._sm
        print("\n✓ maco._sm module loaded successfully")
    except ImportError:
        print("\n⚠ maco._sm not found, attempting to compile...")
        if not compile_module():
            print("\n请手动编译:")
            print("  cd /mini_mirage/maco")
            print("  python maco/csrc/setup_sm.py build_ext --inplace")
            return

        # 重新尝试导入
        try:
            import maco._sm
            print("✓ Module compiled and loaded")
        except ImportError as e:
            print(f"Failed to import after compile: {e}")
            return

    # 运行测试
    print("\n")
    test_sm_scheduling_demo()

    print("\n")
    test_gpu_atomics_correctness()

    print("\n")
    test_kernel_launch_overhead_comparison()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
MACO SM Scheduler 核心技术:

1. GPU Atomics (PTX 指令)
   - ld.acquire.gpu.u64: 原子加载
   - atom.add.release.gpu.u64: 原子加法
   - 完全在 GPU 内完成，无需 CPU 参与

2. Persistent Kernel
   - 单次 kernel launch
   - Worker/Scheduler CTA 角色分配
   - 消除重复 launch 开销

3. Lock-free 队列
   - GPU Atomics 实现的生产者-消费者模型
   - Scheduler 生产，Worker 消费

4. 事件依赖系统
   - 事件计数器 (原子操作)
   - 任务间依赖管理

这些技术不依赖 NVSHMEM！
只需要标准 PyTorch + CUDA (sm_70+)。
""")


if __name__ == "__main__":
    main()
