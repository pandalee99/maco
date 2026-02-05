#!/usr/bin/env python3
"""
TaskGraph API 测试

测试 MACO Phase 2 的核心功能：
1. TaskGraph 创建和编译
2. 任务依赖推断
3. Wave Grouping
4. Stream 模式执行（回退）

运行方式:
    cd /mini_mirage/maco
    CUDA_VISIBLE_DEVICES=1 python3 examples/test_task_graph.py
"""

import sys
import os
import time

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_basic_task_graph():
    """测试基本的 TaskGraph 创建"""
    print("=" * 60)
    print("Test 1: Basic TaskGraph Creation")
    print("=" * 60)

    from maco import TaskGraph, TaskType

    # 创建任务图
    graph = TaskGraph(num_workers=4)

    # 创建测试数据
    x = torch.randn(32, 512, device="cuda")
    w1 = torch.randn(1024, 512, device="cuda")
    w2 = torch.randn(512, 1024, device="cuda")

    # 添加任务
    t1 = graph.linear(x, w1, name="linear1")
    print(f"Created task: {t1.name}, type={t1.task_type.name}")

    t2 = graph.linear(t1.output, w2, name="linear2")
    print(f"Created task: {t2.name}, type={t2.task_type.name}")

    # 检查依赖推断
    graph._infer_dependencies()
    print(f"\nt2 depends on: {[d.name for d in t2.depends_on]}")

    print(f"\n{graph.summary()}")
    print("\n✓ Basic TaskGraph creation passed!")
    return True


def test_dependency_inference():
    """测试依赖推断"""
    print("\n" + "=" * 60)
    print("Test 2: Dependency Inference")
    print("=" * 60)

    from maco import TaskGraph

    graph = TaskGraph(num_workers=4)

    # 创建链式任务
    x = torch.randn(32, 256, device="cuda")
    w1 = torch.randn(512, 256, device="cuda")
    w2 = torch.randn(256, 512, device="cuda")
    w3 = torch.randn(128, 256, device="cuda")

    t1 = graph.linear(x, w1, name="linear1")
    t2 = graph.linear(t1.output, w2, name="linear2")
    t3 = graph.linear(t2.output, w3, name="linear3")

    # 推断依赖
    graph._infer_dependencies()

    # 验证依赖链
    assert len(t1.depends_on) == 0, "t1 should have no dependencies"
    assert t1 in t2.depends_on, "t2 should depend on t1"
    assert t2 in t3.depends_on, "t3 should depend on t2"

    print("Dependency chain:")
    print(f"  t1 ({t1.name}): depends on {[d.name for d in t1.depends_on]}")
    print(f"  t2 ({t2.name}): depends on {[d.name for d in t2.depends_on]}")
    print(f"  t3 ({t3.name}): depends on {[d.name for d in t3.depends_on]}")

    print("\n✓ Dependency inference passed!")
    return True


def test_wave_grouping():
    """测试 Wave Grouping"""
    print("\n" + "=" * 60)
    print("Test 3: Wave Grouping")
    print("=" * 60)

    from maco import TaskGraph

    graph = TaskGraph(num_workers=4)

    # 创建多个计算任务
    x = torch.randn(32, 256, device="cuda")
    tasks = []

    for i in range(8):
        w = torch.randn(256, 256, device="cuda")
        t = graph.linear(x, w, name=f"linear_{i}")
        tasks.append(t)
        x = t.output

    # 创建通信任务
    comm_tensor = torch.randn(32, 256, device="cuda")
    comm_task = graph.allreduce(comm_tensor, name="allreduce")

    # 标记重叠并自动分组
    group = graph.overlap(tasks, [comm_task])
    group.auto_waves()

    print(f"Total compute tasks: {len(tasks)}")
    print(f"Auto-detected waves: {group.num_waves}")
    print(f"Wave assignments:")
    for t in tasks:
        print(f"  {t.name}: wave {t._wave_id}")

    print("\n✓ Wave grouping passed!")
    return True


def test_task_schedule():
    """测试任务调度生成"""
    print("\n" + "=" * 60)
    print("Test 4: Task Schedule Generation")
    print("=" * 60)

    from maco import TaskGraph, TaskSchedule

    graph = TaskGraph(num_workers=4)

    # 创建 DAG 结构的任务
    #     t1
    #    /  \
    #   t2   t3
    #    \  /
    #     t4

    x = torch.randn(32, 256, device="cuda")
    w1 = torch.randn(256, 256, device="cuda")
    w2 = torch.randn(256, 256, device="cuda")
    w3 = torch.randn(256, 256, device="cuda")
    w4 = torch.randn(128, 256, device="cuda")

    t1 = graph.linear(x, w1, name="t1")
    t2 = graph.linear(t1.output, w2, name="t2")
    t3 = graph.linear(t1.output, w3, name="t3")

    # t4 依赖 t2 和 t3
    t4 = graph.linear(t2.output, w4, name="t4")
    t4.add_dependency(t3)

    # 编译生成调度
    graph.compile()

    print(f"Execution waves: {len(graph._schedule.waves)}")
    for i, wave in enumerate(graph._schedule.waves):
        names = [n.name for n in wave]
        print(f"  Wave {i}: {names}")

    # 验证拓扑顺序
    wave_indices = {}
    for i, wave in enumerate(graph._schedule.waves):
        for node in wave:
            wave_indices[node.name] = i

    assert wave_indices["t1"] < wave_indices["t2"], "t1 should be before t2"
    assert wave_indices["t1"] < wave_indices["t3"], "t1 should be before t3"
    assert wave_indices["t2"] < wave_indices["t4"], "t2 should be before t4"
    assert wave_indices["t3"] < wave_indices["t4"], "t3 should be before t4"

    print("\n✓ Task schedule generation passed!")
    return True


def test_stream_execution():
    """测试 Stream 模式执行"""
    print("\n" + "=" * 60)
    print("Test 5: Stream Mode Execution")
    print("=" * 60)

    from maco import TaskGraph

    graph = TaskGraph(num_workers=4)

    # 创建简单的计算链
    x = torch.randn(32, 512, device="cuda")
    w1 = torch.randn(1024, 512, device="cuda")
    w2 = torch.randn(512, 1024, device="cuda")

    t1 = graph.linear(x, w1, name="linear1")
    t2 = graph.linear(t1.output, w2, name="linear2")

    # 编译
    graph.compile()

    # 计算期望结果
    expected = torch.nn.functional.linear(
        torch.nn.functional.linear(x, w1), w2
    )

    # 执行
    torch.cuda.synchronize()
    start = time.perf_counter()

    graph.execute()

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    # 验证结果
    actual = t2.output
    diff = (expected - actual).abs().max().item()

    print(f"Execution time: {elapsed:.3f} ms")
    print(f"Max difference from expected: {diff:.6f}")

    assert diff < 1e-4, f"Result mismatch: {diff}"

    print("\n✓ Stream mode execution passed!")
    return True


def test_benchmark_stream():
    """性能对比测试"""
    print("\n" + "=" * 60)
    print("Test 6: Performance Benchmark")
    print("=" * 60)

    from maco import TaskGraph

    # 测试参数
    batch_size = 64
    hidden_size = 1024
    num_layers = 4
    iterations = 10

    # 准备数据
    x = torch.randn(batch_size, hidden_size, device="cuda")
    weights = [
        torch.randn(hidden_size, hidden_size, device="cuda")
        for _ in range(num_layers)
    ]

    # 1. PyTorch 基线
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        h = x
        for w in weights:
            h = torch.nn.functional.linear(h, w)
        torch.cuda.synchronize()

    baseline_time = (time.perf_counter() - start) / iterations * 1000

    # 2. TaskGraph 执行
    graph = TaskGraph(num_workers=4)

    h = x
    for i, w in enumerate(weights):
        task = graph.linear(h, w, name=f"layer_{i}")
        h = task.output

    graph.compile()

    # Warmup
    for _ in range(3):
        graph.execute()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        graph.execute()
    torch.cuda.synchronize()

    taskgraph_time = (time.perf_counter() - start) / iterations * 1000

    print(f"Batch size: {batch_size}, Hidden: {hidden_size}, Layers: {num_layers}")
    print(f"PyTorch baseline: {baseline_time:.3f} ms")
    print(f"TaskGraph (stream): {taskgraph_time:.3f} ms")
    print(f"Ratio: {taskgraph_time / baseline_time:.2f}x")

    print("\n✓ Benchmark completed!")
    return True


def test_custom_task():
    """测试自定义任务"""
    print("\n" + "=" * 60)
    print("Test 7: Custom Task")
    print("=" * 60)

    from maco import TaskGraph

    graph = TaskGraph(num_workers=4)

    x = torch.randn(32, 256, device="cuda")
    output = torch.empty(32, 256, device="cuda")

    # 自定义函数：ReLU + Scale
    def custom_fn(inp):
        return torch.relu(inp) * 2.0

    t = graph.custom(
        fn=custom_fn,
        inputs=[x],
        outputs=[output],
        name="custom_relu_scale",
    )

    graph.compile()
    graph.execute()

    # 验证
    expected = torch.relu(x) * 2.0
    diff = (expected - output).abs().max().item()

    print(f"Custom task result diff: {diff:.6f}")
    assert diff < 1e-5, f"Custom task result mismatch: {diff}"

    print("\n✓ Custom task passed!")
    return True


def main():
    print("=" * 60)
    print("MACO TaskGraph API Test Suite")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type != "cuda":
        print("CUDA not available, some tests may fail.")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")

    # 运行测试
    tests = [
        test_basic_task_graph,
        test_dependency_inference,
        test_wave_grouping,
        test_task_schedule,
        test_stream_execution,
        test_custom_task,
        test_benchmark_stream,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All tests passed! TaskGraph API is ready.")
    else:
        print(f"\n⚠️ {failed} tests failed.")


if __name__ == "__main__":
    main()
