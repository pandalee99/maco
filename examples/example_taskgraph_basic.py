#!/usr/bin/env python3
"""
TaskGraph 基础示例

演示 MACO TaskGraph API 的基本用法：
1. 创建任务图
2. 添加计算任务
3. 自动依赖推断
4. 编译和执行

运行方式:
    cd /mini_mirage/maco
    CUDA_VISIBLE_DEVICES=0 python3 examples/example_taskgraph_basic.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from maco import TaskGraph


def example_basic_linear():
    """示例 1: 基本线性层执行"""
    print("=" * 60)
    print("Example 1: Basic Linear Layer")
    print("=" * 60)

    # 创建任务图
    graph = TaskGraph(num_workers=8)

    # 准备数据
    x = torch.randn(32, 512, device="cuda")
    w = torch.randn(1024, 512, device="cuda")

    # 添加线性层任务
    task = graph.linear(x, w, name="linear")

    # 编译和执行
    graph.compile()
    graph.execute()

    # 验证结果
    expected = torch.nn.functional.linear(x, w)
    diff = (expected - task.output).abs().max().item()

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {w.shape}")
    print(f"Output shape: {task.output.shape}")
    print(f"Max difference from PyTorch: {diff:.6f}")
    print(f"\n{graph.summary()}")


def example_chain_linear():
    """示例 2: 链式线性层"""
    print("\n" + "=" * 60)
    print("Example 2: Chain of Linear Layers")
    print("=" * 60)

    graph = TaskGraph(num_workers=8)

    # 准备数据
    x = torch.randn(64, 256, device="cuda")
    weights = [
        torch.randn(512, 256, device="cuda"),
        torch.randn(256, 512, device="cuda"),
        torch.randn(128, 256, device="cuda"),
    ]

    # 链式任务（自动推断依赖）
    t1 = graph.linear(x, weights[0], name="linear1")
    t2 = graph.linear(t1.output, weights[1], name="linear2")
    t3 = graph.linear(t2.output, weights[2], name="linear3")

    graph.compile()
    graph.execute()

    # 验证
    h = x
    for w in weights:
        h = torch.nn.functional.linear(h, w)

    diff = (h - t3.output).abs().max().item()
    print(f"3-layer chain output shape: {t3.output.shape}")
    print(f"Max difference: {diff:.6f}")
    print(f"\n{graph.summary()}")


def example_with_overlap():
    """示例 3: 计算与通信重叠"""
    print("\n" + "=" * 60)
    print("Example 3: Compute-Communication Overlap")
    print("=" * 60)

    graph = TaskGraph(num_workers=8)

    # 准备数据
    x = torch.randn(32, 512, device="cuda")
    weights = [torch.randn(512, 512, device="cuda") for _ in range(4)]

    # 计算任务
    compute_tasks = []
    h = x
    for i, w in enumerate(weights):
        t = graph.linear(h, w, name=f"linear_{i}")
        compute_tasks.append(t)
        h = t.output

    # 通信任务（模拟 AllReduce）
    comm_tensor = torch.randn(32, 512, device="cuda")
    comm_task = graph.allreduce(comm_tensor, name="allreduce")

    # 标记重叠
    group = graph.overlap(compute_tasks, [comm_task])
    group.auto_waves()

    print(f"Auto-detected waves: {group.num_waves}")
    for t in compute_tasks:
        print(f"  {t.name}: wave {t._wave_id}")

    graph.compile()
    graph.execute()

    print(f"\n{graph.summary()}")


def example_custom_task():
    """示例 4: 自定义任务"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Task")
    print("=" * 60)

    graph = TaskGraph(num_workers=8)

    x = torch.randn(32, 256, device="cuda")
    output = torch.empty(32, 256, device="cuda")

    # 自定义函数：GELU 激活
    def gelu_fn(inp):
        return torch.nn.functional.gelu(inp)

    task = graph.custom(
        fn=gelu_fn,
        inputs=[x],
        outputs=[output],
        name="gelu"
    )

    graph.compile()
    graph.execute()

    # 验证
    expected = torch.nn.functional.gelu(x)
    diff = (expected - output).abs().max().item()

    print(f"Custom GELU output shape: {output.shape}")
    print(f"Max difference: {diff:.6f}")
    print(f"\n{graph.summary()}")


def main():
    print("MACO TaskGraph Basic Examples")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    example_basic_linear()
    example_chain_linear()
    example_with_overlap()
    example_custom_task()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
