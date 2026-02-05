#!/usr/bin/env python3
"""
All-to-All 通信场景性能测试

测试 MACO TaskGraph 在 All-to-All 通信模式下的性能：
1. 单节点 All-to-All（模拟 Expert Parallel）
2. 计算-通信重叠效果
3. 与 PyTorch NCCL 基线对比

运行方式:
    cd /mini_mirage/maco

    # 单卡测试（模拟 All-to-All）
    CUDA_VISIBLE_DEVICES=1 python3 examples/benchmark_all_to_all.py

    # 双卡测试（真实 All-to-All）
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 examples/benchmark_all_to_all.py --distributed
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist


def benchmark_local_all_to_all(batch_size: int, hidden_size: int, num_experts: int):
    """本地模拟 All-to-All（单卡）"""
    print("\n" + "=" * 60)
    print(f"Benchmark: Local All-to-All Simulation")
    print(f"  Batch: {batch_size}, Hidden: {hidden_size}, Experts: {num_experts}")
    print("=" * 60)

    from maco import TaskGraph

    # 模拟 MoE 场景
    # 输入: [batch, hidden] -> 路由到 num_experts 个专家
    # 每个专家处理 batch/num_experts 的 tokens

    tokens_per_expert = batch_size // num_experts

    # 专家权重
    expert_weights = [
        (torch.randn(hidden_size * 4, hidden_size, device="cuda", dtype=torch.bfloat16),
         torch.randn(hidden_size, hidden_size * 4, device="cuda", dtype=torch.bfloat16))
        for _ in range(num_experts)
    ]

    # 输入数据
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)

    # MACO TaskGraph
    graph = TaskGraph(num_workers=8)

    expert_outputs = []
    for expert_idx in range(num_experts):
        # 每个专家处理一部分 tokens
        start_idx = expert_idx * tokens_per_expert
        end_idx = start_idx + tokens_per_expert
        expert_input = x[start_idx:end_idx]

        w1, w2 = expert_weights[expert_idx]

        # 专家 FFN: input -> w1 -> activation -> w2
        t1 = graph.linear(expert_input, w1, name=f"expert{expert_idx}_up")
        t2 = graph.linear(t1.output, w2, name=f"expert{expert_idx}_down")
        expert_outputs.append(t2)

    # 添加模拟的 All-to-All 通信
    comm_tensor = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    all_to_all_task = graph.all_to_all(comm_tensor, name="all_to_all", scatter_dim=0, gather_dim=0)

    # 标记重叠
    group = graph.overlap(
        [t for t in expert_outputs for t in [t]],  # flatten
        [all_to_all_task]
    )
    group.auto_waves()

    graph.compile()

    # Warmup
    for _ in range(5):
        graph.execute()
    torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        graph.execute()

    torch.cuda.synchronize()
    maco_time = (time.perf_counter() - start) / iterations * 1000

    # PyTorch 基线
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        outputs = []
        for expert_idx in range(num_experts):
            start_idx = expert_idx * tokens_per_expert
            end_idx = start_idx + tokens_per_expert
            expert_input = x[start_idx:end_idx]
            w1, w2 = expert_weights[expert_idx]
            h = torch.nn.functional.linear(expert_input, w1)
            h = torch.nn.functional.linear(h, w2)
            outputs.append(h)
        # 模拟 all_to_all
        _ = torch.cat(outputs, dim=0)

    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / iterations * 1000

    print(f"\nMACO TaskGraph time: {maco_time:.3f} ms")
    print(f"PyTorch baseline time: {pytorch_time:.3f} ms")
    print(f"Speedup: {pytorch_time / maco_time:.2f}x")
    print(f"\n{graph.summary()}")

    return maco_time, pytorch_time


def benchmark_distributed_all_to_all():
    """分布式 All-to-All（多卡）"""
    print("\n" + "=" * 60)
    print("Benchmark: Distributed All-to-All (NCCL)")
    print("=" * 60)

    if not dist.is_initialized():
        print("Distributed not initialized, skipping...")
        return None, None

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank: {rank}, World size: {world_size}")

    batch_size = 1024
    hidden_size = 1024

    # 每个 rank 的数据
    local_data = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    output_data = torch.empty_like(local_data)

    # Warmup
    for _ in range(5):
        dist.all_to_all_single(output_data, local_data)
    torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()

    for _ in range(iterations):
        dist.all_to_all_single(output_data, local_data)

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000

    if rank == 0:
        data_size_mb = batch_size * hidden_size * 2 / (1024 * 1024)  # bfloat16 = 2 bytes
        bandwidth = data_size_mb / (elapsed / 1000)

        print(f"\nAll-to-All time: {elapsed:.3f} ms")
        print(f"Data size: {data_size_mb:.2f} MB")
        print(f"Bandwidth: {bandwidth:.2f} MB/s")

    return elapsed, batch_size * hidden_size


def benchmark_moe_pattern(batch_size: int, hidden_size: int, num_experts: int, top_k: int = 2):
    """MoE 模式性能测试"""
    print("\n" + "=" * 60)
    print(f"Benchmark: MoE Pattern (Experts={num_experts}, Top-K={top_k})")
    print("=" * 60)

    from maco import TaskGraph

    # MoE 配置
    intermediate_size = hidden_size * 4

    # 模拟 top-k 路由后的 token 分配
    # 假设 tokens 均匀分配到各专家
    tokens_per_expert = (batch_size * top_k) // num_experts

    # 专家权重
    expert_up_weights = [
        torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_experts)
    ]
    expert_down_weights = [
        torch.randn(hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_experts)
    ]

    # 输入（已路由后）
    routed_inputs = [
        torch.randn(tokens_per_expert, hidden_size, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_experts)
    ]

    # MACO TaskGraph
    graph = TaskGraph(num_workers=8)

    all_tasks = []
    for expert_idx in range(num_experts):
        up_task = graph.linear(
            routed_inputs[expert_idx],
            expert_up_weights[expert_idx],
            name=f"expert{expert_idx}_up"
        )
        down_task = graph.linear(
            up_task.output,
            expert_down_weights[expert_idx],
            name=f"expert{expert_idx}_down"
        )
        all_tasks.extend([up_task, down_task])

    # All-to-All 通信（routing 前后）
    comm_tensor = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    a2a_task = graph.all_to_all(comm_tensor, name="routing_a2a")

    # 标记重叠
    group = graph.overlap(all_tasks, [a2a_task])
    group.auto_waves()

    graph.compile()

    print(f"Total tasks: {len(all_tasks) + 1}")
    print(f"Waves: {group.num_waves}")
    print(f"Tokens per expert: {tokens_per_expert}")

    # Warmup
    for _ in range(5):
        graph.execute()
    torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        graph.execute()

    torch.cuda.synchronize()
    maco_time = (time.perf_counter() - start) / iterations * 1000

    # PyTorch 基线
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        for expert_idx in range(num_experts):
            h = torch.nn.functional.linear(routed_inputs[expert_idx], expert_up_weights[expert_idx])
            h = torch.nn.functional.linear(h, expert_down_weights[expert_idx])

    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / iterations * 1000

    print(f"\nMACO TaskGraph time: {maco_time:.3f} ms")
    print(f"PyTorch baseline time: {pytorch_time:.3f} ms")
    print(f"Speedup: {pytorch_time / maco_time:.2f}x")

    # 计算理论 FLOPs
    flops_per_expert = 2 * tokens_per_expert * hidden_size * intermediate_size * 2  # up + down
    total_flops = flops_per_expert * num_experts
    tflops = total_flops / (maco_time / 1000) / 1e12

    print(f"Throughput: {tflops:.2f} TFLOPS")

    return maco_time, pytorch_time


def benchmark_overlap_efficiency(hidden_size: int = 1024):
    """计算-通信重叠效率测试"""
    print("\n" + "=" * 60)
    print("Benchmark: Compute-Comm Overlap Efficiency")
    print("=" * 60)

    from maco import TaskGraph

    # 不同 wave 数量的重叠效果
    results = []

    for num_waves in [1, 2, 4, 8]:
        # 创建足够多的计算任务
        num_compute_tasks = num_waves * 4

        graph = TaskGraph(num_workers=8)

        x = torch.randn(256, hidden_size, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.bfloat16)

        compute_tasks = []
        h = x
        for i in range(num_compute_tasks):
            t = graph.linear(h, w, name=f"compute_{i}")
            compute_tasks.append(t)
            h = t.output

        # 通信任务
        comm_tensor = torch.randn(256, hidden_size, device="cuda", dtype=torch.bfloat16)
        comm_task = graph.allreduce(comm_tensor, name="comm")

        # 标记重叠
        group = graph.overlap(compute_tasks, [comm_task])
        group.with_waves(num_waves)

        graph.compile()

        # Benchmark
        for _ in range(5):
            graph.execute()
        torch.cuda.synchronize()

        iterations = 100
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            graph.execute()

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iterations * 1000

        results.append((num_waves, num_compute_tasks, elapsed))
        print(f"Waves={num_waves}, Tasks={num_compute_tasks}: {elapsed:.3f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="All-to-All Benchmark")
    parser.add_argument("--distributed", action="store_true", help="Run distributed test")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden", type=int, default=1024, help="Hidden size")
    parser.add_argument("--experts", type=int, default=8, help="Number of experts")
    args = parser.parse_args()

    print("=" * 60)
    print("MACO All-to-All Performance Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    if args.distributed:
        # 初始化分布式
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        benchmark_distributed_all_to_all()

        dist.destroy_process_group()
    else:
        # 单卡测试
        benchmark_local_all_to_all(args.batch, args.hidden, args.experts)
        benchmark_moe_pattern(args.batch, args.hidden, args.experts)
        benchmark_overlap_efficiency(args.hidden)

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
