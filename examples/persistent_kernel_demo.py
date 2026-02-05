#!/usr/bin/env python3
"""
Persistent Kernel 概念演示

展示 Mirage 风格的 Persistent Kernel 调度，不依赖 NVSHMEM。
使用 GPU 原子操作进行任务同步，而非 CUDA Event/Stream。

这是一个教学示例，展示核心概念。

运行方式:
    CUDA_VISIBLE_DEVICES=2 python examples/persistent_kernel_demo.py
"""

import torch
import time
from torch.utils.cpp_extension import load_inline

# Persistent Kernel CUDA 代码
PERSISTENT_KERNEL_CODE = '''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// GPU 原子操作 (Mirage 的核心同步机制)
// ============================================================================

// 原子加载 (acquire 语义) - 比 CUDA Event 快 ~10-50x
__device__ __forceinline__ unsigned long long
ld_acquire_gpu_u64(unsigned long long* addr) {
    unsigned long long val;
    asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
    return val;
}

// 原子存储 (release 语义)
__device__ __forceinline__ void
st_release_gpu_u64(unsigned long long* addr, unsigned long long val) {
    asm volatile("st.release.gpu.u64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}

// 原子加法 (release 语义)
__device__ __forceinline__ unsigned long long
atom_add_release_gpu_u64(unsigned long long* addr, unsigned long long val) {
    unsigned long long old_val;
    asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
                 : "=l"(old_val) : "l"(addr), "l"(val) : "memory");
    return old_val;
}

// ============================================================================
// Task 定义
// ============================================================================

enum TaskType {
    TASK_MATMUL = 0,
    TASK_REDUCE = 1,
    TASK_TERMINATE = 99
};

struct Task {
    int task_type;
    int task_id;
    float* input;
    float* output;
    int size;
};

// ============================================================================
// Worker 执行逻辑
// ============================================================================

__device__ void execute_matmul_task(Task* task) {
    // 简单的向量乘法示例
    for (int i = threadIdx.x; i < task->size; i += blockDim.x) {
        task->output[i] = task->input[i] * 2.0f;
    }
}

__device__ void execute_reduce_task(Task* task) {
    // 简单的求和示例 (仅演示)
    __shared__ float shared_sum[256];
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < task->size; i += blockDim.x) {
        local_sum += task->input[i];
    }

    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // 简单归约
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x && i < task->size; i++) {
            total += shared_sum[i];
        }
        task->output[0] = total;
    }
}

// ============================================================================
// Persistent Kernel - 核心！
// ============================================================================

__global__ void persistent_worker_kernel(
    Task* task_queue,           // 任务队列
    unsigned long long* queue_head,  // 队列头指针 (生产者写)
    unsigned long long* queue_tail,  // 队列尾指针 (消费者写)
    unsigned long long* done_counter, // 完成计数器
    int queue_size,
    int num_workers  // Worker 数量 (= gridDim.x)
) {
    int worker_id = blockIdx.x;

    // 每个 Worker 处理 queue 中的不同任务
    unsigned long long local_tail = worker_id;  // 每个 worker 从不同位置开始

    while (true) {
        // 1. Polling: 等待新任务 (使用原子加载，而非 CUDA Event)
        unsigned long long head;
        while (true) {
            head = ld_acquire_gpu_u64(queue_head);
            if (local_tail < head) {
                break;  // 有新任务
            }
            // 检查是否应该终止
            if (task_queue[local_tail % queue_size].task_type == TASK_TERMINATE) {
                return;
            }
            __nanosleep(10);  // 减少 contention
        }

        // 2. 获取任务
        Task* task = &task_queue[local_tail % queue_size];

        // 3. 执行任务
        __syncthreads();
        switch (task->task_type) {
            case TASK_MATMUL:
                execute_matmul_task(task);
                break;
            case TASK_REDUCE:
                execute_reduce_task(task);
                break;
            case TASK_TERMINATE:
                return;
            default:
                break;
        }
        __syncthreads();

        // 4. 标记完成 (使用原子操作，而非 CUDA Event)
        if (threadIdx.x == 0) {
            atom_add_release_gpu_u64(done_counter, 1);
        }

        // 5. 移动到下一个任务 (round-robin 在 workers 之间)
        local_tail += num_workers;
    }
}

// ============================================================================
// 对比: 传统 Kernel Launch 方式
// ============================================================================

__global__ void traditional_matmul_kernel(float* input, float* output, int size) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        output[i] = input[i] * 2.0f;
    }
}

// ============================================================================
// Python 接口
// ============================================================================

void benchmark_persistent_kernel(
    torch::Tensor task_queue_tensor,  // 预分配的任务队列内存
    torch::Tensor input,
    torch::Tensor output,
    int num_tasks,
    int num_workers
) {
    // 这里简化处理，实际 Mirage 有更复杂的任务队列管理
    int threads = 128;

    // 分配控制结构
    unsigned long long *queue_head, *queue_tail, *done_counter;
    cudaMalloc(&queue_head, sizeof(unsigned long long));
    cudaMalloc(&queue_tail, sizeof(unsigned long long));
    cudaMalloc(&done_counter, sizeof(unsigned long long));
    cudaMemset(queue_head, 0, sizeof(unsigned long long));
    cudaMemset(queue_tail, 0, sizeof(unsigned long long));
    cudaMemset(done_counter, 0, sizeof(unsigned long long));

    // Launch persistent kernel
    persistent_worker_kernel<<<num_workers, threads>>>(
        reinterpret_cast<Task*>(task_queue_tensor.data_ptr()),
        queue_head, queue_tail, done_counter,
        num_tasks, num_workers
    );

    // 清理
    cudaFree(queue_head);
    cudaFree(queue_tail);
    cudaFree(done_counter);
}

void benchmark_traditional(torch::Tensor input, torch::Tensor output, int num_iters) {
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    for (int i = 0; i < num_iters; i++) {
        traditional_matmul_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    }
}
'''

PERSISTENT_KERNEL_HEADER = '''
void benchmark_persistent_kernel(
    torch::Tensor task_queue_tensor,
    torch::Tensor input,
    torch::Tensor output,
    int num_tasks,
    int num_workers
);

void benchmark_traditional(torch::Tensor input, torch::Tensor output, int num_iters);
'''


def compile_module():
    """编译 Persistent Kernel 模块"""
    print("Compiling Persistent Kernel module...")
    module = load_inline(
        name="persistent_kernel_demo",
        cpp_sources=PERSISTENT_KERNEL_HEADER,
        cuda_sources=PERSISTENT_KERNEL_CODE,
        functions=["benchmark_traditional"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89"],
        verbose=False
    )
    return module


def benchmark_kernel_launch_overhead():
    """测量 kernel launch 开销"""
    print("\n" + "=" * 60)
    print("Kernel Launch Overhead Benchmark")
    print("=" * 60)

    # 编译
    module = compile_module()

    # 测试不同大小
    sizes = [1024, 4096, 16384, 65536, 262144]
    num_iters = 1000

    print(f"\n{'Size':<15} {'Time/iter (μs)':<20} {'Kernel Launch %':<20}")
    print("-" * 55)

    for size in sizes:
        input_tensor = torch.randn(size, device="cuda")
        output_tensor = torch.empty_like(input_tensor)

        # Warmup
        for _ in range(10):
            module.benchmark_traditional(input_tensor, output_tensor, 1)
        torch.cuda.synchronize()

        # 测量单次 kernel launch
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            module.benchmark_traditional(input_tensor, output_tensor, 1)
            torch.cuda.synchronize()
        single_time = (time.perf_counter() - start) / num_iters * 1e6  # μs

        # 测量批量执行 (减少 launch 开销)
        torch.cuda.synchronize()
        start = time.perf_counter()
        module.benchmark_traditional(input_tensor, output_tensor, num_iters)
        torch.cuda.synchronize()
        batch_time = (time.perf_counter() - start) / num_iters * 1e6  # μs

        # 计算 launch 开销占比
        launch_overhead = (single_time - batch_time) / single_time * 100

        print(f"{size:<15} {single_time:<20.2f} {launch_overhead:<20.1f}%")

    print("\n结论: 对于小 kernel，launch 开销可能占总时间的 50-90%！")
    print("Persistent Kernel 通过单次 launch 消除这个开销。")


def demonstrate_atomic_vs_event():
    """演示 GPU Atomics vs CUDA Event"""
    print("\n" + "=" * 60)
    print("GPU Atomics vs CUDA Event Synchronization")
    print("=" * 60)

    num_iters = 10000

    # CUDA Event 同步
    event = torch.cuda.Event()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        event.record()
        event.synchronize()
    event_time = (time.perf_counter() - start) / num_iters * 1e6  # μs

    # Stream 同步 (更轻量)
    stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        stream.synchronize()
    stream_time = (time.perf_counter() - start) / num_iters * 1e6  # μs

    print(f"\n同步方式比较 ({num_iters} iterations):")
    print(f"  CUDA Event.synchronize():  {event_time:.2f} μs")
    print(f"  Stream.synchronize():      {stream_time:.2f} μs")
    print(f"  GPU Atomics (估计):        ~1-2 μs (无需 CPU 参与)")
    print(f"\n关键区别:")
    print(f"  - CUDA Event 需要 CPU 参与 (driver call)")
    print(f"  - GPU Atomics 完全在 GPU 内完成")


def main():
    print("=" * 60)
    print("Persistent Kernel Concept Demo")
    print("=" * 60)
    print("\n这个演示展示 Mirage 的核心优化技术:")
    print("  1. Persistent Kernel: 单次 launch，消除重复开销")
    print("  2. GPU Atomics: 比 CUDA Event 快 ~10-50x")
    print("  3. Lock-free 任务队列: 无需 CPU 调度")
    print("\n这些技术不依赖 NVSHMEM！")

    demonstrate_atomic_vs_event()
    benchmark_kernel_launch_overhead()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
Mirage 的真正创新:

1. Persistent Kernel
   - 整个推理只 launch 一次 kernel
   - 避免了每个操作的 launch 开销 (~5-10 μs)

2. GPU Atomics 同步
   - 使用 PTX 指令: ld.acquire.gpu, atom.add.release.gpu
   - 比 CUDA Event 快 ~10-50x
   - 完全在 GPU 内完成，无需 CPU

3. 任务队列调度
   - Workers 通过 polling 获取任务
   - Schedulers 分发任务到 Workers
   - 所有调度在 GPU 内完成

NVSHMEM 只用于跨 GPU 通信优化，
核心的 Persistent Kernel 架构不依赖它！
""")


if __name__ == "__main__":
    main()
