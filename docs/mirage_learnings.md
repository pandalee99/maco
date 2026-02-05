# 从 Mirage 学到的优化技术

本文档总结了从 [Mirage](https://github.com/mirage-project/mirage) 项目学习并应用到 MACO 的核心技术。

## 目录

1. [Persistent Kernel 架构](#1-persistent-kernel-架构)
2. [GPU Atomics 同步原语](#2-gpu-atomics-同步原语)
3. [Worker/Scheduler CTA 模式](#3-workerscheduler-cta-模式)
4. [Lock-free 任务队列](#4-lock-free-任务队列)
5. [事件依赖系统](#5-事件依赖系统)
6. [关键发现：不依赖 NVSHMEM](#6-关键发现不依赖-nvshmem)

---

## 1. Persistent Kernel 架构

### 来源
- Mirage 代码: `mirage/include/mirage/persistent_kernel/persistent_kernel.cuh`
- 论文: [Mirage: Generating Fast GPU Kernels with Multi-Level Superoptimization](https://arxiv.org/abs/2405.05751)

### 核心思想

传统方式每个操作需要一次 kernel launch：
```
CPU: launch kernel A → wait → launch kernel B → wait → launch kernel C
GPU:     [kernel A]           [kernel B]           [kernel C]
                 ↑                    ↑                    ↑
           launch overhead      launch overhead      launch overhead
```

Persistent Kernel 只需一次 launch：
```
CPU: launch persistent kernel → ... → signal terminate
GPU: [================== persistent kernel ==================]
     [task A][task B][task C][task D]...
           ↑       ↑       ↑
        GPU内部调度，无launch开销
```

### MACO 实现

```cpp
// maco/csrc/maco_kernel.cu
__global__ void maco_persistent_kernel(MacoRuntimeConfig config) {
    if (blockIdx.x < config.num_workers) {
        execute_worker(&config);    // Worker CTA
    } else {
        execute_scheduler(&config); // Scheduler CTA
    }
}
```

### 性能收益

实测结果 (L20 GPU):
- 40 个任务的执行时间
- SM 调度: 0.041 ms
- 传统方式: 0.654 ms
- **加速比: 16.14x**

---

## 2. GPU Atomics 同步原语

### 来源
- Mirage 代码: `mirage/include/mirage/persistent_kernel/mpk_atoms.cuh`
- PTX ISA 文档: Memory Consistency Model

### 核心原语

MACO 从 Mirage 学习并实现了以下原子操作：

```cpp
// maco/csrc/maco_atoms.cuh

// 原子加载 (acquire 语义)
__device__ uint64_t ld_acquire_gpu_u64(uint64_t* addr) {
    uint64_t val;
    asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
    return val;
}

// 原子存储 (release 语义)
__device__ void st_release_gpu_u64(uint64_t* addr, uint64_t val) {
    asm volatile("st.release.gpu.u64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}

// 原子加法 (release 语义)
__device__ uint64_t atom_add_release_gpu_u64(uint64_t* addr, uint64_t val) {
    uint64_t old_val;
    asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
                 : "=l"(old_val) : "l"(addr), "l"(val) : "memory");
    return old_val;
}
```

### 内存顺序语义

| 语义 | 含义 | 使用场景 |
|------|------|----------|
| `acquire` | 确保后续读取能看到之前的写入 | 任务获取后读取任务数据 |
| `release` | 确保之前的写入对其他线程可见 | 任务完成后更新状态 |
| `relaxed` | 无顺序保证，最快 | 简单计数器 |

### 作用域

| 作用域 | 可见性 | 使用场景 |
|--------|--------|----------|
| `.gpu` | GPU 内所有线程 | 单 GPU 内的 SM 间同步 |
| `.sys` | 系统级 (包括 CPU) | 跨 GPU 或 GPU-CPU 同步 |

---

## 3. Worker/Scheduler CTA 模式

### 来源
- Mirage 代码: `persistent_kernel.cuh` Line 527-752 (Worker), Line 755-1000 (Scheduler)

### 架构设计

```
+------------------+     +------------------+
|   Scheduler CTA  |     |   Scheduler CTA  |
| (blockIdx >= N)  |     | (blockIdx >= N)  |
+--------+---------+     +--------+---------+
         |                        |
         | 分发任务               | 监控事件
         v                        v
+--------+---------+     +--------+---------+
|   Worker CTA 0   |     |   Worker CTA 1   |
| (blockIdx = 0)   |     | (blockIdx = 1)   |
+--------+---------+     +--------+---------+
         |                        |
         | 执行任务               | 执行任务
         v                        v
    [Task Queue 0]           [Task Queue 1]
```

### CTA 角色分配

```cpp
// maco/csrc/maco_kernel.cu Line 45-51
__global__ void maco_persistent_kernel(MacoRuntimeConfig config) {
    if (blockIdx.x < config.num_workers) {
        execute_worker(&config);    // Worker: 执行计算
    } else {
        execute_scheduler(&config); // Scheduler: 分发任务
    }
}
```

### Worker 执行流程

```cpp
// maco/csrc/maco_worker.cuh
__device__ void execute_worker(MacoRuntimeConfig* config) {
    while (true) {
        // 1. Polling: 等待新任务
        while (local_tail >= queue_head) {
            if (terminate_flag) return;
            maco_nanosleep(10);  // 低功耗等待
        }

        // 2. 获取任务
        task = queue[local_tail % QUEUE_SIZE];

        // 3. 等待依赖事件
        if (task.dependent_event != INVALID) {
            while (event_counter < needed) {
                maco_nanosleep(10);
            }
        }

        // 4. 执行任务
        execute_task(&task);

        // 5. 触发完成事件
        atom_add_release_gpu_u64(&event_counters[trigger_event], 1);

        // 6. 移动到下一个任务
        local_tail++;
    }
}
```

---

## 4. Lock-free 任务队列

### 来源
- Mirage 代码: `persistent_kernel.cuh` 中的队列管理逻辑

### 设计原理

使用单生产者单消费者 (SPSC) 模型：
- **生产者**: Scheduler CTA 写入队列头
- **消费者**: Worker CTA 读取队列尾

```
         head (Scheduler写)
           ↓
Queue: [task3][task2][task1][    ][    ]
                       ↑
                 tail (Worker读)
```

### 实现代码

```cpp
// Scheduler: 添加任务
uint64_t head = ld_acquire_gpu_u64(&queue_heads[worker_id]);
queue[head % QUEUE_SIZE] = new_task;
st_release_gpu_u64(&queue_heads[worker_id], head + 1);

// Worker: 获取任务
uint64_t head = ld_acquire_gpu_u64(&queue_heads[worker_id]);
if (local_tail < head) {
    task = queue[local_tail % QUEUE_SIZE];
    local_tail++;
    st_release_gpu_u64(&queue_tails[worker_id], local_tail);
}
```

### 为什么不需要锁？

1. **单生产者**: 只有 Scheduler 写 head
2. **单消费者**: 只有 Worker 写 tail
3. **原子语义**: acquire/release 保证内存可见性
4. **环形缓冲**: 使用模运算避免边界问题

---

## 5. 事件依赖系统

### 来源
- Mirage 代码: `persistent_kernel.cuh` Line 613-627

### 设计原理

任务间的依赖通过事件计数器实现：

```
Task A ──触发──> Event 1 ──等待──> Task B
Task C ──触发──> Event 1 ──等待──> Task B
                    ↑
            needed_triggers = 2
```

### 实现代码

```cpp
// 任务完成时触发事件
if (task.trigger_event != EVENT_INVALID_ID) {
    uint64_t old = atom_add_release_gpu_u64(
        &event_counters[task.trigger_event], 1);
}

// 任务开始前等待事件
if (task.dependent_event != EVENT_INVALID_ID) {
    uint64_t needed = events[event_id].needed_triggers;
    while (ld_acquire_gpu_u64(&event_counters[event_id]) < needed) {
        maco_nanosleep(10);
    }
}
```

### 事件类型

```cpp
enum MacoEventType {
    EVENT_TYPE_TASK_DONE = 1,      // 单个任务完成
    EVENT_TYPE_BARRIER = 2,         // 屏障同步
    EVENT_TYPE_LAUNCH_TASK = 3,     // 启动单个任务
    EVENT_TYPE_LAUNCH_MASSIVE = 4,  // 启动大批量任务
};
```

---

## 6. 关键发现：不依赖 NVSHMEM

### Mirage 的两条路径

在分析 Mirage 源码时，我们发现了关键信息：

```cpp
// mirage/include/mirage/persistent_kernel/persistent_kernel.cuh Line 613-627

#ifdef USE_NVSHMEM
    // NVSHMEM 路径: 用于跨 GPU 同步
    nvshmem_signal_wait_until(...);
#else
    // 纯 CUDA 路径: 用于单 GPU 内同步
    while (ld_acquire_sys_u64(event_ptr) < expected) {
        // 轮询等待
    }
#endif
```

### MACO 的设计选择

基于这个发现，MACO 采用了不依赖 NVSHMEM 的设计：

| 组件 | NVSHMEM | MACO 替代方案 |
|------|---------|---------------|
| 跨 GPU 事件 | `nvshmem_signal_wait_until` | PyTorch NCCL |
| GPU 内同步 | 不需要 | GPU Atomics |
| 内存分配 | `nvshmem_malloc` | `torch.cuda` |

### 依赖要求对比

| 依赖 | Mirage (完整) | MACO |
|------|---------------|------|
| PyTorch | >= 2.0 | >= 2.0 |
| CUDA | >= 11.0 | >= 11.0 |
| NVSHMEM | 必需 | 可选 |
| MPI | 必需 | 不需要 |

---

## 总结

从 Mirage 学到的核心优化技术：

1. **Persistent Kernel**: 单次 launch，消除重复开销 → **16x 加速**
2. **GPU Atomics**: 标准 PTX 指令，无硬件依赖
3. **Worker/Scheduler 模式**: CTA 角色分配，高效并行
4. **Lock-free 队列**: SPSC 模型，无锁高效
5. **事件依赖系统**: 细粒度同步，支持复杂任务图

最重要的是：**这些技术不依赖 NVSHMEM**，只需要标准 PyTorch + CUDA (sm_70+)。

---

## 附录 A: Mirage 完整架构概览

### Megakernel 方法

Mirage 将整个 LLM 推理融合成单个 Megakernel：
- 一次 launch，持续运行
- 处理所有计算 (attention, linear, normalization 等)
- 内部管理跨 GPU 通信
- 消除 kernel launch 开销

### 任务图模型

```
Task Graph = (Tasks, Events, Dependencies)
```

- **Tasks**: 独立的计算操作
- **Events**: 任务间的同步点
- **Dependencies**: 数据流依赖关系

### 内存层次优化

```
┌─────────────────────────────────────────────┐
│              Global Memory (HBM)            │
│   - 模型权重, KV cache, 输入输出            │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│            Shared Memory (SRAM)             │
│   - 中间激活, 任务描述符, 归约缓冲          │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│               Registers                     │
│   - 线程本地计算                            │
└─────────────────────────────────────────────┘
```

### 支持的操作类型

| 类型 | 操作 |
|------|------|
| 基础层 | embed, rmsnorm, linear, attention, silu_mul |
| 融合层 | rmsnorm_linear, linear_with_residual |
| 通信 | allreduce, allgather |
| MoE | moe_routing, moe_linear |

### 硬件支持

| GPU | Compute | 特性 |
|-----|---------|------|
| Ampere (A100) | SM 80 | 基础 megakernel |
| Ada (RTX 4090) | SM 86/89 | 改进 shared memory |
| Hopper (H100) | SM 90 | TMA, 225KB SRAM |

---

## 参考文献

1. [Mirage: Generating Fast GPU Kernels with Multi-Level Superoptimization](https://arxiv.org/abs/2405.05751)
2. [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219)
3. [NVIDIA PTX ISA - Memory Consistency Model](https://docs.nvidia.com/cuda/parallel-thread-execution/)
4. [CUDA C++ Programming Guide - Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
5. [Mirage GitHub Repository](https://github.com/mirage-project/mirage)
