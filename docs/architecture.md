# MACO 架构设计

**Multi-GPU Async Communication Optimizer**

## 概述

MACO 是一个通用的 PyTorch 优化框架，通过 SM 级别的任务调度实现计算与通信的重叠，无需特殊硬件配置。

## 设计目标

1. **通用性**: 可接入任意 PyTorch 模型
2. **低耦合**: 不依赖 NVSHMEM，只需标准 PyTorch + CUDA
3. **易用性**: 三层 API 设计，从简单到灵活
4. **高性能**: 利用 Mirage 风格的 SM 调度

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          Python API                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ MacoOptimizer│  │ @overlap_comm│  │     TaskGraph        │  │
│  │  (高层 API)  │  │  (装饰器)    │  │    (低层 API)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Communication Backend                        │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐ │
│  │     NCCL (默认)          │  │    NVSHMEM (可选)            │ │
│  │  - torch.distributed     │  │  - 低延迟跨 GPU              │ │
│  │  - 零配置                │  │  - 需要额外安装              │ │
│  └──────────────────────────┘  └──────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                       CUDA Core Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ maco_kernel  │  │ maco_worker  │  │   maco_scheduler     │  │
│  │ (主入口)     │  │ (计算执行)   │  │   (任务分发)         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────────────────────────────────┐ │
│  │ maco_atoms   │  │            maco_types                    │ │
│  │ (GPU原子操作)│  │          (类型定义)                      │ │
│  └──────────────┘  └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
maco/
├── maco/
│   ├── __init__.py           # 包入口
│   ├── _sm.*.so              # 编译后的 CUDA 扩展
│   ├── optimizer.py          # MacoOptimizer 高层 API
│   ├── task_graph.py         # TaskGraph 低层 API
│   ├── decorators.py         # @overlap_comm 装饰器
│   └── csrc/
│       ├── maco_atoms.cuh    # GPU Atomics 原语
│       ├── maco_types.h      # 类型定义
│       ├── maco_worker.cuh   # Worker CTA 实现
│       ├── maco_scheduler.cuh# Scheduler CTA 实现
│       ├── maco_kernel.cu    # 主 kernel + Python 绑定
│       └── setup_sm.py       # 编译配置
├── examples/
│   ├── test_sm_scheduler.py  # SM 调度测试
│   ├── benchmark_nccl_2gpu.py# NCCL 性能测试
│   └── persistent_kernel_demo.py
├── docs/
│   ├── architecture.md       # 本文档
│   └── mirage_learnings.md   # Mirage 学习总结
├── README.md
├── CLAUDE.md
└── pyproject.toml
```

## 核心组件

### 1. GPU Atomics (maco_atoms.cuh)

提供 SM 间同步的基础原语：

```cpp
// 原子加载 (acquire 语义)
uint64_t ld_acquire_gpu_u64(uint64_t* addr);

// 原子存储 (release 语义)
void st_release_gpu_u64(uint64_t* addr, uint64_t val);

// 原子加法 (release 语义)
uint64_t atom_add_release_gpu_u64(uint64_t* addr, uint64_t val);
```

**关键**: 这些是标准 PTX 指令，任何 sm_70+ GPU 都支持。

### 2. 任务类型 (maco_types.h)

定义任务和事件的数据结构：

```cpp
enum MacoTaskType {
    MACO_TASK_TERMINATE = 0,    // 终止
    MACO_TASK_MATMUL = 100,     // 矩阵乘法
    MACO_TASK_ATTENTION = 101,  // Attention
    MACO_TASK_ALLREDUCE = 200,  // AllReduce
    // ...
};

struct MacoTaskDesc {
    int task_type;
    void* inputs[4];
    void* outputs[2];
    int dims[8];
    uint64_t dependent_event;   // 依赖事件
    uint64_t trigger_event;     // 触发事件
};
```

### 3. Worker CTA (maco_worker.cuh)

执行计算任务的核心逻辑：

```
┌─────────────────────────────────────────┐
│              Worker 执行流程             │
├─────────────────────────────────────────┤
│  1. Polling: 等待新任务                  │
│     └── 使用 ld_acquire_gpu_u64          │
│  2. 获取任务: 从队列读取                 │
│  3. 等待依赖: 检查事件计数器             │
│  4. 执行任务: switch(task_type)          │
│  5. 触发事件: atom_add_release_gpu_u64   │
│  6. 更新尾指针: st_release_gpu_u64       │
│  7. 循环                                 │
└─────────────────────────────────────────┘
```

### 4. Scheduler CTA (maco_scheduler.cuh)

分发任务到 Workers：

```
┌─────────────────────────────────────────┐
│           Scheduler 执行流程             │
├─────────────────────────────────────────┤
│  1. 监听: 等待事件完成                   │
│  2. 分发: 根据事件类型分发任务           │
│     ├── LAUNCH_TASK: 单任务分发          │
│     ├── LAUNCH_MASSIVE: 批量分发         │
│     └── BARRIER: 同步所有 Workers        │
│  3. 负载均衡: Round-robin 或最小负载     │
│  4. 终止检查: 发送终止信号               │
└─────────────────────────────────────────┘
```

### 5. Persistent Kernel (maco_kernel.cu)

主入口，根据 blockIdx 分配 CTA 角色：

```cpp
__global__ void maco_persistent_kernel(MacoRuntimeConfig config) {
    if (blockIdx.x < config.num_workers) {
        execute_worker(&config);    // Worker CTA
    } else {
        execute_scheduler(&config); // Scheduler CTA
    }
}
```

## API 设计

### 高层 API: MacoOptimizer (计划中)

```python
from maco import MacoOptimizer

# 自动优化模型
optimizer = MacoOptimizer(model)
optimized_model = optimizer.optimize()

# 推理
output = optimized_model(input)
```

### 中层 API: @overlap_comm (计划中)

```python
from maco import overlap_comm

@overlap_comm(comm_op="allreduce")
def forward_with_comm(x, weight):
    return x @ weight
```

### 低层 API: TaskGraph (计划中)

```python
from maco import TaskGraph

graph = TaskGraph()
t1 = graph.add_task(TaskType.MATMUL, inputs=[A, B], outputs=[C])
t2 = graph.add_task(TaskType.ALLREDUCE, inputs=[C], outputs=[C_reduced])
graph.add_dependency(t2, t1)  # t2 等待 t1 完成
graph.execute()
```

### 已实现: SM 调度演示

```python
from maco._sm import run_sm_scheduling_demo

# 运行 SM 调度演示
data = torch.randn(1024*1024, device="cuda")
result, completion_order = run_sm_scheduling_demo(
    data,
    num_tasks_per_worker=10,
    num_workers=8,
    threads_per_worker=128
)
```

## 性能特点

### 1. 消除 Kernel Launch 开销

| 方式 | 40 任务耗时 | 加速比 |
|------|-------------|--------|
| 传统 (多次 launch) | 0.654 ms | 1x |
| SM 调度 (单次 launch) | 0.041 ms | **16.14x** |

### 2. Worker 并行扩展

| Workers | 吞吐量 (GB/s) | 扩展效率 |
|---------|---------------|----------|
| 1 | 4.18 | 100% |
| 2 | 7.97 | 95% |
| 4 | 15.12 | 90% |
| 8 | 29.81 | 89% |

### 3. NCCL 通信重叠

| 配置 | 总时间 | 重叠效率 |
|------|--------|----------|
| 串行 | 计算 + 通信 | 0% |
| 重叠 | max(计算, 通信) | 92.2% |

## 依赖要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| PyTorch | >= 2.0 | 核心框架 |
| CUDA | >= 11.0 | sm_70+ GPU |
| NVSHMEM | 可选 | 跨 GPU 低延迟 |

## 编译安装

```bash
cd /path/to/maco
python3 maco/csrc/setup_sm.py build_ext --inplace
```

## 测试验证

```bash
CUDA_VISIBLE_DEVICES=0 python3 examples/test_sm_scheduler.py
```

## 未来规划

1. **Phase 2**: 完善 Python API (MacoOptimizer, TaskGraph)
2. **Phase 3**: 集成到 Self-Forcing 视频生成
3. **Phase 4**: 支持更多模型架构 (LLM, Diffusion)
4. **Phase 5**: 自动任务图生成

## 参考

- [Mirage Project](https://github.com/mirage-project/mirage)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [docs/mirage_learnings.md](mirage_learnings.md)
