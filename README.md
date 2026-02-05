# MACO

<p align="center">
  <b>Multi-GPU Async Communication Optimizer</b><br>
  <i>SM-level Task Scheduling for Compute-Communication Overlap</i>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch 2.0+"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/cuda-11.0+-green.svg" alt="CUDA 11.0+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

MACO 是一个 PyTorch 优化框架，通过 **SM 级别任务调度** 实现计算与通信的高效重叠。核心技术借鉴 [Mirage](https://github.com/mirage-project/mirage)，**无需 NVSHMEM**，仅依赖标准 PyTorch + CUDA。

## ✨ 核心特性

| 特性 | 描述 | 效果 |
|:-----|:-----|:-----|
| **Persistent Kernel** | 单次 launch 执行多任务 | 16x 加速 |
| **GPU Atomics** | PTX 指令实现 SM 间同步 | 1-2μs 延迟 |
| **TaskGraph API** | 细粒度任务依赖控制 | 自动调度 |
| **Compute-Comm Overlap** | 计算与通信并行执行 | 1.27x 加速 |
| **Multi-GPU NCCL** | 异步通信原语 | 4+ GPU 支持 |

## 🏗️ 架构

```
┌──────────────────────────────────────────────────────────────┐
│                        User API                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │  TaskGraph  │  │   linear()  │  │  overlap().auto()   │   │
│  │   compile   │  │   matmul()  │  │                     │   │
│  │   execute   │  │   custom()  │  │                     │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│                     Task Scheduler                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Dependency Inference  →  Wave Grouping  →  Execution   │ │
│  └─────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│                       CUDA Runtime                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Worker CTA  │  │ Scheduler CTA│  │   GPU Atomics    │   │
│  │  (Compute)   │  │   (Dispatch) │  │  (Sync Prims)    │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│                    Communication Backend                      │
│           NCCL (default)    │    NVSHMEM (optional)          │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/your-org/maco.git
cd maco
python3 maco/csrc/setup_sm.py build_ext --inplace
```

**环境要求**: Python 3.8+ / PyTorch 2.0+ / CUDA 11.0+ / GPU sm_70+ (V100/A100/H100)

### 基础用法

```python
import torch
from maco import TaskGraph

# 创建任务图
graph = TaskGraph(num_workers=8)

# 定义计算任务
x = torch.randn(32, 512, device="cuda")
w1 = torch.randn(1024, 512, device="cuda")
w2 = torch.randn(512, 1024, device="cuda")

t1 = graph.linear(x, w1, name="proj_up")
t2 = graph.linear(t1.output, w2, name="proj_down")

# 编译并执行
graph.compile()
graph.execute()

print(t2.output.shape)  # torch.Size([32, 512])
```

### 计算-通信重叠

```python
# 计算任务
compute_tasks = []
h = x
for i, w in enumerate(weights):
    t = graph.linear(h, w, name=f"layer_{i}")
    compute_tasks.append(t)
    h = t.output

# 通信任务
comm_task = graph.allreduce(gradient, name="sync")

# 标记重叠并自动分配 wave
graph.overlap(compute_tasks, [comm_task]).auto_waves()

graph.compile()
graph.execute()
```

## 📊 性能

在 NVIDIA L20 GPU 上测试：

```
┌────────────────────────────────────────────────────────┐
│              SM Scheduling Performance                  │
├──────────────┬─────────────────┬───────────────────────┤
│   Workers    │  Throughput     │      Scaling          │
├──────────────┼─────────────────┼───────────────────────┤
│      1       │    4.18 GB/s    │        1.0x           │
│      4       │   15.12 GB/s    │        3.6x           │
│      8       │   29.81 GB/s    │        7.1x           │
└──────────────┴─────────────────┴───────────────────────┘

Kernel Launch Overhead:     16.14x speedup
Compute-Comm Overlap:       92.2% efficiency
```

## 🧪 测试

```bash
# 单元测试 (单 GPU)
pytest tests/ -v

# 多 GPU 测试 (4x GPU)
torchrun --nproc_per_node=4 -m pytest tests/test_comm.py -v
torchrun --nproc_per_node=4 -m pytest tests/test_overlap.py -v

# 性能验证
torchrun --nproc_per_node=4 python examples/test_real_overlap.py
```

## 📁 项目结构

```
maco/
├── maco/
│   ├── task_graph/              # Python API
│   │   ├── __init__.py          # TaskGraph, TaskNode, TaskSchedule
│   │   ├── runtime.py           # StreamRuntime
│   │   ├── overlap_scheduler.py # OverlapScheduler, OverlapRuntime
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── validation.py        # Input validation
│   ├── comm/                    # Multi-GPU Communication (Phase 3)
│   │   ├── __init__.py          # Module exports
│   │   ├── process_group.py     # ProcessGroupManager
│   │   └── nccl_ops.py          # Async NCCL operations
│   ├── sync/                    # Synchronization Primitives (Phase 3)
│   │   ├── __init__.py          # Module exports
│   │   ├── signal_wait.py       # Signal-Wait, OverlapContext
│   │   └── stream_manager.py    # StreamManager
│   └── csrc/                    # CUDA Core
│       ├── maco_kernel.cu       # Persistent Kernel
│       ├── maco_worker.cuh      # Worker CTA
│       ├── maco_scheduler.cuh   # Scheduler CTA
│       └── maco_atoms.cuh       # GPU Atomics (PTX)
├── tests/                       # Unit tests (55+ tests)
├── examples/                    # Example scripts
└── docs/                        # Documentation
```

## 🗺️ Roadmap

- [x] **Phase 1**: CUDA Core (GPU Atomics, Persistent Kernel)
- [x] **Phase 2**: TaskGraph API + Validation + Tests
- [x] **Phase 3**: Multi-GPU Support (NCCL, Signal-Wait, Compute-Comm Overlap)
- [ ] **Phase 4**: Model Integration (self-forcing, vLLM)

## 📚 Documentation

- [Architecture Design](docs/architecture.md)
- [Technical Internals](docs/mirage_learnings.md)

## 🙏 Acknowledgments

SM scheduling techniques learned from [Mirage](https://github.com/mirage-project/mirage).

## 📄 License

MIT License
