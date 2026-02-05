# MACO

**Multi-GPU Async Communication Optimizer**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![CUDA 11.0+](https://img.shields.io/badge/cuda-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MACO 是一个通用的 PyTorch 优化框架，通过 SM 级别的任务调度实现计算与通信的重叠优化。借鉴 [Mirage](https://github.com/mirage-project/mirage) 的核心技术，但**不依赖 NVSHMEM**，只需标准 PyTorch + CUDA。

## 核心特性

| 特性 | 说明 | 效果 |
|------|------|------|
| **Persistent Kernel** | 单次 launch 执行整个推理 | 消除 **16x** kernel 开销 |
| **GPU Atomics** | PTX 指令实现 SM 间同步 | **1-2μs** 延迟 |
| **Lock-free 队列** | GPU 内部任务调度 | 无需 CPU 参与 |
| **通信重叠** | NCCL 集合操作与计算并行 | **92%** 重叠效率 |

## 性能结果

在 NVIDIA L20 GPU 上的测试：

```
============================================================
SM Scheduling Performance
============================================================
Workers    Throughput (GB/s)    Scaling
1          4.18                 1.0x
2          7.97                 1.9x
4          15.12                3.6x
8          29.81                7.1x

Kernel Launch Overhead Elimination: 16.14x speedup
Compute-Communication Overlap: 92.2% efficiency
```

## 快速开始

### 环境要求

#### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GPU (sm_70+) | V100 / A100 / L20 / H100 |
| GPU 架构 | Volta (sm_70) | Ampere+ (sm_80+) |
| GPU 内存 | 8 GB | 16+ GB |

支持的 GPU 架构：
- **sm_70**: V100
- **sm_75**: T4, RTX 2080
- **sm_80**: A100, A800
- **sm_86**: RTX 3090, A6000
- **sm_89**: L40, RTX 4090
- **sm_90**: H100

#### 软件依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | >= 3.8 | 推荐 3.10+ |
| PyTorch | >= 2.0 | 需要 CUDA 支持 |
| CUDA Toolkit | >= 11.0 | 需要 nvcc 编译器 |
| GCC/G++ | >= 7.0 | C++17 支持 |
| NVSHMEM | 可选 | 跨 GPU 低延迟通信 |

#### 验证环境

```bash
# 检查 CUDA
nvcc --version

# 检查 PyTorch CUDA 支持
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 检查 GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### 安装

```bash
git clone https://github.com/your-org/maco.git
cd maco

# 编译 SM 调度模块 (生成 maco/_sm.*.so)
python3 maco/csrc/setup_sm.py build_ext --inplace
```

**注意**: 编译生成的 `.so` 文件是平台相关的，不要提交到 git。每个用户需要在自己的环境中编译。

#### 常见编译问题

1. **nvcc not found**: 确保 CUDA Toolkit 已安装并在 PATH 中
2. **sm_XX not supported**: 检查 GPU 架构是否 >= sm_70
3. **C++17 errors**: 升级 GCC 到 7.0+

### 运行测试

```bash
# SM 调度测试
CUDA_VISIBLE_DEVICES=0 python3 examples/test_sm_scheduler.py

# NCCL 通信重叠测试 (需要 2 GPU)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/benchmark_nccl_2gpu.py

# Persistent Kernel 概念演示
CUDA_VISIBLE_DEVICES=0 python3 examples/persistent_kernel_demo.py
```

### 基础用法

```python
import torch
from maco._sm import run_sm_scheduling_demo

# 准备数据
data = torch.randn(1024 * 1024, device="cuda")

# 运行 SM 调度
result, completion_order = run_sm_scheduling_demo(
    data,
    num_tasks_per_worker=10,
    num_workers=4,
    threads_per_worker=128
)

print(f"Result shape: {result.shape}")
print(f"Task completion order: {completion_order[:10].tolist()}")
```

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                   Python API                     │
│  MacoOptimizer │ @overlap_comm │ TaskGraph      │
├─────────────────────────────────────────────────┤
│              Communication Backend               │
│         NCCL (默认)  │  NVSHMEM (可选)          │
├─────────────────────────────────────────────────┤
│                  CUDA Core                       │
│  ┌───────────┐ ┌───────────┐ ┌───────────────┐ │
│  │Worker CTA │ │Sched. CTA │ │  GPU Atomics  │ │
│  │(计算执行) │ │(任务分发) │ │  (SM 同步)    │ │
│  └───────────┘ └───────────┘ └───────────────┘ │
└─────────────────────────────────────────────────┘
```

## 核心技术

### 1. Persistent Kernel

传统方式：
```
CPU: launch A → wait → launch B → wait → launch C
       ↓ 8μs        ↓ 8μs        ↓ 8μs
```

MACO Persistent Kernel：
```
CPU: launch persistent_kernel ────────────────→ terminate
GPU: [task A][task B][task C][task D]...
        ↑ GPU 内部调度，0 launch 开销
```

### 2. GPU Atomics

```cpp
// 原子加载 (acquire 语义)
uint64_t val = ld_acquire_gpu_u64(addr);

// 原子存储 (release 语义)
st_release_gpu_u64(addr, val);

// 原子加法 (release 语义)
old = atom_add_release_gpu_u64(addr, 1);
```

这些是标准 PTX 指令，任何 sm_70+ GPU 都支持，**不需要 NVSHMEM**。

### 3. Worker/Scheduler CTA 模式

```cpp
__global__ void maco_persistent_kernel(MacoRuntimeConfig config) {
    if (blockIdx.x < config.num_workers) {
        execute_worker(&config);    // 执行计算任务
    } else {
        execute_scheduler(&config); // 分发任务
    }
}
```

详见 [docs/mirage_learnings.md](docs/mirage_learnings.md)

## 项目结构

```
maco/
├── maco/
│   ├── __init__.py
│   ├── _sm.*.so              # 编译后的 CUDA 扩展
│   └── csrc/
│       ├── maco_atoms.cuh        # GPU Atomics 原语
│       ├── maco_types.h          # 类型定义
│       ├── maco_worker.cuh       # Worker CTA 实现
│       ├── maco_scheduler.cuh    # Scheduler CTA 实现
│       ├── maco_kernel.cu        # 主 Kernel + Python 绑定
│       └── setup_sm.py           # 编译配置
├── examples/
│   ├── test_sm_scheduler.py      # SM 调度测试
│   ├── benchmark_nccl_2gpu.py    # NCCL 性能测试
│   └── persistent_kernel_demo.py # 概念演示
├── docs/
│   ├── architecture.md           # 架构设计
│   └── mirage_learnings.md       # Mirage 技术总结
├── README.md
├── CLAUDE.md
└── pyproject.toml
```

## 路线图

- [x] **Phase 1**: 核心 CUDA 组件
  - [x] GPU Atomics (maco_atoms.cuh)
  - [x] Worker CTA (maco_worker.cuh)
  - [x] Scheduler CTA (maco_scheduler.cuh)
  - [x] Persistent Kernel (maco_kernel.cu)
- [ ] **Phase 2**: Python API
  - [ ] MacoOptimizer (高层封装)
  - [ ] TaskGraph (任务图构建)
  - [ ] @overlap_comm 装饰器
- [ ] **Phase 3**: 模型集成
  - [ ] Self-Forcing 视频生成
  - [ ] LLM 推理优化
- [ ] **Phase 4**: 高级功能
  - [ ] 自动任务图生成
  - [ ] 动态负载均衡

## 文档

- [架构设计](docs/architecture.md)
- [Mirage 技术学习](docs/mirage_learnings.md)

## 致谢

本项目的 SM 调度技术学习自 [Mirage](https://github.com/mirage-project/mirage) 项目。

## 引用

```bibtex
@software{maco2024,
  title = {MACO: Multi-GPU Async Communication Optimizer},
  year = {2024},
  url = {https://github.com/your-org/maco}
}
```

## 许可证

MIT License
