# MACO: Multi-GPU Async Communication Optimizer

MACO 是一个多 GPU 异步通信优化框架，专注于通过计算-通信重叠来加速分布式深度学习推理。

## 核心特性

- **计算-通信重叠**：使用独立 CUDA Stream 并行执行 AllReduce 和矩阵计算
- **简单 API**：一行代码实现通信优化
- **兼容 PyTorch**：与现有 PyTorch 代码无缝集成

## 安装

```bash
cd /mini_mirage/maco
pip install -e .
```

## 快速开始

### 单 GPU（功能验证）

```python
import torch
import maco

# 初始化
maco.init()

print(f"MACO version: {maco.__version__}")
print(f"Rank: {maco.get_rank()}, World Size: {maco.get_world_size()}")
```

### 多 GPU 计算-通信重叠

这是 MACO 的核心功能：**在执行 AllReduce 的同时，并行执行下一层计算**。

```bash
# 使用 2 个 GPU 运行
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 your_script.py
```

```python
import torch
import maco

# 初始化 MACO（自动检测分布式环境）
maco.init(backend="nccl")

device = torch.device(f"cuda:{maco.get_rank()}")

# 准备数据
a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
tensor_to_reduce = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

# 定义计算函数
def compute_fn():
    return torch.matmul(a, b)

# 核心 API：计算和通信并行执行
compute_result, reduced_tensor = maco.overlap_compute_and_comm(
    compute_fn=compute_fn,
    comm_tensor=tensor_to_reduce,
    comm_op="allreduce"
)

# 清理
maco.cleanup()
```

## API 参考

### `maco.init(backend="nccl", overlap_mode="stream")`

初始化 MACO。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `backend` | 通信后端 (`"nccl"` 或 `"gloo"`) | `"nccl"` |
| `overlap_mode` | 重叠模式 (`"stream"` 或 `"none"`) | `"stream"` |

### `maco.overlap_compute_and_comm(compute_fn, comm_tensor, comm_op="allreduce")`

并行执行计算和通信。

| 参数 | 说明 |
|------|------|
| `compute_fn` | 计算函数（无参数，返回计算结果） |
| `comm_tensor` | 需要进行集合通信的张量 |
| `comm_op` | 通信操作类型（目前支持 `"allreduce"`） |

**返回**：`(compute_result, comm_tensor)` 元组

### `maco.get_rank()` / `maco.get_world_size()`

获取当前进程的 rank 和总进程数。

### `maco.cleanup()`

清理 MACO 资源，销毁进程组。

## 性能测试结果

### 计算-通信重叠效果

| 场景 | 顺序执行 | 并行执行 | 加速比 |
|------|---------|---------|--------|
| 小计算 + 中等通信 | 0.71 ms | 0.67 ms | 1.05x |
| 大计算 + 大通信 | 3.72 ms | 2.47 ms | **1.51x** |

平均加速 **1.22x**，最高节省 **33.7%** 时间。

### 适用场景

**高收益场景**：
- 多 GPU 张量并行推理
- 大模型（7B+）
- 大 batch size 解码

**低收益场景**：
- 单 GPU 推理
- 小模型
- 通信量极小的场景

## 运行 Benchmark

```bash
# 计算-通信重叠测试
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 benchmarks/benchmark_overlap.py

# 推理性能对比（PyTorch vs vLLM）
CUDA_VISIBLE_DEVICES=2 python benchmarks/benchmark_inference.py --mode baseline
CUDA_VISIBLE_DEVICES=2 python benchmarks/benchmark_inference.py --mode vllm
```

## 项目结构

```
maco/
├── maco/
│   ├── __init__.py       # 主入口，核心 API
│   ├── core/             # 核心模块（Context, SM 管理）
│   ├── comm/             # 通信模块（NCCL 后端）
│   └── overlap/          # 重叠管理器
├── benchmarks/           # 性能测试脚本
├── examples/             # 使用示例
└── docs/                 # 文档
```

## License

MIT
