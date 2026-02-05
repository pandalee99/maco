# 从 FlashOverlap 学到的通算重叠技术

本文档总结了从 [FlashOverlap](https://github.com/infinigence/FlashOverlap) 项目学习的通算重叠技术，为 MACO 框架提供参考。

## 目录

1. [项目概述](#1-项目概述)
2. [核心架构：Signal-Wait 机制](#2-核心架构signal-wait-机制)
3. [Wave Grouping 波组调度](#3-wave-grouping-波组调度)
4. [Tile Reordering 输出重排](#4-tile-reordering-输出重排)
5. [双流并行架构](#5-双流并行架构)
6. [CUTLASS 集成方式](#6-cutlass-集成方式)
7. [对 MACO 的启发](#7-对-maco-的启发)

---

## 1. 项目概述

### 论文信息

- **论文**: *Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering*
- **会议**: EuroSys'26
- **arXiv**: https://arxiv.org/abs/2504.19519

### 核心思想

FlashOverlap 通过 **轻量级信号机制** 实现 GEMM 计算和 NCCL 通信的细粒度重叠：

```
传统方式:
GEMM: [========全部计算========] → AllReduce: [========全部通信========]
                                  ↑
                          等待全部完成

FlashOverlap:
GEMM: [===Wave1===][===Wave2===][===Wave3===][===Wave4===]
                   ↓           ↓           ↓           ↓
                signal      signal      signal      signal
                   ↓           ↓           ↓           ↓
AllReduce:         [==AR1==]   [==AR2==]   [==AR3==]   [==AR4==]
```

---

## 2. 核心架构：Signal-Wait 机制

### 2.1 信号发送 (GEMM 侧)

在 CUTLASS GEMM 的 Epilogue 阶段，每完成一组 tile 就发送信号：

```cpp
// 来源: src/overlap/gemm_with_signal.h Line 339-361
CUTLASS_DEVICE
void end_epilogue() {
    if (threadIdx.x > 0) { return; }  // 只有 thread 0 发送信号

    // 计算当前 tile 属于哪个通信段
    int tile_idx = (threadblock_offset_.row() / ThreadblockM) * kReorderedColumn +
                   threadblock_offset_.column() / ThreadblockN;

    int idx_bound = kCommu_Seg_Array[0];
    int iter_idx = 0;
    while (idx_bound <= tile_idx) {
        iter_idx += 1;
        idx_bound += kCommu_Seg_Array[iter_idx];
    }

    // 原子计数：该段完成的 tile 数量
    int local_order = atomicAdd(&ptr_Monitored_Matrix[iter_idx], 1);
}
```

### 2.2 信号等待 (通信侧)

使用单独的 kernel 轮询等待信号：

```cpp
// 来源: src/wait.cuh
__global__ __forceinline__ void kernel_wait_flag(const int that, int* addr) {
    while (atomicCAS(addr, that, 0) != that) {
        __nanosleep(100);  // 低功耗等待
    }
}
```

### 2.3 工作原理

```
计算 Stream:
  GEMM kernel 运行，每完成一组 tile 就 atomicAdd(signal_counter, 1)

通信 Stream:
  wait_kernel 轮询 atomicCAS 检查 signal_counter
  当计数达到阈值，wait_kernel 退出
  随后 NCCL AllReduce 开始执行
```

---

## 3. Wave Grouping 波组调度

### 3.1 概念

将 GEMM 的输出 tile 分成多个 "波组" (Wave Group)，每个波组完成后立即开始通信。

```
Output Matrix (TM x TN tiles):
┌───┬───┬───┬───┐
│ W1│ W1│ W2│ W2│  Wave 1: 前 4 个 tile
├───┼───┼───┼───┤  Wave 2: 中间 4 个 tile
│ W2│ W2│ W3│ W3│  Wave 3: 最后 4 个 tile
├───┼───┼───┼───┤
│ W3│ W3│ W4│ W4│
└───┴───┴───┴───┘
```

### 3.2 波组大小调优

FlashOverlap 提供两种搜索方法：

**穷举搜索**:
```bash
python search.py --m 4096 --n 4096 --k 4096 --comm_op all_reduce
```

**预测搜索** (大矩阵推荐):
```bash
# 1. 首先生成带宽曲线
python bandwidth.py --comm_op all_reduce

# 2. 使用预测搜索
python search.py --m 4096 --n 4096 --k 4096 --comm_op all_reduce --predictive_search True
```

### 3.3 配置存储

```json
// configs/gemm_4096x4096x4096.json
{
    "m": 4096, "n": 4096, "k": 4096,
    "algo_idx": 12,
    "wave_groups": [8, 8, 8, 8],  // 每个波组的 tile 数量
    "reorder_array": [0, 4, 8, 12, 1, 5, 9, 13, ...]
}
```

---

## 4. Tile Reordering 输出重排

### 4.1 为什么需要重排？

CUTLASS 的默认 tile 执行顺序可能不连续，需要重排使每个波组的 tile 在内存中连续：

```
默认执行顺序:          重排后:
tile 0 → row 0        tile 0 → row 0
tile 1 → row 2        tile 1 → row 0  (连续)
tile 2 → row 1        tile 2 → row 1
tile 3 → row 3        tile 3 → row 1  (连续)
```

### 4.2 重排实现

```cpp
// 来源: src/overlap/gemm_with_signal.h Line 246-256
CUTLASS_DEVICE
MatrixCoord map_to_d(MatrixCoord const &threadblock_offset) {
    // 计算原始 tile 索引
    int tile_idx = (threadblock_offset.row() / ThreadblockM) * kMonitoredColumn +
                   threadblock_offset.column() / ThreadblockN;

    // 查找重排后的位置
    int reordered_tile_idx = ptr_Reorder_Array[tile_idx];

    // 返回新的输出位置
    return MatrixCoord(
        (reordered_tile_idx / kReorderedColumn) * ThreadblockM,
        (reordered_tile_idx % kReorderedColumn) * ThreadblockN
    );
}
```

### 4.3 后处理 RMSNorm

由于输出被重排，需要在 RMSNorm 中恢复原始顺序：

```python
# 来源: example/RMSNorm.py
class ReorderRMSNorm(nn.Module):
    """支持重排输入的 RMSNorm"""
    def forward(self, x, reorder_array):
        # 根据 reorder_array 恢复原始顺序
        # 然后执行标准 RMSNorm
        pass
```

---

## 5. 双流并行架构

### 5.1 Stream 设置

```cpp
// 来源: src/overlap_impl.cu Line 139-141
void OverlapImpl::OverlapInit() {
    // 创建高优先级的通信流
    cudaStreamCreateWithPriority(&this->comm_stream, cudaStreamNonBlocking, -5);
}
```

### 5.2 重叠执行流程

```cpp
// 来源: src/overlap_impl.cu Line 214-264
void OverlapImpl::GemmAllReduceOverlap(...) {
    // 1. 启动 GEMM (计算流)
    signal_func_table[Algo](M, N, K, ..., this->gemm_stream);

    // 2. 对每个波组启动等待+通信 (通信流)
    for (int iter = 0; iter < SegSize; iter++) {
        int this_seg = cseg_cpu_ptr[iter];
        int commSize = M * N / TileNum * this_seg;

        // 等待信号
        kernel_wait_flag<<<1, 1, 0, this->comm_stream>>>(this_seg, mm_ptr + iter);

        // 启动 AllReduce
        ncclAllReduce(c_ptr + acc_addr, c_ptr + acc_addr, commSize,
                      ncclFloat16, ncclSum, this->comm, this->comm_stream);
        acc_addr += commSize;
    }

    // 3. 同步两个流
    cudaEventRecord(this->gemm_finished, this->comm_stream);
    cudaStreamWaitEvent(this->gemm_stream, this->gemm_finished, 0);
}
```

---

## 6. CUTLASS 集成方式

### 6.1 自定义 Epilogue Visitor

FlashOverlap 通过 CUTLASS 的 Epilogue Visitor 机制注入信号逻辑：

```cpp
// 来源: src/overlap/gemm_with_signal.h
template <...>
class EpilogueVisitorSignaling {
public:
    // 在 Epilogue 结束时发送信号
    CUTLASS_DEVICE void end_epilogue() {
        if (threadIdx.x > 0) return;

        // 计算当前 tile 的波组归属
        int tile_idx = ...;
        int wave_idx = find_wave_group(tile_idx);

        // 原子增加该波组的完成计数
        atomicAdd(&signal_counters[wave_idx], 1);
    }
};
```

### 6.2 GEMM 模板实例化

```cpp
// 来源: src/tiling/signal_tiling.cuh
SignalFuncPtr signal_func_table[] = {
    // ThreadblockShape, WarpShape, InstructionShape, Stages, SwizzleSize
    &cutlass_gemm_signal<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 1, 1>,
    &cutlass_gemm_signal<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 2, 1>,
    // ... 更多配置
};
```

---

## 7. 对 MACO 的启发

### 7.1 可借鉴的技术

| 技术 | FlashOverlap 实现 | MACO 应用方向 |
|------|------------------|--------------|
| **Signal-Wait** | atomicCAS + nanosleep | 可用于 SM 调度器的任务同步 |
| **Wave Grouping** | 静态分组 | 可扩展为动态分组 |
| **双流并行** | 计算流 + 通信流 | 与 SM 调度结合 |
| **Tile Reordering** | Epilogue 重定向 | 优化内存访问 |

### 7.2 核心差异

| 方面 | FlashOverlap | MACO |
|------|--------------|------|
| **调度层级** | Stream 级别 | SM 级别 |
| **通信后端** | NCCL | NCCL (可选 NVSHMEM) |
| **计算引擎** | CUTLASS | 任意 (通用框架) |
| **适用范围** | GEMM + AllReduce/RS | 任意计算 + 任意通信 |

### 7.3 潜在改进方向

1. **动态波组**: 根据运行时负载动态调整波组大小
2. **SM 级信号**: 用 GPU Atomics 实现 SM 间信号传递
3. **通用化**: 不绑定 CUTLASS，支持任意计算 kernel

---

## 8. 关键代码文件

| 文件 | 内容 |
|------|------|
| `src/wait.cuh` | 信号等待 kernel |
| `src/overlap_impl.cu` | 重叠执行主逻辑 |
| `src/overlap/gemm_with_signal.h` | CUTLASS Epilogue Visitor |
| `src/tiling/signal_tiling.cuh` | GEMM 模板实例化 |
| `tune/search.py` | 波组大小搜索 |

---

## 参考文献

1. [FlashOverlap Tech Report](https://arxiv.org/abs/2504.19519)
2. [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
3. [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
