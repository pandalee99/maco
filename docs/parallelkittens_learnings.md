# 从 ThunderKittens/ParallelKittens 学到的多 GPU 技术

本文档总结了从 [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) 项目的 ParallelKittens 部分学习的多 GPU 编程技术，为 MACO 框架提供参考。

## 目录

1. [项目概述](#1-项目概述)
2. [PGL: Parallel Global Layout](#2-pgl-parallel-global-layout)
3. [Multicast Memory 多播内存](#3-multicast-memory-多播内存)
4. [multimem PTX 指令](#4-multimem-ptx-指令)
5. [跨设备屏障同步](#5-跨设备屏障同步)
6. [Compute-Comm SM 分离](#6-compute-comm-sm-分离)
7. [典型算子实现](#7-典型算子实现)
8. [对 MACO 的启发](#8-对-maco-的启发)

---

## 1. 项目概述

### ThunderKittens 简介

ThunderKittens 是 Stanford Hazy Research Lab 开发的 CUDA 内核框架，核心特点：

- **Tile-based**: 围绕 16x16 tile 构建原语
- **Header-only**: 只需 `#include "kittens.cuh"`
- **High Performance**: FlashAttention-3 达到 H100 理论峰值 86%

### ParallelKittens

ParallelKittens 是 ThunderKittens 的多 GPU 扩展，利用 NVLink/NVSwitch 实现：

- **In-network AllReduce**: 使用 NVSwitch 的 multicast 功能
- **PGL (Parallel Global Layout)**: 跨设备内存抽象
- **SM 分离**: 计算 SM 和通信 SM 分工协作

---

## 2. PGL: Parallel Global Layout

### 2.1 设计思想

PGL 将多个 GPU 的内存统一抽象为一个逻辑视图：

```
Device 0          Device 1          Device 2
┌─────────┐      ┌─────────┐      ┌─────────┐
│ gls[0]  │      │ gls[1]  │      │ gls[2]  │
│ (local) │      │ (local) │      │ (local) │
└─────────┘      └─────────┘      └─────────┘
     ↑               ↑               ↑
     └───────────────┴───────────────┘
                     │
              ┌──────┴──────┐
              │   mc_ptr    │  (Multicast 指针)
              │ (NVSwitch)  │
              └─────────────┘
```

### 2.2 PGL 定义

```cpp
// 来源: include/types/system/pgl.cuh
template<
    kittens::ducks::gl::all _GL,  // 底层 Global Layout 类型
    int NUM_DEVICES = 8,          // GPU 数量
    bool MULTICAST = true,        // 是否启用多播
    typename... TMA_Types
>
struct pgl {
    using GL = _GL;
    using T = GL::dtype;

    T *mc_ptr;              // 多播指针 (NVSwitch 地址)
    GL gls[NUM_DEVICES];    // 每个设备的本地 GL

    // 访问特定设备的数据
    __host__ __device__ const GL &operator[](int idx) const {
        return gls[idx];
    }

    // 通过多播指针访问
    __device__ inline T* mc_ptr_at(const coord<> &idx) const {
        const GL &gl = gls[0];
        return &mc_ptr[((idx.b * gl.depth() + idx.d) * gl.rows() + idx.r) * gl.cols() + idx.c];
    }
};
```

### 2.3 使用方式

```cpp
// 定义 PGL 类型
using C_pgl = pgl<gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, true>;

// 在 kernel 中使用
__device__ void kernel(const globals &G) {
    // 访问本地设备数据
    G.C[G.dev_idx][{row, col}] = value;

    // 通过多播写入所有设备
    *G.C.mc_ptr_at({row, col}) = value;
}
```

---

## 3. Multicast Memory 多播内存

### 3.1 NVSwitch 多播原理

NVSwitch 支持将一次写入广播到多个 GPU：

```
           ┌───────────────┐
           │   NVSwitch    │
           │  (Multicast)  │
           └───────┬───────┘
                   │
     ┌─────────────┼─────────────┐
     │             │             │
     ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│  GPU 0  │  │  GPU 1  │  │  GPU 2  │
└─────────┘  └─────────┘  └─────────┘

单次写入 mc_ptr → 所有 GPU 同时收到数据
```

### 3.2 多播地址创建

```python
# 来源: include/pyutils/parallel_tensor.cuh (概念)
class TKParallelTensor:
    def __init__(self, local_tensor, world_size):
        # 1. 收集所有设备的本地指针
        all_ptrs = all_gather_ptrs(local_tensor.data_ptr())

        # 2. 创建多播映射
        self.mc_ptr = create_multicast_mapping(all_ptrs)

        # 3. 存储本地 rank
        self.local_rank = torch.distributed.get_rank()
```

---

## 4. multimem PTX 指令

### 4.1 核心指令

ThunderKittens 封装了 NVIDIA 的 `multimem` PTX 指令，用于跨 GPU 原子操作：

```cpp
// 来源: include/common/multimem.cuh

// 加载并归约 (Load-Reduce)
template <>
struct multimem<bf16_2> {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(bf16_2 &dst, const bf16_2 *src) {
        if constexpr (Op == reduce_op::ADD) {
            asm volatile(
                "multimem.ld_reduce.weak.global.add.acc::f32.bf16x2 %0, [%1];"
                : "=r"(*reinterpret_cast<uint32_t *>(&dst))
                : "l"(src)
                : "memory"
            );
        }
    }

    // 多播存储
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(bf16_2 *dst, const bf16_2 &src) {
        asm volatile(
            "multimem.st.weak.global.bf16x2 [%0], %1;"
            :: "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src))
            : "memory"
        );
    }

    // 远程归约
    template <reduce_op Op>
    __device__ static inline void red(bf16_2 *dst, const bf16_2 &src) {
        asm volatile(
            "multimem.red.release.sys.global.add.bf16x2 [%0], %1;"
            :: "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src))
            : "memory"
        );
    }
};
```

### 4.2 支持的数据类型

| 类型 | 支持的操作 |
|------|-----------|
| `int`, `uint` | ADD, MIN, MAX |
| `float`, `float2` | ADD |
| `bf16`, `bf16_2` | ADD, MIN, MAX |
| `half`, `half_2` | ADD, MIN, MAX |

### 4.3 内存顺序

```cpp
enum class memory_model {
    WEAK = 0,    // 弱顺序，最快
    STRONG = 1   // 强顺序 (acquire/release)
};

// WEAK: multimem.ld_reduce.weak.global.add...
// STRONG: multimem.ld_reduce.acquire.sys.global.add...
```

---

## 5. 跨设备屏障同步

### 5.1 barrier_t 类型

```cpp
// 来源: include/types/system/pgl.cuh Line 174-175
template <int NUM_DEVICES>
using barrier_t = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, true>;
```

### 5.2 barrier_all 实现原理

```cpp
// 概念实现 (简化)
__device__ void barrier_all(barrier_t<N> &barrier, coord idx, int dev_idx) {
    // 1. 每个设备写入自己的 barrier slot
    barrier[dev_idx][idx] = 1;

    // 2. 通过多播指针读取并累加所有设备的值
    int sum;
    multimem<int>::ld_reduce<reduce_op::ADD>(sum, barrier.mc_ptr_at(idx));

    // 3. 等待所有设备到达
    while (sum < N) {
        multimem<int>::ld_reduce<reduce_op::ADD>(sum, barrier.mc_ptr_at(idx));
    }
}
```

### 5.3 使用示例

```cpp
// 来源: kernels/parallel/all_reduce/all_reduce.cu Line 60-64
namespace all_reduce_barrier {
    __device__ inline void kernel(const globals &G) {
        barrier_all(G.barrier, {0}, G.dev_idx);
    }
}
```

---

## 6. Compute-Comm SM 分离

### 6.1 设计思想

ParallelKittens 将 GPU 的 SM 分为两类：
- **计算 SM (Compute SM)**: 执行矩阵乘法等计算任务
- **通信 SM (Comm SM)**: 执行跨 GPU 的 AllReduce 等通信任务

```
GPU SMs (132 total on H100):
┌────────────────────────────────────────────────────────┐
│ Compute SMs (blockIdx < num_comp_sms)                 │
│ ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐     │
│ │ SM0 │ SM1 │ SM2 │ ... │SM120│SM121│SM122│SM123│     │
│ └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘     │
├────────────────────────────────────────────────────────┤
│ Comm SMs (blockIdx >= num_comp_sms)                   │
│ ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐     │
│ │SM124│SM125│SM126│SM127│SM128│SM129│SM130│SM131│     │
│ └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘     │
└────────────────────────────────────────────────────────┘
```

### 6.2 代码实现

```cpp
// 来源: kernels/parallel/gemm_ar/gemm_ar_h100.cu Line 224-228
__device__ inline void main_kernel(const globals &G) {
    if (blockIdx.x < G.num_comp_sms)
        comp_sm(G);    // 计算任务
    else
        comm_sm(G);    // 通信任务
}
```

### 6.3 计算 SM 职责

```cpp
__device__ inline void comp_sm(const globals &G) {
    // Producer warpgroup: 加载数据
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // TMA 异步加载 A, B 矩阵
        tma::load_async(inputs[stage].A[i], G.A, {row_idx, red_idx}, inputs_arrived[stage]);
        tma::load_async(inputs[stage].B, G.B, {red_idx, col_idx}, inputs_arrived[stage]);

        // 存储结果并发送信号
        tma::store_async(G.C[G.dev_idx], outputs.C[i], {row_idx, col_idx});
        signal(G.barrier, {row_idx, col_idx}, signal_dev_idx, 1);  // 通知通信 SM
    }
    // Consumer warpgroup: 执行 MMA
    else {
        warpgroup::mma_AB(C_accum, inputs[stage].A[warpgroup_id], inputs[stage].B);
    }
}
```

### 6.4 通信 SM 职责

```cpp
__device__ inline void comm_sm(const globals &G) {
    for (int task_id = ...; task_id < num_blocks; task_id += ...) {
        // 1. 等待计算 SM 完成
        if (threadIdx.x == 0)
            wait(G.barrier, {row_idx, col_idx}, G.dev_idx, globals::NUM_DEVICES);
        __syncthreads();

        // 2. 执行 in-network AllReduce
        group<config::NUM_WARPS>::all_reduce<ROW_BLOCK, COL_BLOCK, reduce_op::ADD>(
            G.C, {row_idx, col_idx}
        );
    }
}
```

---

## 7. 典型算子实现

### 7.1 AllReduce

```cpp
// 来源: kernels/parallel/all_reduce/all_reduce.cu
__device__ inline void kernel(const globals &G) {
    const size_t idx = N_per_dev * G.dev_idx +
                       NUM_ELEMS_PER_BLOCK * blockIdx.x +
                       NUM_ELEMS_PER_INST * threadIdx.x;

    bf16_2 tmp;
    // 读取并归约所有设备的数据
    multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, &G.tensor.mc_ptr[idx]);
    // 写回
    multimem<bf16_2>::st(&G.tensor.mc_ptr[idx], tmp);
}
```

### 7.2 GEMM + AllReduce

```cpp
// 来源: kernels/parallel/gemm_ar/gemm_ar_h100.cu
void entrypoint(A, B, C, barrier, num_comm_sms) {
    globals G {
        .A = tensor_to_gl<A_gl>(A),
        .B = tensor_to_gl<B_gl>(B),
        .C = parallel_tensor_to_pgl<C_pgl>(C),
        .barrier = parallel_tensor_to_pgl<barrier_pgl>(barrier),
        .dev_idx = barrier.local_rank_,
        .num_comm_sms = num_comm_sms,
        .num_comp_sms = NUM_BLOCKS - num_comm_sms
    };

    launch_kernel<config, globals, main_kernel>(G);
    launch_kernel<config, globals, epilogue_kernel>(G);  // 重置 barrier
}
```

### 7.3 GEMM + ReduceScatter

```cpp
// 来源: kernels/parallel/gemm_rs/gemm_rs_h100.cu
// 关键差异: 使用 tma::store_add_async 原子累加
for (int i = 0; i < 2; i++)
    tma::store_add_async(G.C[dev_idx], outputs.C[i], {row_idx, col_idx});
```

---

## 8. 对 MACO 的启发

### 8.1 可借鉴的技术

| 技术 | ParallelKittens 实现 | MACO 应用方向 |
|------|---------------------|--------------|
| **PGL 抽象** | 多设备统一视图 | 通用多 GPU 内存抽象 |
| **multimem 指令** | PTX inline asm | 直接集成到 SM 调度器 |
| **SM 分离** | 计算/通信 SM | 扩展到任务级别分离 |
| **barrier_t** | 跨设备同步 | Event 系统的多 GPU 扩展 |

### 8.2 关键差异

| 方面 | ParallelKittens | MACO |
|------|-----------------|------|
| **硬件要求** | NVSwitch (H100/B200) | 无特殊要求 (sm_70+) |
| **通信后端** | NVLink + NVSwitch | NCCL (可选 NVSHMEM) |
| **融合粒度** | 算子级融合 | SM 级任务调度 |
| **灵活性** | 特定算子 | 通用框架 |

### 8.3 MACO 可能的扩展

1. **PGL 集成**: 当检测到 NVSwitch 时，使用 PGL 加速多 GPU 通信
2. **混合调度**: 计算 SM 使用 Mirage Worker 模式，通信 SM 使用 ParallelKittens 模式
3. **multimem 支持**: 在支持的硬件上使用 multimem 指令替代 NCCL

---

## 9. 关键代码文件

| 文件 | 内容 |
|------|------|
| `include/types/system/pgl.cuh` | PGL 类型定义 |
| `include/common/multimem.cuh` | multimem PTX 封装 |
| `include/pyutils/parallel_tensor.cuh` | Python 绑定 |
| `kernels/parallel/all_reduce/all_reduce.cu` | AllReduce 实现 |
| `kernels/parallel/gemm_ar/gemm_ar_h100.cu` | GEMM+AR 融合 |
| `kernels/parallel/gemm_rs/gemm_rs_h100.cu` | GEMM+RS 融合 |

---

## 参考文献

1. [ThunderKittens Single GPU Paper](https://arxiv.org/abs/2410.20399)
2. [ThunderKittens Multiple GPUs Paper](https://arxiv.org/abs/2511.13940)
3. [NVIDIA PTX ISA - Multicast](https://docs.nvidia.com/cuda/parallel-thread-execution/)
4. [NVSwitch Architecture](https://www.nvidia.com/en-us/data-center/nvlink/)
5. [ThunderKittens Blog Posts](https://hazyresearch.stanford.edu/blog)
