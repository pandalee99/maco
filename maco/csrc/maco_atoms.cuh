/**
 * MACO GPU Atomics - 标准 PTX 原子操作
 *
 * 这些是 Mirage SM 调度的核心同步原语。
 * 完全不依赖 NVSHMEM，只使用标准 PTX 指令 (sm_70+)。
 *
 * 来源: Mirage mpk_atoms.cuh
 *
 * 内存顺序语义:
 * - acquire: 确保后续读取能看到之前的写入
 * - release: 确保之前的写入对其他线程可见
 * - relaxed: 无顺序保证，最快
 *
 * 作用域:
 * - .gpu: GPU 内所有线程可见
 * - .sys: 系统级 (包括 CPU)，用于跨设备同步
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace maco {

// ============================================================================
// GPU 范围的原子操作 (用于单 GPU 内的 SM 间同步)
// ============================================================================

/**
 * 原子加载 (acquire 语义)
 * 确保后续操作能看到这个地址之前的所有写入
 */
__device__ __forceinline__ uint64_t
ld_acquire_gpu_u64(uint64_t* addr) {
    uint64_t val;
    asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
    return val;
}

/**
 * 原子存储 (release 语义)
 * 确保之前的所有写入对其他线程可见
 */
__device__ __forceinline__ void
st_release_gpu_u64(uint64_t* addr, uint64_t val) {
    asm volatile("st.release.gpu.u64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}

/**
 * 原子加载 (relaxed 语义)
 * 无内存顺序保证，最快
 */
__device__ __forceinline__ uint64_t
ld_relaxed_gpu_u64(uint64_t* addr) {
    uint64_t val;
    asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
    return val;
}

/**
 * 原子存储 (relaxed 语义)
 */
__device__ __forceinline__ void
st_relaxed_gpu_u64(uint64_t* addr, uint64_t val) {
    asm volatile("st.relaxed.gpu.u64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}

/**
 * 原子加法 (release 语义)
 * 返回旧值
 */
__device__ __forceinline__ uint64_t
atom_add_release_gpu_u64(uint64_t* addr, uint64_t val) {
    uint64_t old_val;
    asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
                 : "=l"(old_val) : "l"(addr), "l"(val) : "memory");
    return old_val;
}

/**
 * 原子加法 (relaxed 语义)
 */
__device__ __forceinline__ uint64_t
atom_add_relaxed_gpu_u64(uint64_t* addr, uint64_t val) {
    uint64_t old_val;
    asm volatile("atom.add.relaxed.gpu.u64 %0,[%1],%2;"
                 : "=l"(old_val) : "l"(addr), "l"(val) : "memory");
    return old_val;
}

/**
 * 原子比较并交换 (release 语义)
 * 如果 *addr == compare，则 *addr = val
 * 返回 *addr 的旧值
 */
__device__ __forceinline__ uint64_t
atom_cas_release_gpu_u64(uint64_t* addr, uint64_t compare, uint64_t val) {
    uint64_t old_val;
    asm volatile("atom.cas.release.gpu.u64 %0,[%1],%2,%3;"
                 : "=l"(old_val) : "l"(addr), "l"(compare), "l"(val) : "memory");
    return old_val;
}

// ============================================================================
// 系统范围的原子操作 (用于跨 GPU 或 GPU-CPU 同步)
// ============================================================================

/**
 * 系统级原子加载 (acquire 语义)
 * 用于跨设备事件同步
 */
__device__ __forceinline__ uint64_t
ld_acquire_sys_u64(uint64_t* addr) {
    uint64_t val;
    asm volatile("ld.acquire.sys.u64 %0, [%1];" : "=l"(val) : "l"(addr));
    return val;
}

/**
 * 系统级原子存储 (release 语义)
 */
__device__ __forceinline__ void
st_release_sys_u64(uint64_t* addr, uint64_t val) {
    asm volatile("st.release.sys.u64 [%0], %1;" :: "l"(addr), "l"(val) : "memory");
}

/**
 * 系统级原子加法 (release 语义)
 */
__device__ __forceinline__ uint64_t
atom_add_release_sys_u64(uint64_t* addr, uint64_t val) {
    uint64_t old_val;
    asm volatile("atom.add.release.sys.u64 %0,[%1],%2;"
                 : "=l"(old_val) : "l"(addr), "l"(val) : "memory");
    return old_val;
}

// ============================================================================
// 32-bit 版本 (用于较小的计数器)
// ============================================================================

__device__ __forceinline__ unsigned int
ld_acquire_gpu_u32(unsigned int* addr) {
    unsigned int val;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}

__device__ __forceinline__ void
st_release_gpu_u32(unsigned int* addr, unsigned int val) {
    asm volatile("st.release.gpu.u32 [%0], %1;" :: "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ unsigned int
atom_add_release_gpu_u32(unsigned int* addr, unsigned int val) {
    unsigned int old_val;
    asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;"
                 : "=r"(old_val) : "l"(addr), "r"(val) : "memory");
    return old_val;
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 低功耗等待 (减少轮询时的功耗和竞争)
 * 在现代 GPU 上比空循环更高效
 */
__device__ __forceinline__ void
maco_nanosleep(unsigned ns = 10) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(ns);
#else
    // Fallback: 空循环
    for (volatile int i = 0; i < ns; i++) {}
#endif
}

/**
 * Warp 级别的 leader 选举
 * 返回 warp 中活跃的第一个线程 ID
 */
__device__ __forceinline__ int
elect_warp_leader() {
    unsigned mask = __activemask();
    return __ffs(mask) - 1;  // 找到第一个活跃位
}

/**
 * 检查当前线程是否是 warp leader
 */
__device__ __forceinline__ bool
is_warp_leader() {
    return (threadIdx.x % 32) == elect_warp_leader();
}

}  // namespace maco
