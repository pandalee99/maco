"""
NVSHMEM Backend for MACO

使用 NVSHMEM 替代 NCCL 进行高效 GPU-GPU 通信。
NVSHMEM 提供 GPU 直连内存传输，绕过 CPU 和 PCIe 瓶颈。

参考: Mirage 的 NVSHMEM 实现 (tasks/ampere/allreduce.cuh)

使用方式:
    from maco.comm.nvshmem_backend import init_nvshmem, nvshmem_allreduce

    # 在 MPI 环境下初始化
    init_nvshmem()

    # 执行 allreduce
    result = nvshmem_allreduce(tensor)

运行要求:
    - NVSHMEM >= 3.5.19
    - MPI (OpenMPI 或 MPICH)
    - 使用 mpirun 启动: mpirun -np 8 python script.py
"""

import torch
from typing import Optional, Tuple
import os

# NVSHMEM 初始化状态
_nvshmem_initialized = False
_nvshmem_available = False

# 尝试加载 NVSHMEM C++ 扩展
try:
    from maco._C import (
        nvshmem_init as _nvshmem_init_impl,
        nvshmem_finalize as _nvshmem_finalize_impl,
        nvshmem_allreduce_sum as _nvshmem_allreduce_sum_impl,
        nvshmem_all_to_all_4D as _nvshmem_all_to_all_4D_impl,
        nvshmem_my_pe as _nvshmem_my_pe_impl,
        nvshmem_n_pes as _nvshmem_n_pes_impl,
        nvshmem_barrier as _nvshmem_barrier_impl,
    )
    _nvshmem_available = True
except ImportError as e:
    _nvshmem_available = False
    _import_error = str(e)


def is_nvshmem_available() -> bool:
    """检查 NVSHMEM 是否可用。"""
    return _nvshmem_available


def init_nvshmem() -> bool:
    """
    初始化 NVSHMEM。

    必须在 MPI 环境下运行 (使用 mpirun 启动)。

    Returns:
        bool: 初始化是否成功
    """
    global _nvshmem_initialized

    if _nvshmem_initialized:
        return True

    if not _nvshmem_available:
        print(f"[MACO] NVSHMEM not available: {_import_error}")
        print("[MACO] Please build NVSHMEM extension first:")
        print("       cd /mini_mirage/maco && python maco/csrc/setup.py install")
        return False

    try:
        _nvshmem_init_impl()
        _nvshmem_initialized = True

        # 打印初始化信息
        pe = _nvshmem_my_pe_impl()
        n_pes = _nvshmem_n_pes_impl()
        if pe == 0:
            print(f"[MACO] NVSHMEM initialized with {n_pes} PEs")

        return True
    except Exception as e:
        print(f"[MACO] NVSHMEM initialization failed: {e}")
        return False


def finalize_nvshmem():
    """清理 NVSHMEM 资源。"""
    global _nvshmem_initialized

    if not _nvshmem_initialized:
        return

    if _nvshmem_available:
        try:
            _nvshmem_finalize_impl()
        except:
            pass

    _nvshmem_initialized = False


def nvshmem_my_pe() -> int:
    """获取当前进程的 PE (Processing Element) ID。"""
    if not _nvshmem_initialized:
        raise RuntimeError("NVSHMEM not initialized. Call init_nvshmem() first.")
    return _nvshmem_my_pe_impl()


def nvshmem_n_pes() -> int:
    """获取总 PE 数量。"""
    if not _nvshmem_initialized:
        raise RuntimeError("NVSHMEM not initialized. Call init_nvshmem() first.")
    return _nvshmem_n_pes_impl()


def nvshmem_barrier():
    """NVSHMEM 全局栅栏同步。"""
    if not _nvshmem_initialized:
        raise RuntimeError("NVSHMEM not initialized. Call init_nvshmem() first.")
    _nvshmem_barrier_impl()


def nvshmem_allreduce(
    tensor: torch.Tensor,
    op: str = "sum"
) -> torch.Tensor:
    """
    使用 NVSHMEM 执行 allreduce 操作。

    相比 NCCL，NVSHMEM 的优势:
    - GPU 直连内存传输 (无需 CPU 中转)
    - 细粒度同步 (使用 CUDA atomics)
    - 更低的通信延迟

    Args:
        tensor: 输入张量 (必须在 CUDA 设备上)
        op: 规约操作 ("sum", "max", "min")

    Returns:
        规约后的张量

    Example:
        >>> x = torch.randn(1024, device="cuda")
        >>> result = nvshmem_allreduce(x)  # 所有 GPU 上的 x 求和
    """
    if not _nvshmem_initialized:
        raise RuntimeError("NVSHMEM not initialized. Call init_nvshmem() first.")

    if not tensor.is_cuda:
        raise ValueError("NVSHMEM requires CUDA tensors")

    # 确保张量是连续的
    tensor = tensor.contiguous()

    if op == "sum":
        return _nvshmem_allreduce_sum_impl(tensor)
    else:
        raise ValueError(f"Unsupported reduction op: {op}. Supported: sum")


def nvshmem_all_to_all_4D(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """
    使用 NVSHMEM 执行 all_to_all_4D 操作。

    这是 Sequence Parallel 的核心通信原语，用于在头并行和序列并行之间切换:
    - SP -> HP: scatter_dim=1, gather_dim=2
    - HP -> SP: scatter_dim=2, gather_dim=1

    Args:
        tensor: 4D 输入张量 [batch, seq, heads, head_dim]
        scatter_dim: 分散维度 (将此维度的数据分发到不同 GPU)
        gather_dim: 聚集维度 (从不同 GPU 收集数据到此维度)

    Returns:
        转换后的张量

    Example:
        # Sequence Parallel -> Head Parallel
        # Input: [B, S/P, H, D] -> Output: [B, S, H/P, D]
        >>> x = torch.randn(1, 256, 32, 128, device="cuda")  # S/P=256, H=32
        >>> y = nvshmem_all_to_all_4D(x, scatter_dim=1, gather_dim=2)
        >>> # y.shape = [1, 2048, 4, 128]  # S=2048, H/P=4 (假设 P=8)
    """
    if not _nvshmem_initialized:
        raise RuntimeError("NVSHMEM not initialized. Call init_nvshmem() first.")

    if not tensor.is_cuda:
        raise ValueError("NVSHMEM requires CUDA tensors")

    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")

    # 确保张量是连续的
    tensor = tensor.contiguous()

    return _nvshmem_all_to_all_4D_impl(tensor, scatter_dim, gather_dim)


# Fallback 实现 (当 NVSHMEM 不可用时使用 NCCL)
def nvshmem_allreduce_fallback(
    tensor: torch.Tensor,
    op: str = "sum"
) -> torch.Tensor:
    """
    NCCL fallback 实现。
    当 NVSHMEM 不可用时使用。
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("Neither NVSHMEM nor torch.distributed is initialized")

    tensor = tensor.contiguous()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM if op == "sum" else None)
    return tensor


def nvshmem_all_to_all_4D_fallback(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """
    NCCL fallback 实现。
    当 NVSHMEM 不可用时使用。
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("Neither NVSHMEM nor torch.distributed is initialized")

    world_size = dist.get_world_size()

    # 使用 NCCL all_to_all_single
    # 参考 fastvideo/utils/communications.py 的实现
    input_list = list(torch.chunk(tensor, world_size, dim=scatter_dim))
    output_list = [torch.empty_like(x) for x in input_list]

    dist.all_to_all(output_list, input_list)

    return torch.cat(output_list, dim=gather_dim)
