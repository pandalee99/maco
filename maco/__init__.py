"""
MACO: Multi-GPU Async Communication Optimizer

A framework for optimizing multi-GPU communication in deep learning,
with explicit SM control and compute-communication overlap.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional

from .core.context import Context, PersistentContext
from .core.sm_manager import SMConfig, Role, auto_config
from .comm.patterns import (
    allreduce, allgather, reduce_scatter, send, recv,
    all_to_all, all_to_all_single, all_to_all_4D,
    SUM, AVG, MIN, MAX, PRODUCT
)
from .comm.scheduler import CommScheduler, CommPhase
from .comm.nccl_backend import NCCLBackend, OverlapMode, get_backend, init_backend
from .overlap.manager import OverlapManager, OverlapRegion
from .config import config

# NVSHMEM backend (optional, requires NVSHMEM installation)
try:
    from .comm.nvshmem_backend import (
        is_nvshmem_available,
        init_nvshmem,
        finalize_nvshmem,
        nvshmem_allreduce,
        nvshmem_all_to_all_4D,
        nvshmem_my_pe,
        nvshmem_n_pes,
        nvshmem_barrier,
    )
    _NVSHMEM_AVAILABLE = is_nvshmem_available()
except ImportError:
    _NVSHMEM_AVAILABLE = False

    def is_nvshmem_available():
        return False

    def init_nvshmem():
        raise RuntimeError("NVSHMEM not available. Build extension first.")

    def finalize_nvshmem():
        pass

    def nvshmem_allreduce(tensor, op="sum"):
        raise RuntimeError("NVSHMEM not available. Build extension first.")

    def nvshmem_all_to_all_4D(tensor, scatter_dim, gather_dim):
        raise RuntimeError("NVSHMEM not available. Build extension first.")

    def nvshmem_my_pe():
        raise RuntimeError("NVSHMEM not available. Build extension first.")

    def nvshmem_n_pes():
        raise RuntimeError("NVSHMEM not available. Build extension first.")

    def nvshmem_barrier():
        raise RuntimeError("NVSHMEM not available. Build extension first.")

__version__ = "0.1.0"
__all__ = [
    # Core
    "Context",
    "PersistentContext",
    "SMConfig",
    "Role",
    "auto_config",
    # Communication - basic patterns
    "allreduce",
    "allgather",
    "reduce_scatter",
    "send",
    "recv",
    "CommScheduler",
    "CommPhase",
    # Communication - all-to-all (for Sequence Parallel)
    "all_to_all",
    "all_to_all_single",
    "all_to_all_4D",
    # Reduction ops
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "PRODUCT",
    # Backend
    "NCCLBackend",
    "OverlapMode",
    "get_backend",
    "init_backend",
    # Overlap
    "OverlapManager",
    "OverlapRegion",
    # Config
    "config",
    # NVSHMEM backend
    "is_nvshmem_available",
    "init_nvshmem",
    "finalize_nvshmem",
    "nvshmem_allreduce",
    "nvshmem_all_to_all_4D",
    "nvshmem_my_pe",
    "nvshmem_n_pes",
    "nvshmem_barrier",
    # Utilities
    "init",
    "get_rank",
    "get_world_size",
    "is_initialized",
    "overlap_compute_and_comm",
    "overlap_all_to_all_with_compute",
    "overlap_region",
    "get_comm_stream",
    "get_compute_stream",
    "sync_streams",
    "cleanup",
]


# Global state
_initialized = False
_rank = 0
_world_size = 1
_backend = None
_overlap_manager: Optional[OverlapManager] = None


def init(
    backend: str = "nccl",
    rank: int = None,
    world_size: int = None,
    init_method: str = None,
    overlap_mode: str = "stream",
):
    """
    Initialize MACO.

    Args:
        backend: Communication backend ("nccl", "gloo")
        rank: Current process rank (auto-detected if None)
        world_size: Total number of processes (auto-detected if None)
        init_method: Distributed init method (e.g., "env://", "tcp://...")
        overlap_mode: Overlap mode ("none", "stream")
    """
    global _initialized, _rank, _world_size, _backend, _overlap_manager

    if _initialized:
        return

    # Auto-detect from environment
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    _rank = rank
    _world_size = world_size

    # Initialize distributed if world_size > 1
    if world_size > 1 and not dist.is_initialized():
        if init_method is None:
            init_method = "env://"

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        _rank = dist.get_rank()
        _world_size = dist.get_world_size()

    # Initialize overlap manager
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        _overlap_manager = OverlapManager(device=device)

    # Set backend
    overlap_enum = OverlapMode.STREAM if overlap_mode == "stream" else OverlapMode.NONE
    _backend = init_backend(
        rank=_rank,
        world_size=_world_size,
        overlap_mode=overlap_enum,
    )

    _initialized = True
    print(f"[MACO] Initialized rank={_rank}, world_size={_world_size}, backend={backend}")


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return _rank


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return _world_size


def is_initialized() -> bool:
    """Check if MACO is initialized."""
    return _initialized


def get_overlap_manager() -> Optional[OverlapManager]:
    """Get the global overlap manager."""
    return _overlap_manager


def overlap_compute_and_comm(
    compute_fn,
    comm_tensor: torch.Tensor,
    comm_op: str = "allreduce",
):
    """
    Execute compute and communication in parallel.

    This is the main API for compute-comm overlap.

    Args:
        compute_fn: Function to execute for compute
        comm_tensor: Tensor for communication
        comm_op: Communication operation

    Returns:
        Tuple of (compute_result, comm_tensor)
    """
    global _overlap_manager

    if _overlap_manager is None:
        # Sequential fallback
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(comm_tensor)
        compute_result = compute_fn()
        return compute_result, comm_tensor

    return _overlap_manager.overlap_compute_and_comm(
        compute_fn, comm_tensor, comm_op
    )


def cleanup():
    """Cleanup MACO resources."""
    global _initialized, _overlap_manager

    if _overlap_manager is not None:
        _overlap_manager.synchronize()
        _overlap_manager = None

    if dist.is_initialized():
        dist.destroy_process_group()

    _initialized = False


# ============================================================================
# New APIs for Sequence Parallel (all-to-all) overlap support
# ============================================================================

def get_comm_stream():
    """
    Get the communication CUDA stream.

    Use this for manual stream management when you need fine-grained control.

    Returns:
        torch.cuda.Stream or None if CUDA is not available

    Example:
        comm_stream = maco.get_comm_stream()
        with torch.cuda.stream(comm_stream):
            dist.all_to_all_single(output, input)
    """
    global _overlap_manager
    if _overlap_manager is None:
        if not _initialized:
            init()
        if _overlap_manager is None:
            return None
    return _overlap_manager.get_comm_stream()


def get_compute_stream():
    """
    Get the compute CUDA stream.

    Use this for manual stream management when you need fine-grained control.

    Returns:
        torch.cuda.Stream or None if CUDA is not available

    Example:
        compute_stream = maco.get_compute_stream()
        with torch.cuda.stream(compute_stream):
            result = torch.matmul(a, b)
    """
    global _overlap_manager
    if _overlap_manager is None:
        if not _initialized:
            init()
        if _overlap_manager is None:
            return None
    return _overlap_manager.get_compute_stream()


def sync_streams():
    """
    Synchronize both compute and communication streams with the default stream.

    Call this after using manual stream management to ensure all operations
    are complete before continuing.

    Example:
        # Manual overlap
        with torch.cuda.stream(maco.get_comm_stream()):
            dist.all_to_all_single(output, input)
        with torch.cuda.stream(maco.get_compute_stream()):
            result = torch.matmul(a, b)
        maco.sync_streams()  # Wait for both to complete
    """
    global _overlap_manager
    if _overlap_manager is not None:
        _overlap_manager.sync_streams()


def overlap_all_to_all_with_compute(all_to_all_fn, compute_fn):
    """
    Execute all-to-all communication and compute in parallel.

    This is specifically designed for Sequence Parallel workloads where
    all-to-all is used to redistribute tensors across GPUs.

    Args:
        all_to_all_fn: Function that performs all-to-all communication.
                       Should return the result tensor.
        compute_fn: Function to execute for compute.
                    Should return the compute result.

    Returns:
        Tuple of (compute_result, all_to_all_result)

    Example:
        # Overlap all_to_all with matmul
        compute_result, comm_result = maco.overlap_all_to_all_with_compute(
            all_to_all_fn=lambda: maco.all_to_all_4D(tensor, scatter_dim=2, gather_dim=1),
            compute_fn=lambda: torch.matmul(a, b)
        )
    """
    global _overlap_manager

    if _overlap_manager is None:
        if not _initialized:
            init()

    if _overlap_manager is None:
        # Sequential fallback
        comm_result = all_to_all_fn()
        compute_result = compute_fn()
        return compute_result, comm_result

    return _overlap_manager.overlap_all_to_all_with_compute(all_to_all_fn, compute_fn)


def overlap_region():
    """
    Context manager for overlap region.

    Within this region, submitted compute and comm tasks will be
    executed in parallel on separate CUDA streams.

    Returns:
        OverlapRegion context manager

    Example:
        with maco.overlap_region() as region:
            # These execute in parallel
            region.comm(lambda: dist.all_to_all_single(output, input))
            region.compute(lambda: torch.matmul(a, b))
        # Both are synchronized when exiting the region

        # Get results after the region
        comm_results = region.get_comm_results()
        compute_results = region.get_compute_results()
    """
    global _overlap_manager

    if _overlap_manager is None:
        if not _initialized:
            init()

    if _overlap_manager is None:
        raise RuntimeError("MACO not initialized or CUDA not available")

    return _overlap_manager.overlap_region()
