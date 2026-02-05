"""
MACO Communication Module

提供多 GPU 通信支持，封装 NCCL 操作。
支持异步通信以实现计算-通信重叠。
"""

from .process_group import (
    ProcessGroupManager,
    get_world_size,
    get_rank,
    get_local_rank,
    is_initialized,
    barrier,
)

from .nccl_ops import (
    AsyncHandle,
    async_all_reduce,
    async_all_gather,
    async_all_to_all,
    async_all_to_all_4d,
    async_broadcast,
    sync_stream,
    wait_comm_stream,
)

__all__ = [
    # Process Group
    "ProcessGroupManager",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "is_initialized",
    "barrier",
    # NCCL Ops
    "AsyncHandle",
    "async_all_reduce",
    "async_all_gather",
    "async_all_to_all",
    "async_all_to_all_4d",
    "async_broadcast",
    "sync_stream",
    "wait_comm_stream",
]
