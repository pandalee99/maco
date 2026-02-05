"""
MACO Synchronization Module

提供跨 stream 的同步原语，支持计算-通信重叠。
"""

from .stream_manager import (
    StreamManager,
    get_compute_stream,
    get_comm_stream,
)

from .signal_wait import (
    Signal,
    SignalWait,
    create_signal,
    OverlapContext,
)

__all__ = [
    # Stream Manager
    "StreamManager",
    "get_compute_stream",
    "get_comm_stream",
    # Signal-Wait
    "Signal",
    "SignalWait",
    "create_signal",
    "OverlapContext",
]
