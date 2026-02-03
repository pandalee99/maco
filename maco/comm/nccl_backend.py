"""
NCCL Backend for MACO

Implements optimized communication using NCCL with compute-comm overlap.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, List, Callable, Any
from dataclasses import dataclass
from enum import Enum
import threading
import time


class OverlapMode(Enum):
    """Overlap execution modes."""
    NONE = "none"              # No overlap, sequential execution
    STREAM = "stream"          # Use separate CUDA streams
    PERSISTENT = "persistent"  # Persistent kernel style (advanced)


@dataclass
class CommHandle:
    """Handle for async communication operations."""
    tensor: torch.Tensor
    work: Optional[Any] = None  # dist.Work object
    stream: Optional[torch.cuda.Stream] = None
    event: Optional[torch.cuda.Event] = None
    completed: bool = False

    def wait(self) -> torch.Tensor:
        """Wait for the operation to complete."""
        if self.completed:
            return self.tensor

        if self.work is not None:
            self.work.wait()

        if self.stream is not None and self.event is not None:
            self.event.synchronize()

        self.completed = True
        return self.tensor

    def is_completed(self) -> bool:
        """Check if operation is completed."""
        if self.completed:
            return True
        if self.work is not None:
            return self.work.is_completed()
        return False


class NCCLBackend:
    """
    NCCL communication backend with compute-comm overlap support.

    This backend manages CUDA streams to enable overlapping of
    communication operations with compute operations.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        overlap_mode: OverlapMode = OverlapMode.STREAM,
    ):
        self.rank = rank
        self.world_size = world_size
        self.overlap_mode = overlap_mode
        self._initialized = False

        # CUDA streams for overlap
        self.compute_stream: Optional[torch.cuda.Stream] = None
        self.comm_stream: Optional[torch.cuda.Stream] = None

        # Events for synchronization
        self._events: List[torch.cuda.Event] = []

        # Statistics
        self.stats = {
            "allreduce_count": 0,
            "allreduce_time_ms": 0.0,
            "overlap_time_saved_ms": 0.0,
        }

    def init_process_group(
        self,
        backend: str = "nccl",
        init_method: str = "env://",
    ):
        """Initialize the distributed process group."""
        if self._initialized:
            return

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=self.rank,
                world_size=self.world_size,
            )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Create streams for overlap
        if self.overlap_mode == OverlapMode.STREAM:
            self.compute_stream = torch.cuda.Stream()
            self.comm_stream = torch.cuda.Stream()

        self._initialized = True
        print(f"[MACO-NCCL] Initialized rank {self.rank}/{self.world_size}")

    def allreduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        async_op: bool = False,
    ) -> CommHandle:
        """
        Perform allreduce with optional async execution.

        Args:
            tensor: Tensor to reduce (in-place)
            op: Reduction operation
            async_op: If True, return immediately

        Returns:
            CommHandle for the operation
        """
        if self.world_size == 1:
            return CommHandle(tensor=tensor, completed=True)

        self.stats["allreduce_count"] += 1

        if self.overlap_mode == OverlapMode.STREAM and self.comm_stream is not None:
            # Execute on communication stream
            event_before = torch.cuda.Event(enable_timing=True)
            event_after = torch.cuda.Event(enable_timing=True)

            # Record event on current stream
            event_before.record()

            with torch.cuda.stream(self.comm_stream):
                # Wait for compute to finish producing the tensor
                self.comm_stream.wait_event(event_before)

                # Perform allreduce
                work = dist.all_reduce(tensor, op=op, async_op=True)

                # Record completion event
                event_after.record()

            handle = CommHandle(
                tensor=tensor,
                work=work,
                stream=self.comm_stream,
                event=event_after,
            )

            if not async_op:
                handle.wait()

            return handle
        else:
            # Simple synchronous execution
            work = dist.all_reduce(tensor, op=op, async_op=async_op)

            handle = CommHandle(tensor=tensor, work=work if async_op else None)
            if not async_op:
                handle.completed = True

            return handle

    def allreduce_overlap(
        self,
        tensor: torch.Tensor,
        compute_fn: Callable[[], torch.Tensor],
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> tuple:
        """
        Perform allreduce overlapped with compute.

        This is the key optimization: while the allreduce is running on
        the comm_stream, we execute compute_fn on the compute_stream.

        Args:
            tensor: Tensor to reduce
            compute_fn: Function to execute during communication
            op: Reduction operation

        Returns:
            Tuple of (reduced_tensor, compute_result)
        """
        if self.world_size == 1:
            compute_result = compute_fn()
            return tensor, compute_result

        if self.overlap_mode != OverlapMode.STREAM:
            # Fallback to sequential execution
            dist.all_reduce(tensor, op=op)
            compute_result = compute_fn()
            return tensor, compute_result

        # Record event before starting
        start_event = torch.cuda.Event(enable_timing=True)
        comm_done_event = torch.cuda.Event(enable_timing=True)
        compute_done_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Start allreduce on comm stream
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(start_event)
            work = dist.all_reduce(tensor, op=op, async_op=True)
            comm_done_event.record()

        # Execute compute on compute stream (or default stream)
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(start_event)
            compute_result = compute_fn()
            compute_done_event.record()

        # Wait for both to complete
        work.wait()
        comm_done_event.synchronize()
        compute_done_event.synchronize()

        return tensor, compute_result

    def allgather(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
    ) -> CommHandle:
        """Perform allgather operation."""
        if self.world_size == 1:
            output = tensor.unsqueeze(0)
            return CommHandle(tensor=output, completed=True)

        # Allocate output tensor
        output_list = [
            torch.empty_like(tensor) for _ in range(self.world_size)
        ]

        if self.overlap_mode == OverlapMode.STREAM and self.comm_stream is not None:
            event = torch.cuda.Event()
            event.record()

            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(event)
                work = dist.all_gather(output_list, tensor, async_op=True)
                done_event = torch.cuda.Event()
                done_event.record()

            output = torch.stack(output_list)
            handle = CommHandle(
                tensor=output,
                work=work,
                stream=self.comm_stream,
                event=done_event,
            )

            if not async_op:
                handle.wait()

            return handle
        else:
            work = dist.all_gather(output_list, tensor, async_op=async_op)
            output = torch.stack(output_list)

            handle = CommHandle(tensor=output, work=work if async_op else None)
            if not async_op:
                handle.completed = True

            return handle

    def reduce_scatter(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        async_op: bool = False,
    ) -> CommHandle:
        """Perform reduce_scatter operation."""
        if self.world_size == 1:
            return CommHandle(tensor=tensor, completed=True)

        # Split input and allocate output
        input_list = list(tensor.chunk(self.world_size, dim=0))
        output = torch.empty_like(input_list[0])

        work = dist.reduce_scatter(output, input_list, op=op, async_op=async_op)

        handle = CommHandle(tensor=output, work=work if async_op else None)
        if not async_op:
            handle.completed = True

        return handle

    def barrier(self):
        """Synchronize all processes."""
        if self.world_size > 1:
            dist.barrier()

    def synchronize_streams(self):
        """Synchronize compute and comm streams."""
        if self.compute_stream is not None:
            self.compute_stream.synchronize()
        if self.comm_stream is not None:
            self.comm_stream.synchronize()

    def get_stats(self) -> dict:
        """Get communication statistics."""
        return self.stats.copy()

    def cleanup(self):
        """Cleanup resources."""
        self.synchronize_streams()
        if dist.is_initialized():
            dist.destroy_process_group()


# Global backend instance
_backend: Optional[NCCLBackend] = None


def get_backend() -> NCCLBackend:
    """Get the global NCCL backend instance."""
    global _backend
    if _backend is None:
        _backend = NCCLBackend()
    return _backend


def init_backend(
    rank: int = 0,
    world_size: int = 1,
    overlap_mode: OverlapMode = OverlapMode.STREAM,
):
    """Initialize the global NCCL backend."""
    global _backend
    _backend = NCCLBackend(
        rank=rank,
        world_size=world_size,
        overlap_mode=overlap_mode,
    )
    return _backend
