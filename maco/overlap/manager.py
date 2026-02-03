"""
Overlap Manager

Manages compute-communication overlap using CUDA streams.
"""

import torch
import torch.distributed as dist
from typing import Optional, Callable, Any, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import time


@dataclass
class OverlapTask:
    """A task that can be overlapped."""
    task_type: str  # "compute" or "comm"
    fn: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    result: Any = None
    event: Optional[torch.cuda.Event] = None


class OverlapManager:
    """
    Manages overlapping of compute and communication operations.

    This manager uses separate CUDA streams for compute and communication,
    allowing them to execute concurrently when there are no data dependencies.
    """

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Create streams
        if device.type == "cuda":
            self.default_stream = torch.cuda.current_stream()
            self.compute_stream = torch.cuda.Stream(device=device)
            self.comm_stream = torch.cuda.Stream(device=device)
        else:
            self.default_stream = None
            self.compute_stream = None
            self.comm_stream = None

        # Pending tasks
        self._pending_compute: List[OverlapTask] = []
        self._pending_comm: List[OverlapTask] = []

        # Statistics
        self.stats = {
            "total_compute_time_ms": 0.0,
            "total_comm_time_ms": 0.0,
            "overlapped_time_ms": 0.0,
            "sequential_time_ms": 0.0,
        }

    @contextmanager
    def overlap_region(self):
        """
        Context manager for overlap region.

        Within this region, submitted compute and comm tasks will be
        executed in parallel when possible.
        """
        region = OverlapRegion(self)
        try:
            yield region
        finally:
            region.wait_all()

    def submit_compute(
        self,
        fn: Callable,
        *args,
        **kwargs
    ) -> torch.cuda.Event:
        """
        Submit a compute task to execute on compute stream.

        Returns an event that signals completion.
        """
        if self.compute_stream is None:
            # CPU fallback
            result = fn(*args, **kwargs)
            return result

        event = torch.cuda.Event()

        with torch.cuda.stream(self.compute_stream):
            result = fn(*args, **kwargs)
            event.record()

        return event, result

    def submit_comm(
        self,
        tensor: torch.Tensor,
        op: str = "allreduce",
        **kwargs
    ) -> torch.cuda.Event:
        """
        Submit a communication task to execute on comm stream.

        Returns an event that signals completion.
        """
        if self.comm_stream is None or not dist.is_initialized():
            return torch.cuda.Event(), tensor

        event = torch.cuda.Event()

        # Record current stream event to ensure tensor is ready
        ready_event = torch.cuda.Event()
        ready_event.record()

        with torch.cuda.stream(self.comm_stream):
            # Wait for tensor to be ready
            self.comm_stream.wait_event(ready_event)

            # Execute communication
            if op == "allreduce":
                dist.all_reduce(tensor, **kwargs)
            elif op == "allgather":
                # Handle allgather
                world_size = dist.get_world_size()
                output_list = [torch.empty_like(tensor) for _ in range(world_size)]
                dist.all_gather(output_list, tensor, **kwargs)
                tensor = torch.stack(output_list)

            event.record()

        return event, tensor

    def overlap_compute_and_comm(
        self,
        compute_fn: Callable,
        comm_tensor: torch.Tensor,
        comm_op: str = "allreduce",
    ) -> tuple:
        """
        Execute compute and communication in parallel.

        Args:
            compute_fn: Function to execute for compute
            comm_tensor: Tensor for communication
            comm_op: Communication operation ("allreduce", etc.)

        Returns:
            Tuple of (compute_result, comm_tensor)
        """
        if self.compute_stream is None or self.comm_stream is None:
            # Sequential fallback
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(comm_tensor)
            compute_result = compute_fn()
            return compute_result, comm_tensor

        # Record start event
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Start communication on comm stream
        comm_done_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(start_event)
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(comm_tensor)
            comm_done_event.record()

        # Start compute on compute stream
        compute_done_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(start_event)
            compute_result = compute_fn()
            compute_done_event.record()

        # Wait for both
        comm_done_event.synchronize()
        compute_done_event.synchronize()

        # Record timing
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()

        return compute_result, comm_tensor

    def synchronize(self):
        """Synchronize all streams."""
        if self.compute_stream is not None:
            self.compute_stream.synchronize()
        if self.comm_stream is not None:
            self.comm_stream.synchronize()


class OverlapRegion:
    """
    Region where tasks can be overlapped.

    Usage:
        with manager.overlap_region() as region:
            compute_future = region.submit_compute(my_compute_fn)
            comm_future = region.submit_comm(tensor)
        # Both are synchronized when exiting the region
    """

    def __init__(self, manager: OverlapManager):
        self.manager = manager
        self._compute_events: List[tuple] = []  # (event, result)
        self._comm_events: List[tuple] = []  # (event, tensor)

    def submit_compute(self, fn: Callable, *args, **kwargs):
        """Submit compute task."""
        event, result = self.manager.submit_compute(fn, *args, **kwargs)
        self._compute_events.append((event, result))
        return _FutureResult(event, result)

    def submit_comm(self, tensor: torch.Tensor, op: str = "allreduce", **kwargs):
        """Submit communication task."""
        event, result = self.manager.submit_comm(tensor, op, **kwargs)
        self._comm_events.append((event, result))
        return _FutureResult(event, result)

    def wait_all(self):
        """Wait for all submitted tasks."""
        for event, _ in self._compute_events:
            if isinstance(event, torch.cuda.Event):
                event.synchronize()
        for event, _ in self._comm_events:
            if isinstance(event, torch.cuda.Event):
                event.synchronize()


class _FutureResult:
    """Future result wrapper."""

    def __init__(self, event, result):
        self._event = event
        self._result = result
        self._waited = False

    def wait(self):
        """Wait for result."""
        if not self._waited and isinstance(self._event, torch.cuda.Event):
            self._event.synchronize()
            self._waited = True
        return self._result

    def result(self):
        """Get result (waits if necessary)."""
        return self.wait()
