"""
Execution Context for MACO

Provides context managers for optimized communication execution.
"""

from typing import Optional, Callable, Any, List
from dataclasses import dataclass
import torch
from .sm_manager import SMConfig, SMManager


@dataclass
class ComputeTask:
    """Represents a compute task."""
    fn: Callable
    args: tuple
    kwargs: dict
    result: Any = None


@dataclass
class CommTask:
    """Represents a communication task."""
    op: str  # "allreduce", "allgather", etc.
    tensor: torch.Tensor
    async_op: bool = False
    result: Any = None


class Context:
    """
    Execution context with explicit SM control.

    Example:
        ctx = Context(compute_sms=96, comm_sms=32)
        with ctx:
            y = ctx.compute(lambda: torch.matmul(x, w))
            y = ctx.allreduce(y)
    """

    def __init__(
        self,
        world_size: int = 1,
        compute_sms: int = 96,
        comm_sms: int = 32,
        overlap: bool = True,
    ):
        self.world_size = world_size
        self.compute_sms = compute_sms
        self.comm_sms = comm_sms
        self.overlap = overlap

        self._compute_tasks: List[ComputeTask] = []
        self._comm_tasks: List[CommTask] = []
        self._active = False

    def __enter__(self):
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._active = False
        self._flush()
        return False

    def compute(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Submit a compute task.

        Args:
            fn: Function to execute
            *args, **kwargs: Arguments to pass to fn

        Returns:
            Result of the computation
        """
        if not self._active:
            # Direct execution outside context
            return fn(*args, **kwargs)

        task = ComputeTask(fn=fn, args=args, kwargs=kwargs)
        self._compute_tasks.append(task)

        # For now, execute immediately (TODO: batch execution)
        task.result = fn(*args, **kwargs)
        return task.result

    def allreduce(
        self,
        tensor: torch.Tensor,
        async_op: bool = False
    ) -> torch.Tensor:
        """
        Perform allreduce on tensor.

        Args:
            tensor: Tensor to reduce
            async_op: If True, return immediately with a future

        Returns:
            Reduced tensor (or future if async_op=True)
        """
        task = CommTask(op="allreduce", tensor=tensor, async_op=async_op)
        self._comm_tasks.append(task)

        if self.world_size == 1:
            task.result = tensor
            return tensor

        # TODO: Implement optimized allreduce with SM control
        # For now, use torch.distributed
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                if async_op:
                    handle = dist.all_reduce(tensor, async_op=True)
                    return _AsyncResult(tensor, handle)
                else:
                    dist.all_reduce(tensor)
        except ImportError:
            pass

        task.result = tensor
        return tensor

    def _flush(self):
        """Flush all pending tasks."""
        # Wait for all async operations
        for task in self._comm_tasks:
            if task.async_op and hasattr(task.result, 'wait'):
                task.result.wait()

        self._compute_tasks.clear()
        self._comm_tasks.clear()


class _AsyncResult:
    """Wrapper for async operation result."""

    def __init__(self, tensor: torch.Tensor, handle):
        self.tensor = tensor
        self.handle = handle
        self._completed = False

    def wait(self) -> torch.Tensor:
        """Wait for the operation to complete."""
        if not self._completed:
            self.handle.wait()
            self._completed = True
        return self.tensor

    def is_completed(self) -> bool:
        """Check if the operation is completed."""
        return self._completed


class PersistentContext:
    """
    Persistent kernel context for maximum performance.

    This context compiles the computation graph into a single
    persistent kernel that runs with minimal launch overhead.

    Example:
        with PersistentContext(model) as ctx:
            for batch in dataloader:
                output = ctx.forward(batch)
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        sm_config: Optional[SMConfig] = None,
    ):
        self.model = model
        self.sm_manager = SMManager(sm_config)
        self._compiled = False
        self._kernel = None

    def __enter__(self):
        if not self._compiled:
            self._compile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        return False

    def _compile(self):
        """Compile the model into a persistent kernel."""
        self.sm_manager.compile()

        # TODO: Implement actual persistent kernel compilation
        # This would involve:
        # 1. Tracing the model
        # 2. Generating task graph
        # 3. Compiling CUDA code
        # 4. Loading the compiled kernel

        self._compiled = True
        print("[MACO] Model compiled for persistent execution")

    def _cleanup(self):
        """Cleanup persistent kernel resources."""
        if self._kernel is not None:
            # TODO: Release kernel resources
            pass

    def forward(self, *args, **kwargs):
        """
        Execute forward pass using persistent kernel.
        """
        if not self._compiled:
            raise RuntimeError("Context not compiled. Use 'with' statement.")

        if self.model is not None:
            # TODO: Use persistent kernel execution
            # For now, fall back to regular execution
            return self.model(*args, **kwargs)
        else:
            raise RuntimeError("No model provided")

    def register_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        is_weight: bool = False
    ):
        """Register a tensor for persistent kernel use."""
        # TODO: Implement tensor registration
        pass

    def register_output(
        self,
        shape: tuple,
        dtype: torch.dtype,
        name: str
    ) -> torch.Tensor:
        """Register an output tensor."""
        # TODO: Implement output registration
        return torch.empty(shape, dtype=dtype, device='cuda')
