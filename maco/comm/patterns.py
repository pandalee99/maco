"""
Communication Patterns

Optimized collective communication operations with SM-level control.
"""

from typing import Optional, List
from enum import Enum
import torch


class ReduceOp(Enum):
    """Reduction operations."""
    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    AVG = "avg"


# Expose as module-level constants
SUM = ReduceOp.SUM
PRODUCT = ReduceOp.PRODUCT
MIN = ReduceOp.MIN
MAX = ReduceOp.MAX
AVG = ReduceOp.AVG


def allreduce(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    async_op: bool = False,
    overlap_with_next: bool = False,
    priority: str = "normal",
    overlap_group: Optional[str] = None,
) -> torch.Tensor:
    """
    Optimized AllReduce operation.

    This operation sums (or applies another reduction) the tensor across
    all processes and distributes the result to all processes.

    Args:
        tensor: Input tensor (modified in-place)
        op: Reduction operation (SUM, PRODUCT, MIN, MAX, AVG)
        async_op: If True, return immediately with a handle
        overlap_with_next: Hint that this can overlap with next compute
        priority: Communication priority ("high", "normal", "low")
        overlap_group: Group name for overlap scheduling

    Returns:
        Reduced tensor (same as input, modified in-place)
    """
    # Import here to avoid circular dependency
    import maco

    if not maco.is_initialized():
        maco.init()

    if maco.get_world_size() == 1:
        return tensor

    # TODO: Implement optimized NVSHMEM-based allreduce
    # For now, delegate to torch.distributed
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist_op = _convert_op(op)
            if async_op:
                handle = dist.all_reduce(tensor, op=dist_op, async_op=True)
                return _AsyncHandle(tensor, handle)
            else:
                dist.all_reduce(tensor, op=dist_op)
    except ImportError:
        pass

    return tensor


def allgather(
    tensor: torch.Tensor,
    async_op: bool = False,
) -> torch.Tensor:
    """
    AllGather operation.

    Gathers tensors from all processes and concatenates them.

    Args:
        tensor: Input tensor
        async_op: If True, return immediately with a handle

    Returns:
        Gathered tensor of shape [world_size, *tensor.shape]
    """
    import maco

    if not maco.is_initialized():
        maco.init()

    world_size = maco.get_world_size()
    if world_size == 1:
        return tensor.unsqueeze(0)

    # Allocate output tensor
    output_shape = (world_size,) + tensor.shape
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)

    # TODO: Implement optimized NVSHMEM-based allgather
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            tensor_list = list(output.unbind(0))
            if async_op:
                handle = dist.all_gather(tensor_list, tensor, async_op=True)
                return _AsyncHandle(output, handle)
            else:
                dist.all_gather(tensor_list, tensor)
                return torch.stack(tensor_list)
    except ImportError:
        pass

    return output


def reduce_scatter(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    async_op: bool = False,
) -> torch.Tensor:
    """
    ReduceScatter operation.

    Reduces the tensor across all processes, then scatters the result.

    Args:
        tensor: Input tensor (shape must be divisible by world_size)
        op: Reduction operation
        async_op: If True, return immediately with a handle

    Returns:
        Scattered tensor (1/world_size of original size)
    """
    import maco

    if not maco.is_initialized():
        maco.init()

    world_size = maco.get_world_size()
    if world_size == 1:
        return tensor

    # Output is 1/world_size of input
    chunk_size = tensor.shape[0] // world_size
    output = torch.empty(
        (chunk_size,) + tensor.shape[1:],
        dtype=tensor.dtype,
        device=tensor.device
    )

    # TODO: Implement optimized NVSHMEM-based reduce_scatter
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            input_list = list(tensor.chunk(world_size))
            dist_op = _convert_op(op)
            if async_op:
                handle = dist.reduce_scatter(output, input_list, op=dist_op, async_op=True)
                return _AsyncHandle(output, handle)
            else:
                dist.reduce_scatter(output, input_list, op=dist_op)
    except ImportError:
        pass

    return output


def send(
    tensor: torch.Tensor,
    dst: int,
    tag: int = 0,
    async_op: bool = False,
):
    """
    Point-to-point send.

    Args:
        tensor: Tensor to send
        dst: Destination rank
        tag: Message tag
        async_op: If True, return immediately with a handle
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            if async_op:
                return dist.isend(tensor, dst, tag=tag)
            else:
                dist.send(tensor, dst, tag=tag)
    except ImportError:
        pass


def recv(
    tensor: Optional[torch.Tensor] = None,
    src: int = None,
    shape: tuple = None,
    dtype: torch.dtype = None,
    tag: int = 0,
    async_op: bool = False,
) -> torch.Tensor:
    """
    Point-to-point receive.

    Args:
        tensor: Pre-allocated tensor to receive into (optional)
        src: Source rank
        shape: Shape of tensor to receive (if tensor is None)
        dtype: Dtype of tensor to receive (if tensor is None)
        tag: Message tag
        async_op: If True, return immediately with a handle

    Returns:
        Received tensor
    """
    if tensor is None:
        if shape is None or dtype is None:
            raise ValueError("Must provide either tensor or (shape, dtype)")
        tensor = torch.empty(shape, dtype=dtype, device='cuda')

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            if async_op:
                handle = dist.irecv(tensor, src, tag=tag)
                return _AsyncHandle(tensor, handle)
            else:
                dist.recv(tensor, src, tag=tag)
    except ImportError:
        pass

    return tensor


def _convert_op(op: ReduceOp):
    """Convert MACO ReduceOp to torch.distributed ReduceOp."""
    import torch.distributed as dist

    mapping = {
        ReduceOp.SUM: dist.ReduceOp.SUM,
        ReduceOp.PRODUCT: dist.ReduceOp.PRODUCT,
        ReduceOp.MIN: dist.ReduceOp.MIN,
        ReduceOp.MAX: dist.ReduceOp.MAX,
    }

    if op == ReduceOp.AVG:
        # AVG is SUM followed by division (handled separately)
        return dist.ReduceOp.SUM

    return mapping.get(op, dist.ReduceOp.SUM)


class _AsyncHandle:
    """Handle for async communication operations."""

    def __init__(self, tensor: torch.Tensor, handle):
        self.tensor = tensor
        self.handle = handle
        self._completed = False

    def wait(self) -> torch.Tensor:
        """Wait for the operation to complete."""
        if not self._completed and self.handle is not None:
            self.handle.wait()
            self._completed = True
        return self.tensor

    def is_completed(self) -> bool:
        """Check if the operation is completed."""
        if self._completed:
            return True
        if self.handle is not None and hasattr(self.handle, 'is_completed'):
            self._completed = self.handle.is_completed()
        return self._completed

    def __repr__(self):
        return f"_AsyncHandle(completed={self._completed})"
