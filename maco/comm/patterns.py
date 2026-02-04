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


def all_to_all(
    input_tensor: torch.Tensor,
    output_tensor: Optional[torch.Tensor] = None,
    scatter_dim: int = 0,
    gather_dim: int = 0,
    group=None,
    async_op: bool = False,
) -> torch.Tensor:
    """
    All-to-All communication.

    Each process sends a portion of its tensor to every other process
    and receives a portion from every other process.

    This is commonly used in Sequence Parallel for redistributing
    tensors across the sequence and head dimensions.

    Args:
        input_tensor: Input tensor
        output_tensor: Optional pre-allocated output tensor
        scatter_dim: Dimension to scatter along
        gather_dim: Dimension to gather along
        group: Process group (default: world)
        async_op: If True, return immediately with a handle

    Returns:
        Output tensor after all-to-all communication
    """
    import maco

    if not maco.is_initialized():
        maco.init()

    world_size = maco.get_world_size()
    if world_size == 1:
        return input_tensor

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            # Split input along scatter_dim
            input_list = list(torch.tensor_split(input_tensor, world_size, scatter_dim))
            input_list = [t.contiguous() for t in input_list]

            # Prepare output list
            if output_tensor is not None:
                output_list = list(torch.tensor_split(output_tensor, world_size, gather_dim))
            else:
                output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

            if async_op:
                handle = dist.all_to_all(output_list, input_list, group=group, async_op=True)
                output = torch.cat(output_list, dim=gather_dim)
                return _AsyncHandle(output, handle)
            else:
                dist.all_to_all(output_list, input_list, group=group)
                return torch.cat(output_list, dim=gather_dim).contiguous()
    except ImportError:
        pass

    return input_tensor


def all_to_all_single(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    group=None,
    async_op: bool = False,
):
    """
    All-to-All Single communication (more efficient for equal-sized splits).

    Args:
        output_tensor: Pre-allocated output tensor
        input_tensor: Input tensor
        output_split_sizes: Split sizes for output (optional)
        input_split_sizes: Split sizes for input (optional)
        group: Process group
        async_op: If True, return immediately with a handle

    Returns:
        Work handle if async_op=True, else None
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            if async_op:
                return dist.all_to_all_single(
                    output_tensor, input_tensor,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=True
                )
            else:
                dist.all_to_all_single(
                    output_tensor, input_tensor,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group
                )
    except ImportError:
        output_tensor.copy_(input_tensor)


def all_to_all_4D(
    input_tensor: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    group=None,
) -> torch.Tensor:
    """
    Optimized 4D All-to-All for QKV tensors in Sequence Parallel.

    This is specifically designed for the common pattern in attention:
    - Input: [B, S/P, H, D] (sequence split across P processes)
    - Output: [B, S, H/P, D] (heads split across P processes)

    Args:
        input_tensor: Input tensor of shape [B, S/P, H, D] or similar
        scatter_dim: Dimension to scatter (default 2 for heads)
        gather_dim: Dimension to gather (default 1 for sequence)
        group: Process group

    Returns:
        Output tensor with swapped parallelism

    Example:
        # Before attention: redistribute from seq-parallel to head-parallel
        q_head_parallel = all_to_all_4D(q_seq_parallel, scatter_dim=2, gather_dim=1)

        # After attention: redistribute back from head-parallel to seq-parallel
        out_seq_parallel = all_to_all_4D(out_head_parallel, scatter_dim=1, gather_dim=2)
    """
    import maco

    if not maco.is_initialized():
        maco.init()

    world_size = maco.get_world_size()
    if world_size == 1:
        return input_tensor

    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            return input_tensor

        if group is None:
            world_size = dist.get_world_size()
        else:
            world_size = dist.get_world_size(group)

        # Handle the 4D case: [B, S/P, H, D] -> [B, S, H/P, D]
        if scatter_dim == 2 and gather_dim == 1:
            b, shard_seq, h, d = input_tensor.shape
            seq = shard_seq * world_size
            shard_h = h // world_size

            # Reshape for all-to-all: [B, S/P, P, H/P, D]
            reshaped = input_tensor.reshape(b, shard_seq, world_size, shard_h, d)
            # Transpose to [P, S/P, B, H/P, D]
            transposed = reshaped.transpose(0, 2).contiguous()

            output = torch.empty_like(transposed)
            dist.all_to_all_single(output, transposed, group=group)

            # Reshape back: [S, B, H/P, D] -> [B, S, H/P, D]
            output = output.reshape(seq, b, shard_h, d)
            output = output.transpose(0, 1).contiguous().reshape(b, seq, shard_h, d)
            return output

        elif scatter_dim == 1 and gather_dim == 2:
            b, seq, shard_h, d = input_tensor.shape
            h = shard_h * world_size
            shard_seq = seq // world_size

            # Reshape for all-to-all
            reshaped = input_tensor.reshape(b, world_size, shard_seq, shard_h, d)
            # Transpose to [H/P, P, S/P, B, D] then [P, H/P, S/P, B, D]
            transposed = reshaped.transpose(0, 3).transpose(0, 1).contiguous()
            transposed = transposed.reshape(world_size, shard_h, shard_seq, b, d)

            output = torch.empty_like(transposed)
            dist.all_to_all_single(output, transposed, group=group)

            # Reshape back: [H, S/P, B, D] -> [B, S/P, H, D]
            output = output.reshape(h, shard_seq, b, d)
            output = output.transpose(0, 2).contiguous().reshape(b, shard_seq, h, d)
            return output

        else:
            # Fallback to generic all_to_all
            return all_to_all(input_tensor, scatter_dim=scatter_dim, gather_dim=gather_dim, group=group)

    except ImportError:
        pass

    return input_tensor


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
