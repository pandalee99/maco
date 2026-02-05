"""
MACO NCCL Operations

封装 NCCL 通信操作，支持异步执行以实现计算-通信重叠。
所有操作返回 AsyncHandle 用于后续同步。
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import torch
import torch.distributed as dist

from .process_group import get_world_size, get_rank, is_initialized


@dataclass
class AsyncHandle:
    """
    异步操作句柄
    
    用于追踪异步通信操作的状态，支持等待完成。
    
    Attributes:
        work: torch.distributed 的 Work 对象（如果有）
        stream: 执行操作的 CUDA stream
        event: 操作完成后记录的 CUDA event
        output: 操作的输出 tensor
    """
    work: Optional[dist.Work] = None
    stream: Optional[torch.cuda.Stream] = None
    event: Optional[torch.cuda.Event] = None
    output: Optional[torch.Tensor] = None
    
    def wait(self) -> torch.Tensor:
        """
        等待操作完成并返回结果
        
        Returns:
            输出 tensor
        """
        if self.work is not None:
            self.work.wait()
        if self.event is not None and self.stream is not None:
            self.event.synchronize()
        return self.output
    
    def is_completed(self) -> bool:
        """检查操作是否完成"""
        if self.work is not None:
            return self.work.is_completed()
        if self.event is not None:
            return self.event.query()
        return True


def _get_comm_stream() -> Optional[torch.cuda.Stream]:
    """获取通信专用 stream"""
    if torch.cuda.is_available():
        # 每个线程使用独立的通信 stream
        if not hasattr(_get_comm_stream, "_stream"):
            _get_comm_stream._stream = torch.cuda.Stream()
        return _get_comm_stream._stream
    return None


def async_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = True,
) -> AsyncHandle:
    """
    异步 AllReduce 操作
    
    Args:
        tensor: 输入/输出 tensor（原地操作）
        op: 规约操作类型，默认 SUM
        group: 进程组，默认使用全局组
        async_op: 是否异步执行
        
    Returns:
        AsyncHandle: 异步操作句柄
    """
    if not is_initialized() or get_world_size() == 1:
        # 单 GPU 模式，直接返回
        return AsyncHandle(output=tensor)
    
    stream = _get_comm_stream()
    event = torch.cuda.Event() if stream else None
    
    if stream:
        with torch.cuda.stream(stream):
            work = dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
            if event:
                event.record(stream)
    else:
        work = dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
    
    return AsyncHandle(
        work=work if async_op else None,
        stream=stream,
        event=event,
        output=tensor,
    )


def async_all_gather(
    tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = True,
) -> AsyncHandle:
    """
    异步 AllGather 操作
    
    Args:
        tensor: 输入 tensor
        group: 进程组，默认使用全局组
        async_op: 是否异步执行
        
    Returns:
        AsyncHandle: 异步操作句柄，output 为聚合后的 tensor
    """
    world_size = get_world_size()
    
    if not is_initialized() or world_size == 1:
        return AsyncHandle(output=tensor)
    
    # 准备输出 tensor 列表
    output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    
    stream = _get_comm_stream()
    event = torch.cuda.Event() if stream else None
    
    if stream:
        with torch.cuda.stream(stream):
            work = dist.all_gather(
                output_tensors, tensor, group=group, async_op=async_op
            )
            if event:
                event.record(stream)
    else:
        work = dist.all_gather(
            output_tensors, tensor, group=group, async_op=async_op
        )
    
    # 等待完成后拼接
    if not async_op:
        output = torch.cat(output_tensors, dim=0)
    else:
        # 异步模式下，需要在 wait() 时拼接
        output = output_tensors  # 暂存列表
    
    handle = AsyncHandle(
        work=work if async_op else None,
        stream=stream,
        event=event,
        output=output,
    )
    
    # 重写 wait 方法以支持拼接
    if async_op:
        original_wait = handle.wait
        def wait_and_concat():
            original_wait()
            return torch.cat(handle.output, dim=0)
        handle.wait = wait_and_concat
    
    return handle


def async_all_to_all(
    input_tensor: torch.Tensor,
    scatter_dim: int = 0,
    gather_dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = True,
) -> AsyncHandle:
    """
    异步 AllToAll 操作
    
    将 tensor 沿 scatter_dim 切分发送给所有进程，
    并沿 gather_dim 拼接从所有进程收到的数据。
    
    Args:
        input_tensor: 输入 tensor
        scatter_dim: 切分维度
        gather_dim: 拼接维度
        group: 进程组
        async_op: 是否异步执行
        
    Returns:
        AsyncHandle: 异步操作句柄
    """
    world_size = get_world_size()
    
    if not is_initialized() or world_size == 1:
        return AsyncHandle(output=input_tensor)
    
    # 沿 scatter_dim 切分
    input_list = list(torch.chunk(input_tensor, world_size, dim=scatter_dim))
    # 确保 contiguous
    input_list = [t.contiguous() for t in input_list]
    
    # 准备输出
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    
    stream = _get_comm_stream()
    event = torch.cuda.Event() if stream else None
    
    if stream:
        with torch.cuda.stream(stream):
            work = dist.all_to_all(
                output_list, input_list, group=group, async_op=async_op
            )
            if event:
                event.record(stream)
        # 同步模式下，必须等待 stream 完成再读取结果
        if not async_op:
            stream.synchronize()
    else:
        work = dist.all_to_all(
            output_list, input_list, group=group, async_op=async_op
        )

    handle = AsyncHandle(
        work=work if async_op else None,
        stream=stream,
        event=event,
        output=output_list,
    )

    # 重写 wait 方法以支持拼接
    if async_op:
        original_wait = handle.wait
        def wait_and_concat():
            original_wait()
            return torch.cat(handle.output, dim=gather_dim).contiguous()
        handle.wait = wait_and_concat
    else:
        handle.output = torch.cat(output_list, dim=gather_dim).contiguous()
    
    return handle


def async_all_to_all_4d(
    input_tensor: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = True,
) -> AsyncHandle:
    """
    4D Tensor 的 AllToAll（专为 attention QKV 设计）
    
    适用于 shape [B, S, H, D] 的 tensor：
    - scatter_dim=2 (H): 按 head 切分发送
    - gather_dim=1 (S): 按 seq 拼接接收
    
    Args:
        input_tensor: shape [B, S, H, D]
        scatter_dim: 切分维度（默认 head 维度）
        gather_dim: 拼接维度（默认 seq 维度）
        group: 进程组
        async_op: 是否异步
        
    Returns:
        AsyncHandle
    """
    world_size = get_world_size()
    
    if not is_initialized() or world_size == 1:
        return AsyncHandle(output=input_tensor)
    
    assert input_tensor.dim() == 4, f"Expected 4D tensor, got {input_tensor.dim()}D"
    
    # 使用通用 all_to_all
    return async_all_to_all(
        input_tensor,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        group=group,
        async_op=async_op,
    )


def async_broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = True,
) -> AsyncHandle:
    """
    异步 Broadcast 操作
    
    Args:
        tensor: 输入/输出 tensor
        src: 源进程排名
        group: 进程组
        async_op: 是否异步
        
    Returns:
        AsyncHandle
    """
    if not is_initialized() or get_world_size() == 1:
        return AsyncHandle(output=tensor)
    
    stream = _get_comm_stream()
    event = torch.cuda.Event() if stream else None
    
    if stream:
        with torch.cuda.stream(stream):
            work = dist.broadcast(tensor, src=src, group=group, async_op=async_op)
            if event:
                event.record(stream)
    else:
        work = dist.broadcast(tensor, src=src, group=group, async_op=async_op)
    
    return AsyncHandle(
        work=work if async_op else None,
        stream=stream,
        event=event,
        output=tensor,
    )


def sync_stream(stream: Optional[torch.cuda.Stream] = None) -> None:
    """
    同步指定 stream
    
    Args:
        stream: 要同步的 stream，None 则同步通信 stream
    """
    if stream is None:
        stream = _get_comm_stream()
    if stream is not None:
        stream.synchronize()


def wait_comm_stream(compute_stream: Optional[torch.cuda.Stream] = None) -> None:
    """
    让计算 stream 等待通信 stream 完成
    
    Args:
        compute_stream: 计算 stream，None 则使用当前 stream
    """
    comm_stream = _get_comm_stream()
    if comm_stream is None:
        return
    
    event = torch.cuda.Event()
    event.record(comm_stream)
    
    if compute_stream is None:
        event.synchronize()
    else:
        compute_stream.wait_event(event)
