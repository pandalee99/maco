"""
MACO Signal-Wait Synchronization

实现跨 stream 的 signal-wait 同步机制。

主要用途：
1. 计算完成后通知通信开始
2. 通信完成后通知下一波计算开始
3. Wave 级别的细粒度同步

实现方式：
- 基于 CUDA Events 的轻量级同步
- 可选使用 GPU Atomics 实现更细粒度控制
"""

from typing import Optional, List
from dataclasses import dataclass, field
import torch


@dataclass
class Signal:
    """
    同步信号
    
    封装 CUDA Event，用于跨 stream 同步。
    
    Attributes:
        event: CUDA Event
        stream: 信号记录所在的 stream
        name: 信号名称（用于调试）
        signaled: 是否已发出信号
    """
    event: torch.cuda.Event = field(default_factory=lambda: torch.cuda.Event())
    stream: Optional[torch.cuda.Stream] = None
    name: str = ""
    signaled: bool = False
    
    def record(self, stream: Optional[torch.cuda.Stream] = None) -> "Signal":
        """
        在指定 stream 上记录信号
        
        Args:
            stream: 目标 stream，None 则使用当前 stream
            
        Returns:
            self，支持链式调用
        """
        if stream is not None:
            self.event.record(stream)
        else:
            self.event.record()
        self.stream = stream
        self.signaled = True
        return self
    
    def wait(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        在指定 stream 上等待此信号
        
        Args:
            stream: 等待的 stream，None 则同步等待
        """
        if not self.signaled:
            return
        
        if stream is not None:
            stream.wait_event(self.event)
        else:
            self.event.synchronize()
    
    def is_ready(self) -> bool:
        """检查信号是否就绪"""
        if not self.signaled:
            return False
        return self.event.query()
    
    def synchronize(self) -> None:
        """同步等待信号完成"""
        if self.signaled:
            self.event.synchronize()


class SignalWait:
    """
    Signal-Wait 同步器
    
    管理多个信号，实现复杂的同步模式。
    
    典型用法：
    1. Wave-level 同步：每个 wave 完成后发信号
    2. Pipeline 同步：producer-consumer 模式
    
    Example:
        >>> sw = SignalWait()
        >>> 
        >>> # 在计算 stream 上记录信号
        >>> with torch.cuda.stream(compute_stream):
        ...     result = compute()
        ...     sw.signal("wave_0_done", compute_stream)
        >>> 
        >>> # 在通信 stream 上等待后开始
        >>> with torch.cuda.stream(comm_stream):
        ...     sw.wait("wave_0_done", comm_stream)
        ...     all_reduce(result)
    """
    
    def __init__(self):
        self._signals: dict[str, Signal] = {}
        self._wave_signals: List[Signal] = []
    
    def create_signal(self, name: str) -> Signal:
        """
        创建命名信号
        
        Args:
            name: 信号名称
            
        Returns:
            新创建的信号
        """
        sig = Signal(name=name)
        self._signals[name] = sig
        return sig
    
    def signal(
        self, 
        name: str, 
        stream: Optional[torch.cuda.Stream] = None
    ) -> Signal:
        """
        发出命名信号
        
        Args:
            name: 信号名称
            stream: 记录信号的 stream
            
        Returns:
            信号对象
        """
        if name not in self._signals:
            self.create_signal(name)
        
        sig = self._signals[name]
        sig.record(stream)
        return sig
    
    def wait(
        self, 
        name: str, 
        stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """
        等待命名信号
        
        Args:
            name: 信号名称
            stream: 等待的 stream，None 则同步等待
        """
        if name not in self._signals:
            return  # 信号不存在，跳过
        
        self._signals[name].wait(stream)
    
    def signal_wave(
        self, 
        wave_id: int, 
        stream: Optional[torch.cuda.Stream] = None
    ) -> Signal:
        """
        发出 wave 完成信号
        
        Args:
            wave_id: wave 编号
            stream: 记录信号的 stream
            
        Returns:
            信号对象
        """
        # 确保列表足够长
        while len(self._wave_signals) <= wave_id:
            self._wave_signals.append(Signal(name=f"wave_{len(self._wave_signals)}"))
        
        sig = self._wave_signals[wave_id]
        sig.record(stream)
        return sig
    
    def wait_wave(
        self, 
        wave_id: int, 
        stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """
        等待 wave 完成
        
        Args:
            wave_id: wave 编号
            stream: 等待的 stream
        """
        if wave_id < len(self._wave_signals):
            self._wave_signals[wave_id].wait(stream)
    
    def wait_all_waves(
        self, 
        stream: Optional[torch.cuda.Stream] = None
    ) -> None:
        """
        等待所有 wave 完成
        
        Args:
            stream: 等待的 stream
        """
        for sig in self._wave_signals:
            sig.wait(stream)
    
    def reset(self) -> None:
        """重置所有信号"""
        self._signals.clear()
        self._wave_signals.clear()
    
    def get_signal(self, name: str) -> Optional[Signal]:
        """获取命名信号"""
        return self._signals.get(name)
    
    @property
    def num_waves(self) -> int:
        """已记录的 wave 数量"""
        return len(self._wave_signals)


# 便捷函数
def create_signal(name: str = "") -> Signal:
    """创建新信号"""
    return Signal(name=name)


class OverlapContext:
    """
    重叠执行上下文
    
    封装计算-通信重叠的完整流程。
    
    Example:
        >>> ctx = OverlapContext()
        >>> 
        >>> # Wave 0: 计算
        >>> with ctx.compute():
        ...     result_0 = compute_wave_0()
        >>> ctx.signal_compute_done(0)
        >>> 
        >>> # Wave 0: 通信（与 Wave 1 计算重叠）
        >>> with ctx.comm():
        ...     ctx.wait_compute_done(0)
        ...     all_reduce(result_0)
        >>> 
        >>> # Wave 1: 计算（与 Wave 0 通信重叠）
        >>> with ctx.compute():
        ...     result_1 = compute_wave_1()
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        from .stream_manager import StreamManager
        
        self._stream_manager = StreamManager(device)
        self._signal_wait = SignalWait()
    
    @property
    def compute_stream(self) -> Optional[torch.cuda.Stream]:
        """计算 stream"""
        return self._stream_manager.compute_stream
    
    @property
    def comm_stream(self) -> Optional[torch.cuda.Stream]:
        """通信 stream"""
        return self._stream_manager.comm_stream
    
    def compute(self):
        """计算 stream 上下文"""
        return torch.cuda.stream(self._stream_manager.compute_stream)
    
    def comm(self):
        """通信 stream 上下文"""
        return torch.cuda.stream(self._stream_manager.comm_stream)
    
    def signal_compute_done(self, wave_id: int) -> Signal:
        """标记计算 wave 完成"""
        return self._signal_wait.signal_wave(
            wave_id * 2,  # 偶数为计算信号
            self._stream_manager.compute_stream
        )
    
    def signal_comm_done(self, wave_id: int) -> Signal:
        """标记通信 wave 完成"""
        return self._signal_wait.signal_wave(
            wave_id * 2 + 1,  # 奇数为通信信号
            self._stream_manager.comm_stream
        )
    
    def wait_compute_done(self, wave_id: int) -> None:
        """等待计算 wave 完成"""
        self._signal_wait.wait_wave(
            wave_id * 2,
            self._stream_manager.comm_stream
        )
    
    def wait_comm_done(self, wave_id: int) -> None:
        """等待通信 wave 完成"""
        self._signal_wait.wait_wave(
            wave_id * 2 + 1,
            self._stream_manager.compute_stream
        )
    
    def sync_all(self) -> None:
        """同步所有操作"""
        self._stream_manager.sync_all()
    
    def reset(self) -> None:
        """重置状态"""
        self._signal_wait.reset()
