"""
MACO Stream Manager

管理计算和通信的 CUDA streams，支持并行执行。
"""

from typing import Optional, Dict
import torch


class StreamManager:
    """
    CUDA Stream 管理器
    
    管理计算和通信使用的独立 streams，实现并行执行。
    
    设计原则：
    - 计算任务在 compute_stream 上执行
    - 通信任务在 comm_stream 上执行
    - 两个 stream 可以并行运行
    - 通过 Signal-Wait 机制协调依赖
    
    Example:
        >>> sm = StreamManager()
        >>> with sm.compute_stream:
        ...     # 计算任务
        ...     result = torch.matmul(a, b)
        >>> with sm.comm_stream:
        ...     # 通信任务
        ...     dist.all_reduce(tensor)
    """
    
    _instances: Dict[int, "StreamManager"] = {}
    
    def __new__(cls, device: Optional[torch.device] = None) -> "StreamManager":
        """每个设备一个实例"""
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
        
        device_id = device.index if device.type == "cuda" else -1
        
        if device_id not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[device_id] = instance
        
        return cls._instances[device_id]
    
    def __init__(self, device: Optional[torch.device] = None):
        if self._initialized:
            return
        
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
        
        self._device = device
        self._compute_stream: Optional[torch.cuda.Stream] = None
        self._comm_stream: Optional[torch.cuda.Stream] = None
        self._default_stream: Optional[torch.cuda.Stream] = None
        
        if device.type == "cuda":
            with torch.cuda.device(device):
                # 计算 stream（高优先级）
                self._compute_stream = torch.cuda.Stream(
                    device=device, priority=-1  # 高优先级
                )
                # 通信 stream（普通优先级）
                self._comm_stream = torch.cuda.Stream(
                    device=device, priority=0
                )
                # 默认 stream
                self._default_stream = torch.cuda.current_stream(device)
        
        self._initialized = True
    
    @property
    def device(self) -> torch.device:
        """当前设备"""
        return self._device
    
    @property
    def compute_stream(self) -> Optional[torch.cuda.Stream]:
        """计算 stream"""
        return self._compute_stream
    
    @property
    def comm_stream(self) -> Optional[torch.cuda.Stream]:
        """通信 stream"""
        return self._comm_stream
    
    @property
    def default_stream(self) -> Optional[torch.cuda.Stream]:
        """默认 stream"""
        return self._default_stream
    
    def sync_compute(self) -> None:
        """同步计算 stream"""
        if self._compute_stream is not None:
            self._compute_stream.synchronize()
    
    def sync_comm(self) -> None:
        """同步通信 stream"""
        if self._comm_stream is not None:
            self._comm_stream.synchronize()
    
    def sync_all(self) -> None:
        """同步所有 stream"""
        self.sync_compute()
        self.sync_comm()
        if self._default_stream is not None:
            self._default_stream.synchronize()
    
    def compute_wait_comm(self) -> None:
        """让计算 stream 等待通信 stream"""
        if self._compute_stream is None or self._comm_stream is None:
            return
        
        event = torch.cuda.Event()
        event.record(self._comm_stream)
        self._compute_stream.wait_event(event)
    
    def comm_wait_compute(self) -> None:
        """让通信 stream 等待计算 stream"""
        if self._compute_stream is None or self._comm_stream is None:
            return
        
        event = torch.cuda.Event()
        event.record(self._compute_stream)
        self._comm_stream.wait_event(event)
    
    def default_wait_all(self) -> None:
        """让默认 stream 等待所有其他 stream"""
        if self._default_stream is None:
            return
        
        if self._compute_stream is not None:
            event = torch.cuda.Event()
            event.record(self._compute_stream)
            self._default_stream.wait_event(event)
        
        if self._comm_stream is not None:
            event = torch.cuda.Event()
            event.record(self._comm_stream)
            self._default_stream.wait_event(event)
    
    @classmethod
    def reset(cls) -> None:
        """重置所有实例（用于测试）"""
        cls._instances.clear()


# 便捷函数
_default_manager: Optional[StreamManager] = None


def _get_default_manager() -> StreamManager:
    """获取默认 StreamManager"""
    global _default_manager
    if _default_manager is None:
        _default_manager = StreamManager()
    return _default_manager


def get_compute_stream() -> Optional[torch.cuda.Stream]:
    """获取计算 stream"""
    return _get_default_manager().compute_stream


def get_comm_stream() -> Optional[torch.cuda.Stream]:
    """获取通信 stream"""
    return _get_default_manager().comm_stream
