"""
MACO Process Group Manager

管理分布式进程组，封装 torch.distributed 初始化和查询。
"""

import os
from typing import Optional
import torch
import torch.distributed as dist


class ProcessGroupManager:
    """
    进程组管理器
    
    管理 NCCL 进程组的初始化、查询和销毁。
    支持多种初始化方式：环境变量、手动指定。
    
    Example:
        >>> pgm = ProcessGroupManager()
        >>> pgm.init_process_group()
        >>> print(f"Rank {pgm.rank} / {pgm.world_size}")
        >>> pgm.destroy()
    """
    
    _instance: Optional["ProcessGroupManager"] = None
    
    def __new__(cls) -> "ProcessGroupManager":
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._world_size = 1
        self._rank = 0
        self._local_rank = 0
        self._backend = "nccl"
        self._device = None
        self._comm_stream = None  # 专用通信 stream
        
    def init_process_group(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        """
        初始化进程组
        
        Args:
            backend: 通信后端，默认 "nccl"
            init_method: 初始化方法，默认从环境变量读取
            world_size: 总进程数，默认从环境变量读取
            rank: 当前进程排名，默认从环境变量读取
        """
        if dist.is_initialized():
            self._sync_from_dist()
            self._initialized = True
            return
            
        # 从环境变量获取参数
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        
        self._world_size = world_size
        self._rank = rank
        self._local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        self._backend = backend
        
        if world_size > 1:
            if init_method is None:
                # 使用环境变量初始化
                dist.init_process_group(
                    backend=backend,
                    world_size=world_size,
                    rank=rank,
                )
            else:
                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank,
                )
        
        # 设置当前设备
        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
            self._device = torch.device("cuda", self._local_rank)
            # 创建专用通信 stream
            self._comm_stream = torch.cuda.Stream(device=self._device)
        else:
            self._device = torch.device("cpu")
            
        self._initialized = True
        
    def _sync_from_dist(self) -> None:
        """从已初始化的 torch.distributed 同步状态"""
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._local_rank = int(os.environ.get("LOCAL_RANK", str(self._rank)))
        
        if torch.cuda.is_available():
            self._device = torch.device("cuda", self._local_rank)
            self._comm_stream = torch.cuda.Stream(device=self._device)
        else:
            self._device = torch.device("cpu")
    
    @property
    def world_size(self) -> int:
        """总进程数"""
        return self._world_size
    
    @property
    def rank(self) -> int:
        """当前进程排名"""
        return self._rank
    
    @property
    def local_rank(self) -> int:
        """本地排名（节点内）"""
        return self._local_rank
    
    @property
    def device(self) -> torch.device:
        """当前设备"""
        return self._device
    
    @property
    def comm_stream(self) -> Optional[torch.cuda.Stream]:
        """通信专用 stream"""
        return self._comm_stream
    
    @property
    def is_main_process(self) -> bool:
        """是否为主进程"""
        return self._rank == 0
    
    def barrier(self) -> None:
        """全局同步屏障"""
        if self._world_size > 1 and dist.is_initialized():
            dist.barrier()
    
    def destroy(self) -> None:
        """销毁进程组"""
        if dist.is_initialized():
            dist.destroy_process_group()
        self._initialized = False
        ProcessGroupManager._instance = None


# 便捷函数
_pgm: Optional[ProcessGroupManager] = None


def _get_pgm() -> ProcessGroupManager:
    """获取全局 ProcessGroupManager 实例"""
    global _pgm
    if _pgm is None:
        _pgm = ProcessGroupManager()
    return _pgm


def get_world_size() -> int:
    """获取总进程数"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """获取当前进程排名"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """获取本地排名"""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_initialized() -> bool:
    """检查是否已初始化"""
    return dist.is_initialized()


def barrier() -> None:
    """全局同步屏障"""
    if dist.is_initialized():
        dist.barrier()
