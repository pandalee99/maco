"""
MACO Overlap Scheduler

实现计算-通信重叠调度：
- Wave-level 调度：计算分 wave，每个 wave 完成后开始通信
- Pipeline 调度：producer-consumer 模式
- Signal-Wait 协调：跨 stream 同步
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import torch
import torch.distributed as dist

from ..sync import StreamManager, SignalWait, OverlapContext


class OverlapMode(Enum):
    """重叠模式"""
    NONE = auto()       # 无重叠，串行执行
    WAVE = auto()       # Wave-level 重叠
    PIPELINE = auto()   # Pipeline 重叠


@dataclass
class OverlapPlan:
    """
    重叠执行计划
    
    描述如何安排计算和通信任务以实现重叠。
    """
    mode: OverlapMode
    num_waves: int
    compute_waves: List[List["TaskNode"]] = field(default_factory=list)
    comm_tasks: List["TaskNode"] = field(default_factory=list)
    
    # Wave 到通信的映射
    wave_comm_mapping: Dict[int, List["TaskNode"]] = field(default_factory=dict)


class OverlapScheduler:
    """
    重叠调度器
    
    负责：
    1. 分析任务依赖，确定重叠机会
    2. 生成重叠执行计划
    3. 执行重叠调度
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() 
            else torch.device("cpu")
        )
        self._stream_manager = StreamManager(self.device) if self.device.type == "cuda" else None
        self._signal_wait = SignalWait() if self.device.type == "cuda" else None
        
    def create_plan(
        self,
        compute_tasks: List["TaskNode"],
        comm_tasks: List["TaskNode"],
        mode: OverlapMode = OverlapMode.WAVE,
        num_waves: Optional[int] = None,
    ) -> OverlapPlan:
        """
        创建重叠执行计划
        
        Args:
            compute_tasks: 计算任务列表
            comm_tasks: 通信任务列表
            mode: 重叠模式
            num_waves: wave 数量（None = 自动）
            
        Returns:
            OverlapPlan
        """
        if mode == OverlapMode.NONE:
            return OverlapPlan(
                mode=mode,
                num_waves=1,
                compute_waves=[compute_tasks],
                comm_tasks=comm_tasks,
            )
        
        # 自动确定 wave 数量
        if num_waves is None:
            num_waves = self._auto_determine_waves(compute_tasks, comm_tasks)
        
        # 将计算任务分配到 waves
        compute_waves = self._partition_waves(compute_tasks, num_waves)
        
        # 创建 wave 到通信的映射
        wave_comm_mapping = self._map_waves_to_comm(compute_waves, comm_tasks)
        
        return OverlapPlan(
            mode=mode,
            num_waves=num_waves,
            compute_waves=compute_waves,
            comm_tasks=comm_tasks,
            wave_comm_mapping=wave_comm_mapping,
        )
    
    def _auto_determine_waves(
        self,
        compute_tasks: List["TaskNode"],
        comm_tasks: List["TaskNode"],
    ) -> int:
        """自动确定 wave 数量"""
        n_compute = len(compute_tasks)
        n_comm = len(comm_tasks)
        
        if n_compute <= 1:
            return 1
        
        # 简单策略：尽量让每个 wave 有通信可以重叠
        if n_comm == 0:
            return 1
        
        # 计算任务数 / 通信任务数，至少 1 个 wave
        return max(1, min(n_comm, n_compute // 2))
    
    def _partition_waves(
        self,
        tasks: List["TaskNode"],
        num_waves: int,
    ) -> List[List["TaskNode"]]:
        """将任务分配到 waves"""
        if num_waves <= 0:
            num_waves = 1
        
        waves = [[] for _ in range(num_waves)]
        wave_size = max(1, len(tasks) // num_waves)
        
        for i, task in enumerate(tasks):
            wave_id = min(i // wave_size, num_waves - 1)
            waves[wave_id].append(task)
            task._wave_id = wave_id
        
        return waves
    
    def _map_waves_to_comm(
        self,
        compute_waves: List[List["TaskNode"]],
        comm_tasks: List["TaskNode"],
    ) -> Dict[int, List["TaskNode"]]:
        """映射 wave 到通信任务"""
        mapping = {}
        
        if not comm_tasks:
            return mapping
        
        num_waves = len(compute_waves)
        
        # 简单策略：将通信任务均匀分配给 waves
        for i, task in enumerate(comm_tasks):
            wave_id = i % num_waves
            if wave_id not in mapping:
                mapping[wave_id] = []
            mapping[wave_id].append(task)
        
        return mapping
    
    def execute_plan(
        self,
        plan: OverlapPlan,
        compute_fn: callable,
        comm_fn: callable,
    ) -> None:
        """
        执行重叠计划
        
        Args:
            plan: 重叠执行计划
            compute_fn: 执行计算任务的函数 (task) -> result
            comm_fn: 执行通信任务的函数 (task) -> result
        """
        if plan.mode == OverlapMode.NONE or self._stream_manager is None:
            # 串行执行
            for wave in plan.compute_waves:
                for task in wave:
                    compute_fn(task)
            for task in plan.comm_tasks:
                comm_fn(task)
            return
        
        # Wave-level 重叠执行
        self._execute_wave_overlap(plan, compute_fn, comm_fn)
    
    def _execute_wave_overlap(
        self,
        plan: OverlapPlan,
        compute_fn: callable,
        comm_fn: callable,
    ) -> None:
        """执行 wave-level 重叠"""
        compute_stream = self._stream_manager.compute_stream
        comm_stream = self._stream_manager.comm_stream
        
        for wave_id, wave_tasks in enumerate(plan.compute_waves):
            # 等待上一个 wave 的通信完成（如果有）
            if wave_id > 0:
                self._signal_wait.wait_wave(
                    (wave_id - 1) * 2 + 1,  # 上一个 wave 的通信信号
                    compute_stream
                )
            
            # 执行计算任务
            with torch.cuda.stream(compute_stream):
                for task in wave_tasks:
                    compute_fn(task)
            
            # 标记计算完成
            self._signal_wait.signal_wave(wave_id * 2, compute_stream)
            
            # 启动对应的通信任务
            if wave_id in plan.wave_comm_mapping:
                with torch.cuda.stream(comm_stream):
                    # 等待计算完成
                    self._signal_wait.wait_wave(wave_id * 2, comm_stream)
                    
                    # 执行通信
                    for task in plan.wave_comm_mapping[wave_id]:
                        comm_fn(task)
                
                # 标记通信完成
                self._signal_wait.signal_wave(wave_id * 2 + 1, comm_stream)
        
        # 同步所有操作
        self._stream_manager.sync_all()


class OverlapRuntime:
    """
    重叠执行运行时
    
    整合 OverlapScheduler 和 NCCL 通信，
    提供简单的 API 执行重叠计算-通信。
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.scheduler = OverlapScheduler(self.device)
        self._overlap_context = OverlapContext(self.device) if self.device.type == "cuda" else None
    
    def execute_overlap(
        self,
        compute_tasks: List["TaskNode"],
        comm_tasks: List["TaskNode"],
        compute_executor: callable,
        comm_executor: callable,
        mode: OverlapMode = OverlapMode.WAVE,
        num_waves: Optional[int] = None,
    ) -> None:
        """
        执行重叠计算-通信
        
        Args:
            compute_tasks: 计算任务
            comm_tasks: 通信任务
            compute_executor: 计算执行器
            comm_executor: 通信执行器
            mode: 重叠模式
            num_waves: wave 数量
        """
        plan = self.scheduler.create_plan(
            compute_tasks=compute_tasks,
            comm_tasks=comm_tasks,
            mode=mode,
            num_waves=num_waves,
        )
        
        self.scheduler.execute_plan(
            plan=plan,
            compute_fn=compute_executor,
            comm_fn=comm_executor,
        )
    
    @property
    def compute_stream(self) -> Optional[torch.cuda.Stream]:
        """计算 stream"""
        if self._overlap_context:
            return self._overlap_context.compute_stream
        return None
    
    @property
    def comm_stream(self) -> Optional[torch.cuda.Stream]:
        """通信 stream"""
        if self._overlap_context:
            return self._overlap_context.comm_stream
        return None
