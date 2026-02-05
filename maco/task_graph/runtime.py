"""
MACO TaskGraph 运行时

提供两种运行时实现：
1. MacoRuntime: 使用 maco._sm 的 SM 级别调度
2. StreamRuntime: 基于 CUDA Stream 的回退实现

技术借鉴：
- FlashOverlap: Signal-Wait 同步
- Mirage: Persistent Kernel 任务分发
"""

from typing import List, Dict, Optional, TYPE_CHECKING
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from . import TaskNode, TaskSchedule


class MacoRuntime:
    """
    SM 调度运行时

    使用 maco._sm 的 Persistent Kernel 执行任务图：
    - Worker CTA 执行计算任务
    - 通过 GPU Atomics 同步
    - Signal-Wait 实现 wave 间重叠
    """

    def __init__(
        self,
        num_workers: int,
        queue_size: int,
        num_events: int,
        device: torch.device,
    ):
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.num_events = num_events
        self.device = device

        # 尝试加载 SM 调度模块
        try:
            from maco._sm import init_simple_persistent_kernel_runtime

            # 初始化运行时结构
            (
                self.task_queues,
                self.queue_heads,
                self.queue_tails,
                self.event_counters,
                self.event_descs,
                self.control,
            ) = init_simple_persistent_kernel_runtime(
                num_workers, queue_size, num_events, device
            )
            self._sm_available = True

        except ImportError:
            print("[MACO] Warning: maco._sm not available, will use stream fallback")
            self._sm_available = False
            self._stream_runtime = StreamRuntime(device)

    def execute(self, schedule: "TaskSchedule", stream: torch.cuda.Stream = None):
        """
        执行任务计划

        Args:
            schedule: 编译后的任务计划
            stream: CUDA stream（可选）
        """
        if not self._sm_available:
            return self._stream_runtime.execute(schedule, stream)

        from maco._sm import (
            add_task_to_queue,
            launch_simple_persistent_kernel,
            set_terminate_flag,
            TASK_TERMINATE,
        )

        # 重置控制变量
        self.control.zero_()
        self.event_counters.zero_()
        self.queue_heads.zero_()
        self.queue_tails.zero_()

        # 1. 将任务加入队列
        for wave_idx, wave in enumerate(schedule.waves):
            for node in wave:
                worker_id = schedule.worker_assignments[node._node_id]

                # 获取依赖事件
                dep_event = -1
                if node.depends_on:
                    dep_event = node.depends_on[0]._event_id

                # 准备输入/输出张量
                input_tensor = node.inputs[0] if node.inputs else torch.empty(0, device=self.device)
                output_tensor = node.outputs[0] if node.outputs else torch.empty(0, device=self.device)

                add_task_to_queue(
                    self.task_queues,
                    self.queue_heads,
                    worker_id,
                    self.queue_size,
                    node.task_type.value,
                    input_tensor,
                    output_tensor,
                    node.dims,
                    dep_event,
                    node._event_id,
                )

        # 2. 添加终止任务
        for worker_id in range(self.num_workers):
            add_task_to_queue(
                self.task_queues,
                self.queue_heads,
                worker_id,
                self.queue_size,
                TASK_TERMINATE,
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
                [],
                -1,
                -1,
            )

        # 3. 启动 Persistent Kernel
        launch_simple_persistent_kernel(
            self.task_queues,
            self.queue_heads,
            self.queue_tails,
            self.event_counters,
            self.event_descs,
            self.control,
            self.queue_size,
            self.num_workers,
            128,  # threads per worker
        )

        torch.cuda.synchronize()


class StreamRuntime:
    """
    Stream 回退运行时

    当 maco._sm 不可用时，使用简单的顺序执行：
    - 按拓扑顺序执行每个波
    - 计算任务直接执行
    - 通信任务调用 dist 函数
    """

    def __init__(self, device: torch.device):
        self.device = device

    def execute(self, schedule: "TaskSchedule", stream: torch.cuda.Stream = None):
        """
        执行任务计划（顺序模式）

        Args:
            schedule: 编译后的任务计划
            stream: CUDA stream（可选，暂不使用）
        """
        # 按波顺序执行
        for wave_idx, wave in enumerate(schedule.waves):
            for node in wave:
                self._execute_task(node)

    def _execute_task(self, node: "TaskNode"):
        """执行单个任务"""
        from . import TaskType

        if node.task_type == TaskType.LINEAR:
            self._execute_linear(node)
        elif node.task_type == TaskType.MATMUL:
            self._execute_matmul(node)
        elif node.task_type == TaskType.ALLREDUCE:
            self._execute_allreduce(node)
        elif node.task_type == TaskType.ALL_TO_ALL:
            self._execute_all_to_all(node)
        elif node.task_type in (TaskType.CUSTOM_COMPUTE, TaskType.CUSTOM_COMM):
            self._execute_custom(node)

    def _execute_linear(self, node: "TaskNode"):
        """执行 Linear"""
        input_tensor = node.inputs[0]
        weight = node.inputs[1]
        bias = node.inputs[2] if len(node.inputs) > 2 else None
        output = node.outputs[0]

        # 使用 torch.mm 直接写入输出
        # input: [batch, in_features], weight: [out_features, in_features]
        # output: [batch, out_features]
        torch.mm(input_tensor, weight.t(), out=output)
        if bias is not None:
            output.add_(bias)

    def _execute_matmul(self, node: "TaskNode"):
        """执行矩阵乘法"""
        a = node.inputs[0]
        b = node.inputs[1]
        output = node.outputs[0]

        # 直接使用 mm 写入输出
        torch.mm(a, b, out=output)

    def _execute_allreduce(self, node: "TaskNode"):
        """执行 AllReduce"""
        tensor = node.inputs[0]

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(tensor)

    def _execute_all_to_all(self, node: "TaskNode"):
        """执行 All-to-All"""
        input_tensor = node.inputs[0]
        output_tensor = node.outputs[0]

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_to_all_single(output_tensor, input_tensor)
        else:
            output_tensor.copy_(input_tensor)

    def _execute_custom(self, node: "TaskNode"):
        """执行自定义任务"""
        if node.custom_fn is not None:
            result = node.custom_fn(*node.inputs)
            if result is not None and node.outputs:
                if isinstance(result, torch.Tensor):
                    node.outputs[0].copy_(result)
