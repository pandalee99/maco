"""
MACO TaskGraph - SM 级别任务调度 API

TaskGraph 提供细粒度的任务依赖控制，将计算和通信任务编排到 SM 调度器执行。

核心概念：
- TaskNode: 单个任务节点（计算或通信）
- TaskGraph: 任务依赖图
- TaskSchedule: 编译后的执行计划（包含 Wave Grouping）

技术借鉴：
- FlashOverlap: Signal-Wait 机制, Wave Grouping
- ParallelKittens: Compute/Comm SM 分离
- Mirage: Persistent Kernel, GPU Atomics

Example:
    from maco.task_graph import TaskGraph

    # 创建任务图
    graph = TaskGraph(num_workers=8)

    # 添加任务
    t1 = graph.linear(x, w1, name="linear1")
    t2 = graph.allreduce(t1.output, name="ar1")
    t3 = graph.linear(t2.output, w2, name="linear2")

    # 标记重叠
    graph.overlap([t1], [t2])

    # 编译和执行
    graph.compile()
    graph.execute()
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple, Union
import torch
import torch.distributed as dist

from .exceptions import (
    TaskGraphError,
    ValidationError,
    NullInputError,
    DeviceError,
    ShapeMismatchError,
    DimensionError,
    DTypeMismatchError,
    CyclicDependencyError,
    NotCompiledError,
    AlreadyCompiledError,
    CUDANotAvailableError,
)
from .validation import (
    validate_tensor,
    validate_linear_shapes,
    validate_matmul_shapes,
    validate_dtype_match,
    validate_same_device,
    infer_device,
    infer_dtype,
)


class TaskType(Enum):
    """任务类型，对应 maco_types.h 中的 MacoTaskType"""

    # 终止任务
    TERMINATE = 0
    NOP = 1

    # 计算任务 (100-199)
    MATMUL = 100
    LINEAR = 101
    ATTENTION = 102
    LAYERNORM = 103
    RMSNORM = 104
    GELU = 105
    SILU = 106
    SOFTMAX = 107
    CUSTOM_COMPUTE = 150

    # 通信任务 (200-299)
    ALLREDUCE = 200
    ALLGATHER = 201
    REDUCE_SCATTER = 202
    ALL_TO_ALL = 203
    BROADCAST = 204
    CUSTOM_COMM = 250


@dataclass
class TaskNode:
    """
    任务节点 - 表示一个计算或通信操作

    Attributes:
        name: 任务名称（用于调试）
        task_type: 任务类型
        inputs: 输入张量列表
        outputs: 输出张量列表
        dims: 维度参数（任务特定）
        depends_on: 依赖的前置任务
        priority: 优先级（越大越优先）
        target_worker: 目标 Worker ID（-1 表示自动分配）
    """

    name: str
    task_type: TaskType
    inputs: List[torch.Tensor] = field(default_factory=list)
    outputs: List[torch.Tensor] = field(default_factory=list)
    dims: List[int] = field(default_factory=list)

    # 依赖关系
    depends_on: List["TaskNode"] = field(default_factory=list)

    # 调度提示
    priority: int = 0
    target_worker: int = -1

    # 自定义函数（用于 CUSTOM_COMPUTE/CUSTOM_COMM）
    custom_fn: Optional[Callable] = None

    # 内部状态（由 TaskGraph 设置）
    _node_id: int = -1
    _event_id: int = -1
    _wave_id: int = -1
    _overlap_role: Optional[str] = None  # "compute" or "comm"

    def add_dependency(self, *tasks: "TaskNode") -> "TaskNode":
        """添加依赖（fluent API）"""
        for task in tasks:
            if task not in self.depends_on:
                self.depends_on.append(task)
        return self

    @property
    def output(self) -> Optional[torch.Tensor]:
        """获取第一个输出张量（便捷属性）"""
        return self.outputs[0] if self.outputs else None

    @property
    def is_compute(self) -> bool:
        """是否为计算任务"""
        return 100 <= self.task_type.value < 200

    @property
    def is_comm(self) -> bool:
        """是否为通信任务"""
        return 200 <= self.task_type.value < 300


class OverlapGroup:
    """
    重叠组 - 标记需要重叠执行的计算和通信任务

    实现 FlashOverlap 风格的 Wave Grouping：
    - 计算任务分组为多个 waves
    - 每个 wave 完成后立即开始对应通信
    """

    def __init__(
        self,
        graph: "TaskGraph",
        compute_tasks: List[TaskNode],
        comm_tasks: List[TaskNode],
    ):
        self.graph = graph
        self.compute_tasks = compute_tasks
        self.comm_tasks = comm_tasks
        self.num_waves = 1  # 默认 1 个 wave

        # 标记任务角色
        for task in compute_tasks:
            task._overlap_role = "compute"

        for task in comm_tasks:
            task._overlap_role = "comm"

    def with_waves(self, num_waves: int) -> "OverlapGroup":
        """设置 wave 数量（手动）"""
        self.num_waves = num_waves
        self._assign_waves()
        return self

    def auto_waves(self) -> "OverlapGroup":
        """自动检测 wave 数量（根据计算量）"""
        # 简单策略：根据计算任务数量
        n = len(self.compute_tasks)
        if n <= 2:
            self.num_waves = 1
        elif n <= 8:
            self.num_waves = 2
        else:
            self.num_waves = 4

        self._assign_waves()
        return self

    def _assign_waves(self):
        """分配任务到 waves"""
        n = len(self.compute_tasks)
        wave_size = max(1, n // self.num_waves)

        for i, task in enumerate(self.compute_tasks):
            task._wave_id = min(i // wave_size, self.num_waves - 1)


class TaskSchedule:
    """
    任务执行计划 - 编译后的调度信息

    包含：
    - waves: 按依赖关系排序的任务波
    - worker_assignments: 任务到 Worker 的分配
    - event_assignments: 任务到事件 ID 的分配
    """

    def __init__(self, nodes: List[TaskNode], num_workers: int):
        self.nodes = nodes
        self.num_workers = num_workers

        # 计算执行波（拓扑排序）
        self.waves = self._compute_waves()

        # 分配 Workers
        self.worker_assignments = self._assign_workers()

    def _compute_waves(self) -> List[List[TaskNode]]:
        """使用 Kahn 算法计算执行波（拓扑排序）"""
        # 计算入度
        in_degree = {n._node_id: len(n.depends_on) for n in self.nodes}
        ready = [n for n in self.nodes if in_degree[n._node_id] == 0]

        waves = []
        processed = set()

        while ready:
            waves.append(ready)
            processed.update(n._node_id for n in ready)

            next_ready = []
            for node in ready:
                for candidate in self.nodes:
                    if candidate._node_id in processed:
                        continue
                    # 检查是否所有依赖都已处理
                    if all(dep._node_id in processed for dep in candidate.depends_on):
                        if candidate._node_id not in [n._node_id for n in next_ready]:
                            in_degree[candidate._node_id] = 0
                            next_ready.append(candidate)

            ready = next_ready

        return waves

    def _assign_workers(self) -> Dict[int, int]:
        """分配任务到 Workers（round-robin with hints）"""
        assignments = {}
        worker_idx = 0

        for node in self.nodes:
            if node.target_worker >= 0:
                assignments[node._node_id] = node.target_worker % self.num_workers
            else:
                assignments[node._node_id] = worker_idx
                worker_idx = (worker_idx + 1) % self.num_workers

        return assignments


class TaskGraph:
    """
    任务依赖图 - MACO SM 调度的核心 API

    用法:
        graph = TaskGraph(num_workers=8)
        t1 = graph.linear(x, w, name="linear")
        t2 = graph.allreduce(t1.output, name="ar")
        graph.compile()
        graph.execute()
    """

    def __init__(
        self,
        num_workers: int = None,
        device: torch.device = None,
    ):
        """
        创建任务图

        Args:
            num_workers: Worker 数量（None = 自动检测）
            device: 执行设备
        """
        self._nodes: List[TaskNode] = []
        self._node_counter = 0
        self._event_counter = 0
        self._overlap_groups: List[OverlapGroup] = []

        # 配置
        self.num_workers = num_workers or self._auto_detect_workers()
        self.device = device or self._get_default_device()

        # 编译状态
        self._compiled = False
        self._schedule: Optional[TaskSchedule] = None
        self._runtime = None

    # ========== 计算任务 ==========

    def linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        name: str = None,
        output: torch.Tensor = None,
    ) -> TaskNode:
        """
        创建 Linear（矩阵乘法）任务

        Args:
            input: 输入张量 [batch, in_features]
            weight: 权重张量 [out_features, in_features]
            bias: 偏置张量 [out_features]（可选）
            name: 任务名称
            output: 输出张量（可选，自动创建）

        Returns:
            TaskNode

        Raises:
            NullInputError: input 或 weight 为 None
            DeviceError: tensor 不在 CUDA 设备上
            ShapeMismatchError: 形状不兼容
            AlreadyCompiledError: 图已编译
        """
        self._check_not_compiled()

        # 验证输入
        validate_tensor(input, "input", "linear", min_ndim=1)
        validate_tensor(weight, "weight", "linear", ndim=2)
        if bias is not None:
            validate_tensor(bias, "bias", "linear", ndim=1)

        # 验证形状兼容性
        validate_linear_shapes(input, weight, bias)

        # 验证 dtype 和 device
        validate_dtype_match(input, "input", weight, "weight")
        validate_same_device(input, "input", weight, "weight")

        # 自动推断 device
        if self.device is None:
            self.device = infer_device(input, weight)

        # 自动创建输出
        if output is None:
            out_shape = list(input.shape)
            out_shape[-1] = weight.shape[0]
            output = torch.empty(out_shape, dtype=input.dtype, device=self.device)

        inputs = [input, weight]
        if bias is not None:
            inputs.append(bias)

        node = TaskNode(
            name=name or f"linear_{self._node_counter}",
            task_type=TaskType.LINEAR,
            inputs=inputs,
            outputs=[output],
            dims=[input.shape[-1], weight.shape[0]],  # K, N
        )

        return self._add_node(node)

    def matmul(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        name: str = None,
        output: torch.Tensor = None,
    ) -> TaskNode:
        """
        创建矩阵乘法任务

        Args:
            a: 第一个 tensor [..., M, K]
            b: 第二个 tensor [..., K, N]
            name: 任务名称
            output: 输出张量（可选，自动创建）

        Returns:
            TaskNode

        Raises:
            NullInputError: a 或 b 为 None
            DeviceError: tensor 不在 CUDA 设备上
            ShapeMismatchError: 形状不兼容
            AlreadyCompiledError: 图已编译
        """
        self._check_not_compiled()

        # 验证输入
        validate_tensor(a, "a", "matmul", min_ndim=1)
        validate_tensor(b, "b", "matmul", min_ndim=1)

        # 验证形状兼容性
        validate_matmul_shapes(a, b)

        # 验证 dtype 和 device
        validate_dtype_match(a, "a", b, "b")
        validate_same_device(a, "a", b, "b")

        # 自动推断 device
        if self.device is None:
            self.device = infer_device(a, b)

        if output is None:
            out_shape = list(a.shape[:-1]) + [b.shape[-1]]
            output = torch.empty(out_shape, dtype=a.dtype, device=self.device)

        node = TaskNode(
            name=name or f"matmul_{self._node_counter}",
            task_type=TaskType.MATMUL,
            inputs=[a, b],
            outputs=[output],
            dims=list(a.shape) + list(b.shape),
        )

        return self._add_node(node)

    def custom(
        self,
        fn: Callable,
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        name: str = None,
        is_comm: bool = False,
    ) -> TaskNode:
        """
        创建自定义任务

        Args:
            fn: 自定义函数
            inputs: 输入张量列表
            outputs: 输出张量列表
            name: 任务名称
            is_comm: 是否为通信任务

        Returns:
            TaskNode

        Raises:
            NullInputError: fn, inputs 或 outputs 为 None
            AlreadyCompiledError: 图已编译
        """
        self._check_not_compiled()

        # 验证输入
        if fn is None:
            raise NullInputError("fn", "custom")
        if inputs is None:
            raise NullInputError("inputs", "custom")
        if outputs is None:
            raise NullInputError("outputs", "custom")
        if not callable(fn):
            raise ValidationError("'fn' must be callable in custom()")

        # 验证 inputs 和 outputs 中的 tensor
        for i, inp in enumerate(inputs):
            if inp is not None:
                validate_tensor(inp, f"inputs[{i}]", "custom")

        for i, out in enumerate(outputs):
            if out is not None:
                validate_tensor(out, f"outputs[{i}]", "custom")

        # 自动推断 device
        if self.device is None:
            all_tensors = [t for t in inputs + outputs if t is not None]
            if all_tensors:
                self.device = infer_device(*all_tensors)

        task_type = TaskType.CUSTOM_COMM if is_comm else TaskType.CUSTOM_COMPUTE

        node = TaskNode(
            name=name or f"custom_{self._node_counter}",
            task_type=task_type,
            inputs=inputs,
            outputs=outputs,
            dims=[],
            custom_fn=fn,
        )

        return self._add_node(node)

    # ========== 通信任务 ==========

    def allreduce(
        self,
        tensor: torch.Tensor,
        name: str = None,
        op: str = "sum",
    ) -> TaskNode:
        """
        创建 AllReduce 通信任务

        Args:
            tensor: 要 AllReduce 的张量
            name: 任务名称
            op: 归约操作（"sum", "avg", "min", "max"）

        Returns:
            TaskNode

        Raises:
            NullInputError: tensor 为 None
            DeviceError: tensor 不在 CUDA 设备上
            AlreadyCompiledError: 图已编译
        """
        self._check_not_compiled()

        # 验证输入
        validate_tensor(tensor, "tensor", "allreduce")

        # 自动推断 device
        if self.device is None:
            self.device = infer_device(tensor)

        node = TaskNode(
            name=name or f"allreduce_{self._node_counter}",
            task_type=TaskType.ALLREDUCE,
            inputs=[tensor],
            outputs=[tensor],  # in-place
            dims=list(tensor.shape),
        )

        return self._add_node(node)

    def all_to_all(
        self,
        input: torch.Tensor,
        name: str = None,
        scatter_dim: int = 2,
        gather_dim: int = 1,
    ) -> TaskNode:
        """
        创建 All-to-All 通信任务

        Args:
            input: 输入张量
            scatter_dim: 分散维度
            gather_dim: 聚集维度

        Returns:
            TaskNode

        Raises:
            NullInputError: input 为 None
            DeviceError: tensor 不在 CUDA 设备上
            AlreadyCompiledError: 图已编译
        """
        self._check_not_compiled()

        # 验证输入
        validate_tensor(input, "input", "all_to_all")

        # 自动推断 device
        if self.device is None:
            self.device = infer_device(input)

        output = torch.empty_like(input)

        node = TaskNode(
            name=name or f"all_to_all_{self._node_counter}",
            task_type=TaskType.ALL_TO_ALL,
            inputs=[input],
            outputs=[output],
            dims=[scatter_dim, gather_dim] + list(input.shape),
        )

        return self._add_node(node)

    # ========== 重叠标记 ==========

    def overlap(
        self,
        compute_tasks: List[TaskNode],
        comm_tasks: List[TaskNode],
    ) -> OverlapGroup:
        """
        标记计算和通信任务为重叠执行

        Args:
            compute_tasks: 计算任务列表
            comm_tasks: 通信任务列表

        Returns:
            OverlapGroup（可链式调用 .with_waves() 或 .auto_waves()）
        """
        group = OverlapGroup(self, compute_tasks, comm_tasks)
        self._overlap_groups.append(group)
        return group

    # ========== 编译和执行 ==========

    def compile(self):
        """
        编译任务图

        步骤:
        1. 推断数据依赖
        2. 检测循环依赖
        3. 分配事件 ID
        4. 生成执行计划（Wave Grouping）
        5. 初始化运行时

        Raises:
            CyclicDependencyError: 存在循环依赖
        """
        if self._compiled:
            return

        # 1. 推断依赖
        self._infer_dependencies()

        # 2. 检测循环依赖
        self._detect_cycles()

        # 3. 分配事件
        self._assign_events()

        # 4. 自动 wave grouping（如果没有手动设置）
        for group in self._overlap_groups:
            if group.num_waves == 1 and len(group.compute_tasks) > 1:
                group.auto_waves()

        # 5. 生成执行计划
        self._schedule = TaskSchedule(self._nodes, self.num_workers)

        # 6. 初始化运行时
        self._init_runtime()

        self._compiled = True

    def execute(self, stream: torch.cuda.Stream = None):
        """
        执行任务图

        Args:
            stream: CUDA stream（可选）
        """
        if not self._compiled:
            self.compile()

        self._runtime.execute(self._schedule, stream)

    # ========== 内部方法 ==========

    def _check_not_compiled(self) -> None:
        """检查图是否已编译（编译后不允许修改）"""
        if self._compiled:
            raise AlreadyCompiledError()

    def _add_node(self, node: TaskNode) -> TaskNode:
        """添加节点到图"""
        node._node_id = self._node_counter
        self._node_counter += 1
        self._nodes.append(node)
        return node

    def _infer_dependencies(self):
        """从数据流推断依赖关系"""
        # tensor.data_ptr() -> producer node
        tensor_producers: Dict[int, TaskNode] = {}

        for node in self._nodes:
            # 检查输入是否由其他节点产生
            for inp in node.inputs:
                if inp is None:
                    continue
                tensor_id = inp.data_ptr()
                if tensor_id in tensor_producers:
                    producer = tensor_producers[tensor_id]
                    if producer not in node.depends_on:
                        node.depends_on.append(producer)

            # 注册输出
            for out in node.outputs:
                if out is not None:
                    tensor_producers[out.data_ptr()] = node

    def _detect_cycles(self):
        """检测循环依赖（使用 DFS）"""
        # 状态: 0=未访问, 1=访问中, 2=已完成
        state = {n._node_id: 0 for n in self._nodes}
        path = []

        def dfs(node: TaskNode) -> bool:
            """返回 True 表示发现环"""
            state[node._node_id] = 1  # 访问中
            path.append(node.name)

            for dep in node.depends_on:
                if state[dep._node_id] == 1:
                    # 发现环
                    cycle_start = path.index(dep.name)
                    cycle = path[cycle_start:] + [dep.name]
                    raise CyclicDependencyError(cycle)
                elif state[dep._node_id] == 0:
                    dfs(dep)

            state[node._node_id] = 2  # 已完成
            path.pop()

        for node in self._nodes:
            if state[node._node_id] == 0:
                dfs(node)

    def _assign_events(self):
        """分配事件 ID"""
        for node in self._nodes:
            node._event_id = self._event_counter
            self._event_counter += 1

    def _init_runtime(self):
        """初始化运行时"""
        # 当前默认使用 StreamRuntime（简单可靠）
        # MacoRuntime (SM 调度) 将在后续版本启用
        from .runtime import StreamRuntime
        self._runtime = StreamRuntime(self.device)

    def _auto_detect_workers(self) -> int:
        """自动检测 Worker 数量"""
        if not torch.cuda.is_available():
            return 1

        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        # 保留一些 SM 给调度器和通信
        return max(1, sm_count - 8)

    def _get_default_device(self) -> torch.device:
        """获取默认设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ========== 调试/信息 ==========

    def __repr__(self) -> str:
        return (
            f"TaskGraph(nodes={len(self._nodes)}, "
            f"workers={self.num_workers}, "
            f"compiled={self._compiled})"
        )

    def summary(self) -> str:
        """返回任务图摘要"""
        compute_nodes = sum(1 for n in self._nodes if n.is_compute)
        comm_nodes = sum(1 for n in self._nodes if n.is_comm)

        lines = [
            "TaskGraph Summary:",
            f"  Total nodes: {len(self._nodes)}",
            f"  Compute nodes: {compute_nodes}",
            f"  Comm nodes: {comm_nodes}",
            f"  Overlap groups: {len(self._overlap_groups)}",
            f"  Num workers: {self.num_workers}",
            f"  Device: {self.device}",
        ]

        if self._compiled and self._schedule:
            lines.append(f"  Execution waves: {len(self._schedule.waves)}")

        return "\n".join(lines)
