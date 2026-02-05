"""
MACO TaskGraph 自定义异常

提供清晰的错误消息，帮助用户快速定位问题。
"""


class TaskGraphError(Exception):
    """TaskGraph 基础异常类"""

    pass


class ValidationError(TaskGraphError):
    """输入验证错误"""

    pass


class NullInputError(ValidationError):
    """输入为 None"""

    def __init__(self, param_name: str, method_name: str):
        self.param_name = param_name
        self.method_name = method_name
        super().__init__(
            f"'{param_name}' cannot be None in {method_name}()\n"
            f"Hint: Ensure all input tensors are properly initialized before calling graph.{method_name}()"
        )


class DeviceError(ValidationError):
    """设备错误（如 CPU tensor）"""

    def __init__(self, param_name: str, actual_device: str, expected_device: str = "cuda"):
        self.param_name = param_name
        self.actual_device = actual_device
        self.expected_device = expected_device
        super().__init__(
            f"'{param_name}' is on {actual_device}, expected {expected_device}\n"
            f"Hint: Move tensor to GPU with tensor.to('cuda') or tensor.cuda()"
        )


class ShapeMismatchError(ValidationError):
    """形状不匹配"""

    def __init__(self, op: str, details: str, hint: str = None):
        self.op = op
        self.details = details
        message = f"Shape mismatch in '{op}': {details}"
        if hint:
            message += f"\nHint: {hint}"
        super().__init__(message)


class DimensionError(ValidationError):
    """维度数量错误"""

    def __init__(self, param_name: str, expected_ndim: int, actual_ndim: int):
        self.param_name = param_name
        self.expected_ndim = expected_ndim
        self.actual_ndim = actual_ndim
        super().__init__(
            f"'{param_name}' has {actual_ndim} dimensions, expected {expected_ndim}\n"
            f"Hint: Reshape tensor to have {expected_ndim} dimensions"
        )


class DTypeMismatchError(ValidationError):
    """数据类型不匹配"""

    def __init__(self, param1: str, dtype1: str, param2: str, dtype2: str):
        self.param1 = param1
        self.dtype1 = dtype1
        self.param2 = param2
        self.dtype2 = dtype2
        super().__init__(
            f"dtype mismatch: '{param1}' is {dtype1}, '{param2}' is {dtype2}\n"
            f"Hint: Convert tensors to the same dtype with tensor.to(dtype)"
        )


class CompilationError(TaskGraphError):
    """编译错误"""

    pass


class CyclicDependencyError(CompilationError):
    """循环依赖"""

    def __init__(self, cycle_nodes: list = None):
        self.cycle_nodes = cycle_nodes
        message = "Cyclic dependency detected in task graph"
        if cycle_nodes:
            node_names = " -> ".join(str(n) for n in cycle_nodes)
            message += f": {node_names}"
        message += "\nHint: Check task dependencies and remove circular references"
        super().__init__(message)


class NotCompiledError(TaskGraphError):
    """未编译就执行"""

    def __init__(self):
        super().__init__(
            "TaskGraph has not been compiled\n"
            "Hint: Call graph.compile() before graph.execute()"
        )


class AlreadyCompiledError(TaskGraphError):
    """编译后修改"""

    def __init__(self):
        super().__init__(
            "Cannot modify TaskGraph after compilation\n"
            "Hint: Create a new TaskGraph for different operations"
        )


class RuntimeError(TaskGraphError):
    """运行时错误"""

    pass


class CUDANotAvailableError(RuntimeError):
    """CUDA 不可用"""

    def __init__(self):
        super().__init__(
            "CUDA is not available\n"
            "Hint: Check that PyTorch is installed with CUDA support and GPU drivers are installed"
        )
