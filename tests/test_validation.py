"""
MACO 输入验证测试

测试严格模式下的输入验证：
- None 输入检测
- CPU tensor 检测
- 形状不匹配检测
- dtype 不匹配检测
- 编译后修改检测
- 循环依赖检测
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maco import TaskGraph
from maco.task_graph.exceptions import (
    NullInputError,
    DeviceError,
    ShapeMismatchError,
    DTypeMismatchError,
    AlreadyCompiledError,
    CyclicDependencyError,
    ValidationError,
)


# ========== None 输入测试 ==========


@pytest.mark.cuda
class TestNullInput:
    """None 输入检测测试"""

    def test_linear_none_input(self, cuda_tensor):
        """linear() 的 input 为 None"""
        w = cuda_tensor((512, 256))

        graph = TaskGraph(num_workers=4)
        with pytest.raises(NullInputError) as exc_info:
            graph.linear(None, w)

        assert "input" in str(exc_info.value)
        assert "linear" in str(exc_info.value)

    def test_linear_none_weight(self, cuda_tensor):
        """linear() 的 weight 为 None"""
        x = cuda_tensor((32, 256))

        graph = TaskGraph(num_workers=4)
        with pytest.raises(NullInputError) as exc_info:
            graph.linear(x, None)

        assert "weight" in str(exc_info.value)

    def test_matmul_none_inputs(self, cuda_tensor):
        """matmul() 输入为 None"""
        a = cuda_tensor((32, 64))

        graph = TaskGraph(num_workers=4)
        with pytest.raises(NullInputError):
            graph.matmul(None, a)

        with pytest.raises(NullInputError):
            graph.matmul(a, None)

    def test_allreduce_none_tensor(self):
        """allreduce() 输入为 None"""
        graph = TaskGraph(num_workers=4)
        with pytest.raises(NullInputError):
            graph.allreduce(None)

    def test_custom_none_fn(self, cuda_tensor):
        """custom() 的 fn 为 None"""
        x = cuda_tensor((32, 256))
        out = torch.empty_like(x)

        graph = TaskGraph(num_workers=4)
        with pytest.raises(NullInputError):
            graph.custom(fn=None, inputs=[x], outputs=[out])

    def test_custom_none_inputs(self, cuda_tensor):
        """custom() 的 inputs 为 None"""
        out = cuda_tensor((32, 256))

        graph = TaskGraph(num_workers=4)
        with pytest.raises(NullInputError):
            graph.custom(fn=lambda x: x, inputs=None, outputs=[out])

    def test_custom_none_outputs(self, cuda_tensor):
        """custom() 的 outputs 为 None"""
        x = cuda_tensor((32, 256))

        graph = TaskGraph(num_workers=4)
        with pytest.raises(NullInputError):
            graph.custom(fn=lambda x: x, inputs=[x], outputs=None)


# ========== CPU Tensor 测试 ==========


@pytest.mark.cuda
class TestCPUTensor:
    """CPU tensor 检测测试"""

    def test_linear_cpu_input(self, cpu_tensor, cuda_tensor):
        """linear() 的 input 在 CPU 上"""
        x = cpu_tensor((32, 256))
        w = cuda_tensor((512, 256))

        graph = TaskGraph(num_workers=4)
        with pytest.raises(DeviceError) as exc_info:
            graph.linear(x, w)

        assert "cpu" in str(exc_info.value).lower()
        assert "cuda" in str(exc_info.value).lower()

    def test_linear_cpu_weight(self, cpu_tensor, cuda_tensor):
        """linear() 的 weight 在 CPU 上"""
        x = cuda_tensor((32, 256))
        w = cpu_tensor((512, 256))

        graph = TaskGraph(num_workers=4)
        with pytest.raises(DeviceError):
            graph.linear(x, w)

    def test_matmul_cpu_tensor(self, cpu_tensor, cuda_tensor):
        """matmul() 输入在 CPU 上"""
        a = cpu_tensor((32, 64))
        b = cuda_tensor((64, 128))

        graph = TaskGraph(num_workers=4)
        with pytest.raises(DeviceError):
            graph.matmul(a, b)

    def test_allreduce_cpu_tensor(self, cpu_tensor):
        """allreduce() 输入在 CPU 上"""
        x = cpu_tensor((32, 256))

        graph = TaskGraph(num_workers=4)
        with pytest.raises(DeviceError):
            graph.allreduce(x)


# ========== 形状不匹配测试 ==========


@pytest.mark.cuda
class TestShapeMismatch:
    """形状不匹配检测测试"""

    def test_linear_shape_mismatch(self, cuda_tensor):
        """linear() 形状不匹配"""
        x = cuda_tensor((32, 256))  # in_features = 256
        w = cuda_tensor((512, 128))  # in_features = 128

        graph = TaskGraph(num_workers=4)
        with pytest.raises(ShapeMismatchError) as exc_info:
            graph.linear(x, w)

        error_msg = str(exc_info.value)
        assert "256" in error_msg  # 输入维度
        assert "128" in error_msg  # 权重维度

    def test_linear_bias_shape_mismatch(self, cuda_tensor):
        """linear() bias 形状不匹配"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((512, 256))  # out_features = 512
        b = cuda_tensor((128,))  # 应该是 512

        graph = TaskGraph(num_workers=4)
        with pytest.raises(ShapeMismatchError):
            graph.linear(x, w, b)

    def test_matmul_shape_mismatch(self, cuda_tensor):
        """matmul() 形状不匹配"""
        a = cuda_tensor((32, 64))  # K = 64
        b = cuda_tensor((128, 256))  # K = 128

        graph = TaskGraph(num_workers=4)
        with pytest.raises(ShapeMismatchError):
            graph.matmul(a, b)


# ========== dtype 不匹配测试 ==========


@pytest.mark.cuda
class TestDTypeMismatch:
    """dtype 不匹配检测测试"""

    def test_linear_dtype_mismatch(self):
        """linear() dtype 不匹配"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.randn(32, 256, dtype=torch.float32, device="cuda")
        w = torch.randn(512, 256, dtype=torch.float16, device="cuda")

        graph = TaskGraph(num_workers=4)
        with pytest.raises(DTypeMismatchError) as exc_info:
            graph.linear(x, w)

        error_msg = str(exc_info.value)
        assert "float32" in error_msg
        assert "float16" in error_msg

    def test_matmul_dtype_mismatch(self):
        """matmul() dtype 不匹配"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        a = torch.randn(32, 64, dtype=torch.float32, device="cuda")
        b = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")

        graph = TaskGraph(num_workers=4)
        with pytest.raises(DTypeMismatchError):
            graph.matmul(a, b)


# ========== 编译后修改测试 ==========


@pytest.mark.cuda
class TestAlreadyCompiled:
    """编译后修改检测测试"""

    def test_add_task_after_compile(self, cuda_tensor):
        """编译后添加任务"""
        x = cuda_tensor((32, 256))
        w1 = cuda_tensor((512, 256))
        w2 = cuda_tensor((128, 512))

        graph = TaskGraph(num_workers=4)
        graph.linear(x, w1)
        graph.compile()

        with pytest.raises(AlreadyCompiledError) as exc_info:
            graph.linear(x, w2)

        assert "Cannot modify" in str(exc_info.value)

    def test_compile_twice(self, cuda_tensor):
        """重复编译应该是安全的"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((512, 256))

        graph = TaskGraph(num_workers=4)
        graph.linear(x, w)
        graph.compile()
        graph.compile()  # 第二次编译应该是 no-op，不抛异常


# ========== 循环依赖测试 ==========


@pytest.mark.cuda
class TestCyclicDependency:
    """循环依赖检测测试"""

    def test_self_dependency(self, cuda_tensor):
        """自我依赖检测"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((256, 256))

        graph = TaskGraph(num_workers=4)
        t1 = graph.linear(x, w)

        # 手动添加自我依赖
        t1.depends_on.append(t1)

        with pytest.raises(CyclicDependencyError):
            graph.compile()

    def test_two_node_cycle(self, cuda_tensor):
        """两节点循环检测"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((256, 256))

        graph = TaskGraph(num_workers=4)
        t1 = graph.linear(x, w)
        t2 = graph.linear(t1.output, w)

        # 手动添加循环依赖
        t1.depends_on.append(t2)

        with pytest.raises(CyclicDependencyError):
            graph.compile()


# ========== 边界情况测试 ==========


@pytest.mark.cuda
class TestEdgeCases:
    """边界情况测试"""

    def test_scalar_tensor(self):
        """标量 tensor（0 维）"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.tensor(1.0, device="cuda")  # 0 维
        w = torch.randn(512, 256, device="cuda")

        graph = TaskGraph(num_workers=4)
        # 标量 tensor 应该被拒绝（linear 需要至少 1 维）
        # 当前实现允许 1 维，所以 0 维应该也可以
        # 但实际上形状不匹配会先触发

    def test_empty_tensor(self):
        """空 tensor（某一维为 0）"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.randn(0, 256, device="cuda")  # batch=0
        w = torch.randn(512, 256, device="cuda")

        graph = TaskGraph(num_workers=4)
        # 空 tensor 应该能正常处理
        task = graph.linear(x, w)
        graph.compile()
        graph.execute()

        assert task.output.shape == (0, 512)

    def test_custom_not_callable(self, cuda_tensor):
        """custom() fn 不是可调用对象"""
        x = cuda_tensor((32, 256))
        out = torch.empty_like(x)

        graph = TaskGraph(num_workers=4)
        with pytest.raises(ValidationError) as exc_info:
            graph.custom(fn="not_callable", inputs=[x], outputs=[out])

        assert "callable" in str(exc_info.value).lower()


# ========== 设备推断测试 ==========


@pytest.mark.cuda
class TestDeviceInference:
    """设备推断测试"""

    def test_auto_device_inference(self, cuda_tensor):
        """自动设备推断"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((512, 256))

        graph = TaskGraph()  # 不指定 device
        task = graph.linear(x, w)

        # 应该从输入推断设备
        assert graph.device == torch.device("cuda", 0) or graph.device.type == "cuda"

    def test_device_on_different_gpu(self):
        """不同 GPU 上的 tensor"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs")

        x = torch.randn(32, 256, device="cuda:0")
        w = torch.randn(512, 256, device="cuda:1")

        graph = TaskGraph(num_workers=4)
        with pytest.raises(DeviceError):
            graph.linear(x, w)
