"""
MACO Runtime 测试

测试 StreamRuntime 和执行正确性：
- 任务执行顺序
- 结果正确性
- 多种数据类型支持
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maco import TaskGraph
from maco.task_graph.runtime import StreamRuntime


# ========== StreamRuntime 测试 ==========


@pytest.mark.cuda
class TestStreamRuntime:
    """StreamRuntime 测试"""

    def test_linear_correctness_float32(self, cuda_tensor):
        """float32 linear 正确性"""
        x = cuda_tensor((64, 256))
        w = cuda_tensor((512, 256))

        graph = TaskGraph(num_workers=4)
        task = graph.linear(x, w)
        graph.compile()
        graph.execute()

        expected = torch.nn.functional.linear(x, w)
        diff = (expected - task.output).abs().max().item()
        assert diff < 1e-5, f"Max diff: {diff}"

    def test_linear_correctness_float16(self):
        """float16 linear 正确性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.randn(64, 256, dtype=torch.float16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.float16, device="cuda")

        graph = TaskGraph(num_workers=4)
        task = graph.linear(x, w)
        graph.compile()
        graph.execute()

        expected = torch.nn.functional.linear(x, w)
        diff = (expected - task.output).abs().max().item()
        assert diff < 1e-2, f"Max diff: {diff}"  # float16 精度较低

    def test_linear_correctness_bfloat16(self):
        """bfloat16 linear 正确性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")

        graph = TaskGraph(num_workers=4)
        task = graph.linear(x, w)
        graph.compile()
        graph.execute()

        expected = torch.nn.functional.linear(x, w)
        diff = (expected - task.output).abs().max().item()
        assert diff < 1e-1, f"Max diff: {diff}"  # bfloat16 精度较低

    def test_matmul_correctness(self, cuda_tensor):
        """matmul 正确性"""
        a = cuda_tensor((32, 64))
        b = cuda_tensor((64, 128))

        graph = TaskGraph(num_workers=4)
        task = graph.matmul(a, b)
        graph.compile()
        graph.execute()

        expected = torch.matmul(a, b)
        diff = (expected - task.output).abs().max().item()
        assert diff < 1e-5, f"Max diff: {diff}"

    def test_custom_task_execution(self, cuda_tensor):
        """自定义任务执行"""
        x = cuda_tensor((32, 256))
        output = torch.empty_like(x)

        def relu_fn(inp):
            return torch.nn.functional.relu(inp)

        graph = TaskGraph(num_workers=4)
        task = graph.custom(
            fn=relu_fn,
            inputs=[x],
            outputs=[output],
        )
        graph.compile()
        graph.execute()

        expected = torch.nn.functional.relu(x)
        diff = (expected - output).abs().max().item()
        assert diff < 1e-6


# ========== 执行顺序测试 ==========


@pytest.mark.cuda
class TestExecutionOrder:
    """执行顺序测试"""

    def test_dependency_order(self, cuda_tensor):
        """依赖顺序正确执行"""
        x = cuda_tensor((32, 256))
        w1 = cuda_tensor((512, 256))
        w2 = cuda_tensor((128, 512))
        w3 = cuda_tensor((64, 128))

        graph = TaskGraph(num_workers=4)
        t1 = graph.linear(x, w1)
        t2 = graph.linear(t1.output, w2)
        t3 = graph.linear(t2.output, w3)
        graph.compile()
        graph.execute()

        # PyTorch 基线
        h1 = torch.nn.functional.linear(x, w1)
        h2 = torch.nn.functional.linear(h1, w2)
        h3 = torch.nn.functional.linear(h2, w3)

        diff = (h3 - t3.output).abs().max().item()
        assert diff < 1e-4, f"Max diff: {diff}"

    def test_parallel_execution(self, cuda_tensor):
        """并行任务执行"""
        x1 = cuda_tensor((32, 256))
        x2 = cuda_tensor((32, 256))
        w1 = cuda_tensor((512, 256))
        w2 = cuda_tensor((128, 256))

        graph = TaskGraph(num_workers=4)
        t1 = graph.linear(x1, w1)  # 独立任务
        t2 = graph.linear(x2, w2)  # 独立任务
        graph.compile()
        graph.execute()

        # 验证两个结果
        expected1 = torch.nn.functional.linear(x1, w1)
        expected2 = torch.nn.functional.linear(x2, w2)

        diff1 = (expected1 - t1.output).abs().max().item()
        diff2 = (expected2 - t2.output).abs().max().item()

        assert diff1 < 1e-5
        assert diff2 < 1e-5


# ========== 大规模测试 ==========


@pytest.mark.cuda
class TestLargeScale:
    """大规模测试"""

    def test_many_tasks(self, cuda_tensor):
        """大量任务（50 个）"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((256, 256))  # 输入输出维度相同

        graph = TaskGraph(num_workers=8)
        current = x
        tasks = []
        for i in range(50):
            t = graph.linear(current, w)
            tasks.append(t)
            current = t.output

        graph.compile()
        graph.execute()

        # 简单验证：形状正确
        assert tasks[-1].output.shape == (32, 256)

    def test_large_tensors(self):
        """大 tensor"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # 较大的 tensor
        x = torch.randn(1024, 2048, device="cuda")
        w = torch.randn(2048, 2048, device="cuda")

        graph = TaskGraph(num_workers=8)
        task = graph.linear(x, w)
        graph.compile()
        graph.execute()

        expected = torch.nn.functional.linear(x, w)
        diff = (expected - task.output).abs().max().item()
        assert diff < 1e-4


# ========== 内存管理测试 ==========


@pytest.mark.cuda
class TestMemoryManagement:
    """内存管理测试"""

    def test_output_tensor_allocation(self, cuda_tensor):
        """输出 tensor 自动分配"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((512, 256))

        graph = TaskGraph(num_workers=4)
        task = graph.linear(x, w)

        # 检查输出 tensor 已分配
        assert task.output is not None
        assert task.output.shape == (32, 512)
        assert task.output.device.type == "cuda"

    def test_preallocated_output(self):
        """预分配输出 tensor"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x = torch.randn(32, 256, device="cuda")
        w = torch.randn(512, 256, device="cuda")
        out = torch.zeros(32, 512, device="cuda")

        graph = TaskGraph(num_workers=4)
        task = graph.linear(x, w, output=out)
        graph.compile()
        graph.execute()

        # 输出应该写入预分配的 tensor
        assert task.output is out
        assert out.sum().item() != 0  # 应该有数据


# ========== 带 bias 的 linear 测试 ==========


@pytest.mark.cuda
class TestLinearWithBias:
    """带 bias 的 linear 测试"""

    def test_linear_with_bias(self, cuda_tensor):
        """linear() 带 bias"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((512, 256))
        b = cuda_tensor((512,))

        graph = TaskGraph(num_workers=4)
        task = graph.linear(x, w, b)
        graph.compile()
        graph.execute()

        expected = torch.nn.functional.linear(x, w, b)
        diff = (expected - task.output).abs().max().item()
        assert diff < 1e-5
