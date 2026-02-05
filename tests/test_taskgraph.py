"""
MACO TaskGraph 基础测试

测试 TaskGraph 的基本功能：
- 创建任务图
- 添加任务
- 编译和执行
- 依赖推断
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maco import TaskGraph
from maco.task_graph import TaskNode, TaskType


# ========== 基础功能测试 ==========


@pytest.mark.cuda
class TestTaskGraphBasic:
    """TaskGraph 基本功能测试"""

    def test_create_empty_graph(self):
        """空图创建"""
        graph = TaskGraph(num_workers=4)
        assert len(graph._nodes) == 0
        assert not graph._compiled

    def test_compile_empty_graph(self):
        """空图编译应该成功"""
        graph = TaskGraph(num_workers=4)
        graph.compile()
        assert graph._compiled

    def test_execute_empty_graph(self):
        """空图执行应该成功"""
        graph = TaskGraph(num_workers=4)
        graph.compile()
        graph.execute()  # 不应抛出异常

    def test_single_linear_task(self, cuda_tensor):
        """单个 linear 任务"""
        x = cuda_tensor((32, 512))
        w = cuda_tensor((1024, 512))

        graph = TaskGraph(num_workers=4)
        task = graph.linear(x, w, name="linear1")

        assert task.name == "linear1"
        assert task.task_type == TaskType.LINEAR
        assert len(graph._nodes) == 1

        graph.compile()
        graph.execute()

        # 验证输出形状
        assert task.output.shape == (32, 1024)

    def test_single_matmul_task(self, cuda_tensor):
        """单个 matmul 任务"""
        a = cuda_tensor((32, 64))
        b = cuda_tensor((64, 128))

        graph = TaskGraph(num_workers=4)
        task = graph.matmul(a, b, name="matmul1")

        graph.compile()
        graph.execute()

        assert task.output.shape == (32, 128)


@pytest.mark.cuda
class TestLinearChain:
    """线性链测试"""

    def test_two_task_chain(self, cuda_tensor):
        """两个任务的链"""
        x = cuda_tensor((32, 256))
        w1 = cuda_tensor((512, 256))
        w2 = cuda_tensor((128, 512))

        graph = TaskGraph(num_workers=4)
        t1 = graph.linear(x, w1, name="linear1")
        t2 = graph.linear(t1.output, w2, name="linear2")

        graph.compile()
        graph.execute()

        assert t2.output.shape == (32, 128)

    def test_three_task_chain(self, cuda_tensor):
        """三个任务的链"""
        x = cuda_tensor((64, 256))
        weights = [
            cuda_tensor((512, 256)),
            cuda_tensor((256, 512)),
            cuda_tensor((128, 256)),
        ]

        graph = TaskGraph(num_workers=4)
        t1 = graph.linear(x, weights[0])
        t2 = graph.linear(t1.output, weights[1])
        t3 = graph.linear(t2.output, weights[2])

        graph.compile()
        graph.execute()

        assert t3.output.shape == (64, 128)

    def test_chain_correctness(self, cuda_tensor):
        """链式执行正确性验证"""
        x = cuda_tensor((32, 256))
        w1 = cuda_tensor((512, 256))
        w2 = cuda_tensor((128, 512))

        # TaskGraph 执行
        graph = TaskGraph(num_workers=4)
        t1 = graph.linear(x, w1)
        t2 = graph.linear(t1.output, w2)
        graph.compile()
        graph.execute()

        # PyTorch 基线
        expected1 = torch.nn.functional.linear(x, w1)
        expected2 = torch.nn.functional.linear(expected1, w2)

        # 验证结果
        diff = (expected2 - t2.output).abs().max().item()
        assert diff < 1e-4, f"Max difference: {diff}"


@pytest.mark.cuda
class TestDependencyInference:
    """依赖推断测试"""

    def test_auto_dependency(self, cuda_tensor):
        """自动依赖推断"""
        x = cuda_tensor((32, 256))
        w1 = cuda_tensor((512, 256))
        w2 = cuda_tensor((128, 512))

        graph = TaskGraph(num_workers=4)
        t1 = graph.linear(x, w1)
        t2 = graph.linear(t1.output, w2)  # t2 依赖 t1

        graph.compile()

        # t2 应该依赖 t1
        assert t1 in t2.depends_on

    def test_no_dependency_for_independent_tasks(self, cuda_tensor):
        """独立任务无依赖"""
        x1 = cuda_tensor((32, 256))
        x2 = cuda_tensor((32, 256))
        w1 = cuda_tensor((512, 256))
        w2 = cuda_tensor((128, 256))

        graph = TaskGraph(num_workers=4)
        t1 = graph.linear(x1, w1)
        t2 = graph.linear(x2, w2)  # t2 不依赖 t1

        graph.compile()

        assert t1 not in t2.depends_on
        assert t2 not in t1.depends_on


@pytest.mark.cuda
class TestWaveGrouping:
    """Wave Grouping 测试"""

    def test_schedule_waves(self, cuda_tensor):
        """执行波计算"""
        x = cuda_tensor((32, 256))
        weights = [cuda_tensor((256, 256)) for _ in range(4)]

        graph = TaskGraph(num_workers=4)
        current = x
        for i, w in enumerate(weights):
            t = graph.linear(current, w)
            current = t.output

        graph.compile()

        # 链式依赖应该产生多个波
        assert len(graph._schedule.waves) >= 1


@pytest.mark.cuda
class TestCustomTask:
    """自定义任务测试"""

    def test_custom_gelu(self, cuda_tensor):
        """自定义 GELU 激活"""
        x = cuda_tensor((32, 256))
        output = torch.empty_like(x)

        def gelu_fn(inp):
            return torch.nn.functional.gelu(inp)

        graph = TaskGraph(num_workers=4)
        task = graph.custom(
            fn=gelu_fn,
            inputs=[x],
            outputs=[output],
            name="gelu",
        )

        graph.compile()
        graph.execute()

        # 验证正确性
        expected = torch.nn.functional.gelu(x)
        diff = (expected - output).abs().max().item()
        assert diff < 1e-5


@pytest.mark.cuda
class TestAllReduce:
    """AllReduce 测试（单 GPU 模式）"""

    def test_allreduce_single_gpu(self, cuda_tensor):
        """单 GPU AllReduce（应该是 no-op）"""
        x = cuda_tensor((32, 256))

        graph = TaskGraph(num_workers=4)
        task = graph.allreduce(x, name="ar")

        graph.compile()
        graph.execute()

        # 单 GPU 模式下，输出应该与输入相同
        assert task.output is x


@pytest.mark.cuda
class TestOverlapGroup:
    """重叠组测试"""

    def test_overlap_marking(self, cuda_tensor):
        """重叠标记"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((256, 256))
        comm_tensor = cuda_tensor((32, 256))

        graph = TaskGraph(num_workers=4)

        # 计算任务
        t1 = graph.linear(x, w)

        # 通信任务
        t2 = graph.allreduce(comm_tensor)

        # 标记重叠
        group = graph.overlap([t1], [t2])

        assert t1._overlap_role == "compute"
        assert t2._overlap_role == "comm"

    def test_auto_waves(self, cuda_tensor):
        """自动 wave 分配"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((256, 256))
        comm_tensor = cuda_tensor((32, 256))

        graph = TaskGraph(num_workers=4)

        # 多个计算任务
        tasks = []
        current = x
        for i in range(4):
            t = graph.linear(current, w)
            tasks.append(t)
            current = t.output

        # 通信任务
        comm = graph.allreduce(comm_tensor)

        # 标记重叠并自动分配 waves
        group = graph.overlap(tasks, [comm])
        group.auto_waves()

        # 应该有至少 1 个 wave
        assert group.num_waves >= 1


@pytest.mark.cuda
class TestGraphReuse:
    """图重用测试"""

    def test_execute_multiple_times(self, cuda_tensor):
        """多次执行同一图"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((512, 256))

        graph = TaskGraph(num_workers=4)
        task = graph.linear(x, w)
        graph.compile()

        # 多次执行
        for _ in range(3):
            graph.execute()

        # 不应抛出异常


@pytest.mark.cuda
class TestSummary:
    """摘要功能测试"""

    def test_summary_uncompiled(self, cuda_tensor):
        """未编译图的摘要"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((512, 256))

        graph = TaskGraph(num_workers=4)
        graph.linear(x, w)

        summary = graph.summary()
        assert "Total nodes: 1" in summary
        assert "Compute nodes: 1" in summary

    def test_summary_compiled(self, cuda_tensor):
        """已编译图的摘要"""
        x = cuda_tensor((32, 256))
        w = cuda_tensor((512, 256))

        graph = TaskGraph(num_workers=4)
        graph.linear(x, w)
        graph.compile()

        summary = graph.summary()
        assert "Execution waves" in summary
