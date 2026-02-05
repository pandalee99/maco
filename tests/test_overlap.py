"""
MACO Overlap Scheduler Tests

测试计算-通信重叠调度：
- 单 GPU 模式测试
- 多 GPU 模式测试（需要 torchrun）

运行方式:
- 单 GPU: pytest tests/test_overlap.py -v
- 多 GPU: torchrun --nproc_per_node=4 tests/test_overlap.py
"""

import pytest
import torch
import torch.distributed as dist
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cuda_available():
    return torch.cuda.is_available()


def is_distributed():
    return os.environ.get("WORLD_SIZE") is not None


def setup_distributed():
    if dist.is_initialized():
        return
    
    if not is_distributed():
        return
    
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)


# ========== OverlapScheduler Tests ==========


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
class TestOverlapScheduler:
    """重叠调度器测试"""
    
    def test_create_plan_no_overlap(self):
        """创建无重叠计划"""
        from maco.task_graph.overlap_scheduler import OverlapScheduler, OverlapMode
        from maco.task_graph import TaskNode, TaskType
        
        scheduler = OverlapScheduler()
        
        # 模拟任务
        compute_tasks = [
            TaskNode(name=f"compute_{i}", task_type=TaskType.LINEAR)
            for i in range(4)
        ]
        comm_tasks = []
        
        plan = scheduler.create_plan(
            compute_tasks=compute_tasks,
            comm_tasks=comm_tasks,
            mode=OverlapMode.NONE,
        )
        
        assert plan.mode == OverlapMode.NONE
        assert plan.num_waves == 1
        assert len(plan.compute_waves) == 1
        assert len(plan.compute_waves[0]) == 4
    
    def test_create_plan_wave_overlap(self):
        """创建 wave 重叠计划"""
        from maco.task_graph.overlap_scheduler import OverlapScheduler, OverlapMode
        from maco.task_graph import TaskNode, TaskType
        
        scheduler = OverlapScheduler()
        
        compute_tasks = [
            TaskNode(name=f"compute_{i}", task_type=TaskType.LINEAR)
            for i in range(8)
        ]
        comm_tasks = [
            TaskNode(name=f"comm_{i}", task_type=TaskType.ALLREDUCE)
            for i in range(2)
        ]
        
        plan = scheduler.create_plan(
            compute_tasks=compute_tasks,
            comm_tasks=comm_tasks,
            mode=OverlapMode.WAVE,
            num_waves=2,
        )
        
        assert plan.mode == OverlapMode.WAVE
        assert plan.num_waves == 2
        assert len(plan.compute_waves) == 2
    
    def test_auto_determine_waves(self):
        """自动确定 wave 数量"""
        from maco.task_graph.overlap_scheduler import OverlapScheduler, OverlapMode
        from maco.task_graph import TaskNode, TaskType
        
        scheduler = OverlapScheduler()
        
        # 4 个计算任务，2 个通信任务 -> 应该是 2 个 waves
        compute_tasks = [
            TaskNode(name=f"compute_{i}", task_type=TaskType.LINEAR)
            for i in range(4)
        ]
        comm_tasks = [
            TaskNode(name=f"comm_{i}", task_type=TaskType.ALLREDUCE)
            for i in range(2)
        ]
        
        plan = scheduler.create_plan(
            compute_tasks=compute_tasks,
            comm_tasks=comm_tasks,
            mode=OverlapMode.WAVE,
        )
        
        assert plan.num_waves >= 1


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
class TestOverlapExecution:
    """重叠执行测试"""
    
    def test_execute_no_overlap(self):
        """执行无重叠计划"""
        from maco.task_graph.overlap_scheduler import OverlapScheduler, OverlapMode, OverlapPlan
        from maco.task_graph import TaskNode, TaskType
        
        scheduler = OverlapScheduler()
        
        # 追踪执行顺序
        execution_order = []
        
        def compute_fn(task):
            execution_order.append(("compute", task.name))
            # 模拟计算
            x = torch.randn(100, 100, device="cuda")
            _ = torch.matmul(x, x.T)
        
        def comm_fn(task):
            execution_order.append(("comm", task.name))
        
        compute_tasks = [
            TaskNode(name=f"c{i}", task_type=TaskType.LINEAR)
            for i in range(2)
        ]
        comm_tasks = [
            TaskNode(name="comm0", task_type=TaskType.ALLREDUCE)
        ]
        
        plan = scheduler.create_plan(
            compute_tasks=compute_tasks,
            comm_tasks=comm_tasks,
            mode=OverlapMode.NONE,
        )
        
        scheduler.execute_plan(plan, compute_fn, comm_fn)
        
        # 验证执行顺序
        assert len(execution_order) == 3
        assert execution_order[0] == ("compute", "c0")
        assert execution_order[1] == ("compute", "c1")
        assert execution_order[2] == ("comm", "comm0")
    
    def test_execute_wave_overlap(self):
        """执行 wave 重叠计划"""
        from maco.task_graph.overlap_scheduler import OverlapScheduler, OverlapMode
        from maco.task_graph import TaskNode, TaskType
        
        scheduler = OverlapScheduler()
        
        compute_count = [0]
        comm_count = [0]
        
        def compute_fn(task):
            compute_count[0] += 1
            # 模拟计算
            x = torch.randn(500, 500, device="cuda")
            for _ in range(5):
                x = torch.matmul(x, x.T)
        
        def comm_fn(task):
            comm_count[0] += 1
            # 模拟通信（复制）
            x = torch.randn(500, 500, device="cuda")
            _ = x.clone()
        
        compute_tasks = [
            TaskNode(name=f"c{i}", task_type=TaskType.LINEAR)
            for i in range(4)
        ]
        comm_tasks = [
            TaskNode(name=f"comm{i}", task_type=TaskType.ALLREDUCE)
            for i in range(2)
        ]
        
        plan = scheduler.create_plan(
            compute_tasks=compute_tasks,
            comm_tasks=comm_tasks,
            mode=OverlapMode.WAVE,
            num_waves=2,
        )
        
        scheduler.execute_plan(plan, compute_fn, comm_fn)
        
        # 验证所有任务都执行了
        assert compute_count[0] == 4
        assert comm_count[0] == 2


# ========== OverlapRuntime Tests ==========


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
class TestOverlapRuntime:
    """重叠运行时测试"""
    
    def test_runtime_basic(self):
        """基本运行时测试"""
        from maco.task_graph.overlap_scheduler import OverlapRuntime, OverlapMode
        from maco.task_graph import TaskNode, TaskType
        
        runtime = OverlapRuntime()
        
        results = []
        
        def compute_executor(task):
            x = torch.randn(100, 100, device="cuda")
            result = torch.matmul(x, x.T)
            results.append(result)
        
        def comm_executor(task):
            if results:
                _ = results[-1].clone()
        
        compute_tasks = [
            TaskNode(name=f"c{i}", task_type=TaskType.LINEAR)
            for i in range(3)
        ]
        comm_tasks = [
            TaskNode(name="comm", task_type=TaskType.ALLREDUCE)
        ]
        
        runtime.execute_overlap(
            compute_tasks=compute_tasks,
            comm_tasks=comm_tasks,
            compute_executor=compute_executor,
            comm_executor=comm_executor,
            mode=OverlapMode.WAVE,
        )
        
        assert len(results) == 3


# ========== Multi-GPU Integration Tests ==========


class TestMultiGPUOverlap:
    """多 GPU 重叠测试"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        setup_distributed()
        yield
    
    @pytest.mark.skipif(
        not (cuda_available() and is_distributed()),
        reason="Multi-GPU not available"
    )
    def test_overlap_with_real_comm(self):
        """真实通信的重叠测试"""
        from maco.task_graph.overlap_scheduler import OverlapRuntime, OverlapMode
        from maco.task_graph import TaskNode, TaskType
        from maco.comm import async_all_reduce
        
        runtime = OverlapRuntime()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        compute_results = []
        comm_tensors = []
        
        def compute_executor(task):
            # 模拟计算
            x = torch.randn(256, 256, device="cuda")
            for _ in range(10):
                x = torch.matmul(x, x.T)
            compute_results.append(x)
            # 准备通信数据
            comm_tensors.append(torch.full((64, 64), float(rank + 1), device="cuda"))
        
        def comm_executor(task):
            if comm_tensors:
                tensor = comm_tensors[-1]
                handle = async_all_reduce(tensor, async_op=True)
                handle.wait()
        
        compute_tasks = [
            TaskNode(name=f"c{i}", task_type=TaskType.LINEAR)
            for i in range(4)
        ]
        comm_tasks = [
            TaskNode(name=f"comm{i}", task_type=TaskType.ALLREDUCE)
            for i in range(2)
        ]
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        runtime.execute_overlap(
            compute_tasks=compute_tasks,
            comm_tasks=comm_tasks,
            compute_executor=compute_executor,
            comm_executor=comm_executor,
            mode=OverlapMode.WAVE,
            num_waves=2,
        )
        end.record()
        
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        
        if rank == 0:
            print(f"  Overlap execution time: {elapsed:.2f} ms")
        
        # 验证计算结果
        assert len(compute_results) == 4
        
        # 验证通信结果
        if comm_tensors:
            expected_value = world_size * (world_size + 1) / 2
            assert torch.allclose(
                comm_tensors[-1],
                torch.full((64, 64), expected_value, device="cuda")
            )


# ========== Main Entry ==========


if __name__ == "__main__":
    """
    直接运行测试
    用法: 
      单 GPU: python tests/test_overlap.py
      多 GPU: torchrun --nproc_per_node=4 tests/test_overlap.py
    """
    setup_distributed()
    
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if rank == 0:
        print(f"Running overlap tests (world_size={world_size})...")
    
    # 测试 OverlapScheduler
    from maco.task_graph.overlap_scheduler import OverlapScheduler, OverlapMode
    from maco.task_graph import TaskNode, TaskType
    
    scheduler = OverlapScheduler()
    
    compute_tasks = [
        TaskNode(name=f"c{i}", task_type=TaskType.LINEAR)
        for i in range(4)
    ]
    comm_tasks = [
        TaskNode(name=f"comm{i}", task_type=TaskType.ALLREDUCE)
        for i in range(2)
    ]
    
    plan = scheduler.create_plan(
        compute_tasks=compute_tasks,
        comm_tasks=comm_tasks,
        mode=OverlapMode.WAVE,
        num_waves=2,
    )
    
    if rank == 0:
        print(f"  [PASS] Create plan: {plan.num_waves} waves")
    
    # 执行计划
    count = [0]
    
    def compute_fn(task):
        count[0] += 1
        x = torch.randn(100, 100, device="cuda")
        _ = torch.matmul(x, x.T)
    
    def comm_fn(task):
        if world_size > 1:
            tensor = torch.ones(10, 10, device="cuda")
            dist.all_reduce(tensor)
    
    scheduler.execute_plan(plan, compute_fn, comm_fn)
    
    if rank == 0:
        print(f"  [PASS] Execute plan: {count[0]} compute tasks")
    
    # 测试带真实通信的重叠
    if world_size > 1:
        from maco.task_graph.overlap_scheduler import OverlapRuntime
        
        runtime = OverlapRuntime()
        results = []
        
        def compute_executor(task):
            x = torch.randn(256, 256, device="cuda")
            for _ in range(5):
                x = torch.matmul(x, x.T)
            results.append(x)
        
        def comm_executor(task):
            tensor = torch.full((64, 64), float(rank + 1), device="cuda")
            dist.all_reduce(tensor)
        
        runtime.execute_overlap(
            compute_tasks=compute_tasks,
            comm_tasks=comm_tasks,
            compute_executor=compute_executor,
            comm_executor=comm_executor,
            mode=OverlapMode.WAVE,
        )
        
        if rank == 0:
            print(f"  [PASS] Runtime with real comm: {len(results)} results")
    
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print("\n[SUCCESS] All overlap tests passed!")
