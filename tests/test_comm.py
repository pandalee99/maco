"""
MACO Communication Module Tests

测试多 GPU 通信操作的正确性。
运行方式: torchrun --nproc_per_node=4 -m pytest tests/test_comm.py -v
"""

import pytest
import torch
import torch.distributed as dist
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_distributed_available():
    """检查分布式环境是否可用"""
    return (
        torch.cuda.is_available() 
        and torch.cuda.device_count() >= 1
        and os.environ.get("WORLD_SIZE") is not None
    )


def setup_distributed():
    """设置分布式环境"""
    if dist.is_initialized():
        return
    
    if not is_distributed_available():
        return
    
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )


def teardown_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


# ========== Process Group Tests ==========


class TestProcessGroup:
    """进程组管理测试"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前设置"""
        setup_distributed()
        yield
    
    def test_get_world_size(self):
        """测试获取 world_size"""
        from maco.comm import get_world_size
        
        world_size = get_world_size()
        expected = int(os.environ.get("WORLD_SIZE", "1"))
        assert world_size == expected
    
    def test_get_rank(self):
        """测试获取 rank"""
        from maco.comm import get_rank
        
        rank = get_rank()
        expected = int(os.environ.get("RANK", "0"))
        assert rank == expected
    
    def test_is_initialized(self):
        """测试初始化状态"""
        from maco.comm import is_initialized
        
        if is_distributed_available():
            assert is_initialized()


# ========== AllReduce Tests ==========


class TestAsyncAllReduce:
    """异步 AllReduce 测试"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        setup_distributed()
        yield
    
    @pytest.mark.skipif(
        not is_distributed_available(),
        reason="Distributed environment not available"
    )
    def test_all_reduce_sum(self):
        """测试 AllReduce SUM"""
        from maco.comm import async_all_reduce, get_rank, get_world_size
        
        rank = get_rank()
        world_size = get_world_size()
        
        # 每个进程创建值为 rank+1 的 tensor
        tensor = torch.full((4, 4), rank + 1.0, device="cuda")
        
        handle = async_all_reduce(tensor, async_op=True)
        result = handle.wait()
        
        # SUM 结果应该是 1+2+...+world_size = world_size*(world_size+1)/2
        expected_value = world_size * (world_size + 1) / 2
        expected = torch.full((4, 4), expected_value, device="cuda")
        
        assert torch.allclose(result, expected), f"Expected {expected_value}, got {result[0,0].item()}"
    
    @pytest.mark.skipif(
        not is_distributed_available(),
        reason="Distributed environment not available"
    )
    def test_all_reduce_sync(self):
        """测试同步 AllReduce"""
        from maco.comm import async_all_reduce, get_world_size
        
        world_size = get_world_size()
        tensor = torch.ones(4, 4, device="cuda")
        
        handle = async_all_reduce(tensor, async_op=False)
        result = handle.output
        
        expected = torch.full((4, 4), float(world_size), device="cuda")
        assert torch.allclose(result, expected)


# ========== AllGather Tests ==========


class TestAsyncAllGather:
    """异步 AllGather 测试"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        setup_distributed()
        yield
    
    @pytest.mark.skipif(
        not is_distributed_available(),
        reason="Distributed environment not available"
    )
    def test_all_gather_basic(self):
        """测试基本 AllGather"""
        from maco.comm import async_all_gather, get_rank, get_world_size
        
        rank = get_rank()
        world_size = get_world_size()
        
        # 每个进程创建不同的 tensor
        tensor = torch.full((2, 4), float(rank), device="cuda")
        
        handle = async_all_gather(tensor, async_op=True)
        result = handle.wait()
        
        # 结果应该是 [rank0_data, rank1_data, ..., rankN_data] 拼接
        assert result.shape == (2 * world_size, 4)
        
        # 验证每个 rank 的数据
        for r in range(world_size):
            expected_chunk = torch.full((2, 4), float(r), device="cuda")
            actual_chunk = result[r*2:(r+1)*2]
            assert torch.allclose(actual_chunk, expected_chunk), \
                f"Rank {r} data mismatch"


# ========== AllToAll Tests ==========


class TestAsyncAllToAll:
    """异步 AllToAll 测试"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        setup_distributed()
        yield
    
    @pytest.mark.skipif(
        not is_distributed_available(),
        reason="Distributed environment not available"
    )
    def test_all_to_all_basic(self):
        """测试基本 AllToAll"""
        from maco.comm import async_all_to_all, get_rank, get_world_size
        
        rank = get_rank()
        world_size = get_world_size()
        
        # 创建 tensor: 每个进程有 world_size 个 chunk
        # 第 i 个 chunk 的值为 rank * 10 + i
        chunks = []
        for i in range(world_size):
            chunk = torch.full((2, 4), float(rank * 10 + i), device="cuda")
            chunks.append(chunk)
        tensor = torch.cat(chunks, dim=0)  # shape: [2*world_size, 4]
        
        handle = async_all_to_all(tensor, scatter_dim=0, gather_dim=0, async_op=True)
        result = handle.wait()
        
        # AllToAll 后，rank i 应该收到所有进程的第 i 个 chunk
        # 即 rank i 收到: [0*10+i, 1*10+i, 2*10+i, ..., (N-1)*10+i]
        assert result.shape == tensor.shape
        
        for src_rank in range(world_size):
            expected_value = float(src_rank * 10 + rank)
            actual_chunk = result[src_rank*2:(src_rank+1)*2]
            expected_chunk = torch.full((2, 4), expected_value, device="cuda")
            assert torch.allclose(actual_chunk, expected_chunk), \
                f"From rank {src_rank}, expected {expected_value}, got {actual_chunk[0,0].item()}"
    
    @pytest.mark.skipif(
        not is_distributed_available(),
        reason="Distributed environment not available"
    )
    def test_all_to_all_4d(self):
        """测试 4D AllToAll（attention QKV 场景）"""
        from maco.comm import async_all_to_all_4d, get_rank, get_world_size
        
        rank = get_rank()
        world_size = get_world_size()
        
        # 模拟 attention tensor: [B, S, H, D]
        B, S, H, D = 2, 8, world_size * 4, 64  # H 必须能被 world_size 整除
        
        tensor = torch.randn(B, S, H, D, device="cuda")
        # 标记每个 head 属于哪个进程（用于验证）
        for h in range(H):
            tensor[:, :, h, :] = rank * 100 + h
        
        # scatter head (dim=2), gather seq (dim=1)
        handle = async_all_to_all_4d(
            tensor, scatter_dim=2, gather_dim=1, async_op=True
        )
        result = handle.wait()
        
        # 验证形状：H 减少为 H/world_size，S 增加为 S*world_size
        expected_shape = (B, S * world_size, H // world_size, D)
        assert result.shape == expected_shape, \
            f"Expected {expected_shape}, got {result.shape}"


# ========== Broadcast Tests ==========


class TestAsyncBroadcast:
    """异步 Broadcast 测试"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        setup_distributed()
        yield
    
    @pytest.mark.skipif(
        not is_distributed_available(),
        reason="Distributed environment not available"
    )
    def test_broadcast_from_rank0(self):
        """测试从 rank 0 广播"""
        from maco.comm import async_broadcast, get_rank
        
        rank = get_rank()
        
        if rank == 0:
            tensor = torch.full((4, 4), 42.0, device="cuda")
        else:
            tensor = torch.zeros(4, 4, device="cuda")
        
        handle = async_broadcast(tensor, src=0, async_op=True)
        result = handle.wait()
        
        expected = torch.full((4, 4), 42.0, device="cuda")
        assert torch.allclose(result, expected)


# ========== Single GPU Fallback Tests ==========


class TestSingleGPUFallback:
    """单 GPU 回退测试"""
    
    def test_all_reduce_single_gpu(self):
        """单 GPU 下 AllReduce 应该直接返回"""
        # 临时禁用分布式
        if dist.is_initialized():
            pytest.skip("Distributed is initialized, skip single GPU test")
        
        from maco.comm.nccl_ops import async_all_reduce
        
        if torch.cuda.is_available():
            tensor = torch.ones(4, 4, device="cuda")
            handle = async_all_reduce(tensor, async_op=True)
            result = handle.wait()
            assert torch.allclose(result, tensor)


# ========== Main Entry ==========


if __name__ == "__main__":
    """
    直接运行测试（非 pytest 模式）
    用法: torchrun --nproc_per_node=4 tests/test_comm.py
    """
    setup_distributed()
    
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    print(f"[Rank {rank}/{world_size}] Running communication tests...")
    
    # 测试 AllReduce
    from maco.comm import async_all_reduce
    tensor = torch.full((4, 4), rank + 1.0, device="cuda")
    handle = async_all_reduce(tensor, async_op=True)
    result = handle.wait()
    expected_value = world_size * (world_size + 1) / 2
    assert torch.allclose(result, torch.full((4, 4), expected_value, device="cuda"))
    if rank == 0:
        print(f"  [PASS] AllReduce SUM: {result[0,0].item()}")
    
    # 测试 AllGather
    from maco.comm import async_all_gather
    tensor = torch.full((2, 4), float(rank), device="cuda")
    handle = async_all_gather(tensor, async_op=True)
    result = handle.wait()
    assert result.shape == (2 * world_size, 4)
    if rank == 0:
        print(f"  [PASS] AllGather: shape {result.shape}")
    
    # 测试 AllToAll
    from maco.comm import async_all_to_all
    chunks = [torch.full((2, 4), float(rank * 10 + i), device="cuda") for i in range(world_size)]
    tensor = torch.cat(chunks, dim=0)
    handle = async_all_to_all(tensor, scatter_dim=0, gather_dim=0, async_op=True)
    result = handle.wait()
    if rank == 0:
        print(f"  [PASS] AllToAll: shape {result.shape}")
    
    # 测试 AllToAll 4D
    from maco.comm import async_all_to_all_4d
    B, S, H, D = 2, 8, world_size * 4, 64
    tensor = torch.randn(B, S, H, D, device="cuda")
    handle = async_all_to_all_4d(tensor, scatter_dim=2, gather_dim=1, async_op=True)
    result = handle.wait()
    expected_shape = (B, S * world_size, H // world_size, D)
    assert result.shape == expected_shape
    if rank == 0:
        print(f"  [PASS] AllToAll 4D: {tensor.shape} -> {result.shape}")
    
    # 测试 Broadcast
    from maco.comm import async_broadcast
    tensor = torch.full((4, 4), 42.0 if rank == 0 else 0.0, device="cuda")
    handle = async_broadcast(tensor, src=0, async_op=True)
    result = handle.wait()
    assert torch.allclose(result, torch.full((4, 4), 42.0, device="cuda"))
    if rank == 0:
        print(f"  [PASS] Broadcast: {result[0,0].item()}")
    
    dist.barrier()
    if rank == 0:
        print("\n[SUCCESS] All communication tests passed!")
    
    teardown_distributed()
