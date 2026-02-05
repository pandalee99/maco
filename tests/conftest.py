"""
MACO Test Configuration

提供 pytest fixtures 和测试工具。
"""

import pytest
import torch
import sys
import os

# 确保 maco 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """Pytest 配置钩子"""
    config.addinivalue_line("markers", "cuda: marks tests as requiring CUDA")
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(config, items):
    """根据 CUDA 可用性跳过测试"""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)


# ========== Fixtures ==========


@pytest.fixture
def cuda_available():
    """检查 CUDA 是否可用"""
    return torch.cuda.is_available()


@pytest.fixture
def device():
    """返回测试设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """返回 CUDA 设备（如果不可用则跳过测试）"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def random_tensor(device):
    """创建随机 tensor 的工厂函数"""
    def _create(shape, dtype=torch.float32):
        return torch.randn(shape, dtype=dtype, device=device)
    return _create


@pytest.fixture
def cuda_tensor():
    """创建 CUDA tensor 的工厂函数"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def _create(shape, dtype=torch.float32):
        return torch.randn(shape, dtype=dtype, device="cuda")
    return _create


@pytest.fixture
def cpu_tensor():
    """创建 CPU tensor 的工厂函数"""
    def _create(shape, dtype=torch.float32):
        return torch.randn(shape, dtype=dtype, device="cpu")
    return _create


@pytest.fixture
def task_graph():
    """创建一个新的 TaskGraph 实例"""
    from maco import TaskGraph

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    return TaskGraph(num_workers=4)


@pytest.fixture
def simple_linear_inputs(cuda_tensor):
    """创建简单的 linear 层输入"""
    x = cuda_tensor((32, 512))
    w = cuda_tensor((1024, 512))
    return x, w


@pytest.fixture
def chain_linear_inputs(cuda_tensor):
    """创建链式 linear 层输入"""
    x = cuda_tensor((64, 256))
    weights = [
        cuda_tensor((512, 256)),
        cuda_tensor((256, 512)),
        cuda_tensor((128, 256)),
    ]
    return x, weights


# ========== 辅助函数 ==========


def assert_tensors_close(a: torch.Tensor, b: torch.Tensor, rtol=1e-5, atol=1e-5):
    """断言两个 tensor 接近"""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert torch.allclose(a, b, rtol=rtol, atol=atol), \
        f"Tensors not close. Max diff: {(a - b).abs().max().item()}"


def assert_tensor_on_cuda(tensor: torch.Tensor):
    """断言 tensor 在 CUDA 上"""
    assert tensor.is_cuda, f"Expected CUDA tensor, got {tensor.device}"
