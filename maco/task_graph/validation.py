"""
MACO TaskGraph 输入验证工具

提供严格模式的输入验证，确保早期发现错误。
"""

import torch
from typing import Optional, List, Tuple, Union

from .exceptions import (
    NullInputError,
    DeviceError,
    ShapeMismatchError,
    DimensionError,
    DTypeMismatchError,
)


def validate_tensor(
    tensor: torch.Tensor,
    name: str,
    method_name: str,
    *,
    ndim: Optional[int] = None,
    min_ndim: Optional[int] = None,
    max_ndim: Optional[int] = None,
    require_cuda: bool = True,
) -> torch.Tensor:
    """
    验证单个 tensor

    Args:
        tensor: 要验证的 tensor
        name: 参数名称（用于错误消息）
        method_name: 方法名称（用于错误消息）
        ndim: 要求的精确维度数
        min_ndim: 最小维度数
        max_ndim: 最大维度数
        require_cuda: 是否要求在 CUDA 设备上

    Returns:
        验证后的 tensor

    Raises:
        NullInputError: tensor 为 None
        DeviceError: tensor 不在要求的设备上
        DimensionError: 维度数不符合要求
    """
    # 检查 None
    if tensor is None:
        raise NullInputError(name, method_name)

    # 检查是否为 tensor
    if not isinstance(tensor, torch.Tensor):
        raise NullInputError(name, method_name)

    # 检查设备
    if require_cuda and not tensor.is_cuda:
        raise DeviceError(name, str(tensor.device), "cuda")

    # 检查维度数
    actual_ndim = tensor.ndim
    if ndim is not None and actual_ndim != ndim:
        raise DimensionError(name, ndim, actual_ndim)

    if min_ndim is not None and actual_ndim < min_ndim:
        raise DimensionError(name, f">={min_ndim}", actual_ndim)

    if max_ndim is not None and actual_ndim > max_ndim:
        raise DimensionError(name, f"<={max_ndim}", actual_ndim)

    return tensor


def validate_linear_shapes(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> None:
    """
    验证 linear 操作的形状兼容性

    Args:
        input: 输入 tensor [..., in_features]
        weight: 权重 tensor [out_features, in_features]
        bias: 偏置 tensor [out_features]（可选）

    Raises:
        ShapeMismatchError: 形状不兼容
    """
    in_features = input.shape[-1]
    weight_in = weight.shape[-1]

    if in_features != weight_in:
        raise ShapeMismatchError(
            "linear",
            f"input last dim ({in_features}) != weight last dim ({weight_in})",
            f"Input shape: {list(input.shape)}, Weight shape: {list(weight.shape)}. "
            f"For linear(input, weight), input[..., K] @ weight.T[K, N] requires matching K.",
        )

    if bias is not None:
        out_features = weight.shape[0]
        if bias.shape[0] != out_features:
            raise ShapeMismatchError(
                "linear",
                f"bias size ({bias.shape[0]}) != weight out_features ({out_features})",
                f"Bias should have shape [{out_features}]",
            )


def validate_matmul_shapes(a: torch.Tensor, b: torch.Tensor) -> None:
    """
    验证 matmul 操作的形状兼容性

    Args:
        a: 第一个 tensor [..., M, K]
        b: 第二个 tensor [..., K, N]

    Raises:
        ShapeMismatchError: 形状不兼容
    """
    if a.ndim < 1 or b.ndim < 1:
        raise ShapeMismatchError(
            "matmul",
            f"tensors must have at least 1 dimension",
            f"Got a.ndim={a.ndim}, b.ndim={b.ndim}",
        )

    # 对于 2D+ 矩阵乘法，检查内积维度
    if a.ndim >= 2 and b.ndim >= 2:
        k_a = a.shape[-1]
        k_b = b.shape[-2]
        if k_a != k_b:
            raise ShapeMismatchError(
                "matmul",
                f"a last dim ({k_a}) != b second-to-last dim ({k_b})",
                f"For matmul(a, b), a[..., M, K] @ b[..., K, N] requires matching K. "
                f"Got a.shape={list(a.shape)}, b.shape={list(b.shape)}",
            )


def validate_dtype_match(
    tensor1: torch.Tensor,
    name1: str,
    tensor2: torch.Tensor,
    name2: str,
) -> None:
    """
    验证两个 tensor 的 dtype 是否匹配

    Args:
        tensor1: 第一个 tensor
        name1: 第一个 tensor 的名称
        tensor2: 第二个 tensor
        name2: 第二个 tensor 的名称

    Raises:
        DTypeMismatchError: dtype 不匹配
    """
    if tensor1.dtype != tensor2.dtype:
        raise DTypeMismatchError(name1, str(tensor1.dtype), name2, str(tensor2.dtype))


def validate_same_device(
    tensor1: torch.Tensor,
    name1: str,
    tensor2: torch.Tensor,
    name2: str,
) -> None:
    """
    验证两个 tensor 在同一设备上

    Args:
        tensor1: 第一个 tensor
        name1: 第一个 tensor 的名称
        tensor2: 第二个 tensor
        name2: 第二个 tensor 的名称

    Raises:
        DeviceError: 设备不一致
    """
    if tensor1.device != tensor2.device:
        raise DeviceError(
            f"{name1}/{name2}",
            f"{name1}={tensor1.device}, {name2}={tensor2.device}",
            "same device",
        )


def infer_device(*tensors: torch.Tensor) -> torch.device:
    """
    从多个 tensor 推断设备

    Args:
        tensors: tensor 列表

    Returns:
        推断出的设备

    Raises:
        DeviceError: 设备不一致
    """
    device = None
    for t in tensors:
        if t is not None and isinstance(t, torch.Tensor):
            if device is None:
                device = t.device
            elif t.device != device:
                raise DeviceError(
                    "tensors",
                    f"mixed devices: {device} and {t.device}",
                    "same device",
                )

    if device is None:
        # 默认 CUDA 如果可用
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return device


def infer_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """
    从多个 tensor 推断 dtype

    Args:
        tensors: tensor 列表

    Returns:
        推断出的 dtype（使用第一个非 None tensor 的 dtype）
    """
    for t in tensors:
        if t is not None and isinstance(t, torch.Tensor):
            return t.dtype

    # 默认 float32
    return torch.float32
