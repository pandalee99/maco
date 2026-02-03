"""
Tensor Registry for MACO

Manages tensor registration and lifecycle for persistent kernels.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import torch


@dataclass
class TensorInfo:
    """Information about a registered tensor."""
    name: str
    tensor: torch.Tensor
    shape: Tuple[int, ...]
    dtype: torch.dtype
    is_weight: bool
    is_input: bool
    is_output: bool
    device_ptr: Optional[int] = None  # GPU memory pointer


class TensorRegistry:
    """
    Registry for managing tensors in persistent kernel execution.

    This registry tracks all tensors that need to be accessed by
    the persistent kernel, including weights, inputs, outputs, and
    intermediate activations.
    """

    def __init__(self):
        self._tensors: Dict[str, TensorInfo] = {}
        self._inputs: Dict[str, TensorInfo] = {}
        self._outputs: Dict[str, TensorInfo] = {}
        self._weights: Dict[str, TensorInfo] = {}

    def register(
        self,
        tensor: torch.Tensor,
        name: str,
        is_weight: bool = False,
        is_input: bool = False,
        is_output: bool = False,
    ) -> TensorInfo:
        """
        Register a tensor.

        Args:
            tensor: The tensor to register
            name: Unique name for this tensor
            is_weight: Whether this is a model weight
            is_input: Whether this is an input tensor
            is_output: Whether this is an output tensor

        Returns:
            TensorInfo object
        """
        if name in self._tensors:
            raise ValueError(f"Tensor '{name}' is already registered")

        if not tensor.is_cuda:
            tensor = tensor.cuda()

        info = TensorInfo(
            name=name,
            tensor=tensor,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            is_weight=is_weight,
            is_input=is_input,
            is_output=is_output,
            device_ptr=tensor.data_ptr(),
        )

        self._tensors[name] = info

        if is_weight:
            self._weights[name] = info
        if is_input:
            self._inputs[name] = info
        if is_output:
            self._outputs[name] = info

        return info

    def register_input(self, tensor: torch.Tensor, name: str) -> TensorInfo:
        """Register an input tensor."""
        return self.register(tensor, name, is_input=True)

    def register_output(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        name: str
    ) -> torch.Tensor:
        """
        Register an output tensor (allocates new tensor).

        Args:
            shape: Shape of the output tensor
            dtype: Data type of the output tensor
            name: Unique name for this tensor

        Returns:
            The allocated output tensor
        """
        tensor = torch.empty(shape, dtype=dtype, device='cuda')
        self.register(tensor, name, is_output=True)
        return tensor

    def register_weight(self, tensor: torch.Tensor, name: str) -> TensorInfo:
        """Register a weight tensor."""
        return self.register(tensor, name, is_weight=True)

    def get(self, name: str) -> Optional[TensorInfo]:
        """Get tensor info by name."""
        return self._tensors.get(name)

    def get_tensor(self, name: str) -> Optional[torch.Tensor]:
        """Get the actual tensor by name."""
        info = self._tensors.get(name)
        return info.tensor if info else None

    @property
    def inputs(self) -> Dict[str, TensorInfo]:
        """Get all input tensors."""
        return self._inputs

    @property
    def outputs(self) -> Dict[str, TensorInfo]:
        """Get all output tensors."""
        return self._outputs

    @property
    def weights(self) -> Dict[str, TensorInfo]:
        """Get all weight tensors."""
        return self._weights

    def get_all_pointers(self) -> Dict[str, int]:
        """Get device pointers for all tensors."""
        return {
            name: info.device_ptr
            for name, info in self._tensors.items()
            if info.device_ptr is not None
        }

    def clear(self):
        """Clear all registered tensors."""
        self._tensors.clear()
        self._inputs.clear()
        self._outputs.clear()
        self._weights.clear()

    def __len__(self) -> int:
        return len(self._tensors)

    def __contains__(self, name: str) -> bool:
        return name in self._tensors
