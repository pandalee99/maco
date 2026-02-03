"""Core components of MACO."""

from .context import Context, PersistentContext
from .sm_manager import SMConfig, SMManager, Role
from .tensor_registry import TensorRegistry

__all__ = [
    "Context",
    "PersistentContext",
    "SMConfig",
    "SMManager",
    "Role",
    "TensorRegistry",
]
