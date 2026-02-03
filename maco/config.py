"""
Global Configuration for MACO
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MacoConfig:
    """Global configuration for MACO."""

    # Communication backend
    backend: str = "nvshmem"  # "nvshmem" | "nccl" | "gloo"

    # SM allocation policy
    sm_policy: str = "auto"  # "auto" | "manual" | "dynamic"

    # Overlap strategy
    overlap_strategy: str = "aggressive"  # "none" | "conservative" | "aggressive"

    # Memory management
    comm_buffer_size: str = "256MB"
    use_pinned_memory: bool = True

    # Debug options
    debug: bool = False
    profile: bool = False
    verbose: bool = False

    def set(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    def get(self, key: str, default=None):
        """Get a configuration value."""
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "backend": self.backend,
            "sm_policy": self.sm_policy,
            "overlap_strategy": self.overlap_strategy,
            "comm_buffer_size": self.comm_buffer_size,
            "use_pinned_memory": self.use_pinned_memory,
            "debug": self.debug,
            "profile": self.profile,
            "verbose": self.verbose,
        }

    def __repr__(self):
        return f"MacoConfig({self.to_dict()})"


# Global config instance
config = MacoConfig()
