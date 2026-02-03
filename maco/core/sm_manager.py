"""
SM (Streaming Multiprocessor) Role Management

This module provides explicit control over how SMs are assigned
to different roles: compute, local scheduling, and remote scheduling.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch


class Role(Enum):
    """SM roles in the persistent kernel."""
    COMPUTE = auto()           # Execute compute tasks (matmul, attention, etc.)
    LOCAL_SCHEDULER = auto()   # Schedule tasks within a single GPU
    REMOTE_SCHEDULER = auto()  # Handle cross-GPU communication


@dataclass
class SMAssignment:
    """Assignment of SMs to a specific role."""
    sm_range: Tuple[int, int]  # (start, end) SM indices
    role: Role
    task_types: List[str] = field(default_factory=list)  # Optional: specific tasks


@dataclass
class SMConfig:
    """Configuration for SM role assignment."""
    total_sms: int
    assignments: List[SMAssignment] = field(default_factory=list)

    def __post_init__(self):
        # Detect total SMs if not specified
        if self.total_sms <= 0:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.total_sms = props.multi_processor_count

    def assign_role(
        self,
        sm_range: Tuple[int, int],
        role: Role,
        task_types: Optional[List[str]] = None
    ):
        """
        Assign a role to a range of SMs.

        Args:
            sm_range: Tuple of (start_sm, end_sm), exclusive end
            role: The role to assign
            task_types: Optional list of task types this SM group handles
        """
        if sm_range[1] > self.total_sms:
            raise ValueError(
                f"SM range {sm_range} exceeds total SMs ({self.total_sms})"
            )

        # Check for overlapping assignments
        for existing in self.assignments:
            if (sm_range[0] < existing.sm_range[1] and
                sm_range[1] > existing.sm_range[0]):
                raise ValueError(
                    f"SM range {sm_range} overlaps with existing assignment "
                    f"{existing.sm_range}"
                )

        self.assignments.append(SMAssignment(
            sm_range=sm_range,
            role=role,
            task_types=task_types or []
        ))

    @property
    def compute_sms(self) -> int:
        """Number of SMs assigned to compute."""
        return sum(
            a.sm_range[1] - a.sm_range[0]
            for a in self.assignments
            if a.role == Role.COMPUTE
        )

    @property
    def local_scheduler_sms(self) -> int:
        """Number of SMs assigned to local scheduling."""
        return sum(
            a.sm_range[1] - a.sm_range[0]
            for a in self.assignments
            if a.role == Role.LOCAL_SCHEDULER
        )

    @property
    def remote_scheduler_sms(self) -> int:
        """Number of SMs assigned to remote scheduling."""
        return sum(
            a.sm_range[1] - a.sm_range[0]
            for a in self.assignments
            if a.role == Role.REMOTE_SCHEDULER
        )

    def validate(self):
        """Validate the SM configuration."""
        assigned = set()
        for a in self.assignments:
            for sm in range(a.sm_range[0], a.sm_range[1]):
                if sm in assigned:
                    raise ValueError(f"SM {sm} is assigned multiple times")
                assigned.add(sm)

        unassigned = set(range(self.total_sms)) - assigned
        if unassigned:
            print(f"[MACO Warning] {len(unassigned)} SMs are unassigned: "
                  f"{sorted(unassigned)[:5]}...")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_sms": self.total_sms,
            "assignments": [
                {
                    "sm_range": a.sm_range,
                    "role": a.role.name,
                    "task_types": a.task_types
                }
                for a in self.assignments
            ]
        }


class SMManager:
    """
    Manages SM allocation and provides runtime control.
    """

    def __init__(self, config: Optional[SMConfig] = None):
        self.config = config or self._auto_config()
        self._compiled = False

    def _auto_config(self) -> SMConfig:
        """Auto-generate SM configuration based on GPU properties."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        props = torch.cuda.get_device_properties(0)
        total_sms = props.multi_processor_count

        config = SMConfig(total_sms=total_sms)

        # Default allocation: 75% compute, 18.75% local sched, 6.25% remote sched
        compute_end = int(total_sms * 0.75)
        local_sched_end = int(total_sms * 0.9375)

        config.assign_role((0, compute_end), Role.COMPUTE)
        config.assign_role((compute_end, local_sched_end), Role.LOCAL_SCHEDULER)
        config.assign_role((local_sched_end, total_sms), Role.REMOTE_SCHEDULER)

        return config

    def compile(self):
        """Compile the SM configuration for runtime use."""
        self.config.validate()
        self._compiled = True

    def get_worker_config(self) -> dict:
        """Get configuration for worker SMs."""
        if not self._compiled:
            self.compile()

        return {
            "num_workers": self.config.compute_sms,
            "num_local_schedulers": self.config.local_scheduler_sms,
            "num_remote_schedulers": self.config.remote_scheduler_sms,
        }


def auto_config(
    compute_intensity: str = "high",
    comm_intensity: str = "medium",
    gpu_arch: str = "auto"
) -> SMConfig:
    """
    Automatically generate SM configuration based on workload characteristics.

    Args:
        compute_intensity: "high", "medium", or "low"
        comm_intensity: "high", "medium", or "low"
        gpu_arch: GPU architecture ("hopper", "ampere", or "auto")

    Returns:
        Recommended SMConfig
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    props = torch.cuda.get_device_properties(0)
    total_sms = props.multi_processor_count

    # Determine ratios based on intensity
    intensity_map = {
        ("high", "low"): (0.90, 0.08, 0.02),
        ("high", "medium"): (0.80, 0.15, 0.05),
        ("high", "high"): (0.70, 0.20, 0.10),
        ("medium", "low"): (0.85, 0.12, 0.03),
        ("medium", "medium"): (0.75, 0.18, 0.07),
        ("medium", "high"): (0.65, 0.23, 0.12),
        ("low", "low"): (0.80, 0.15, 0.05),
        ("low", "medium"): (0.70, 0.20, 0.10),
        ("low", "high"): (0.60, 0.25, 0.15),
    }

    compute_ratio, local_ratio, remote_ratio = intensity_map.get(
        (compute_intensity, comm_intensity),
        (0.75, 0.18, 0.07)  # default
    )

    config = SMConfig(total_sms=total_sms)

    compute_end = int(total_sms * compute_ratio)
    local_end = int(total_sms * (compute_ratio + local_ratio))

    config.assign_role((0, compute_end), Role.COMPUTE)
    config.assign_role((compute_end, local_end), Role.LOCAL_SCHEDULER)
    config.assign_role((local_end, total_sms), Role.REMOTE_SCHEDULER)

    return config
