"""
Communication Scheduler

Schedules communication operations for optimal overlap with compute.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import torch


class CommOpType(Enum):
    """Types of communication operations."""
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    REDUCE_SCATTER = "reduce_scatter"
    SEND = "send"
    RECV = "recv"
    BARRIER = "barrier"


@dataclass
class CommOp:
    """A single communication operation."""
    op_type: CommOpType
    tensor: torch.Tensor
    kwargs: Dict[str, Any] = field(default_factory=dict)
    handle: Any = None


@dataclass
class CommPhase:
    """A phase of communication operations that can execute together."""
    name: str
    ops: List[CommOp] = field(default_factory=list)
    depends_on: Optional['CommPhase'] = None
    _completed: bool = False

    def allreduce(self, tensor: torch.Tensor, **kwargs) -> 'CommPhase':
        """Add an allreduce operation to this phase."""
        self.ops.append(CommOp(
            op_type=CommOpType.ALLREDUCE,
            tensor=tensor,
            kwargs=kwargs
        ))
        return self

    def allgather(self, tensor: torch.Tensor, **kwargs) -> 'CommPhase':
        """Add an allgather operation to this phase."""
        self.ops.append(CommOp(
            op_type=CommOpType.ALLGATHER,
            tensor=tensor,
            kwargs=kwargs
        ))
        return self

    def reduce_scatter(self, tensor: torch.Tensor, **kwargs) -> 'CommPhase':
        """Add a reduce_scatter operation to this phase."""
        self.ops.append(CommOp(
            op_type=CommOpType.REDUCE_SCATTER,
            tensor=tensor,
            kwargs=kwargs
        ))
        return self

    def send(self, tensor: torch.Tensor, dst: int, **kwargs) -> 'CommPhase':
        """Add a send operation to this phase."""
        self.ops.append(CommOp(
            op_type=CommOpType.SEND,
            tensor=tensor,
            kwargs={"dst": dst, **kwargs}
        ))
        return self

    def recv(self, tensor: torch.Tensor, src: int, **kwargs) -> 'CommPhase':
        """Add a recv operation to this phase."""
        self.ops.append(CommOp(
            op_type=CommOpType.RECV,
            tensor=tensor,
            kwargs={"src": src, **kwargs}
        ))
        return self


class CommScheduler:
    """
    Scheduler for communication operations.

    Manages phases of communication operations and their dependencies,
    enabling efficient overlap with compute operations.

    Example:
        scheduler = CommScheduler(num_sms=32)

        phase1 = scheduler.phase("attn_comm")
        phase1.allreduce(q_proj)
        phase1.allreduce(k_proj)

        phase2 = scheduler.phase("ffn_comm", depends_on=phase1)
        phase2.allreduce(ffn_out)

        scheduler.compile()
        scheduler.execute()
    """

    def __init__(self, num_sms: int = 32):
        """
        Initialize the scheduler.

        Args:
            num_sms: Number of SMs dedicated to communication
        """
        self.num_sms = num_sms
        self.phases: Dict[str, CommPhase] = {}
        self._phase_order: List[str] = []
        self._compiled = False

    def phase(
        self,
        name: str,
        depends_on: Optional[CommPhase] = None
    ) -> CommPhase:
        """
        Create a new communication phase.

        Args:
            name: Unique name for this phase
            depends_on: Optional phase that must complete before this one

        Returns:
            New CommPhase object
        """
        if name in self.phases:
            raise ValueError(f"Phase '{name}' already exists")

        phase = CommPhase(name=name, depends_on=depends_on)
        self.phases[name] = phase
        self._phase_order.append(name)
        return phase

    def compile(self):
        """
        Compile the communication schedule.

        This analyzes dependencies and optimizes the execution order.
        """
        # Topological sort of phases based on dependencies
        sorted_phases = []
        visited = set()
        temp_visited = set()

        def visit(phase_name: str):
            if phase_name in temp_visited:
                raise ValueError(f"Circular dependency detected at '{phase_name}'")
            if phase_name in visited:
                return

            temp_visited.add(phase_name)
            phase = self.phases[phase_name]

            if phase.depends_on:
                visit(phase.depends_on.name)

            temp_visited.remove(phase_name)
            visited.add(phase_name)
            sorted_phases.append(phase_name)

        for name in self._phase_order:
            if name not in visited:
                visit(name)

        self._phase_order = sorted_phases
        self._compiled = True

        print(f"[MACO] Compiled {len(self.phases)} communication phases")

    def execute(self, async_mode: bool = False):
        """
        Execute the compiled communication schedule.

        Args:
            async_mode: If True, launch all operations asynchronously
        """
        if not self._compiled:
            self.compile()

        from . import patterns

        for phase_name in self._phase_order:
            phase = self.phases[phase_name]

            # Wait for dependency
            if phase.depends_on and not phase.depends_on._completed:
                self._wait_phase(phase.depends_on)

            # Execute operations in this phase
            for op in phase.ops:
                if op.op_type == CommOpType.ALLREDUCE:
                    op.handle = patterns.allreduce(
                        op.tensor,
                        async_op=async_mode,
                        **op.kwargs
                    )
                elif op.op_type == CommOpType.ALLGATHER:
                    op.handle = patterns.allgather(
                        op.tensor,
                        async_op=async_mode,
                        **op.kwargs
                    )
                elif op.op_type == CommOpType.REDUCE_SCATTER:
                    op.handle = patterns.reduce_scatter(
                        op.tensor,
                        async_op=async_mode,
                        **op.kwargs
                    )
                elif op.op_type == CommOpType.SEND:
                    op.handle = patterns.send(
                        op.tensor,
                        async_op=async_mode,
                        **op.kwargs
                    )
                elif op.op_type == CommOpType.RECV:
                    op.handle = patterns.recv(
                        tensor=op.tensor,
                        async_op=async_mode,
                        **op.kwargs
                    )

            if not async_mode:
                phase._completed = True

    def _wait_phase(self, phase: CommPhase):
        """Wait for all operations in a phase to complete."""
        for op in phase.ops:
            if op.handle is not None and hasattr(op.handle, 'wait'):
                op.handle.wait()
        phase._completed = True

    def wait_all(self):
        """Wait for all phases to complete."""
        for phase in self.phases.values():
            if not phase._completed:
                self._wait_phase(phase)

    def reset(self):
        """Reset the scheduler for reuse."""
        for phase in self.phases.values():
            phase._completed = False
            for op in phase.ops:
                op.handle = None

    def get_schedule_info(self) -> Dict[str, Any]:
        """Get information about the compiled schedule."""
        return {
            "num_phases": len(self.phases),
            "phase_order": self._phase_order,
            "num_sms": self.num_sms,
            "phases": {
                name: {
                    "num_ops": len(phase.ops),
                    "depends_on": phase.depends_on.name if phase.depends_on else None,
                    "ops": [op.op_type.value for op in phase.ops]
                }
                for name, phase in self.phases.items()
            }
        }
