from __future__ import annotations

from typing import List, Optional, Dict, Any
from copy import deepcopy

from .models import Phase, PhaseTask, PhaseStatus, AIAgent, _now


# Default phase scaffold used in tests
IDAW_PHASES: List[Phase] = [
    Phase(
        id=1,
        name="Core Systems",
        description="Phase 1 - Core Systems",
        status=PhaseStatus.IN_PROGRESS,
        tasks=[PhaseTask(id="", name="Bootstrap", description="")],
    ),
    Phase(
        id=2,
        name="Expansion & Integration",
        description="Phase 2 - Expansion",
        status=PhaseStatus.NOT_STARTED,
    ),
    Phase(
        id=3,
        name="Advanced Features & C++ Transition",
        description="Phase 3 - Advanced",
        status=PhaseStatus.NOT_STARTED,
    ),
]


class PhaseManager:
    def __init__(self, phases: Optional[List[Phase]] = None):
        self.phases: List[Phase] = deepcopy(phases) if phases is not None else deepcopy(IDAW_PHASES)
        in_progress = next((p for p in self.phases if p.status == PhaseStatus.IN_PROGRESS), None)
        self.current_phase_id = in_progress.id if in_progress else (self.phases[0].id if self.phases else 1)

    def get_current_phase(self) -> Optional[Phase]:
        return self.get_phase(self.current_phase_id)

    def get_phase(self, phase_id: int) -> Optional[Phase]:
        return next((p for p in self.phases if p.id == phase_id), None)

    def update_task_status(
        self,
        phase_id: int,
        task_id: str,
        status: PhaseStatus,
        progress: Optional[float] = None,
        notes: Optional[str] = None,
    ):
        phase = self.get_phase(phase_id)
        if not phase:
            return
        task = next((t for t in phase.tasks if t.id == task_id), None)
        if not task:
            return
        task.status = status
        if progress is not None:
            task.progress = progress
        if notes is not None:
            task.notes = notes
        if status == PhaseStatus.COMPLETED:
            task.progress = 1.0
            task.completed_at = task.completed_at or _now()
        phase.update_progress()

    def assign_task(self, phase_id: int, task_id: str, agent: AIAgent):
        phase = self.get_phase(phase_id)
        if not phase:
            return
        task = next((t for t in phase.tasks if t.id == task_id), None)
        if not task:
            return
        task.assigned_to = agent
        if task.status == PhaseStatus.NOT_STARTED:
            task.status = PhaseStatus.IN_PROGRESS
        phase.update_progress()

    def advance_phase(self) -> bool:
        current = self.get_current_phase()
        if not current or current.progress < 1.0:
            return False
        current.status = PhaseStatus.VERIFIED
        idx = self.phases.index(current)
        if idx + 1 < len(self.phases):
            next_phase = self.phases[idx + 1]
            next_phase.status = PhaseStatus.IN_PROGRESS
            self.current_phase_id = next_phase.id
            return True
        return False

    def get_incomplete_tasks(self, phase_id: Optional[int] = None) -> List[PhaseTask]:
        tasks: List[PhaseTask] = []
        for phase in self.phases:
            if phase_id and phase.id != phase_id:
                continue
            tasks.extend([t for t in phase.tasks if t.status != PhaseStatus.COMPLETED])
        return tasks

    def get_blocked_tasks(self) -> List[PhaseTask]:
        tasks: List[PhaseTask] = []
        for phase in self.phases:
            tasks.extend([t for t in phase.tasks if t.status == PhaseStatus.BLOCKED or t.blockers])
        return tasks

    def get_phase_summary(self, phase_id: Optional[int] = None) -> Dict[str, Any]:
        phases = self.phases if phase_id is None else [p for p in self.phases if p.id == phase_id]
        summary = {
            "current_phase": self.current_phase_id,
            "phases": [
                {
                    "phase_id": p.id,
                    "name": p.name,
                    "status": p.status.value,
                    "progress": p.progress,
                }
                for p in phases
            ],
        }
        if phases:
            summary["overall_progress"] = sum(p.progress for p in phases) / len(phases)
        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phases": [p.to_dict() for p in self.phases],
            "current_phase_id": self.current_phase_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseManager":
        phases = [Phase.from_dict(p) for p in data.get("phases", [])]
        mgr = cls(phases=phases)
        mgr.current_phase_id = data.get("current_phase_id", mgr.current_phase_id)
        return mgr


def format_phase_progress(manager: PhaseManager) -> str:
    lines = []
    for phase in manager.phases:
        lines.append(f"{phase.name} [{phase.status.value}] {phase.progress:.0%}")
        for task in phase.tasks:
            lines.append(f"  - {task.name} ({task.status.value})")
    return "\n".join(lines)


def get_next_actions(manager: PhaseManager) -> List[str]:
    actions: List[str] = []
    # continue or start tasks
    for task in manager.get_incomplete_tasks():
        if task.status == PhaseStatus.IN_PROGRESS:
            actions.append(f"CONTINUE {task.name}")
        elif task.status == PhaseStatus.NOT_STARTED:
            actions.append(f"START {task.name}")
        elif task.status == PhaseStatus.BLOCKED:
            actions.append(f"UNBLOCK {task.name}")
    # ready to advance
    current = manager.get_current_phase()
    if current and current.progress >= 1.0:
        actions.append("READY TO ADVANCE")
    return actions
