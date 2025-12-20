from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


def _now() -> str:
    return datetime.utcnow().isoformat()


def _gen_id() -> str:
    return uuid.uuid4().hex[:8]


class AIAgent(Enum):
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    GEMINI = "gemini"
    GITHUB_COPILOT = "github_copilot"

    @property
    def display_name(self) -> str:
        return {
            AIAgent.CLAUDE: "Anthropic Claude",
            AIAgent.CHATGPT: "OpenAI ChatGPT",
            AIAgent.GEMINI: "Google Gemini",
            AIAgent.GITHUB_COPILOT: "GitHub Copilot",
        }.get(self, self.value)


class ProposalStatus(Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    DEFERRED = "deferred"


class ProposalCategory(Enum):
    ARCHITECTURE = "architecture"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    FEATURE_NEW = "feature_new"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    AUDIO_PROCESSING = "audio_processing"
    MIDI_HANDLING = "midi_handling"
    DSP_ALGORITHM = "dsp_algorithm"
    CPP_PORT = "cpp_port"
    CPP_OPTIMIZATION = "cpp_optimization"
    DOCUMENTATION = "documentation"


class PhaseStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    VERIFIED = "verified"


@dataclass
class ProposalVote:
    agent: AIAgent
    proposal_id: str
    vote: int
    comment: Optional[str] = None
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent.value,
            "proposal_id": self.proposal_id,
            "vote": self.vote,
            "comment": self.comment,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposalVote":
        return cls(
            agent=AIAgent(data["agent"]),
            proposal_id=data["proposal_id"],
            vote=data["vote"],
            comment=data.get("comment"),
            timestamp=data.get("timestamp", _now()),
        )


@dataclass
class Proposal:
    id: str
    agent: AIAgent
    title: str
    description: str
    category: ProposalCategory
    status: ProposalStatus = ProposalStatus.DRAFT
    priority: int = 5
    estimated_effort: str = "medium"
    phase_target: int = 1
    implementation_notes: str = ""
    dependencies: List[str] = field(default_factory=list)
    votes: Dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self):
        if not self.id:
            self.id = _gen_id()

    @property
    def vote_score(self) -> int:
        return sum(self.votes.values())

    def add_vote(self, agent: AIAgent, value: int):
        clamped = max(-1, min(1, value))
        self.votes[agent.value] = clamped
        self.updated_at = _now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent.value,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "status": self.status.value,
            "priority": self.priority,
            "estimated_effort": self.estimated_effort,
            "phase_target": self.phase_target,
            "implementation_notes": self.implementation_notes,
            "dependencies": self.dependencies,
            "votes": self.votes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Proposal":
        return cls(
            id=data.get("id", _gen_id()),
            agent=AIAgent(data["agent"]),
            title=data.get("title", ""),
            description=data.get("description", ""),
            category=ProposalCategory(data["category"]),
            status=ProposalStatus(data.get("status", ProposalStatus.DRAFT.value)),
            priority=data.get("priority", 5),
            estimated_effort=data.get("estimated_effort", "medium"),
            phase_target=data.get("phase_target", 1),
            implementation_notes=data.get("implementation_notes", ""),
            dependencies=data.get("dependencies", []) or [],
            votes=data.get("votes", {}) or {},
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
        )


@dataclass
class PhaseTask:
    id: str
    name: str
    description: str
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    progress: float = 0.0
    assigned_to: Optional[AIAgent] = None
    blockers: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = _gen_id()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "progress": self.progress,
            "assigned_to": self.assigned_to.value if self.assigned_to else None,
            "blockers": self.blockers,
            "notes": self.notes,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseTask":
        assigned = data.get("assigned_to")
        return cls(
            id=data.get("id", _gen_id()),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=PhaseStatus(data.get("status", PhaseStatus.NOT_STARTED.value)),
            progress=data.get("progress", 0.0),
            assigned_to=AIAgent(assigned) if assigned else None,
            blockers=data.get("blockers", []) or [],
            notes=data.get("notes"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class Phase:
    id: int
    name: str
    description: str
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    progress: float = 0.0
    tasks: List[PhaseTask] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    actual_completion: Optional[str] = None

    def update_progress(self):
        if not self.tasks:
            return
        completed = sum(1 for t in self.tasks if t.status == PhaseStatus.COMPLETED)
        self.progress = completed / len(self.tasks)
        if self.progress >= 1.0:
            self.status = PhaseStatus.COMPLETED
            if not self.actual_completion:
                self.actual_completion = _now()
        elif completed > 0 or any(t.status != PhaseStatus.NOT_STARTED for t in self.tasks):
            self.status = PhaseStatus.IN_PROGRESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "progress": self.progress,
            "tasks": [t.to_dict() for t in self.tasks],
            "milestones": self.milestones,
            "deliverables": self.deliverables,
            "actual_completion": self.actual_completion,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase":
        tasks = [PhaseTask.from_dict(td) for td in data.get("tasks", [])]
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=PhaseStatus(data.get("status", PhaseStatus.NOT_STARTED.value)),
            progress=data.get("progress", 0.0),
            tasks=tasks,
            milestones=data.get("milestones", []) or [],
            deliverables=data.get("deliverables", []) or [],
            actual_completion=data.get("actual_completion"),
        )


@dataclass
class WorkstationState:
    proposals: List[Proposal] = field(default_factory=list)
    phases: List[Phase] = field(default_factory=list)
    active_agents: List[AIAgent] = field(default_factory=list)
    current_phase: int = 1
    session_id: str = field(default_factory=_gen_id)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposals": [p.to_dict() for p in self.proposals],
            "phases": [p.to_dict() for p in self.phases],
            "active_agents": [a.value for a in self.active_agents],
            "current_phase": self.current_phase,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkstationState":
        proposals = [Proposal.from_dict(p) for p in data.get("proposals", [])]
        phases = [Phase.from_dict(p) for p in data.get("phases", [])]
        agents = [AIAgent(a) for a in data.get("active_agents", [])]
        return cls(
            proposals=proposals,
            phases=phases,
            active_agents=agents,
            current_phase=data.get("current_phase", 1),
            session_id=data.get("session_id", _gen_id()),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
        )

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "WorkstationState":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
