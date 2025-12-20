"""
Tests for MCP Workstation Data Models.

Tests AIAgent, Proposal, Phase, PhaseTask, and WorkstationState models.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from mcp_workstation.models import (
    AIAgent,
    ProposalStatus,
    ProposalCategory,
    PhaseStatus,
    Proposal,
    PhaseTask,
    Phase,
    WorkstationState,
)


class TestAIAgent:
    """Tests for AIAgent enum."""

    def test_agent_values(self):
        """Test that all agent values are defined."""
        assert AIAgent.CLAUDE.value == "claude"
        assert AIAgent.CHATGPT.value == "chatgpt"
        assert AIAgent.GEMINI.value == "gemini"
        assert AIAgent.GITHUB_COPILOT.value == "github_copilot"

    def test_agent_from_string(self):
        """Test creating agent from string."""
        assert AIAgent("claude") == AIAgent.CLAUDE
        assert AIAgent("chatgpt") == AIAgent.CHATGPT

    def test_agent_display_names(self):
        """Test agent display names."""
        assert "Anthropic" in AIAgent.CLAUDE.display_name
        assert "OpenAI" in AIAgent.CHATGPT.display_name
        assert "Google" in AIAgent.GEMINI.display_name
        assert "GitHub" in AIAgent.GITHUB_COPILOT.display_name

    def test_agent_invalid_value(self):
        """Test that invalid agent raises ValueError."""
        with pytest.raises(ValueError):
            AIAgent("invalid")


class TestProposalStatus:
    """Tests for ProposalStatus enum."""

    def test_status_values(self):
        """Test that all status values are defined."""
        assert ProposalStatus.DRAFT.value == "draft"
        assert ProposalStatus.SUBMITTED.value == "submitted"
        assert ProposalStatus.UNDER_REVIEW.value == "under_review"
        assert ProposalStatus.APPROVED.value == "approved"
        assert ProposalStatus.REJECTED.value == "rejected"
        assert ProposalStatus.IMPLEMENTED.value == "implemented"
        assert ProposalStatus.DEFERRED.value == "deferred"


class TestProposalCategory:
    """Tests for ProposalCategory enum."""

    def test_core_categories(self):
        """Test core system categories."""
        assert ProposalCategory.ARCHITECTURE.value == "architecture"
        assert ProposalCategory.PERFORMANCE.value == "performance"
        assert ProposalCategory.RELIABILITY.value == "reliability"

    def test_feature_categories(self):
        """Test feature categories."""
        assert ProposalCategory.FEATURE_NEW.value == "feature_new"
        assert ProposalCategory.FEATURE_ENHANCEMENT.value == "feature_enhancement"

    def test_audio_categories(self):
        """Test audio/music categories."""
        assert ProposalCategory.AUDIO_PROCESSING.value == "audio_processing"
        assert ProposalCategory.MIDI_HANDLING.value == "midi_handling"
        assert ProposalCategory.DSP_ALGORITHM.value == "dsp_algorithm"

    def test_cpp_categories(self):
        """Test C++ transition categories."""
        assert ProposalCategory.CPP_PORT.value == "cpp_port"
        assert ProposalCategory.CPP_OPTIMIZATION.value == "cpp_optimization"


class TestPhaseStatus:
    """Tests for PhaseStatus enum."""

    def test_status_values(self):
        """Test that all phase status values are defined."""
        assert PhaseStatus.NOT_STARTED.value == "not_started"
        assert PhaseStatus.IN_PROGRESS.value == "in_progress"
        assert PhaseStatus.BLOCKED.value == "blocked"
        assert PhaseStatus.COMPLETED.value == "completed"
        assert PhaseStatus.VERIFIED.value == "verified"


class TestProposal:
    """Tests for Proposal dataclass."""

    def test_proposal_creation(self):
        """Test creating a proposal."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="Test Proposal",
            description="Description",
            category=ProposalCategory.FEATURE_NEW,
        )
        assert proposal.title == "Test Proposal"
        assert len(proposal.id) == 8
        assert proposal.agent == AIAgent.CLAUDE
        assert proposal.status == ProposalStatus.DRAFT

    def test_proposal_defaults(self):
        """Test proposal default values."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="Test",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        assert proposal.priority == 5
        assert proposal.estimated_effort == "medium"
        assert proposal.phase_target == 1
        assert proposal.dependencies == []
        assert proposal.votes == {}

    def test_proposal_auto_timestamps(self):
        """Test that timestamps are auto-generated."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="Test",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        assert proposal.created_at != ""
        assert proposal.updated_at != ""

    def test_proposal_to_dict(self):
        """Test serializing proposal to dictionary."""
        proposal = Proposal(
            id="test123",
            agent=AIAgent.CLAUDE,
            title="Test Proposal",
            description="Description",
            category=ProposalCategory.ARCHITECTURE,
            priority=8,
        )
        data = proposal.to_dict()

        assert data["id"] == "test123"
        assert data["agent"] == "claude"
        assert data["title"] == "Test Proposal"
        assert data["category"] == "architecture"
        assert data["priority"] == 8
        assert data["status"] == "draft"

    def test_proposal_from_dict(self):
        """Test deserializing proposal from dictionary."""
        data = {
            "id": "abc123",
            "agent": "chatgpt",
            "title": "Loaded Proposal",
            "description": "Loaded description",
            "category": "performance",
            "status": "approved",
            "priority": 7,
            "estimated_effort": "high",
            "phase_target": 2,
            "votes": {"claude": 1},
        }
        proposal = Proposal.from_dict(data)

        assert proposal.id == "abc123"
        assert proposal.agent == AIAgent.CHATGPT
        assert proposal.title == "Loaded Proposal"
        assert proposal.category == ProposalCategory.PERFORMANCE
        assert proposal.status == ProposalStatus.APPROVED
        assert proposal.votes == {"claude": 1}

    def test_proposal_roundtrip(self):
        """Test that to_dict/from_dict preserves data."""
        original = Proposal(
            id="round",
            agent=AIAgent.GEMINI,
            title="Roundtrip Test",
            description="Testing roundtrip",
            category=ProposalCategory.DSP_ALGORITHM,
            priority=9,
            votes={"claude": 1, "chatgpt": -1},
        )
        data = original.to_dict()
        restored = Proposal.from_dict(data)

        assert restored.id == original.id
        assert restored.agent == original.agent
        assert restored.title == original.title
        assert restored.category == original.category
        assert restored.votes == original.votes

    def test_vote_score(self):
        """Test vote score calculation."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="Test",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        proposal.votes = {"chatgpt": 1, "gemini": 1, "github_copilot": -1}
        assert proposal.vote_score == 1

    def test_add_vote(self):
        """Test adding a vote."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="Test",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        proposal.add_vote(AIAgent.CHATGPT, 1)
        assert proposal.votes["chatgpt"] == 1

    def test_add_vote_clamped(self):
        """Test that votes are clamped to [-1, 1]."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="Test",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        proposal.add_vote(AIAgent.CHATGPT, 5)
        assert proposal.votes["chatgpt"] == 1

        proposal.add_vote(AIAgent.GEMINI, -10)
        assert proposal.votes["gemini"] == -1

    def test_add_vote_updates_timestamp(self):
        """Test that adding vote updates timestamp."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="Test",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        original_updated = proposal.updated_at

        import time
        time.sleep(0.01)

        proposal.add_vote(AIAgent.CHATGPT, 1)
        assert proposal.updated_at != original_updated


class TestPhaseTask:
    """Tests for PhaseTask dataclass."""

    def test_task_creation(self):
        """Test creating a phase task."""
        task = PhaseTask(
            id="",
            name="Test Task",
            description="Task description",
        )
        assert task.name == "Test Task"
        assert len(task.id) == 8
        assert task.status == PhaseStatus.NOT_STARTED
        assert task.progress == 0.0

    def test_task_with_assignment(self):
        """Test task with agent assignment."""
        task = PhaseTask(
            id="t1",
            name="Assigned Task",
            description="Has assignee",
            assigned_to=AIAgent.CLAUDE,
        )
        assert task.assigned_to == AIAgent.CLAUDE

    def test_task_to_dict(self):
        """Test serializing task to dictionary."""
        task = PhaseTask(
            id="task1",
            name="Test Task",
            description="Description",
            status=PhaseStatus.IN_PROGRESS,
            progress=0.5,
            assigned_to=AIAgent.CLAUDE,
        )
        data = task.to_dict()

        assert data["id"] == "task1"
        assert data["name"] == "Test Task"
        assert data["status"] == "in_progress"
        assert data["progress"] == 0.5
        assert data["assigned_to"] == "claude"

    def test_task_from_dict(self):
        """Test deserializing task from dictionary."""
        data = {
            "id": "t123",
            "name": "Loaded Task",
            "description": "Loaded",
            "status": "completed",
            "progress": 1.0,
            "assigned_to": "gemini",
        }
        task = PhaseTask.from_dict(data)

        assert task.id == "t123"
        assert task.name == "Loaded Task"
        assert task.status == PhaseStatus.COMPLETED
        assert task.assigned_to == AIAgent.GEMINI

    def test_task_blockers(self):
        """Test task with blockers."""
        task = PhaseTask(
            id="t1",
            name="Blocked Task",
            description="Has blockers",
            blockers=["Waiting for API", "Need review"],
        )
        assert len(task.blockers) == 2


class TestPhase:
    """Tests for Phase dataclass."""

    def test_phase_creation(self):
        """Test creating a phase."""
        phase = Phase(
            id=1,
            name="Phase 1",
            description="First phase",
        )
        assert phase.id == 1
        assert phase.name == "Phase 1"
        assert phase.status == PhaseStatus.NOT_STARTED
        assert phase.progress == 0.0
        assert phase.tasks == []

    def test_phase_with_tasks(self):
        """Test phase with tasks."""
        tasks = [
            PhaseTask(id="t1", name="Task 1", description="First"),
            PhaseTask(id="t2", name="Task 2", description="Second"),
        ]
        phase = Phase(
            id=1,
            name="With Tasks",
            description="Has tasks",
            tasks=tasks,
        )
        assert len(phase.tasks) == 2

    def test_phase_to_dict(self):
        """Test serializing phase to dictionary."""
        phase = Phase(
            id=1,
            name="Test Phase",
            description="Description",
            milestones=["M1", "M2"],
            deliverables=["D1"],
        )
        data = phase.to_dict()

        assert data["id"] == 1
        assert data["name"] == "Test Phase"
        assert data["milestones"] == ["M1", "M2"]
        assert data["deliverables"] == ["D1"]

    def test_phase_from_dict(self):
        """Test deserializing phase from dictionary."""
        data = {
            "id": 2,
            "name": "Loaded Phase",
            "description": "Loaded",
            "status": "in_progress",
            "progress": 0.5,
            "tasks": [
                {"id": "t1", "name": "Task", "description": "Desc", "status": "completed"},
            ],
            "milestones": ["Done"],
        }
        phase = Phase.from_dict(data)

        assert phase.id == 2
        assert phase.name == "Loaded Phase"
        assert phase.status == PhaseStatus.IN_PROGRESS
        assert len(phase.tasks) == 1
        assert phase.tasks[0].status == PhaseStatus.COMPLETED

    def test_phase_update_progress(self):
        """Test updating phase progress based on tasks."""
        tasks = [
            PhaseTask(id="t1", name="T1", description="", status=PhaseStatus.COMPLETED),
            PhaseTask(id="t2", name="T2", description="", status=PhaseStatus.COMPLETED),
            PhaseTask(id="t3", name="T3", description="", status=PhaseStatus.NOT_STARTED),
            PhaseTask(id="t4", name="T4", description="", status=PhaseStatus.NOT_STARTED),
        ]
        phase = Phase(id=1, name="Test", description="", tasks=tasks)
        phase.update_progress()

        assert phase.progress == 0.5
        assert phase.status == PhaseStatus.IN_PROGRESS

    def test_phase_update_progress_completed(self):
        """Test phase completion when all tasks done."""
        tasks = [
            PhaseTask(id="t1", name="T1", description="", status=PhaseStatus.COMPLETED),
            PhaseTask(id="t2", name="T2", description="", status=PhaseStatus.COMPLETED),
        ]
        phase = Phase(id=1, name="Test", description="", tasks=tasks)
        phase.update_progress()

        assert phase.progress == 1.0
        assert phase.status == PhaseStatus.COMPLETED
        assert phase.actual_completion is not None

    def test_phase_update_progress_empty(self):
        """Test updating progress with no tasks."""
        phase = Phase(id=1, name="Empty", description="")
        phase.update_progress()
        # Should not crash


class TestWorkstationState:
    """Tests for WorkstationState dataclass."""

    def test_state_creation(self):
        """Test creating workstation state."""
        state = WorkstationState()
        assert len(state.session_id) == 8
        assert state.proposals == []
        assert state.phases == []
        assert state.active_agents == []
        assert state.current_phase == 1

    def test_state_with_data(self):
        """Test state with proposals and phases."""
        proposals = [
            Proposal(
                id="p1",
                agent=AIAgent.CLAUDE,
                title="Test",
                description="Desc",
                category=ProposalCategory.FEATURE_NEW,
            )
        ]
        phases = [
            Phase(id=1, name="Phase 1", description="First")
        ]
        state = WorkstationState(
            proposals=proposals,
            phases=phases,
            active_agents=[AIAgent.CLAUDE, AIAgent.CHATGPT],
        )
        assert len(state.proposals) == 1
        assert len(state.phases) == 1
        assert len(state.active_agents) == 2

    def test_state_to_dict(self):
        """Test serializing state to dictionary."""
        state = WorkstationState(
            current_phase=2,
            active_agents=[AIAgent.CLAUDE],
        )
        data = state.to_dict()

        assert data["current_phase"] == 2
        assert data["active_agents"] == ["claude"]
        assert "session_id" in data

    def test_state_from_dict(self):
        """Test deserializing state from dictionary."""
        data = {
            "proposals": [],
            "phases": [],
            "active_agents": ["claude", "chatgpt"],
            "current_phase": 3,
            "session_id": "test123",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
        state = WorkstationState.from_dict(data)

        assert state.current_phase == 3
        assert state.session_id == "test123"
        assert AIAgent.CLAUDE in state.active_agents
        assert AIAgent.CHATGPT in state.active_agents

    def test_state_save_and_load(self):
        """Test saving and loading state to/from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Create and save
            original = WorkstationState(
                current_phase=2,
                active_agents=[AIAgent.CLAUDE],
            )
            original.save(temp_path)

            # Load and verify
            loaded = WorkstationState.load(temp_path)
            assert loaded.current_phase == 2
            assert AIAgent.CLAUDE in loaded.active_agents
        finally:
            Path(temp_path).unlink()

    def test_state_roundtrip(self):
        """Test complete roundtrip of state with complex data."""
        proposals = [
            Proposal(
                id="p1",
                agent=AIAgent.CLAUDE,
                title="Test Proposal",
                description="Description",
                category=ProposalCategory.ARCHITECTURE,
                votes={"chatgpt": 1},
            )
        ]
        tasks = [
            PhaseTask(id="t1", name="Task 1", description="Desc")
        ]
        phases = [
            Phase(id=1, name="Phase 1", description="First", tasks=tasks)
        ]

        original = WorkstationState(
            proposals=proposals,
            phases=phases,
            active_agents=[AIAgent.CLAUDE, AIAgent.GEMINI],
            current_phase=1,
        )

        data = original.to_dict()
        restored = WorkstationState.from_dict(data)

        assert len(restored.proposals) == 1
        assert restored.proposals[0].title == "Test Proposal"
        assert len(restored.phases) == 1
        assert len(restored.phases[0].tasks) == 1


class TestProposalEdgeCases:
    """Edge case tests for Proposal models."""

    def test_proposal_empty_description(self):
        """Test proposal with empty description."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="No Description",
            description="",
            category=ProposalCategory.FEATURE_NEW,
        )
        assert proposal.description == ""

    def test_proposal_many_votes(self):
        """Test proposal with votes from all agents."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="Test",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        for agent in AIAgent:
            if agent != AIAgent.CLAUDE:
                proposal.add_vote(agent, 1)

        assert proposal.vote_score == 3  # 3 other agents

    def test_proposal_all_rejections(self):
        """Test proposal with all rejections."""
        proposal = Proposal(
            id="",
            agent=AIAgent.CLAUDE,
            title="Test",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        for agent in AIAgent:
            if agent != AIAgent.CLAUDE:
                proposal.add_vote(agent, -1)

        assert proposal.vote_score == -3

    def test_proposal_dependencies(self):
        """Test proposal with dependencies."""
        proposal = Proposal(
            id="p1",
            agent=AIAgent.CLAUDE,
            title="Dependent",
            description="Needs others",
            category=ProposalCategory.FEATURE_NEW,
            dependencies=["p2", "p3"],
        )
        assert len(proposal.dependencies) == 2


class TestPhaseEdgeCases:
    """Edge case tests for Phase models."""

    def test_phase_many_tasks(self):
        """Test phase with many tasks."""
        tasks = [
            PhaseTask(id=f"t{i}", name=f"Task {i}", description="")
            for i in range(100)
        ]
        phase = Phase(id=1, name="Many Tasks", description="", tasks=tasks)
        assert len(phase.tasks) == 100

    def test_phase_mixed_task_statuses(self):
        """Test phase progress with mixed task statuses."""
        tasks = [
            PhaseTask(id="t1", name="T1", description="", status=PhaseStatus.COMPLETED),
            PhaseTask(id="t2", name="T2", description="", status=PhaseStatus.IN_PROGRESS),
            PhaseTask(id="t3", name="T3", description="", status=PhaseStatus.BLOCKED),
            PhaseTask(id="t4", name="T4", description="", status=PhaseStatus.NOT_STARTED),
        ]
        phase = Phase(id=1, name="Mixed", description="", tasks=tasks)
        phase.update_progress()

        # Only completed tasks count
        assert phase.progress == 0.25
