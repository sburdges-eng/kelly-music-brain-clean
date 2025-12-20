"""
Tests for MCP Workstation Phase Management.

Tests the PhaseManager class and phase-related utilities.
"""

import pytest
from unittest.mock import patch, MagicMock

from mcp_workstation.models import Phase, PhaseTask, PhaseStatus, AIAgent
from mcp_workstation.phases import (
    PhaseManager,
    IDAW_PHASES,
    format_phase_progress,
    get_next_actions,
)


@pytest.fixture
def manager():
    """Create a PhaseManager with default phases."""
    return PhaseManager()


@pytest.fixture
def custom_phases():
    """Create custom phases for testing."""
    return [
        Phase(
            id=1,
            name="Phase 1",
            description="First phase",
            status=PhaseStatus.IN_PROGRESS,
            progress=0.5,
            tasks=[
                PhaseTask(id="t1", name="Task 1", description="First task", status=PhaseStatus.COMPLETED),
                PhaseTask(id="t2", name="Task 2", description="Second task", status=PhaseStatus.IN_PROGRESS),
            ],
        ),
        Phase(
            id=2,
            name="Phase 2",
            description="Second phase",
            status=PhaseStatus.NOT_STARTED,
            tasks=[
                PhaseTask(id="t3", name="Task 3", description="Third task"),
            ],
        ),
    ]


@pytest.fixture
def custom_manager(custom_phases):
    """Create a PhaseManager with custom phases."""
    return PhaseManager(phases=custom_phases)


class TestIDAWPhases:
    """Tests for the predefined IDAW phases."""

    def test_idaw_phases_defined(self):
        """Test that IDAW phases are defined."""
        assert len(IDAW_PHASES) == 3

    def test_phase_1_core_systems(self):
        """Test Phase 1 - Core Systems."""
        phase1 = IDAW_PHASES[0]
        assert phase1.id == 1
        assert phase1.name == "Core Systems"
        assert phase1.status == PhaseStatus.IN_PROGRESS
        assert len(phase1.tasks) > 0

    def test_phase_2_expansion(self):
        """Test Phase 2 - Expansion & Integration."""
        phase2 = IDAW_PHASES[1]
        assert phase2.id == 2
        assert "Expansion" in phase2.name
        assert phase2.status == PhaseStatus.NOT_STARTED

    def test_phase_3_advanced(self):
        """Test Phase 3 - Advanced Features & C++ Transition."""
        phase3 = IDAW_PHASES[2]
        assert phase3.id == 3
        assert "C++" in phase3.name
        assert phase3.status == PhaseStatus.NOT_STARTED


class TestPhaseManagerInit:
    """Tests for PhaseManager initialization."""

    def test_default_init(self, manager):
        """Test default initialization with IDAW phases."""
        assert len(manager.phases) == 3
        assert manager.current_phase_id == 1

    def test_custom_phases_init(self, custom_manager, custom_phases):
        """Test initialization with custom phases."""
        assert len(custom_manager.phases) == 2

    def test_finds_current_phase(self, custom_manager):
        """Test that current phase is found based on status."""
        assert custom_manager.current_phase_id == 1

    def test_no_in_progress_phase(self):
        """Test initialization when no phase is in progress."""
        phases = [
            Phase(id=1, name="P1", description="", status=PhaseStatus.NOT_STARTED),
        ]
        manager = PhaseManager(phases=phases)
        # Should default to first phase
        assert manager.current_phase_id == 1


class TestPhaseManagerGetPhase:
    """Tests for getting phases."""

    def test_get_current_phase(self, custom_manager):
        """Test getting the current phase."""
        current = custom_manager.get_current_phase()
        assert current.id == 1
        assert current.status == PhaseStatus.IN_PROGRESS

    def test_get_phase_by_id(self, custom_manager):
        """Test getting a phase by ID."""
        phase = custom_manager.get_phase(2)
        assert phase.id == 2
        assert phase.name == "Phase 2"

    def test_get_nonexistent_phase(self, custom_manager):
        """Test getting a non-existent phase returns None."""
        phase = custom_manager.get_phase(999)
        assert phase is None


class TestPhaseManagerUpdateTask:
    """Tests for updating task status."""

    def test_update_task_status(self, custom_manager):
        """Test updating a task's status."""
        custom_manager.update_task_status(
            phase_id=1,
            task_id="t2",
            status=PhaseStatus.COMPLETED,
        )
        phase = custom_manager.get_phase(1)
        task = next(t for t in phase.tasks if t.id == "t2")
        assert task.status == PhaseStatus.COMPLETED

    def test_update_task_progress(self, custom_manager):
        """Test updating a task's progress."""
        custom_manager.update_task_status(
            phase_id=1,
            task_id="t2",
            status=PhaseStatus.IN_PROGRESS,
            progress=0.75,
        )
        phase = custom_manager.get_phase(1)
        task = next(t for t in phase.tasks if t.id == "t2")
        assert task.progress == 0.75

    def test_update_task_notes(self, custom_manager):
        """Test updating a task's notes."""
        custom_manager.update_task_status(
            phase_id=1,
            task_id="t2",
            status=PhaseStatus.IN_PROGRESS,
            notes="Working on it",
        )
        phase = custom_manager.get_phase(1)
        task = next(t for t in phase.tasks if t.id == "t2")
        assert task.notes == "Working on it"

    def test_update_task_completed_sets_timestamp(self, custom_manager):
        """Test that completing a task sets completed_at."""
        custom_manager.update_task_status(
            phase_id=1,
            task_id="t2",
            status=PhaseStatus.COMPLETED,
        )
        phase = custom_manager.get_phase(1)
        task = next(t for t in phase.tasks if t.id == "t2")
        assert task.completed_at is not None
        assert task.progress == 1.0

    def test_update_task_updates_phase_progress(self, custom_manager):
        """Test that updating task updates phase progress."""
        custom_manager.update_task_status(
            phase_id=1,
            task_id="t2",
            status=PhaseStatus.COMPLETED,
        )
        phase = custom_manager.get_phase(1)
        assert phase.progress == 1.0  # Both tasks now completed

    def test_update_task_invalid_phase(self, custom_manager):
        """Test updating task in invalid phase."""
        # Should not raise, just log warning
        custom_manager.update_task_status(
            phase_id=999,
            task_id="t1",
            status=PhaseStatus.COMPLETED,
        )


class TestPhaseManagerAssignTask:
    """Tests for assigning tasks to agents."""

    def test_assign_task(self, custom_manager):
        """Test assigning a task to an agent."""
        custom_manager.assign_task(
            phase_id=1,
            task_id="t2",
            agent=AIAgent.CLAUDE,
        )
        phase = custom_manager.get_phase(1)
        task = next(t for t in phase.tasks if t.id == "t2")
        assert task.assigned_to == AIAgent.CLAUDE

    def test_assign_task_starts_task(self, custom_manager):
        """Test that assigning a not-started task starts it."""
        custom_manager.assign_task(
            phase_id=2,
            task_id="t3",
            agent=AIAgent.GEMINI,
        )
        phase = custom_manager.get_phase(2)
        task = next(t for t in phase.tasks if t.id == "t3")
        assert task.status == PhaseStatus.IN_PROGRESS

    def test_assign_task_invalid_phase(self, custom_manager):
        """Test assigning task in invalid phase."""
        # Should not raise
        custom_manager.assign_task(
            phase_id=999,
            task_id="t1",
            agent=AIAgent.CLAUDE,
        )


class TestPhaseManagerAdvance:
    """Tests for advancing phases."""

    def test_advance_phase_when_complete(self):
        """Test advancing to next phase when current is complete."""
        phases = [
            Phase(
                id=1,
                name="Phase 1",
                description="",
                status=PhaseStatus.IN_PROGRESS,
                progress=1.0,
                tasks=[
                    PhaseTask(id="t1", name="T1", description="", status=PhaseStatus.COMPLETED),
                ],
            ),
            Phase(
                id=2,
                name="Phase 2",
                description="",
                status=PhaseStatus.NOT_STARTED,
            ),
        ]
        manager = PhaseManager(phases=phases)

        result = manager.advance_phase()
        assert result is True
        assert manager.current_phase_id == 2
        assert manager.get_phase(1).status == PhaseStatus.VERIFIED
        assert manager.get_phase(2).status == PhaseStatus.IN_PROGRESS

    def test_advance_phase_when_incomplete(self, custom_manager):
        """Test that advance fails when current phase is incomplete."""
        result = custom_manager.advance_phase()
        assert result is False
        assert custom_manager.current_phase_id == 1

    def test_advance_phase_no_next(self):
        """Test advancing when on last phase."""
        phases = [
            Phase(
                id=1,
                name="Last Phase",
                description="",
                status=PhaseStatus.IN_PROGRESS,
                progress=1.0,
            ),
        ]
        manager = PhaseManager(phases=phases)

        result = manager.advance_phase()
        assert result is False


class TestPhaseManagerQueries:
    """Tests for query methods."""

    def test_get_incomplete_tasks(self, custom_manager):
        """Test getting incomplete tasks."""
        tasks = custom_manager.get_incomplete_tasks()
        # t2 is in_progress, t3 is not_started
        assert len(tasks) >= 2

    def test_get_incomplete_tasks_by_phase(self, custom_manager):
        """Test getting incomplete tasks for specific phase."""
        tasks = custom_manager.get_incomplete_tasks(phase_id=1)
        assert len(tasks) == 1  # Only t2

    def test_get_blocked_tasks(self):
        """Test getting blocked tasks."""
        phases = [
            Phase(
                id=1,
                name="P1",
                description="",
                tasks=[
                    PhaseTask(id="t1", name="T1", description="", status=PhaseStatus.BLOCKED),
                    PhaseTask(id="t2", name="T2", description="", blockers=["API down"]),
                ],
            ),
        ]
        manager = PhaseManager(phases=phases)
        blocked = manager.get_blocked_tasks()
        assert len(blocked) == 2

    def test_get_phase_summary(self, custom_manager):
        """Test getting phase summary."""
        summary = custom_manager.get_phase_summary()

        assert "current_phase" in summary
        assert "phases" in summary
        assert "overall_progress" in summary
        assert len(summary["phases"]) == 2

    def test_get_phase_summary_single(self, custom_manager):
        """Test getting summary for single phase."""
        summary = custom_manager.get_phase_summary(phase_id=1)

        assert summary["current_phase"] == 1
        assert len(summary["phases"]) == 1
        assert summary["phases"][0]["phase_id"] == 1


class TestPhaseManagerSerialization:
    """Tests for serialization."""

    def test_to_dict(self, custom_manager):
        """Test serializing manager to dictionary."""
        data = custom_manager.to_dict()

        assert "phases" in data
        assert "current_phase_id" in data
        assert len(data["phases"]) == 2

    def test_from_dict(self):
        """Test deserializing manager from dictionary."""
        data = {
            "phases": [
                {
                    "id": 1,
                    "name": "Phase 1",
                    "description": "First",
                    "status": "in_progress",
                    "progress": 0.5,
                    "tasks": [],
                    "milestones": [],
                    "deliverables": [],
                },
            ],
            "current_phase_id": 1,
        }
        manager = PhaseManager.from_dict(data)

        assert len(manager.phases) == 1
        assert manager.current_phase_id == 1

    def test_roundtrip(self, custom_manager):
        """Test serialization roundtrip."""
        data = custom_manager.to_dict()
        restored = PhaseManager.from_dict(data)

        assert len(restored.phases) == len(custom_manager.phases)
        assert restored.current_phase_id == custom_manager.current_phase_id


class TestFormatPhaseProgress:
    """Tests for format_phase_progress function."""

    def test_format_includes_phases(self, custom_manager):
        """Test that format includes all phases."""
        output = format_phase_progress(custom_manager)

        assert "Phase 1" in output
        assert "Phase 2" in output

    def test_format_includes_progress(self, custom_manager):
        """Test that format includes progress bars."""
        output = format_phase_progress(custom_manager)

        # Should contain block characters for progress
        assert "" in output or "" in output

    def test_format_includes_tasks(self, custom_manager):
        """Test that format includes tasks."""
        output = format_phase_progress(custom_manager)

        assert "Task 1" in output
        assert "Task 2" in output

    def test_format_includes_status_icons(self, custom_manager):
        """Test that format includes status icons."""
        output = format_phase_progress(custom_manager)

        # Should contain various status indicators
        assert any(c in output for c in ["", "", "", "", ""])


class TestGetNextActions:
    """Tests for get_next_actions function."""

    def test_next_actions_includes_continue(self, custom_manager):
        """Test that in-progress tasks are listed."""
        actions = get_next_actions(custom_manager)

        assert any("CONTINUE" in a or "Task 2" in a for a in actions)

    def test_next_actions_includes_start(self):
        """Test that not-started tasks are suggested."""
        phases = [
            Phase(
                id=1,
                name="P1",
                description="",
                status=PhaseStatus.IN_PROGRESS,
                tasks=[
                    PhaseTask(id="t1", name="New Task", description="", status=PhaseStatus.NOT_STARTED),
                ],
            ),
        ]
        manager = PhaseManager(phases=phases)
        actions = get_next_actions(manager)

        assert any("START" in a for a in actions)

    def test_next_actions_includes_unblock(self):
        """Test that blocked tasks are flagged."""
        phases = [
            Phase(
                id=1,
                name="P1",
                description="",
                status=PhaseStatus.IN_PROGRESS,
                tasks=[
                    PhaseTask(
                        id="t1",
                        name="Blocked",
                        description="",
                        status=PhaseStatus.BLOCKED,
                        blockers=["API issue"],
                    ),
                ],
            ),
        ]
        manager = PhaseManager(phases=phases)
        actions = get_next_actions(manager)

        assert any("UNBLOCK" in a for a in actions)

    def test_next_actions_includes_advance(self):
        """Test that phase advance is suggested when ready."""
        phases = [
            Phase(
                id=1,
                name="P1",
                description="",
                status=PhaseStatus.IN_PROGRESS,
                progress=1.0,
                tasks=[
                    PhaseTask(id="t1", name="Done", description="", status=PhaseStatus.COMPLETED),
                ],
            ),
            Phase(id=2, name="P2", description="", status=PhaseStatus.NOT_STARTED),
        ]
        manager = PhaseManager(phases=phases)
        actions = get_next_actions(manager)

        assert any("READY" in a or "Advance" in a for a in actions)


class TestPhaseManagerIntegration:
    """Integration tests for PhaseManager."""

    def test_full_workflow(self):
        """Test a complete phase workflow."""
        phases = [
            Phase(
                id=1,
                name="Phase 1",
                description="First",
                status=PhaseStatus.NOT_STARTED,
                tasks=[
                    PhaseTask(id="t1", name="Task 1", description=""),
                    PhaseTask(id="t2", name="Task 2", description=""),
                ],
            ),
            Phase(
                id=2,
                name="Phase 2",
                description="Second",
                status=PhaseStatus.NOT_STARTED,
            ),
        ]
        manager = PhaseManager(phases=phases)

        # Start working
        manager.assign_task(1, "t1", AIAgent.CLAUDE)
        assert manager.get_phase(1).tasks[0].status == PhaseStatus.IN_PROGRESS

        # Complete tasks
        manager.update_task_status(1, "t1", PhaseStatus.COMPLETED)
        manager.update_task_status(1, "t2", PhaseStatus.COMPLETED)

        # Verify phase completed
        assert manager.get_phase(1).progress == 1.0

        # Advance
        result = manager.advance_phase()
        assert result is True
        assert manager.current_phase_id == 2

    def test_multi_agent_workflow(self):
        """Test multiple agents working on tasks."""
        phases = [
            Phase(
                id=1,
                name="Phase 1",
                description="",
                status=PhaseStatus.IN_PROGRESS,
                tasks=[
                    PhaseTask(id="t1", name="Task 1", description=""),
                    PhaseTask(id="t2", name="Task 2", description=""),
                    PhaseTask(id="t3", name="Task 3", description=""),
                ],
            ),
        ]
        manager = PhaseManager(phases=phases)

        # Assign different agents
        manager.assign_task(1, "t1", AIAgent.CLAUDE)
        manager.assign_task(1, "t2", AIAgent.CHATGPT)
        manager.assign_task(1, "t3", AIAgent.GEMINI)

        phase = manager.get_phase(1)
        assignees = [t.assigned_to for t in phase.tasks]

        assert AIAgent.CLAUDE in assignees
        assert AIAgent.CHATGPT in assignees
        assert AIAgent.GEMINI in assignees
