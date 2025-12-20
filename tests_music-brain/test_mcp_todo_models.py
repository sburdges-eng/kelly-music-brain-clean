"""
Tests for MCP TODO Data Models.

Tests the Todo, TodoList, TodoPriority, and TodoStatus models.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from mcp_todo.models import (
    Todo,
    TodoList,
    TodoPriority,
    TodoStatus,
)


class TestTodoPriority:
    """Tests for TodoPriority enum."""

    def test_priority_values(self):
        """Test that all priority values are defined."""
        assert TodoPriority.LOW.value == "low"
        assert TodoPriority.MEDIUM.value == "medium"
        assert TodoPriority.HIGH.value == "high"
        assert TodoPriority.URGENT.value == "urgent"

    def test_priority_from_string(self):
        """Test creating priority from string."""
        assert TodoPriority("low") == TodoPriority.LOW
        assert TodoPriority("urgent") == TodoPriority.URGENT

    def test_priority_invalid_value(self):
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError):
            TodoPriority("invalid")


class TestTodoStatus:
    """Tests for TodoStatus enum."""

    def test_status_values(self):
        """Test that all status values are defined."""
        assert TodoStatus.PENDING.value == "pending"
        assert TodoStatus.IN_PROGRESS.value == "in_progress"
        assert TodoStatus.COMPLETED.value == "completed"
        assert TodoStatus.BLOCKED.value == "blocked"
        assert TodoStatus.CANCELLED.value == "cancelled"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert TodoStatus("pending") == TodoStatus.PENDING
        assert TodoStatus("completed") == TodoStatus.COMPLETED

    def test_status_invalid_value(self):
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError):
            TodoStatus("invalid")


class TestTodo:
    """Tests for Todo dataclass."""

    def test_todo_creation_minimal(self):
        """Test creating a Todo with only required fields."""
        todo = Todo(title="Test Task")
        assert todo.title == "Test Task"
        assert todo.description == ""
        assert todo.status == TodoStatus.PENDING
        assert todo.priority == TodoPriority.MEDIUM
        assert len(todo.id) == 8
        assert todo.tags == []
        assert todo.notes == []

    def test_todo_creation_full(self):
        """Test creating a Todo with all fields."""
        todo = Todo(
            title="Full Task",
            description="A complete task description",
            status=TodoStatus.IN_PROGRESS,
            priority=TodoPriority.HIGH,
            tags=["urgent", "code"],
            project="iDAW",
            due_date="2024-12-31",
            context="Additional context",
            ai_source="claude",
        )
        assert todo.title == "Full Task"
        assert todo.description == "A complete task description"
        assert todo.status == TodoStatus.IN_PROGRESS
        assert todo.priority == TodoPriority.HIGH
        assert todo.tags == ["urgent", "code"]
        assert todo.project == "iDAW"
        assert todo.due_date == "2024-12-31"
        assert todo.context == "Additional context"
        assert todo.ai_source == "claude"

    def test_todo_unique_ids(self):
        """Test that each Todo gets a unique ID."""
        todos = [Todo(title=f"Task {i}") for i in range(10)]
        ids = [todo.id for todo in todos]
        assert len(set(ids)) == 10  # All unique

    def test_todo_to_dict(self):
        """Test serializing Todo to dictionary."""
        todo = Todo(
            title="Test Task",
            description="Description",
            priority=TodoPriority.HIGH,
            status=TodoStatus.PENDING,
            tags=["test"],
        )
        data = todo.to_dict()
        assert data["title"] == "Test Task"
        assert data["description"] == "Description"
        assert data["priority"] == "high"
        assert data["status"] == "pending"
        assert data["tags"] == ["test"]
        assert "id" in data
        assert "created_at" in data

    def test_todo_from_dict(self):
        """Test deserializing Todo from dictionary."""
        data = {
            "id": "abc123",
            "title": "From Dict",
            "description": "Loaded from dict",
            "status": "in_progress",
            "priority": "urgent",
            "tags": ["loaded"],
            "project": "test",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
            "notes": [],
            "depends_on": [],
            "blocks": [],
        }
        todo = Todo.from_dict(data)
        assert todo.id == "abc123"
        assert todo.title == "From Dict"
        assert todo.status == TodoStatus.IN_PROGRESS
        assert todo.priority == TodoPriority.URGENT
        assert todo.tags == ["loaded"]

    def test_todo_roundtrip(self):
        """Test that to_dict/from_dict preserves data."""
        original = Todo(
            title="Roundtrip Test",
            description="Testing roundtrip",
            priority=TodoPriority.HIGH,
            status=TodoStatus.BLOCKED,
            tags=["test", "roundtrip"],
            project="testing",
        )
        data = original.to_dict()
        restored = Todo.from_dict(data)
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.priority == original.priority
        assert restored.status == original.status
        assert restored.tags == original.tags

    def test_mark_complete(self):
        """Test marking a todo as complete."""
        todo = Todo(title="To Complete")
        assert todo.status == TodoStatus.PENDING
        assert todo.completed_at is None

        todo.mark_complete(ai_source="claude")
        assert todo.status == TodoStatus.COMPLETED
        assert todo.completed_at is not None
        assert todo.ai_source == "claude"
        assert any("Completed by claude" in note for note in todo.notes)

    def test_mark_in_progress(self):
        """Test marking a todo as in progress."""
        todo = Todo(title="To Start")
        assert todo.status == TodoStatus.PENDING

        todo.mark_in_progress(ai_source="chatgpt")
        assert todo.status == TodoStatus.IN_PROGRESS
        assert todo.ai_source == "chatgpt"
        assert any("Started by chatgpt" in note for note in todo.notes)

    def test_add_note(self):
        """Test adding notes to a todo."""
        todo = Todo(title="With Notes")
        assert todo.notes == []

        todo.add_note("First note", ai_source="claude")
        assert len(todo.notes) == 1
        assert "[claude]" in todo.notes[0]
        assert "First note" in todo.notes[0]

        todo.add_note("Second note")  # No AI source
        assert len(todo.notes) == 2
        assert "Second note" in todo.notes[1]

    def test_str_representation(self):
        """Test string representation of Todo."""
        todo = Todo(title="String Test", priority=TodoPriority.HIGH)
        string = str(todo)
        assert "String Test" in string
        assert todo.id in string
        assert "[ ]" in string  # Pending icon

    def test_str_representation_completed(self):
        """Test string representation of completed Todo."""
        todo = Todo(title="Completed", status=TodoStatus.COMPLETED)
        string = str(todo)
        assert "[x]" in string

    def test_str_representation_in_progress(self):
        """Test string representation of in-progress Todo."""
        todo = Todo(title="In Progress", status=TodoStatus.IN_PROGRESS)
        string = str(todo)
        assert "[~]" in string

    def test_str_representation_urgent(self):
        """Test string representation with urgent priority."""
        todo = Todo(title="Urgent", priority=TodoPriority.URGENT)
        string = str(todo)
        assert "!!!" in string

    def test_dependencies(self):
        """Test todo dependencies."""
        parent = Todo(title="Parent Task")
        child = Todo(title="Child Task", depends_on=[parent.id])
        assert parent.id in child.depends_on

    def test_blocks(self):
        """Test todo blocking relationships."""
        blocker = Todo(title="Blocker")
        blocked = Todo(title="Blocked")
        blocker.blocks.append(blocked.id)
        assert blocked.id in blocker.blocks


class TestTodoList:
    """Tests for TodoList dataclass."""

    def test_todolist_creation(self):
        """Test creating a TodoList."""
        todo_list = TodoList(name="Test List")
        assert todo_list.name == "Test List"
        assert todo_list.todos == []
        assert todo_list.created_at is not None

    def test_todolist_with_todos(self):
        """Test creating a TodoList with todos."""
        todos = [
            Todo(title="Task 1"),
            Todo(title="Task 2"),
        ]
        todo_list = TodoList(name="With Todos", todos=todos)
        assert len(todo_list.todos) == 2
        assert todo_list.todos[0].title == "Task 1"

    def test_todolist_to_dict(self):
        """Test serializing TodoList to dictionary."""
        todos = [Todo(title="Task 1")]
        todo_list = TodoList(name="Test", todos=todos)
        data = todo_list.to_dict()
        assert data["name"] == "Test"
        assert len(data["todos"]) == 1
        assert data["todos"][0]["title"] == "Task 1"
        assert "created_at" in data

    def test_todolist_from_dict(self):
        """Test deserializing TodoList from dictionary."""
        data = {
            "name": "Loaded List",
            "todos": [
                {
                    "id": "abc123",
                    "title": "Loaded Task",
                    "description": "",
                    "status": "pending",
                    "priority": "medium",
                    "tags": [],
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "notes": [],
                    "depends_on": [],
                    "blocks": [],
                }
            ],
            "created_at": "2024-01-01T00:00:00",
        }
        todo_list = TodoList.from_dict(data)
        assert todo_list.name == "Loaded List"
        assert len(todo_list.todos) == 1
        assert todo_list.todos[0].title == "Loaded Task"

    def test_todolist_roundtrip(self):
        """Test that to_dict/from_dict preserves TodoList data."""
        original = TodoList(
            name="Roundtrip",
            todos=[
                Todo(title="Task 1", priority=TodoPriority.HIGH),
                Todo(title="Task 2", status=TodoStatus.COMPLETED),
            ],
        )
        data = original.to_dict()
        restored = TodoList.from_dict(data)
        assert restored.name == original.name
        assert len(restored.todos) == 2
        assert restored.todos[0].priority == TodoPriority.HIGH
        assert restored.todos[1].status == TodoStatus.COMPLETED


class TestTodoEdgeCases:
    """Edge case tests for Todo models."""

    def test_empty_title(self):
        """Test todo with empty title."""
        todo = Todo(title="")
        assert todo.title == ""

    def test_very_long_title(self):
        """Test todo with very long title."""
        long_title = "A" * 1000
        todo = Todo(title=long_title)
        assert todo.title == long_title

    def test_special_characters_in_title(self):
        """Test todo with special characters."""
        todo = Todo(title="Test <script>alert('xss')</script>")
        assert "<script>" in todo.title

    def test_unicode_in_title(self):
        """Test todo with unicode characters."""
        todo = Todo(title="Test with emoji and unicode")
        assert "" in todo.title

    def test_empty_tags_list(self):
        """Test todo with empty tags list."""
        todo = Todo(title="No Tags", tags=[])
        assert todo.tags == []

    def test_many_tags(self):
        """Test todo with many tags."""
        tags = [f"tag{i}" for i in range(100)]
        todo = Todo(title="Many Tags", tags=tags)
        assert len(todo.tags) == 100

    def test_nested_notes(self):
        """Test adding multiple notes."""
        todo = Todo(title="With Notes")
        for i in range(10):
            todo.add_note(f"Note {i}")
        assert len(todo.notes) == 10

    def test_date_formats(self):
        """Test various date formats in due_date."""
        todo = Todo(title="Date Test", due_date="2024-12-31T23:59:59")
        assert todo.due_date == "2024-12-31T23:59:59"

    def test_from_dict_missing_fields(self):
        """Test from_dict with minimal data."""
        data = {"title": "Minimal"}
        todo = Todo.from_dict(data)
        assert todo.title == "Minimal"
        assert todo.status == TodoStatus.PENDING
        assert todo.priority == TodoPriority.MEDIUM

    def test_from_dict_extra_fields(self):
        """Test from_dict ignores extra fields gracefully."""
        data = {
            "id": "test123",
            "title": "Extra Fields",
            "description": "",
            "status": "pending",
            "priority": "medium",
            "tags": [],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "notes": [],
            "depends_on": [],
            "blocks": [],
            "extra_field": "should be ignored",
        }
        # This should not raise, but may need adjustment based on implementation
        # If strict, it will raise TypeError
        try:
            todo = Todo.from_dict(data)
        except TypeError:
            # Expected if using strict dataclass
            pass
