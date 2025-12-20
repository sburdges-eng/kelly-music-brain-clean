"""
Tests for MCP TODO Storage Backend.

Tests the TodoStorage class with JSON file-based persistence.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from mcp_todo.storage import TodoStorage
from mcp_todo.models import Todo, TodoPriority, TodoStatus


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def storage(temp_storage_dir):
    """Create a TodoStorage instance with temporary directory."""
    return TodoStorage(storage_dir=temp_storage_dir)


class TestTodoStorageInit:
    """Tests for TodoStorage initialization."""

    def test_storage_creates_directory(self, temp_storage_dir):
        """Test that storage creates directory if it doesn't exist."""
        new_dir = Path(temp_storage_dir) / "new_storage"
        storage = TodoStorage(storage_dir=str(new_dir))
        assert new_dir.exists()

    def test_storage_creates_default_file(self, storage, temp_storage_dir):
        """Test that storage creates default file on init."""
        default_file = Path(temp_storage_dir) / "todos.json"
        assert default_file.exists()

    def test_storage_default_file_structure(self, storage, temp_storage_dir):
        """Test default file has correct structure."""
        default_file = Path(temp_storage_dir) / "todos.json"
        with open(default_file) as f:
            data = json.load(f)
        assert "lists" in data
        assert "default" in data["lists"]

    def test_storage_preserves_existing_data(self, temp_storage_dir):
        """Test storage doesn't overwrite existing data."""
        file_path = Path(temp_storage_dir) / "todos.json"
        existing_data = {
            "lists": {
                "default": {
                    "name": "default",
                    "todos": [{"id": "existing", "title": "Existing Task"}]
                }
            }
        }
        with open(file_path, "w") as f:
            json.dump(existing_data, f)

        storage = TodoStorage(storage_dir=temp_storage_dir)
        data = storage._load_data()
        assert len(data["lists"]["default"]["todos"]) == 1


class TestTodoStorageAdd:
    """Tests for adding todos."""

    def test_add_basic_todo(self, storage):
        """Test adding a basic todo."""
        todo = storage.add(title="Test Task")
        assert todo.title == "Test Task"
        assert todo.id is not None
        assert len(todo.id) == 8

    def test_add_full_todo(self, storage):
        """Test adding a todo with all fields."""
        todo = storage.add(
            title="Full Task",
            description="A complete description",
            priority="high",
            tags=["test", "urgent"],
            project="iDAW",
            due_date="2024-12-31",
            context="Test context",
            ai_source="claude",
        )
        assert todo.title == "Full Task"
        assert todo.description == "A complete description"
        assert todo.priority == TodoPriority.HIGH
        assert todo.tags == ["test", "urgent"]
        assert todo.project == "iDAW"
        assert todo.due_date == "2024-12-31"
        assert todo.context == "Test context"
        assert todo.ai_source == "claude"

    def test_add_persists_to_file(self, storage, temp_storage_dir):
        """Test that added todos are persisted to file."""
        storage.add(title="Persistent Task")

        # Reload from file
        with open(Path(temp_storage_dir) / "todos.json") as f:
            data = json.load(f)

        todos = data["lists"]["default"]["todos"]
        assert len(todos) == 1
        assert todos[0]["title"] == "Persistent Task"

    def test_add_multiple_todos(self, storage):
        """Test adding multiple todos."""
        storage.add(title="Task 1")
        storage.add(title="Task 2")
        storage.add(title="Task 3")

        todos = storage.list_all()
        assert len(todos) == 3

    def test_add_to_project(self, storage):
        """Test adding todos to different projects."""
        storage.add(title="Default Task")
        storage.add(title="Project Task", project="iDAW")

        default_todos = storage.list_all(project="default")
        project_todos = storage.list_all(project="iDAW")

        assert len(default_todos) == 1
        assert len(project_todos) == 1


class TestTodoStorageGet:
    """Tests for getting todos."""

    def test_get_existing_todo(self, storage):
        """Test getting an existing todo by ID."""
        created = storage.add(title="Retrievable")
        retrieved = storage.get(created.id)
        assert retrieved is not None
        assert retrieved.title == "Retrievable"

    def test_get_nonexistent_todo(self, storage):
        """Test getting a non-existent todo returns None."""
        result = storage.get("nonexistent")
        assert result is None

    def test_get_from_project(self, storage):
        """Test getting todo from specific project."""
        created = storage.add(title="Project Task", project="myproject")
        retrieved = storage.get(created.id, project="myproject")
        assert retrieved is not None
        assert retrieved.title == "Project Task"


class TestTodoStorageUpdate:
    """Tests for updating todos."""

    def test_update_title(self, storage):
        """Test updating a todo's title."""
        todo = storage.add(title="Original")
        updated = storage.update(todo.id, title="Updated")
        assert updated.title == "Updated"

    def test_update_status(self, storage):
        """Test updating a todo's status."""
        todo = storage.add(title="To Update")
        updated = storage.update(todo.id, status="in_progress")
        assert updated.status == TodoStatus.IN_PROGRESS

    def test_update_priority(self, storage):
        """Test updating a todo's priority."""
        todo = storage.add(title="Priority Change")
        updated = storage.update(todo.id, priority="urgent")
        assert updated.priority == TodoPriority.URGENT

    def test_update_multiple_fields(self, storage):
        """Test updating multiple fields at once."""
        todo = storage.add(title="Multi Update")
        updated = storage.update(
            todo.id,
            title="New Title",
            description="New Description",
            priority="high",
        )
        assert updated.title == "New Title"
        assert updated.description == "New Description"
        assert updated.priority == TodoPriority.HIGH

    def test_update_nonexistent_todo(self, storage):
        """Test updating a non-existent todo returns None."""
        result = storage.update("nonexistent", title="New")
        assert result is None

    def test_update_sets_updated_at(self, storage):
        """Test that update sets updated_at timestamp."""
        todo = storage.add(title="Timestamp Test")
        original_updated_at = todo.updated_at

        import time
        time.sleep(0.01)  # Small delay to ensure different timestamp

        updated = storage.update(todo.id, title="New Title")
        assert updated.updated_at != original_updated_at

    def test_update_completion_sets_completed_at(self, storage):
        """Test that completing a todo sets completed_at."""
        todo = storage.add(title="To Complete")
        assert todo.completed_at is None

        updated = storage.update(todo.id, status="completed")
        assert updated.completed_at is not None


class TestTodoStorageDelete:
    """Tests for deleting todos."""

    def test_delete_existing_todo(self, storage):
        """Test deleting an existing todo."""
        todo = storage.add(title="To Delete")
        assert storage.get(todo.id) is not None

        result = storage.delete(todo.id)
        assert result is True
        assert storage.get(todo.id) is None

    def test_delete_nonexistent_todo(self, storage):
        """Test deleting a non-existent todo returns False."""
        result = storage.delete("nonexistent")
        assert result is False

    def test_delete_from_project(self, storage):
        """Test deleting todo from specific project."""
        todo = storage.add(title="Project Delete", project="myproject")
        result = storage.delete(todo.id, project="myproject")
        assert result is True


class TestTodoStorageComplete:
    """Tests for completing todos."""

    def test_complete_todo(self, storage):
        """Test completing a todo."""
        todo = storage.add(title="To Complete")
        completed = storage.complete(todo.id)
        assert completed.status == TodoStatus.COMPLETED
        assert completed.completed_at is not None

    def test_complete_with_ai_source(self, storage):
        """Test completing a todo with AI source tracking."""
        todo = storage.add(title="AI Complete")
        completed = storage.complete(todo.id, ai_source="claude")
        assert completed.ai_source == "claude"


class TestTodoStorageStart:
    """Tests for starting todos."""

    def test_start_todo(self, storage):
        """Test starting a todo."""
        todo = storage.add(title="To Start")
        started = storage.start(todo.id)
        assert started.status == TodoStatus.IN_PROGRESS

    def test_start_with_ai_source(self, storage):
        """Test starting a todo with AI source tracking."""
        todo = storage.add(title="AI Start")
        started = storage.start(todo.id, ai_source="gemini")
        assert started.ai_source == "gemini"


class TestTodoStorageListAll:
    """Tests for listing todos."""

    def test_list_all_empty(self, storage):
        """Test listing when no todos exist."""
        todos = storage.list_all()
        assert len(todos) == 0

    def test_list_all_returns_all(self, storage):
        """Test list_all returns all todos."""
        storage.add(title="Task 1")
        storage.add(title="Task 2")
        storage.add(title="Task 3")

        todos = storage.list_all()
        assert len(todos) == 3

    def test_list_filter_by_status(self, storage):
        """Test filtering by status."""
        storage.add(title="Pending 1")
        storage.add(title="Pending 2")
        todo3 = storage.add(title="In Progress")
        storage.start(todo3.id)

        pending = storage.list_all(status="pending")
        in_progress = storage.list_all(status="in_progress")

        assert len(pending) == 2
        assert len(in_progress) == 1

    def test_list_filter_by_priority(self, storage):
        """Test filtering by priority."""
        storage.add(title="Low", priority="low")
        storage.add(title="High", priority="high")
        storage.add(title="High 2", priority="high")

        high = storage.list_all(priority="high")
        low = storage.list_all(priority="low")

        assert len(high) == 2
        assert len(low) == 1

    def test_list_filter_by_tags(self, storage):
        """Test filtering by tags."""
        storage.add(title="Tagged 1", tags=["urgent", "code"])
        storage.add(title="Tagged 2", tags=["urgent"])
        storage.add(title="Not Tagged", tags=["other"])

        urgent = storage.list_all(tags=["urgent"])
        code = storage.list_all(tags=["code"])

        assert len(urgent) == 2
        assert len(code) == 1

    def test_list_exclude_completed(self, storage):
        """Test excluding completed todos."""
        storage.add(title="Pending")
        todo2 = storage.add(title="Completed")
        storage.complete(todo2.id)

        all_todos = storage.list_all()
        active = storage.list_all(include_completed=False)

        assert len(all_todos) == 2
        assert len(active) == 1

    def test_list_filter_by_project(self, storage):
        """Test filtering by project."""
        storage.add(title="Default", project="default")
        storage.add(title="Project", project="iDAW")

        default = storage.list_all(project="default")
        idaw = storage.list_all(project="iDAW")

        assert len(default) == 1
        assert len(idaw) == 1


class TestTodoStorageSearch:
    """Tests for searching todos."""

    def test_search_by_title(self, storage):
        """Test searching by title."""
        storage.add(title="Fix bug in parser")
        storage.add(title="Add new feature")
        storage.add(title="Update documentation")

        results = storage.search("bug")
        assert len(results) == 1
        assert results[0].title == "Fix bug in parser"

    def test_search_by_description(self, storage):
        """Test searching by description."""
        storage.add(title="Task 1", description="This involves fixing bugs")
        storage.add(title="Task 2", description="This adds features")

        results = storage.search("bugs")
        assert len(results) == 1

    def test_search_case_insensitive(self, storage):
        """Test that search is case-insensitive."""
        storage.add(title="Fix BUG in parser")

        results = storage.search("bug")
        assert len(results) == 1

    def test_search_no_results(self, storage):
        """Test search with no results."""
        storage.add(title="Some task")

        results = storage.search("nonexistent")
        assert len(results) == 0


class TestTodoStorageGetByTags:
    """Tests for getting todos by tags."""

    def test_get_by_single_tag(self, storage):
        """Test getting todos by a single tag."""
        storage.add(title="Tagged", tags=["urgent"])
        storage.add(title="Not Tagged", tags=["other"])

        results = storage.get_by_tags(["urgent"])
        assert len(results) == 1

    def test_get_by_multiple_tags(self, storage):
        """Test getting todos by multiple tags (any match)."""
        storage.add(title="Urgent", tags=["urgent"])
        storage.add(title="Important", tags=["important"])
        storage.add(title="Other", tags=["other"])

        results = storage.get_by_tags(["urgent", "important"])
        assert len(results) == 2


class TestTodoStoragePending:
    """Tests for getting pending todos."""

    def test_get_pending(self, storage):
        """Test getting pending todos."""
        storage.add(title="Pending 1")
        storage.add(title="Pending 2")
        todo3 = storage.add(title="Started")
        storage.start(todo3.id)

        pending = storage.get_pending()
        assert len(pending) == 2


class TestTodoStorageInProgress:
    """Tests for getting in-progress todos."""

    def test_get_in_progress(self, storage):
        """Test getting in-progress todos."""
        storage.add(title="Pending")
        todo2 = storage.add(title="In Progress")
        storage.start(todo2.id)

        in_progress = storage.get_in_progress()
        assert len(in_progress) == 1


class TestTodoStorageSummary:
    """Tests for getting todo summary."""

    def test_summary_empty(self, storage):
        """Test summary with no todos."""
        summary = storage.get_summary()
        assert summary["total"] == 0

    def test_summary_counts(self, storage):
        """Test summary with various todos."""
        storage.add(title="Pending", priority="low")
        todo2 = storage.add(title="In Progress", priority="high")
        storage.start(todo2.id)
        todo3 = storage.add(title="Completed", priority="high")
        storage.complete(todo3.id)

        summary = storage.get_summary()
        assert summary["total"] == 3
        assert summary["pending"] == 1
        assert summary["in_progress"] == 1
        assert summary["completed"] == 1
        assert summary["by_priority"]["high"] == 2
        assert summary["by_priority"]["low"] == 1


class TestTodoStorageSubtasks:
    """Tests for subtask management."""

    def test_add_subtask(self, storage):
        """Test adding a subtask."""
        parent = storage.add(title="Parent Task")
        subtask = storage.add_subtask(parent.id, title="Subtask")

        assert subtask is not None
        assert subtask.parent_id == parent.id

    def test_add_subtask_to_nonexistent_parent(self, storage):
        """Test adding subtask to non-existent parent returns None."""
        result = storage.add_subtask("nonexistent", title="Orphan")
        assert result is None


class TestTodoStorageClearCompleted:
    """Tests for clearing completed todos."""

    def test_clear_completed(self, storage):
        """Test clearing completed todos."""
        storage.add(title="Pending")
        todo2 = storage.add(title="Completed 1")
        todo3 = storage.add(title="Completed 2")
        storage.complete(todo2.id)
        storage.complete(todo3.id)

        count = storage.clear_completed()
        assert count == 2
        assert len(storage.list_all()) == 1

    def test_clear_completed_none(self, storage):
        """Test clearing when no completed todos exist."""
        storage.add(title="Pending")

        count = storage.clear_completed()
        assert count == 0


class TestTodoStorageExportMarkdown:
    """Tests for markdown export."""

    def test_export_empty(self, storage):
        """Test exporting empty list."""
        markdown = storage.export_markdown()
        assert "# TODO List" in markdown

    def test_export_with_todos(self, storage):
        """Test exporting list with todos."""
        storage.add(title="Pending Task")
        todo2 = storage.add(title="Completed Task")
        storage.complete(todo2.id)

        markdown = storage.export_markdown()
        assert "Pending Task" in markdown
        assert "Completed Task" in markdown
        assert "## Pending" in markdown
        assert "## Completed" in markdown

    def test_export_includes_priority(self, storage):
        """Test that export shows priority markers."""
        storage.add(title="High Priority", priority="high")

        markdown = storage.export_markdown()
        assert "!!" in markdown or "High Priority" in markdown


class TestTodoStorageBackup:
    """Tests for backup functionality."""

    def test_backup_created_on_save(self, storage, temp_storage_dir):
        """Test that backup is created when saving."""
        storage.add(title="First Task")
        storage.add(title="Second Task")

        backup_file = Path(temp_storage_dir) / "todos.json.bak"
        assert backup_file.exists()


class TestTodoStorageConcurrency:
    """Tests for concurrent access patterns."""

    def test_rapid_adds(self, storage):
        """Test rapid sequential adds don't corrupt data."""
        for i in range(100):
            storage.add(title=f"Task {i}")

        todos = storage.list_all()
        assert len(todos) == 100

    def test_rapid_updates(self, storage):
        """Test rapid sequential updates don't corrupt data."""
        todo = storage.add(title="Update Target")

        for i in range(50):
            storage.update(todo.id, description=f"Update {i}")

        final = storage.get(todo.id)
        assert "Update 49" in final.description


class TestTodoStorageProjectIsolation:
    """Tests for project data isolation."""

    def test_projects_are_isolated(self, storage):
        """Test that projects don't interfere with each other."""
        storage.add(title="Default 1")
        storage.add(title="Default 2")
        storage.add(title="Project 1", project="project_a")
        storage.add(title="Project 2", project="project_b")

        default = storage.list_all(project="default")
        proj_a = storage.list_all(project="project_a")
        proj_b = storage.list_all(project="project_b")

        assert len(default) == 2
        assert len(proj_a) == 1
        assert len(proj_b) == 1

    def test_delete_from_wrong_project_fails(self, storage):
        """Test that delete from wrong project doesn't affect todo."""
        todo = storage.add(title="Project Task", project="correct")

        # Try to delete from wrong project
        result = storage.delete(todo.id, project="wrong")
        assert result is False

        # Todo should still exist
        assert storage.get(todo.id, project="correct") is not None
