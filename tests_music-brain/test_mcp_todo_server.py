"""
Tests for MCP TODO Server.

Tests the MCPTodoServer class and its MCP protocol implementation.
"""

import pytest
import json
import tempfile
import shutil
import asyncio
from pathlib import Path

from mcp_todo.server import MCPTodoServer
from mcp_todo.models import TodoStatus, TodoPriority


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def server(temp_storage_dir):
    """Create an MCPTodoServer instance with temporary storage."""
    return MCPTodoServer(storage_dir=temp_storage_dir)


class TestMCPServerInit:
    """Tests for server initialization."""

    def test_server_init(self, server):
        """Test server initialization."""
        assert server.server_info["name"] == "mcp-todo"
        assert server.server_info["version"] == "1.0.0"
        assert server.storage is not None

    def test_server_with_custom_storage(self, temp_storage_dir):
        """Test server with custom storage directory."""
        server = MCPTodoServer(storage_dir=temp_storage_dir)
        assert (Path(temp_storage_dir) / "todos.json").exists()


class TestMCPServerGetTools:
    """Tests for get_tools method."""

    def test_get_tools_returns_list(self, server):
        """Test that get_tools returns a list."""
        tools = server.get_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_tools_has_required_tools(self, server):
        """Test that all required tools are present."""
        tools = server.get_tools()
        tool_names = [t["name"] for t in tools]

        required_tools = [
            "todo_add",
            "todo_list",
            "todo_get",
            "todo_complete",
            "todo_start",
            "todo_update",
            "todo_delete",
            "todo_search",
            "todo_summary",
            "todo_add_subtask",
            "todo_add_note",
            "todo_clear_completed",
            "todo_export",
        ]

        for tool in required_tools:
            assert tool in tool_names, f"Missing tool: {tool}"

    def test_tool_schemas_valid(self, server):
        """Test that tool schemas are valid."""
        tools = server.get_tools()

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"


class TestMCPServerToolAdd:
    """Tests for todo_add tool."""

    def test_add_basic(self, server):
        """Test adding a basic todo."""
        result = server.handle_tool_call(
            "todo_add",
            {"title": "Test Task"}
        )
        assert result["success"] is True
        assert "todo" in result
        assert result["todo"]["title"] == "Test Task"

    def test_add_full(self, server):
        """Test adding a todo with all fields."""
        result = server.handle_tool_call(
            "todo_add",
            {
                "title": "Full Task",
                "description": "Complete description",
                "priority": "high",
                "tags": ["test", "urgent"],
                "project": "iDAW",
                "due_date": "2024-12-31",
                "context": "Testing context",
            },
            ai_source="claude"
        )
        assert result["success"] is True
        todo = result["todo"]
        assert todo["title"] == "Full Task"
        assert todo["priority"] == "high"
        assert todo["tags"] == ["test", "urgent"]

    def test_add_returns_id(self, server):
        """Test that add returns a todo with ID."""
        result = server.handle_tool_call(
            "todo_add",
            {"title": "ID Test"}
        )
        assert "id" in result["todo"]
        assert len(result["todo"]["id"]) == 8


class TestMCPServerToolList:
    """Tests for todo_list tool."""

    def test_list_empty(self, server):
        """Test listing empty todo list."""
        result = server.handle_tool_call("todo_list", {})
        assert result["success"] is True
        assert result["count"] == 0
        assert result["todos"] == []

    def test_list_with_todos(self, server):
        """Test listing todos."""
        server.handle_tool_call("todo_add", {"title": "Task 1"})
        server.handle_tool_call("todo_add", {"title": "Task 2"})

        result = server.handle_tool_call("todo_list", {})
        assert result["success"] is True
        assert result["count"] == 2

    def test_list_filter_by_status(self, server):
        """Test filtering by status."""
        server.handle_tool_call("todo_add", {"title": "Pending"})
        add_result = server.handle_tool_call("todo_add", {"title": "Started"})
        server.handle_tool_call(
            "todo_start",
            {"id": add_result["todo"]["id"]}
        )

        result = server.handle_tool_call(
            "todo_list",
            {"status": "in_progress"}
        )
        assert result["count"] == 1
        assert result["todos"][0]["title"] == "Started"

    def test_list_filter_by_priority(self, server):
        """Test filtering by priority."""
        server.handle_tool_call("todo_add", {"title": "Low", "priority": "low"})
        server.handle_tool_call("todo_add", {"title": "High", "priority": "high"})

        result = server.handle_tool_call(
            "todo_list",
            {"priority": "high"}
        )
        assert result["count"] == 1


class TestMCPServerToolGet:
    """Tests for todo_get tool."""

    def test_get_existing(self, server):
        """Test getting an existing todo."""
        add_result = server.handle_tool_call(
            "todo_add",
            {"title": "Get Test"}
        )
        todo_id = add_result["todo"]["id"]

        result = server.handle_tool_call("todo_get", {"id": todo_id})
        assert result["success"] is True
        assert result["todo"]["title"] == "Get Test"

    def test_get_nonexistent(self, server):
        """Test getting a non-existent todo."""
        result = server.handle_tool_call("todo_get", {"id": "nonexistent"})
        assert result["success"] is False
        assert "error" in result


class TestMCPServerToolComplete:
    """Tests for todo_complete tool."""

    def test_complete_existing(self, server):
        """Test completing an existing todo."""
        add_result = server.handle_tool_call(
            "todo_add",
            {"title": "To Complete"}
        )
        todo_id = add_result["todo"]["id"]

        result = server.handle_tool_call(
            "todo_complete",
            {"id": todo_id},
            ai_source="claude"
        )
        assert result["success"] is True
        assert result["todo"]["status"] == "completed"

    def test_complete_nonexistent(self, server):
        """Test completing a non-existent todo."""
        result = server.handle_tool_call(
            "todo_complete",
            {"id": "nonexistent"}
        )
        assert result["success"] is False


class TestMCPServerToolStart:
    """Tests for todo_start tool."""

    def test_start_existing(self, server):
        """Test starting an existing todo."""
        add_result = server.handle_tool_call(
            "todo_add",
            {"title": "To Start"}
        )
        todo_id = add_result["todo"]["id"]

        result = server.handle_tool_call("todo_start", {"id": todo_id})
        assert result["success"] is True
        assert result["todo"]["status"] == "in_progress"

    def test_start_nonexistent(self, server):
        """Test starting a non-existent todo."""
        result = server.handle_tool_call("todo_start", {"id": "nonexistent"})
        assert result["success"] is False


class TestMCPServerToolUpdate:
    """Tests for todo_update tool."""

    def test_update_title(self, server):
        """Test updating todo title."""
        add_result = server.handle_tool_call(
            "todo_add",
            {"title": "Original"}
        )
        todo_id = add_result["todo"]["id"]

        result = server.handle_tool_call(
            "todo_update",
            {"id": todo_id, "title": "Updated"}
        )
        assert result["success"] is True
        assert result["todo"]["title"] == "Updated"

    def test_update_multiple_fields(self, server):
        """Test updating multiple fields."""
        add_result = server.handle_tool_call(
            "todo_add",
            {"title": "Multi Update"}
        )
        todo_id = add_result["todo"]["id"]

        result = server.handle_tool_call(
            "todo_update",
            {
                "id": todo_id,
                "title": "New Title",
                "description": "New Description",
                "priority": "urgent",
            }
        )
        assert result["success"] is True
        assert result["todo"]["title"] == "New Title"
        assert result["todo"]["priority"] == "urgent"

    def test_update_nonexistent(self, server):
        """Test updating a non-existent todo."""
        result = server.handle_tool_call(
            "todo_update",
            {"id": "nonexistent", "title": "New"}
        )
        assert result["success"] is False


class TestMCPServerToolDelete:
    """Tests for todo_delete tool."""

    def test_delete_existing(self, server):
        """Test deleting an existing todo."""
        add_result = server.handle_tool_call(
            "todo_add",
            {"title": "To Delete"}
        )
        todo_id = add_result["todo"]["id"]

        result = server.handle_tool_call("todo_delete", {"id": todo_id})
        assert result["success"] is True

        # Verify deleted
        get_result = server.handle_tool_call("todo_get", {"id": todo_id})
        assert get_result["success"] is False

    def test_delete_nonexistent(self, server):
        """Test deleting a non-existent todo."""
        result = server.handle_tool_call("todo_delete", {"id": "nonexistent"})
        assert result["success"] is False


class TestMCPServerToolSearch:
    """Tests for todo_search tool."""

    def test_search_finds_match(self, server):
        """Test searching finds matching todos."""
        server.handle_tool_call("todo_add", {"title": "Fix the bug"})
        server.handle_tool_call("todo_add", {"title": "Add feature"})

        result = server.handle_tool_call("todo_search", {"query": "bug"})
        assert result["success"] is True
        assert result["count"] == 1

    def test_search_no_match(self, server):
        """Test searching with no matches."""
        server.handle_tool_call("todo_add", {"title": "Some task"})

        result = server.handle_tool_call("todo_search", {"query": "nonexistent"})
        assert result["success"] is True
        assert result["count"] == 0


class TestMCPServerToolSummary:
    """Tests for todo_summary tool."""

    def test_summary_empty(self, server):
        """Test summary with no todos."""
        result = server.handle_tool_call("todo_summary", {})
        assert result["success"] is True
        assert result["summary"]["total"] == 0

    def test_summary_with_todos(self, server):
        """Test summary with various todos."""
        server.handle_tool_call("todo_add", {"title": "Pending"})
        add_result = server.handle_tool_call(
            "todo_add",
            {"title": "Started", "priority": "high"}
        )
        server.handle_tool_call("todo_start", {"id": add_result["todo"]["id"]})

        result = server.handle_tool_call("todo_summary", {})
        summary = result["summary"]
        assert summary["total"] == 2
        assert summary["pending"] == 1
        assert summary["in_progress"] == 1


class TestMCPServerToolSubtask:
    """Tests for todo_add_subtask tool."""

    def test_add_subtask(self, server):
        """Test adding a subtask."""
        parent = server.handle_tool_call(
            "todo_add",
            {"title": "Parent Task"}
        )
        parent_id = parent["todo"]["id"]

        result = server.handle_tool_call(
            "todo_add_subtask",
            {"parent_id": parent_id, "title": "Subtask"}
        )
        assert result["success"] is True
        assert result["todo"]["title"] == "Subtask"

    def test_add_subtask_nonexistent_parent(self, server):
        """Test adding subtask to non-existent parent."""
        result = server.handle_tool_call(
            "todo_add_subtask",
            {"parent_id": "nonexistent", "title": "Orphan"}
        )
        assert result["success"] is False


class TestMCPServerToolNote:
    """Tests for todo_add_note tool."""

    def test_add_note(self, server):
        """Test adding a note to a todo."""
        add_result = server.handle_tool_call(
            "todo_add",
            {"title": "With Note"}
        )
        todo_id = add_result["todo"]["id"]

        result = server.handle_tool_call(
            "todo_add_note",
            {"id": todo_id, "note": "This is a note"},
            ai_source="claude"
        )
        assert result["success"] is True

    def test_add_note_nonexistent(self, server):
        """Test adding note to non-existent todo."""
        result = server.handle_tool_call(
            "todo_add_note",
            {"id": "nonexistent", "note": "Note"}
        )
        assert result["success"] is False


class TestMCPServerToolClearCompleted:
    """Tests for todo_clear_completed tool."""

    def test_clear_completed(self, server):
        """Test clearing completed todos."""
        server.handle_tool_call("todo_add", {"title": "Pending"})
        add_result = server.handle_tool_call(
            "todo_add",
            {"title": "Completed"}
        )
        server.handle_tool_call(
            "todo_complete",
            {"id": add_result["todo"]["id"]}
        )

        result = server.handle_tool_call("todo_clear_completed", {})
        assert result["success"] is True
        assert "1" in result["message"]  # Cleared 1 todo

    def test_clear_completed_none(self, server):
        """Test clearing when no completed todos."""
        server.handle_tool_call("todo_add", {"title": "Pending"})

        result = server.handle_tool_call("todo_clear_completed", {})
        assert result["success"] is True
        assert "0" in result["message"]


class TestMCPServerToolExport:
    """Tests for todo_export tool."""

    def test_export_markdown(self, server):
        """Test exporting as markdown."""
        server.handle_tool_call("todo_add", {"title": "Export Test"})

        result = server.handle_tool_call("todo_export", {})
        assert result["success"] is True
        assert result["format"] == "markdown"
        assert "Export Test" in result["content"]


class TestMCPServerUnknownTool:
    """Tests for unknown tool handling."""

    def test_unknown_tool(self, server):
        """Test handling unknown tool."""
        result = server.handle_tool_call("unknown_tool", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]


class TestMCPServerErrorHandling:
    """Tests for error handling."""

    def test_handles_exception(self, server):
        """Test that exceptions are caught and returned as errors."""
        # This should cause an error - missing required 'title'
        result = server.handle_tool_call("todo_add", {})
        assert result["success"] is False
        assert "error" in result


class TestMCPProtocolMessages:
    """Tests for MCP protocol message handling."""

    @pytest.mark.asyncio
    async def test_initialize_message(self, server):
        """Test handling initialize message."""
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        response = await server.handle_message(message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"

    @pytest.mark.asyncio
    async def test_tools_list_message(self, server):
        """Test handling tools/list message."""
        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        response = await server.handle_message(message)

        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0

    @pytest.mark.asyncio
    async def test_tools_call_message(self, server):
        """Test handling tools/call message."""
        message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "todo_add",
                "arguments": {"title": "Protocol Test"}
            }
        }
        response = await server.handle_message(message)

        assert response["id"] == 3
        assert "result" in response
        assert "content" in response["result"]

        content = json.loads(response["result"]["content"][0]["text"])
        assert content["success"] is True

    @pytest.mark.asyncio
    async def test_initialized_notification(self, server):
        """Test handling initialized notification."""
        message = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        response = await server.handle_message(message)
        assert response is None  # No response for notifications

    @pytest.mark.asyncio
    async def test_unknown_method(self, server):
        """Test handling unknown method."""
        message = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "unknown/method",
            "params": {}
        }
        response = await server.handle_message(message)

        assert response["id"] == 4
        assert "error" in response
        assert response["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_tools_call_with_ai_source(self, server):
        """Test that AI source is passed through from meta."""
        message = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "todo_add",
                "arguments": {"title": "AI Source Test"},
                "_meta": {"ai_source": "claude"}
            }
        }
        response = await server.handle_message(message)

        content = json.loads(response["result"]["content"][0]["text"])
        assert content["success"] is True


class TestMCPServerAISourceTracking:
    """Tests for AI source tracking."""

    def test_add_tracks_ai_source(self, server):
        """Test that add tracks AI source."""
        result = server.handle_tool_call(
            "todo_add",
            {"title": "AI Tracked"},
            ai_source="claude"
        )
        assert result["todo"].get("ai_source") == "claude"

    def test_complete_tracks_ai_source(self, server):
        """Test that complete tracks AI source."""
        add = server.handle_tool_call("todo_add", {"title": "To Complete"})
        todo_id = add["todo"]["id"]

        result = server.handle_tool_call(
            "todo_complete",
            {"id": todo_id},
            ai_source="chatgpt"
        )
        assert result["todo"].get("ai_source") == "chatgpt"


class TestMCPServerIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, server):
        """Test a complete todo workflow."""
        # Create
        add_result = server.handle_tool_call(
            "todo_add",
            {
                "title": "Integration Test",
                "priority": "high",
                "tags": ["test"],
            }
        )
        assert add_result["success"] is True
        todo_id = add_result["todo"]["id"]

        # Start
        start_result = server.handle_tool_call("todo_start", {"id": todo_id})
        assert start_result["success"] is True

        # Add note
        note_result = server.handle_tool_call(
            "todo_add_note",
            {"id": todo_id, "note": "Working on it"}
        )
        assert note_result["success"] is True

        # Complete
        complete_result = server.handle_tool_call(
            "todo_complete",
            {"id": todo_id}
        )
        assert complete_result["success"] is True

        # Verify final state
        get_result = server.handle_tool_call("todo_get", {"id": todo_id})
        assert get_result["todo"]["status"] == "completed"

    def test_subtask_workflow(self, server):
        """Test subtask workflow."""
        # Create parent
        parent = server.handle_tool_call(
            "todo_add",
            {"title": "Parent Task"}
        )
        parent_id = parent["todo"]["id"]

        # Add subtasks
        for i in range(3):
            server.handle_tool_call(
                "todo_add_subtask",
                {"parent_id": parent_id, "title": f"Subtask {i+1}"}
            )

        # List all
        list_result = server.handle_tool_call("todo_list", {})
        assert list_result["count"] == 4  # 1 parent + 3 subtasks

    def test_multi_project_workflow(self, server):
        """Test multi-project workflow."""
        # Add to different projects
        server.handle_tool_call(
            "todo_add",
            {"title": "Default Task"}
        )
        server.handle_tool_call(
            "todo_add",
            {"title": "Project A Task", "project": "project_a"}
        )
        server.handle_tool_call(
            "todo_add",
            {"title": "Project B Task", "project": "project_b"}
        )

        # Get summary (should include all)
        summary = server.handle_tool_call("todo_summary", {})
        assert summary["summary"]["total"] >= 1
