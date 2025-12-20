from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .storage import TodoStorage
from .models import TodoPriority, TodoStatus


class MCPTodoServer:
    def __init__(self, storage_dir: str):
        self.server_info = {"name": "mcp-todo", "version": "1.0.0"}
        self.storage = TodoStorage(storage_dir=storage_dir)

    def get_tools(self) -> List[Dict[str, Any]]:
        tool_names = [
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
        return [
            {
                "name": name,
                "description": name.replace("_", " "),
                "inputSchema": {"type": "object"},
            }
            for name in tool_names
        ]

    def _serialize_todo(self, todo):
        return todo.to_dict() if todo else None

    def handle_tool_call(self, name: str, arguments: Dict[str, Any], ai_source: Optional[str] = None) -> Dict[str, Any]:
        try:
            if name == "todo_add":
                title = arguments.get("title")
                if not title:
                    raise ValueError("title is required")
                todo = self.storage.add(
                    title=title,
                    description=arguments.get("description", ""),
                    priority=arguments.get("priority", "medium"),
                    tags=arguments.get("tags"),
                    project=arguments.get("project", "default"),
                    due_date=arguments.get("due_date"),
                    context=arguments.get("context"),
                    ai_source=ai_source,
                )
                return {"success": True, "todo": todo.to_dict()}

            if name == "todo_list":
                todos = self.storage.list_all(
                    status=arguments.get("status"),
                    priority=arguments.get("priority"),
                    project=arguments.get("project"),
                )
                return {"success": True, "count": len(todos), "todos": [t.to_dict() for t in todos]}

            if name == "todo_get":
                todo = self.storage.get(arguments.get("id", ""), project=arguments.get("project"))
                if not todo:
                    return {"success": False, "error": "Todo not found"}
                return {"success": True, "todo": todo.to_dict()}

            if name == "todo_complete":
                todo = self.storage.complete(arguments.get("id", ""), ai_source=ai_source)
                if not todo:
                    return {"success": False, "error": "Todo not found"}
                return {"success": True, "todo": todo.to_dict()}

            if name == "todo_start":
                todo = self.storage.start(arguments.get("id", ""), ai_source=ai_source)
                if not todo:
                    return {"success": False, "error": "Todo not found"}
                return {"success": True, "todo": todo.to_dict()}

            if name == "todo_update":
                todo = self.storage.update(arguments.get("id", ""), **arguments)
                if not todo:
                    return {"success": False, "error": "Todo not found"}
                return {"success": True, "todo": todo.to_dict()}

            if name == "todo_delete":
                if self.storage.delete(arguments.get("id", "")):
                    return {"success": True}
                return {"success": False, "error": "Todo not found"}

            if name == "todo_search":
                todos = self.storage.search(arguments.get("query", ""))
                return {"success": True, "count": len(todos), "todos": [t.to_dict() for t in todos]}

            if name == "todo_summary":
                summary = self.storage.get_summary()
                return {"success": True, "summary": summary}

            if name == "todo_add_subtask":
                todo = self.storage.add_subtask(arguments.get("parent_id", ""), arguments.get("title", ""))
                if not todo:
                    return {"success": False, "error": "Parent not found"}
                return {"success": True, "todo": todo.to_dict()}

            if name == "todo_add_note":
                todo = self.storage.add_note(arguments.get("id", ""), arguments.get("note", ""), ai_source=ai_source)
                if not todo:
                    return {"success": False, "error": "Todo not found"}
                return {"success": True, "todo": todo.to_dict()}

            if name == "todo_clear_completed":
                removed = self.storage.clear_completed()
                return {"success": True, "message": f"Cleared {removed} completed todos"}

            if name == "todo_export":
                content = self.storage.export_markdown()
                return {"success": True, "format": "markdown", "content": content}

            return {"success": False, "error": f"Unknown tool: {name}"}
        except Exception as exc:  # pragma: no cover - defensive
            return {"success": False, "error": str(exc)}

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = message.get("method")
        msg_id = message.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": self.server_info,
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.get_tools()},
            }

        if method == "tools/call":
            params = message.get("params", {})
            ai_source = None
            meta = params.get("_meta") or {}
            if isinstance(meta, dict):
                ai_source = meta.get("ai_source")
            result = self.handle_tool_call(params.get("name", ""), params.get("arguments", {}), ai_source=ai_source)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                },
            }

        if method == "notifications/initialized":
            return None

        # Unknown method
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Unknown method {method}"},
        }
