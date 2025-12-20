from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _gen_id() -> str:
    return uuid.uuid4().hex[:8]


class TodoPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TodoStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class Todo:
    title: str
    description: str = ""
    status: TodoStatus = TodoStatus.PENDING
    priority: TodoPriority = TodoPriority.MEDIUM
    tags: List[str] = field(default_factory=list)
    project: str = "default"
    due_date: Optional[str] = None
    context: Optional[str] = None
    ai_source: Optional[str] = None
    parent_id: Optional[str] = None
    id: str = field(default_factory=_gen_id)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    completed_at: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

    def _touch(self):
        self.updated_at = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "tags": self.tags,
            "project": self.project,
            "due_date": self.due_date,
            "context": self.context,
            "ai_source": self.ai_source,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Todo":
        return cls(
            id=data.get("id", _gen_id()),
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=TodoStatus(data.get("status", TodoStatus.PENDING.value)),
            priority=TodoPriority(data.get("priority", TodoPriority.MEDIUM.value)),
            tags=data.get("tags", []) or [],
            project=data.get("project", "default"),
            due_date=data.get("due_date"),
            context=data.get("context"),
            ai_source=data.get("ai_source"),
            parent_id=data.get("parent_id"),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            completed_at=data.get("completed_at"),
            notes=data.get("notes", []) or [],
            depends_on=data.get("depends_on", []) or [],
            blocks=data.get("blocks", []) or [],
        )

    def mark_complete(self, ai_source: Optional[str] = None):
        self.status = TodoStatus.COMPLETED
        self.completed_at = _now_iso()
        if ai_source:
            self.ai_source = ai_source
            self.add_note(f"Completed by {ai_source}", ai_source=ai_source)
        self._touch()

    def mark_in_progress(self, ai_source: Optional[str] = None):
        self.status = TodoStatus.IN_PROGRESS
        if ai_source:
            self.ai_source = ai_source
            self.add_note(f"Started by {ai_source}", ai_source=ai_source)
        self._touch()

    def add_note(self, note: str, ai_source: Optional[str] = None):
        prefix = f"[{ai_source}] " if ai_source else ""
        self.notes.append(f"{prefix}{note}")
        self._touch()

    def __str__(self) -> str:
        icon = {
            TodoStatus.PENDING: "[ ]",
            TodoStatus.IN_PROGRESS: "[~]",
            TodoStatus.COMPLETED: "[x]",
            TodoStatus.BLOCKED: "[!]",
            TodoStatus.CANCELLED: "[x]",
        }.get(self.status, "[ ]")
        urgency = " !!!" if self.priority == TodoPriority.URGENT else ""
        return f"{icon} {self.title} ({self.id}){urgency}"


@dataclass
class TodoList:
    name: str
    todos: List[Todo] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "todos": [t.to_dict() for t in self.todos],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoList":
        todos = [Todo.from_dict(td) for td in data.get("todos", [])]
        return cls(
            name=data.get("name", "default"),
            todos=todos,
            created_at=data.get("created_at", _now_iso()),
        )

    def add(self, todo: Todo):
        self.todos.append(todo)
        return todo
