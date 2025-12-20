from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from .models import Todo, TodoList, TodoPriority, TodoStatus, _now_iso


class TodoStorage:
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / "todos.json"
        if not self.file_path.exists():
            default = {"lists": {"default": TodoList(name="default").to_dict()}}
            self.file_path.write_text(json.dumps(default, indent=2))
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        with open(self.file_path) as f:
            return json.load(f)

    def _save_data(self):
        if self.file_path.exists():
            bak = self.file_path.with_suffix(self.file_path.suffix + ".bak")
            bak.write_text(self.file_path.read_text())
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _get_or_create_list(self, project: str) -> TodoList:
        lists = self.data.setdefault("lists", {})
        if project not in lists:
            lists[project] = TodoList(name=project).to_dict()
        return TodoList.from_dict(lists[project])

    def _write_list(self, todo_list: TodoList):
        self.data["lists"][todo_list.name] = todo_list.to_dict()
        self._save_data()

    def add(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        status: str = "pending",
        tags: Optional[List[str]] = None,
        project: str = "default",
        due_date: Optional[str] = None,
        context: Optional[str] = None,
        ai_source: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Todo:
        todo = Todo(
            title=title,
            description=description,
            priority=TodoPriority(priority),
            status=TodoStatus(status),
            tags=tags or [],
            project=project,
            due_date=due_date,
            context=context,
            ai_source=ai_source,
            parent_id=parent_id,
            depends_on=[parent_id] if parent_id else [],
        )
        todo_list = self._get_or_create_list(project)
        todo_list.add(todo)
        self._write_list(todo_list)
        return todo

    def _iter_all(self):
        for list_data in self.data.get("lists", {}).values():
            for td in list_data.get("todos", []):
                yield list_data["name"], Todo.from_dict(td)

    def get(self, todo_id: str, project: Optional[str] = None) -> Optional[Todo]:
        if project:
            lst = self._get_or_create_list(project)
            for todo in lst.todos:
                if todo.id == todo_id:
                    return todo
            return None
        for _, todo in self._iter_all():
            if todo.id == todo_id:
                return todo
        return None

    def list_all(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        include_completed: bool = True,
    ) -> List[Todo]:
        results = []
        for list_name, todo in self._iter_all():
            if project and list_name != project:
                continue
            if status and todo.status != TodoStatus(status):
                continue
            if priority and todo.priority != TodoPriority(priority):
                continue
            if tags and not (set(tags) & set(todo.tags)):
                continue
            if not include_completed and todo.status == TodoStatus.COMPLETED:
                continue
            results.append(todo)
        return results

    def _update_and_save(self, todo: Todo):
        # rewrite lists to persist current todo state
        for list_name, _ in list(self._iter_all()):
            lst = self._get_or_create_list(list_name)
            for idx, td in enumerate(lst.todos):
                if td.id == todo.id:
                    lst.todos[idx] = todo
                    self._write_list(lst)
                    return

    def update(self, todo_id: str, **fields) -> Optional[Todo]:
        target = None
        for list_name, todo in self._iter_all():
            if todo.id == todo_id:
                target = todo
                break
        if not target:
            return None

        if "title" in fields and fields["title"] is not None:
            target.title = fields["title"]
        if "description" in fields and fields["description"] is not None:
            target.description = fields["description"]
        if "priority" in fields and fields["priority"] is not None:
            target.priority = TodoPriority(fields["priority"])
        if "status" in fields and fields["status"] is not None:
            target.status = TodoStatus(fields["status"])
            if target.status == TodoStatus.COMPLETED:
                target.completed_at = target.completed_at or _now_iso()
        if "tags" in fields and fields["tags"] is not None:
            target.tags = fields["tags"]
        target._touch()
        self._update_and_save(target)
        return target

    def delete(self, todo_id: str, project: Optional[str] = None) -> bool:
        changed = False
        lists = self.data.get("lists", {})
        for name, list_data in lists.items():
            if project and name != project:
                continue
            todos = list_data.get("todos", [])
            new_todos = [td for td in todos if td.get("id") != todo_id]
            if len(new_todos) != len(todos):
                list_data["todos"] = new_todos
                changed = True
        if changed:
            self._save_data()
        return changed

    def complete(self, todo_id: str, ai_source: Optional[str] = None) -> Optional[Todo]:
        todo = self.get(todo_id)
        if not todo:
            return None
        todo.mark_complete(ai_source=ai_source)
        self._update_and_save(todo)
        return todo

    def start(self, todo_id: str, ai_source: Optional[str] = None) -> Optional[Todo]:
        todo = self.get(todo_id)
        if not todo:
            return None
        todo.mark_in_progress(ai_source=ai_source)
        self._update_and_save(todo)
        return todo

    def add_subtask(self, parent_id: str, title: str) -> Optional[Todo]:
        parent = self.get(parent_id)
        if not parent:
            return None
        return self.add(title=title, parent_id=parent_id, project=parent.project)

    def add_note(self, todo_id: str, note: str, ai_source: Optional[str] = None) -> Optional[Todo]:
        todo = self.get(todo_id)
        if not todo:
            return None
        todo.add_note(note, ai_source=ai_source)
        self._update_and_save(todo)
        return todo

    def clear_completed(self) -> int:
        removed = 0
        lists = self.data.get("lists", {})
        for name, list_data in lists.items():
            todos = [Todo.from_dict(td) for td in list_data.get("todos", [])]
            remaining = [t for t in todos if t.status != TodoStatus.COMPLETED]
            removed += len(todos) - len(remaining)
            list_data["todos"] = [t.to_dict() for t in remaining]
        self._save_data()
        return removed

    def search(self, query: str) -> List[Todo]:
        q = query.lower()
        return [
            todo
            for _, todo in self._iter_all()
            if q in todo.title.lower() or q in (todo.description or "").lower()
        ]

    def get_by_tags(self, tags: List[str]) -> List[Todo]:
        return [t for t in self.list_all() if set(tags) & set(t.tags)]

    def get_pending(self) -> List[Todo]:
        return self.list_all(status=TodoStatus.PENDING.value)

    def get_in_progress(self) -> List[Todo]:
        return self.list_all(status=TodoStatus.IN_PROGRESS.value)

    def get_summary(self) -> Dict[str, Any]:
        todos = [todo for _, todo in self._iter_all()]
        summary = {
            "total": len(todos),
            "pending": sum(1 for t in todos if t.status == TodoStatus.PENDING),
            "in_progress": sum(1 for t in todos if t.status == TodoStatus.IN_PROGRESS),
            "completed": sum(1 for t in todos if t.status == TodoStatus.COMPLETED),
            "blocked": sum(1 for t in todos if t.status == TodoStatus.BLOCKED),
            "by_priority": {},
        }
        for t in todos:
            summary["by_priority"][t.priority.value] = summary["by_priority"].get(t.priority.value, 0) + 1
        return summary

    def export_markdown(self) -> str:
        pending = [t for t in self.list_all(status=TodoStatus.PENDING.value)]
        in_progress = [t for t in self.list_all(status=TodoStatus.IN_PROGRESS.value)]
        completed = [t for t in self.list_all(status=TodoStatus.COMPLETED.value)]

        lines = ["# TODO List"]
        lines.append("## Pending")
        for t in pending:
            lines.append(f"- [ ] {t.title} ({t.id})")
        lines.append("## In Progress")
        for t in in_progress:
            lines.append(f"- [~] {t.title} ({t.id})")
        lines.append("## Completed")
        for t in completed:
            lines.append(f"- [x] {t.title} ({t.id})")
        return "\n".join(lines)
