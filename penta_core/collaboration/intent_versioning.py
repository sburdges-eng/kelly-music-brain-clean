"""
Intent Version Control System for iDAW.

Provides Git-like version control for intent documents,
enabling history tracking, branching, merging, and rollback.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Type of change in a diff."""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"


@dataclass
class Change:
    """A single change in a commit."""
    change_type: ChangeType
    path: str
    old_value: Any = None
    new_value: Any = None

    def to_dict(self) -> dict:
        return {
            "type": self.change_type.value,
            "path": self.path,
            "old": self.old_value,
            "new": self.new_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Change":
        return cls(
            change_type=ChangeType(data["type"]),
            path=data["path"],
            old_value=data.get("old"),
            new_value=data.get("new"),
        )


@dataclass
class Commit:
    """A commit in the intent version history."""
    commit_id: str
    parent_id: Optional[str]
    timestamp: datetime
    author: str
    message: str
    changes: list[Change]
    intent_hash: str  # Hash of the full intent state after this commit

    def to_dict(self) -> dict:
        return {
            "id": self.commit_id,
            "parent": self.parent_id,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "message": self.message,
            "changes": [c.to_dict() for c in self.changes],
            "hash": self.intent_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Commit":
        return cls(
            commit_id=data["id"],
            parent_id=data.get("parent"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            author=data["author"],
            message=data["message"],
            changes=[Change.from_dict(c) for c in data["changes"]],
            intent_hash=data["hash"],
        )


@dataclass
class Branch:
    """A branch in the intent version history."""
    name: str
    head_commit: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_from: Optional[str] = None  # Commit ID where branch was created

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "head": self.head_commit,
            "createdAt": self.created_at.isoformat(),
            "createdFrom": self.created_from,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Branch":
        return cls(
            name=data["name"],
            head_commit=data["head"],
            created_at=datetime.fromisoformat(data["createdAt"]),
            created_from=data.get("createdFrom"),
        )


@dataclass
class Tag:
    """A tag marking a specific commit."""
    name: str
    commit_id: str
    message: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "commit": self.commit_id,
            "message": self.message,
            "createdAt": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Tag":
        return cls(
            name=data["name"],
            commit_id=data["commit"],
            message=data.get("message", ""),
            created_at=datetime.fromisoformat(data["createdAt"]),
        )


class IntentVersionControl:
    """
    Version control system for intent documents.

    Provides:
    - Commit history with diff tracking
    - Branching for experimental changes
    - Tags for marking important versions
    - Merge support for collaborative editing
    - Rollback to any previous commit
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the version control system.

        Args:
            storage_path: Path to store version history.
                         Defaults to ~/.idaw/intent_history/
        """
        if storage_path is None:
            storage_path = Path.home() / ".idaw" / "intent_history"

        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.commits: dict[str, Commit] = {}
        self.branches: dict[str, Branch] = {}
        self.tags: dict[str, Tag] = {}
        self.snapshots: dict[str, dict] = {}  # commit_id -> full intent state

        self.current_branch: str = "main"
        self.working_intent: dict = self._empty_intent()
        self.staged_changes: list[Change] = []

        self._load_history()

    def _empty_intent(self) -> dict:
        """Return an empty intent structure."""
        return {
            "core": {
                "event": "",
                "resistance": "",
                "longing": "",
            },
            "emotional": {
                "moodPrimary": "",
                "moodSecondary": [],
                "vulnerabilityScale": 5,
                "narrativeArc": "standard",
            },
            "technical": {
                "genre": "",
                "key": "C",
                "tempo": 120,
                "timeSignature": "4/4",
                "rulesToBreak": [],
            },
        }

    def _load_history(self):
        """Load version history from storage."""
        history_file = self.storage_path / "history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)

                for commit_data in data.get("commits", []):
                    commit = Commit.from_dict(commit_data)
                    self.commits[commit.commit_id] = commit

                for branch_data in data.get("branches", []):
                    branch = Branch.from_dict(branch_data)
                    self.branches[branch.name] = branch

                for tag_data in data.get("tags", []):
                    tag = Tag.from_dict(tag_data)
                    self.tags[tag.name] = tag

                self.current_branch = data.get("currentBranch", "main")

                # Load snapshots
                for commit_id in self.commits:
                    snapshot_file = self.storage_path / f"snapshots/{commit_id}.json"
                    if snapshot_file.exists():
                        with open(snapshot_file, "r") as f:
                            self.snapshots[commit_id] = json.load(f)

                # Reconstruct working intent
                if self.current_branch in self.branches:
                    head = self.branches[self.current_branch].head_commit
                    if head in self.snapshots:
                        self.working_intent = self.snapshots[head].copy()

                logger.info(f"Loaded {len(self.commits)} commits from history")

            except Exception as e:
                logger.error(f"Failed to load history: {e}")
                self._init_empty_repo()
        else:
            self._init_empty_repo()

    def _init_empty_repo(self):
        """Initialize an empty repository."""
        self.branches["main"] = Branch(
            name="main",
            head_commit="",
            created_at=datetime.utcnow(),
        )
        self.current_branch = "main"
        self.working_intent = self._empty_intent()

    def _save_history(self):
        """Save version history to storage."""
        history_file = self.storage_path / "history.json"

        data = {
            "commits": [c.to_dict() for c in self.commits.values()],
            "branches": [b.to_dict() for b in self.branches.values()],
            "tags": [t.to_dict() for t in self.tags.values()],
            "currentBranch": self.current_branch,
        }

        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

        # Save snapshots
        snapshots_dir = self.storage_path / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)

        for commit_id, snapshot in self.snapshots.items():
            snapshot_file = snapshots_dir / f"{commit_id}.json"
            with open(snapshot_file, "w") as f:
                json.dump(snapshot, f, indent=2)

    def _compute_hash(self, intent: dict) -> str:
        """Compute a hash of an intent state."""
        content = json.dumps(intent, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _generate_commit_id(self) -> str:
        """Generate a unique commit ID."""
        timestamp = datetime.utcnow().isoformat()
        content = f"{timestamp}-{len(self.commits)}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    def _diff_intents(self, old: dict, new: dict, prefix: str = "") -> list[Change]:
        """
        Compute the diff between two intent states.

        Args:
            old: The previous intent state
            new: The new intent state
            prefix: Path prefix for nested fields

        Returns:
            List of changes
        """
        changes = []

        all_keys = set(old.keys()) | set(new.keys())

        for key in all_keys:
            path = f"{prefix}.{key}" if prefix else key
            old_val = old.get(key)
            new_val = new.get(key)

            if key not in old:
                changes.append(Change(
                    change_type=ChangeType.ADD,
                    path=path,
                    new_value=new_val,
                ))
            elif key not in new:
                changes.append(Change(
                    change_type=ChangeType.DELETE,
                    path=path,
                    old_value=old_val,
                ))
            elif isinstance(old_val, dict) and isinstance(new_val, dict):
                # Recurse into nested dicts
                changes.extend(self._diff_intents(old_val, new_val, path))
            elif old_val != new_val:
                changes.append(Change(
                    change_type=ChangeType.MODIFY,
                    path=path,
                    old_value=old_val,
                    new_value=new_val,
                ))

        return changes

    def _apply_changes(self, base: dict, changes: list[Change]) -> dict:
        """Apply a list of changes to a base intent."""
        result = json.loads(json.dumps(base))  # Deep copy

        for change in changes:
            parts = change.path.split(".")
            target = result

            # Navigate to parent
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]

            final_key = parts[-1]

            if change.change_type == ChangeType.DELETE:
                if final_key in target:
                    del target[final_key]
            else:
                target[final_key] = change.new_value

        return result

    # Public API

    def status(self) -> dict:
        """
        Get the current status.

        Returns:
            Dict with branch, staged changes, and uncommitted changes
        """
        head_commit = self.branches[self.current_branch].head_commit
        base_intent = self.snapshots.get(head_commit, self._empty_intent())

        uncommitted = self._diff_intents(base_intent, self.working_intent)

        return {
            "branch": self.current_branch,
            "head": head_commit,
            "staged": len(self.staged_changes),
            "uncommitted": len(uncommitted),
            "changes": [c.to_dict() for c in uncommitted],
        }

    def stage(self, path: Optional[str] = None):
        """
        Stage changes for commit.

        Args:
            path: Specific field path to stage, or None for all changes
        """
        head_commit = self.branches[self.current_branch].head_commit
        base_intent = self.snapshots.get(head_commit, self._empty_intent())

        all_changes = self._diff_intents(base_intent, self.working_intent)

        if path:
            # Stage only matching changes
            for change in all_changes:
                if change.path == path or change.path.startswith(f"{path}."):
                    if change not in self.staged_changes:
                        self.staged_changes.append(change)
        else:
            # Stage all changes
            self.staged_changes = all_changes

        logger.info(f"Staged {len(self.staged_changes)} changes")

    def unstage(self, path: Optional[str] = None):
        """
        Unstage changes.

        Args:
            path: Specific field path to unstage, or None for all
        """
        if path:
            self.staged_changes = [
                c for c in self.staged_changes
                if not (c.path == path or c.path.startswith(f"{path}."))
            ]
        else:
            self.staged_changes = []

    def commit(self, message: str, author: str = "user") -> str:
        """
        Commit staged changes.

        Args:
            message: Commit message
            author: Author name

        Returns:
            The new commit ID
        """
        if not self.staged_changes:
            raise ValueError("No changes staged for commit")

        head_commit = self.branches[self.current_branch].head_commit

        # Create new commit
        commit_id = self._generate_commit_id()
        intent_hash = self._compute_hash(self.working_intent)

        commit = Commit(
            commit_id=commit_id,
            parent_id=head_commit if head_commit else None,
            timestamp=datetime.utcnow(),
            author=author,
            message=message,
            changes=self.staged_changes.copy(),
            intent_hash=intent_hash,
        )

        # Store commit and snapshot
        self.commits[commit_id] = commit
        self.snapshots[commit_id] = json.loads(json.dumps(self.working_intent))

        # Update branch head
        self.branches[self.current_branch].head_commit = commit_id

        # Clear staged changes
        self.staged_changes = []

        self._save_history()

        logger.info(f"Created commit {commit_id}: {message}")
        return commit_id

    def log(self, limit: int = 10) -> list[dict]:
        """
        Get commit history.

        Args:
            limit: Maximum number of commits to return

        Returns:
            List of commit dicts, most recent first
        """
        head = self.branches[self.current_branch].head_commit
        if not head:
            return []

        history = []
        current = head

        while current and len(history) < limit:
            commit = self.commits.get(current)
            if not commit:
                break

            history.append({
                "id": commit.commit_id,
                "timestamp": commit.timestamp.isoformat(),
                "author": commit.author,
                "message": commit.message,
                "changes": len(commit.changes),
            })

            current = commit.parent_id

        return history

    def show(self, commit_id: str) -> Optional[dict]:
        """
        Show details of a commit.

        Args:
            commit_id: The commit ID

        Returns:
            Commit details or None if not found
        """
        commit = self.commits.get(commit_id)
        if not commit:
            return None

        return {
            "id": commit.commit_id,
            "parent": commit.parent_id,
            "timestamp": commit.timestamp.isoformat(),
            "author": commit.author,
            "message": commit.message,
            "changes": [c.to_dict() for c in commit.changes],
            "hash": commit.intent_hash,
        }

    def diff(self, from_ref: Optional[str] = None, to_ref: Optional[str] = None) -> list[dict]:
        """
        Show diff between two commits or working state.

        Args:
            from_ref: Source commit ID (default: HEAD)
            to_ref: Target commit ID (default: working intent)

        Returns:
            List of changes
        """
        if from_ref is None:
            from_ref = self.branches[self.current_branch].head_commit

        from_intent = self.snapshots.get(from_ref, self._empty_intent())

        if to_ref is None:
            to_intent = self.working_intent
        else:
            to_intent = self.snapshots.get(to_ref, self._empty_intent())

        changes = self._diff_intents(from_intent, to_intent)
        return [c.to_dict() for c in changes]

    def checkout(self, ref: str):
        """
        Checkout a branch or commit.

        Args:
            ref: Branch name or commit ID
        """
        if ref in self.branches:
            # Checkout branch
            self.current_branch = ref
            head = self.branches[ref].head_commit
            if head and head in self.snapshots:
                self.working_intent = json.loads(json.dumps(self.snapshots[head]))
            else:
                self.working_intent = self._empty_intent()
        elif ref in self.commits:
            # Checkout commit (detached HEAD state)
            # For simplicity, create a temp branch
            self.working_intent = json.loads(json.dumps(self.snapshots[ref]))
        else:
            raise ValueError(f"Unknown ref: {ref}")

        self.staged_changes = []
        self._save_history()
        logger.info(f"Checked out {ref}")

    def branch(self, name: str, from_ref: Optional[str] = None) -> Branch:
        """
        Create a new branch.

        Args:
            name: Branch name
            from_ref: Commit ID to branch from (default: current HEAD)

        Returns:
            The new branch
        """
        if name in self.branches:
            raise ValueError(f"Branch {name} already exists")

        if from_ref is None:
            from_ref = self.branches[self.current_branch].head_commit

        branch = Branch(
            name=name,
            head_commit=from_ref,
            created_at=datetime.utcnow(),
            created_from=from_ref,
        )

        self.branches[name] = branch
        self._save_history()

        logger.info(f"Created branch {name} from {from_ref}")
        return branch

    def list_branches(self) -> list[dict]:
        """List all branches."""
        return [
            {
                "name": b.name,
                "head": b.head_commit,
                "current": b.name == self.current_branch,
            }
            for b in self.branches.values()
        ]

    def delete_branch(self, name: str):
        """Delete a branch."""
        if name == self.current_branch:
            raise ValueError("Cannot delete current branch")
        if name == "main":
            raise ValueError("Cannot delete main branch")
        if name in self.branches:
            del self.branches[name]
            self._save_history()

    def tag(self, name: str, commit_id: Optional[str] = None, message: str = "") -> Tag:
        """
        Create a tag.

        Args:
            name: Tag name
            commit_id: Commit to tag (default: current HEAD)
            message: Tag message

        Returns:
            The new tag
        """
        if name in self.tags:
            raise ValueError(f"Tag {name} already exists")

        if commit_id is None:
            commit_id = self.branches[self.current_branch].head_commit

        tag = Tag(
            name=name,
            commit_id=commit_id,
            message=message,
            created_at=datetime.utcnow(),
        )

        self.tags[name] = tag
        self._save_history()

        logger.info(f"Created tag {name} at {commit_id}")
        return tag

    def list_tags(self) -> list[dict]:
        """List all tags."""
        return [
            {
                "name": t.name,
                "commit": t.commit_id,
                "message": t.message,
            }
            for t in self.tags.values()
        ]

    def revert(self, commit_id: str, author: str = "user") -> str:
        """
        Revert a commit by creating an inverse commit.

        Args:
            commit_id: The commit to revert
            author: Author name

        Returns:
            The new revert commit ID
        """
        commit = self.commits.get(commit_id)
        if not commit:
            raise ValueError(f"Commit {commit_id} not found")

        # Create inverse changes
        inverse_changes = []
        for change in commit.changes:
            if change.change_type == ChangeType.ADD:
                inverse_changes.append(Change(
                    change_type=ChangeType.DELETE,
                    path=change.path,
                    old_value=change.new_value,
                ))
            elif change.change_type == ChangeType.DELETE:
                inverse_changes.append(Change(
                    change_type=ChangeType.ADD,
                    path=change.path,
                    new_value=change.old_value,
                ))
            else:
                inverse_changes.append(Change(
                    change_type=ChangeType.MODIFY,
                    path=change.path,
                    old_value=change.new_value,
                    new_value=change.old_value,
                ))

        # Apply inverse to working intent
        self.working_intent = self._apply_changes(self.working_intent, inverse_changes)
        self.staged_changes = inverse_changes

        return self.commit(f"Revert \"{commit.message}\"", author)

    def reset(self, commit_id: str, mode: str = "soft"):
        """
        Reset to a specific commit.

        Args:
            commit_id: Target commit ID
            mode: "soft" (keep changes), "hard" (discard changes)
        """
        if commit_id not in self.commits:
            raise ValueError(f"Commit {commit_id} not found")

        self.branches[self.current_branch].head_commit = commit_id

        if mode == "hard":
            self.working_intent = json.loads(json.dumps(self.snapshots[commit_id]))
            self.staged_changes = []

        self._save_history()
        logger.info(f"Reset {self.current_branch} to {commit_id} ({mode})")

    def merge(self, source_branch: str, author: str = "user") -> str:
        """
        Merge another branch into the current branch.

        Args:
            source_branch: Branch to merge from
            author: Author name

        Returns:
            The merge commit ID
        """
        if source_branch not in self.branches:
            raise ValueError(f"Branch {source_branch} not found")

        source_head = self.branches[source_branch].head_commit
        target_head = self.branches[self.current_branch].head_commit

        if not source_head:
            raise ValueError(f"Branch {source_branch} has no commits")

        # Get source state
        source_intent = self.snapshots.get(source_head, self._empty_intent())
        target_intent = self.snapshots.get(target_head, self._empty_intent())

        # Simple merge: apply source changes on top of target
        # Real implementation would need 3-way merge with common ancestor
        changes = self._diff_intents(target_intent, source_intent)

        if not changes:
            return target_head  # Already up to date

        self.working_intent = source_intent.copy()
        self.staged_changes = changes

        return self.commit(
            f"Merge branch '{source_branch}' into {self.current_branch}",
            author
        )

    # Intent manipulation

    def set_field(self, path: str, value: Any):
        """
        Set a field in the working intent.

        Args:
            path: Dot-separated path (e.g., "emotional.moodPrimary")
            value: The value to set
        """
        parts = path.split(".")
        target = self.working_intent

        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

    def get_field(self, path: str) -> Any:
        """
        Get a field from the working intent.

        Args:
            path: Dot-separated path

        Returns:
            The field value or None
        """
        parts = path.split(".")
        target = self.working_intent

        for part in parts:
            if not isinstance(target, dict) or part not in target:
                return None
            target = target[part]

        return target

    def get_intent(self) -> dict:
        """Get the current working intent."""
        return json.loads(json.dumps(self.working_intent))

    def set_intent(self, intent: dict):
        """Set the entire working intent."""
        self.working_intent = json.loads(json.dumps(intent))
