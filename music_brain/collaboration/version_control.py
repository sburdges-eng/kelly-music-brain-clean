"""
Version Control - Song intent version history.

Provides git-like version control for song intents and project state.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import hashlib
import json
import uuid

from music_brain.utils.path_utils import safe_path, ensure_path_exists


@dataclass
class IntentDiff:
    """Represents changes between two intent versions."""
    added: Dict[str, Any] = field(default_factory=dict)
    removed: Dict[str, Any] = field(default_factory=dict)
    modified: Dict[str, tuple] = field(default_factory=dict)  # (old, new)

    def is_empty(self) -> bool:
        """Check if diff is empty (no changes)."""
        return not self.added and not self.removed and not self.modified

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "added": self.added,
            "removed": self.removed,
            "modified": self.modified,
        }

    @classmethod
    def compute(cls, old: Dict[str, Any], new: Dict[str, Any]) -> "IntentDiff":
        """Compute diff between two intent dictionaries."""
        diff = cls()

        old_keys = set(old.keys())
        new_keys = set(new.keys())

        # Added keys
        for key in new_keys - old_keys:
            diff.added[key] = new[key]

        # Removed keys
        for key in old_keys - new_keys:
            diff.removed[key] = old[key]

        # Modified keys
        for key in old_keys & new_keys:
            if old[key] != new[key]:
                diff.modified[key] = (old[key], new[key])

        return diff


@dataclass
class IntentVersion:
    """A versioned snapshot of a song intent."""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Author info
    author_id: str = ""
    author_name: str = ""

    # Version metadata
    message: str = ""
    tags: List[str] = field(default_factory=list)

    # Content
    intent_data: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    # Diff from parent
    diff: Optional[IntentDiff] = None

    def compute_hash(self) -> str:
        """Compute content hash."""
        content = json.dumps(self.intent_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "author_id": self.author_id,
            "author_name": self.author_name,
            "message": self.message,
            "tags": self.tags,
            "intent_data": self.intent_data,
            "content_hash": self.content_hash,
            "diff": self.diff.to_dict() if self.diff else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentVersion":
        """Create from dictionary."""
        version = cls(
            version_id=data.get("version_id", str(uuid.uuid4())),
            parent_id=data.get("parent_id"),
            author_id=data.get("author_id", ""),
            author_name=data.get("author_name", ""),
            message=data.get("message", ""),
            tags=data.get("tags", []),
            intent_data=data.get("intent_data", {}),
            content_hash=data.get("content_hash", ""),
        )

        if "created_at" in data:
            version.created_at = datetime.fromisoformat(data["created_at"])

        return version


@dataclass
class IntentBranch:
    """A branch in the version history."""
    name: str
    head_id: str  # Latest version ID on this branch
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


class IntentVersionControl:
    """
    Version control system for song intents.

    Provides git-like functionality for tracking changes.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self._versions: Dict[str, IntentVersion] = {}
        self._branches: Dict[str, IntentBranch] = {}
        self._current_branch: str = "main"
        self._head_id: Optional[str] = None

        if storage_path:
            self._storage_path = safe_path(storage_path)
            self._storage_path.ensure_dir()
            self._load()
        else:
            self._storage_path = None

        # Initialize main branch
        if "main" not in self._branches:
            self._branches["main"] = IntentBranch(name="main", head_id="")

    def commit(
        self,
        intent_data: Dict[str, Any],
        message: str,
        author_id: str = "",
        author_name: str = "",
        tags: Optional[List[str]] = None,
    ) -> IntentVersion:
        """
        Create a new version (commit).

        Args:
            intent_data: The intent data to version
            message: Commit message
            author_id: Author's user ID
            author_name: Author's display name
            tags: Optional tags for the version

        Returns:
            The created version
        """
        # Compute diff from parent
        diff = None
        if self._head_id and self._head_id in self._versions:
            parent = self._versions[self._head_id]
            diff = IntentDiff.compute(parent.intent_data, intent_data)

        version = IntentVersion(
            parent_id=self._head_id,
            author_id=author_id,
            author_name=author_name,
            message=message,
            tags=tags or [],
            intent_data=intent_data,
            diff=diff,
        )
        version.content_hash = version.compute_hash()

        # Store version
        self._versions[version.version_id] = version

        # Update branch head
        self._head_id = version.version_id
        self._branches[self._current_branch].head_id = version.version_id

        # Persist if storage path set
        if self._storage_path:
            self._save()

        return version

    def get_version(self, version_id: str) -> Optional[IntentVersion]:
        """Get a specific version."""
        return self._versions.get(version_id)

    def get_current(self) -> Optional[IntentVersion]:
        """Get the current version (HEAD)."""
        if self._head_id:
            return self._versions.get(self._head_id)
        return None

    def get_history(
        self,
        limit: int = 100,
        from_version: Optional[str] = None,
    ) -> List[IntentVersion]:
        """
        Get version history.

        Args:
            limit: Maximum number of versions to return
            from_version: Start from this version (default: HEAD)

        Returns:
            List of versions, newest first
        """
        history = []
        current_id = from_version or self._head_id

        while current_id and len(history) < limit:
            if current_id in self._versions:
                version = self._versions[current_id]
                history.append(version)
                current_id = version.parent_id
            else:
                break

        return history

    def restore(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore intent data from a specific version.

        Args:
            version_id: Version to restore

        Returns:
            The intent data from that version
        """
        version = self._versions.get(version_id)
        if version:
            return version.intent_data.copy()
        return None

    def checkout(self, version_id: str) -> bool:
        """
        Move HEAD to a specific version.

        Args:
            version_id: Version to checkout

        Returns:
            True if successful
        """
        if version_id in self._versions:
            self._head_id = version_id
            return True
        return False

    def create_branch(self, name: str, from_version: Optional[str] = None) -> IntentBranch:
        """
        Create a new branch.

        Args:
            name: Branch name
            from_version: Version to branch from (default: HEAD)

        Returns:
            The created branch
        """
        start_id = from_version or self._head_id or ""

        branch = IntentBranch(
            name=name,
            head_id=start_id,
        )
        self._branches[name] = branch

        if self._storage_path:
            self._save()

        return branch

    def switch_branch(self, name: str) -> bool:
        """
        Switch to a different branch.

        Args:
            name: Branch name

        Returns:
            True if successful
        """
        if name in self._branches:
            self._current_branch = name
            self._head_id = self._branches[name].head_id
            return True
        return False

    def list_branches(self) -> List[IntentBranch]:
        """List all branches."""
        return list(self._branches.values())

    def get_diff(
        self,
        from_version: str,
        to_version: str,
    ) -> Optional[IntentDiff]:
        """
        Get diff between two versions.

        Args:
            from_version: Source version ID
            to_version: Target version ID

        Returns:
            Diff between versions
        """
        from_v = self._versions.get(from_version)
        to_v = self._versions.get(to_version)

        if from_v and to_v:
            return IntentDiff.compute(from_v.intent_data, to_v.intent_data)
        return None

    def find_by_tag(self, tag: str) -> List[IntentVersion]:
        """Find versions with a specific tag."""
        return [v for v in self._versions.values() if tag in v.tags]

    def find_by_author(self, author_id: str) -> List[IntentVersion]:
        """Find versions by author."""
        return [v for v in self._versions.values() if v.author_id == author_id]

    def _save(self) -> None:
        """Save version history to storage."""
        if not self._storage_path:
            return

        data = {
            "versions": {vid: v.to_dict() for vid, v in self._versions.items()},
            "branches": {
                name: {
                    "name": b.name,
                    "head_id": b.head_id,
                    "created_at": b.created_at.isoformat(),
                    "description": b.description,
                }
                for name, b in self._branches.items()
            },
            "current_branch": self._current_branch,
            "head_id": self._head_id,
        }

        versions_file = self._storage_path / "versions.json"
        with open(str(versions_file), 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load version history from storage."""
        if not self._storage_path:
            return

        versions_file = self._storage_path / "versions.json"
        if not versions_file.to_path().exists():
            return

        with open(str(versions_file)) as f:
            data = json.load(f)

        # Load versions
        for vid, vdata in data.get("versions", {}).items():
            self._versions[vid] = IntentVersion.from_dict(vdata)

        # Load branches
        for name, bdata in data.get("branches", {}).items():
            self._branches[name] = IntentBranch(
                name=bdata["name"],
                head_id=bdata["head_id"],
                created_at=datetime.fromisoformat(bdata["created_at"]),
                description=bdata.get("description", ""),
            )

        self._current_branch = data.get("current_branch", "main")
        self._head_id = data.get("head_id")


# Convenience functions
_default_vc: Optional[IntentVersionControl] = None


def get_version_control(storage_path: Optional[str] = None) -> IntentVersionControl:
    """Get or create the default version control instance."""
    global _default_vc
    if _default_vc is None:
        _default_vc = IntentVersionControl(storage_path)
    return _default_vc


def create_version(
    intent_data: Dict[str, Any],
    message: str,
    author_id: str = "",
    author_name: str = "",
) -> IntentVersion:
    """Create a new version (commit)."""
    return get_version_control().commit(
        intent_data, message, author_id, author_name
    )


def get_history(limit: int = 100) -> List[IntentVersion]:
    """Get version history."""
    return get_version_control().get_history(limit)


def restore_version(version_id: str) -> Optional[Dict[str, Any]]:
    """Restore intent data from a specific version."""
    return get_version_control().restore(version_id)
