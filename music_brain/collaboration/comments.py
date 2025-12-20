"""
Comments and Annotations - Collaborative feedback system.

Provides commenting and annotation features for collaborative music creation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import uuid


class AnnotationType(Enum):
    """Types of annotations."""
    COMMENT = "comment"
    SUGGESTION = "suggestion"
    QUESTION = "question"
    APPROVAL = "approval"
    ISSUE = "issue"


class AnnotationTarget(Enum):
    """What the annotation targets."""
    MEASURE = "measure"
    NOTE = "note"
    CHORD = "chord"
    SECTION = "section"
    TRACK = "track"
    INTENT = "intent"
    GENERAL = "general"


@dataclass
class Comment:
    """A comment in a thread."""
    comment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    author_id: str = ""
    author_name: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    edited_at: Optional[datetime] = None

    # Reactions
    reactions: Dict[str, List[str]] = field(default_factory=dict)  # emoji -> user_ids

    # Mentions
    mentions: List[str] = field(default_factory=list)  # user_ids mentioned

    def add_reaction(self, emoji: str, user_id: str) -> None:
        """Add a reaction."""
        if emoji not in self.reactions:
            self.reactions[emoji] = []
        if user_id not in self.reactions[emoji]:
            self.reactions[emoji].append(user_id)

    def remove_reaction(self, emoji: str, user_id: str) -> None:
        """Remove a reaction."""
        if emoji in self.reactions and user_id in self.reactions[emoji]:
            self.reactions[emoji].remove(user_id)

    def edit(self, new_text: str) -> None:
        """Edit the comment."""
        self.text = new_text
        self.edited_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "comment_id": self.comment_id,
            "text": self.text,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "created_at": self.created_at.isoformat(),
            "edited_at": self.edited_at.isoformat() if self.edited_at else None,
            "reactions": self.reactions,
            "mentions": self.mentions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Comment":
        """Create from dictionary."""
        comment = cls(
            comment_id=data.get("comment_id", str(uuid.uuid4())),
            text=data.get("text", ""),
            author_id=data.get("author_id", ""),
            author_name=data.get("author_name", ""),
            reactions=data.get("reactions", {}),
            mentions=data.get("mentions", []),
        )

        if "created_at" in data:
            comment.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("edited_at"):
            comment.edited_at = datetime.fromisoformat(data["edited_at"])

        return comment


@dataclass
class Annotation:
    """
    An annotation attached to a specific location.

    Can reference measures, notes, chords, or other elements.
    """
    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    annotation_type: AnnotationType = AnnotationType.COMMENT
    target_type: AnnotationTarget = AnnotationTarget.GENERAL

    # Location info
    measure: Optional[int] = None
    beat: Optional[float] = None
    track: Optional[str] = None
    element_id: Optional[str] = None  # Specific element ID

    # Visual position (for free-form annotations)
    x: Optional[float] = None
    y: Optional[float] = None

    # Color/style
    color: str = "#ffeb3b"  # Yellow highlight by default

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    author_id: str = ""
    author_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "annotation_type": self.annotation_type.value,
            "target_type": self.target_type.value,
            "measure": self.measure,
            "beat": self.beat,
            "track": self.track,
            "element_id": self.element_id,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "created_at": self.created_at.isoformat(),
            "author_id": self.author_id,
            "author_name": self.author_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """Create from dictionary."""
        annotation = cls(
            annotation_id=data.get("annotation_id", str(uuid.uuid4())),
            annotation_type=AnnotationType(data.get("annotation_type", "comment")),
            target_type=AnnotationTarget(data.get("target_type", "general")),
            measure=data.get("measure"),
            beat=data.get("beat"),
            track=data.get("track"),
            element_id=data.get("element_id"),
            x=data.get("x"),
            y=data.get("y"),
            color=data.get("color", "#ffeb3b"),
            author_id=data.get("author_id", ""),
            author_name=data.get("author_name", ""),
        )

        if "created_at" in data:
            annotation.created_at = datetime.fromisoformat(data["created_at"])

        return annotation


@dataclass
class CommentThread:
    """
    A thread of comments attached to an annotation.

    Threads can be resolved or reopened.
    """
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    annotation: Annotation = field(default_factory=Annotation)
    comments: List[Comment] = field(default_factory=list)

    # Thread state
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_comment(
        self,
        text: str,
        author_id: str,
        author_name: str,
    ) -> Comment:
        """Add a comment to the thread."""
        comment = Comment(
            text=text,
            author_id=author_id,
            author_name=author_name,
        )
        self.comments.append(comment)
        return comment

    def resolve(self, user_id: str) -> None:
        """Resolve the thread."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolved_by = user_id

    def reopen(self) -> None:
        """Reopen a resolved thread."""
        self.resolved = False
        self.resolved_at = None
        self.resolved_by = None

    def get_participants(self) -> List[str]:
        """Get list of participant user IDs."""
        return list(set(c.author_id for c in self.comments))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "thread_id": self.thread_id,
            "annotation": self.annotation.to_dict(),
            "comments": [c.to_dict() for c in self.comments],
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommentThread":
        """Create from dictionary."""
        thread = cls(
            thread_id=data.get("thread_id", str(uuid.uuid4())),
            annotation=Annotation.from_dict(data.get("annotation", {})),
            comments=[Comment.from_dict(c) for c in data.get("comments", [])],
            resolved=data.get("resolved", False),
            resolved_by=data.get("resolved_by"),
        )

        if "created_at" in data:
            thread.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("resolved_at"):
            thread.resolved_at = datetime.fromisoformat(data["resolved_at"])

        return thread


class CommentManager:
    """
    Manager for comment threads in a project.

    Handles creation, retrieval, and resolution of comment threads.
    """

    def __init__(self):
        self._threads: Dict[str, CommentThread] = {}

    def create_thread(
        self,
        text: str,
        author_id: str,
        author_name: str,
        annotation: Optional[Annotation] = None,
        annotation_type: AnnotationType = AnnotationType.COMMENT,
        target_type: AnnotationTarget = AnnotationTarget.GENERAL,
        measure: Optional[int] = None,
        beat: Optional[float] = None,
        track: Optional[str] = None,
    ) -> CommentThread:
        """
        Create a new comment thread.

        Args:
            text: Initial comment text
            author_id: Author's user ID
            author_name: Author's display name
            annotation: Pre-created annotation (optional)
            annotation_type: Type of annotation
            target_type: What the annotation targets
            measure: Measure number
            beat: Beat position
            track: Track name

        Returns:
            The created thread
        """
        if annotation is None:
            annotation = Annotation(
                annotation_type=annotation_type,
                target_type=target_type,
                measure=measure,
                beat=beat,
                track=track,
                author_id=author_id,
                author_name=author_name,
            )

        thread = CommentThread(annotation=annotation)
        thread.add_comment(text, author_id, author_name)

        self._threads[thread.thread_id] = thread
        return thread

    def get_thread(self, thread_id: str) -> Optional[CommentThread]:
        """Get a thread by ID."""
        return self._threads.get(thread_id)

    def get_threads(
        self,
        resolved: Optional[bool] = None,
        measure: Optional[int] = None,
        track: Optional[str] = None,
    ) -> List[CommentThread]:
        """
        Get threads with optional filters.

        Args:
            resolved: Filter by resolved state
            measure: Filter by measure number
            track: Filter by track name

        Returns:
            List of matching threads
        """
        threads = list(self._threads.values())

        if resolved is not None:
            threads = [t for t in threads if t.resolved == resolved]

        if measure is not None:
            threads = [t for t in threads if t.annotation.measure == measure]

        if track is not None:
            threads = [t for t in threads if t.annotation.track == track]

        return sorted(threads, key=lambda t: t.created_at, reverse=True)

    def add_comment(
        self,
        thread_id: str,
        text: str,
        author_id: str,
        author_name: str,
    ) -> Optional[Comment]:
        """Add a comment to an existing thread."""
        thread = self._threads.get(thread_id)
        if thread:
            return thread.add_comment(text, author_id, author_name)
        return None

    def resolve_thread(self, thread_id: str, user_id: str) -> bool:
        """Resolve a thread."""
        thread = self._threads.get(thread_id)
        if thread:
            thread.resolve(user_id)
            return True
        return False

    def reopen_thread(self, thread_id: str) -> bool:
        """Reopen a resolved thread."""
        thread = self._threads.get(thread_id)
        if thread:
            thread.reopen()
            return True
        return False

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread."""
        if thread_id in self._threads:
            del self._threads[thread_id]
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert all threads to dictionary."""
        return {
            "threads": [t.to_dict() for t in self._threads.values()]
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load threads from dictionary."""
        self._threads.clear()
        for thread_data in data.get("threads", []):
            thread = CommentThread.from_dict(thread_data)
            self._threads[thread.thread_id] = thread


# Convenience functions
_comment_manager: Optional[CommentManager] = None


def get_comment_manager() -> CommentManager:
    """Get or create the comment manager."""
    global _comment_manager
    if _comment_manager is None:
        _comment_manager = CommentManager()
    return _comment_manager


def add_comment(
    text: str,
    author_id: str,
    author_name: str,
    thread_id: Optional[str] = None,
    measure: Optional[int] = None,
    beat: Optional[float] = None,
    track: Optional[str] = None,
) -> Optional[Comment]:
    """
    Add a comment.

    If thread_id is provided, adds to existing thread.
    Otherwise, creates a new thread.
    """
    manager = get_comment_manager()

    if thread_id:
        return manager.add_comment(thread_id, text, author_id, author_name)
    else:
        thread = manager.create_thread(
            text=text,
            author_id=author_id,
            author_name=author_name,
            measure=measure,
            beat=beat,
            track=track,
        )
        return thread.comments[0] if thread.comments else None


def resolve_thread(thread_id: str, user_id: str) -> bool:
    """Resolve a comment thread."""
    return get_comment_manager().resolve_thread(thread_id, user_id)


def get_threads(
    resolved: Optional[bool] = None,
    measure: Optional[int] = None,
) -> List[CommentThread]:
    """Get comment threads with optional filters."""
    return get_comment_manager().get_threads(resolved=resolved, measure=measure)
