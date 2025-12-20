"""
Session Management - Collaborative session handling.

Manages multi-user sessions for real-time music collaboration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import uuid
import json


class SessionRole(Enum):
    """Participant roles in a collaborative session."""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"


class SessionState(Enum):
    """Session lifecycle states."""
    CREATING = "creating"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


@dataclass
class SessionParticipant:
    """A participant in a collaborative session."""
    user_id: str
    username: str
    role: SessionRole = SessionRole.VIEWER
    color: str = "#4a90d9"  # User's cursor/highlight color

    # Presence info
    is_online: bool = True
    last_seen: datetime = field(default_factory=datetime.utcnow)

    # Cursor/selection state
    cursor_position: Optional[Dict[str, Any]] = None  # Track, bar, beat
    selection: Optional[Dict[str, Any]] = None

    # Connection info
    connection_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role.value,
            "color": self.color,
            "is_online": self.is_online,
            "last_seen": self.last_seen.isoformat(),
            "cursor_position": self.cursor_position,
            "selection": self.selection,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionParticipant":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            role=SessionRole(data.get("role", "viewer")),
            color=data.get("color", "#4a90d9"),
            is_online=data.get("is_online", True),
            last_seen=datetime.fromisoformat(data.get("last_seen", datetime.utcnow().isoformat())),
            cursor_position=data.get("cursor_position"),
            selection=data.get("selection"),
        )


@dataclass
class CollaborativeSession:
    """
    A collaborative music creation session.

    Supports multiple participants working on the same project
    with real-time synchronization.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Session"
    created_at: datetime = field(default_factory=datetime.utcnow)
    state: SessionState = SessionState.CREATING

    # Project reference
    project_id: Optional[str] = None
    intent_id: Optional[str] = None

    # Participants
    participants: Dict[str, SessionParticipant] = field(default_factory=dict)
    owner_id: Optional[str] = None

    # Session settings
    max_participants: int = 10
    allow_anonymous: bool = False
    require_approval: bool = False

    # Session data
    shared_state: Dict[str, Any] = field(default_factory=dict)
    pending_operations: List[Dict[str, Any]] = field(default_factory=list)

    # Callbacks
    _on_join: Optional[Callable] = None
    _on_leave: Optional[Callable] = None
    _on_update: Optional[Callable] = None

    def add_participant(
        self,
        user_id: str,
        username: str,
        role: SessionRole = SessionRole.VIEWER,
    ) -> SessionParticipant:
        """Add a participant to the session."""
        if len(self.participants) >= self.max_participants:
            raise ValueError("Session is full")

        # Assign unique color
        colors = [
            "#4a90d9", "#e74c3c", "#2ecc71", "#f1c40f",
            "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
            "#16a085", "#c0392b",
        ]
        used_colors = {p.color for p in self.participants.values()}
        color = next((c for c in colors if c not in used_colors), colors[0])

        participant = SessionParticipant(
            user_id=user_id,
            username=username,
            role=role,
            color=color,
        )

        self.participants[user_id] = participant

        if self._on_join:
            self._on_join(participant)

        return participant

    def remove_participant(self, user_id: str) -> bool:
        """Remove a participant from the session."""
        if user_id in self.participants:
            participant = self.participants.pop(user_id)

            if self._on_leave:
                self._on_leave(participant)

            return True
        return False

    def get_participant(self, user_id: str) -> Optional[SessionParticipant]:
        """Get a participant by ID."""
        return self.participants.get(user_id)

    def update_presence(
        self,
        user_id: str,
        cursor_position: Optional[Dict] = None,
        selection: Optional[Dict] = None,
    ) -> None:
        """Update a participant's presence info."""
        if user_id in self.participants:
            participant = self.participants[user_id]
            participant.last_seen = datetime.utcnow()
            participant.is_online = True

            if cursor_position is not None:
                participant.cursor_position = cursor_position
            if selection is not None:
                participant.selection = selection

            if self._on_update:
                self._on_update(participant)

    def set_participant_offline(self, user_id: str) -> None:
        """Mark a participant as offline."""
        if user_id in self.participants:
            self.participants[user_id].is_online = False
            self.participants[user_id].last_seen = datetime.utcnow()

    def get_online_participants(self) -> List[SessionParticipant]:
        """Get list of online participants."""
        return [p for p in self.participants.values() if p.is_online]

    def can_edit(self, user_id: str) -> bool:
        """Check if a user can edit the session."""
        participant = self.participants.get(user_id)
        if not participant:
            return False
        return participant.role in [SessionRole.OWNER, SessionRole.EDITOR]

    def change_role(self, user_id: str, new_role: SessionRole) -> bool:
        """Change a participant's role."""
        if user_id in self.participants:
            self.participants[user_id].role = new_role
            return True
        return False

    def update_shared_state(self, key: str, value: Any) -> None:
        """Update shared session state."""
        self.shared_state[key] = value

    def get_shared_state(self, key: str, default: Any = None) -> Any:
        """Get shared session state."""
        return self.shared_state.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "state": self.state.value,
            "project_id": self.project_id,
            "intent_id": self.intent_id,
            "participants": {
                uid: p.to_dict() for uid, p in self.participants.items()
            },
            "owner_id": self.owner_id,
            "max_participants": self.max_participants,
            "shared_state": self.shared_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollaborativeSession":
        """Create session from dictionary."""
        session = cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            name=data.get("name", "Untitled Session"),
            state=SessionState(data.get("state", "creating")),
            project_id=data.get("project_id"),
            intent_id=data.get("intent_id"),
            owner_id=data.get("owner_id"),
            max_participants=data.get("max_participants", 10),
        )

        if "created_at" in data:
            session.created_at = datetime.fromisoformat(data["created_at"])

        for uid, pdata in data.get("participants", {}).items():
            session.participants[uid] = SessionParticipant.from_dict(pdata)

        session.shared_state = data.get("shared_state", {})

        return session


class SessionManager:
    """Manager for multiple collaborative sessions."""

    def __init__(self):
        self._sessions: Dict[str, CollaborativeSession] = {}

    def create_session(
        self,
        name: str,
        owner_id: str,
        owner_name: str,
        project_id: Optional[str] = None,
    ) -> CollaborativeSession:
        """Create a new collaborative session."""
        session = CollaborativeSession(
            name=name,
            owner_id=owner_id,
            project_id=project_id,
            state=SessionState.ACTIVE,
        )

        # Add owner as first participant
        session.add_participant(owner_id, owner_name, SessionRole.OWNER)

        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[CollaborativeSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self, user_id: Optional[str] = None) -> List[CollaborativeSession]:
        """List sessions, optionally filtered by participant."""
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if user_id in s.participants]
        return sessions

    def close_session(self, session_id: str) -> bool:
        """Close a session."""
        if session_id in self._sessions:
            self._sessions[session_id].state = SessionState.CLOSED
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# Singleton session manager
_session_manager = SessionManager()


def create_session(
    name: str,
    owner_id: str,
    owner_name: str,
    project_id: Optional[str] = None,
) -> CollaborativeSession:
    """Create a new collaborative session."""
    return _session_manager.create_session(name, owner_id, owner_name, project_id)


def join_session(
    session_id: str,
    user_id: str,
    username: str,
    role: SessionRole = SessionRole.VIEWER,
) -> Optional[SessionParticipant]:
    """Join an existing session."""
    session = _session_manager.get_session(session_id)
    if session:
        return session.add_participant(user_id, username, role)
    return None


def leave_session(session_id: str, user_id: str) -> bool:
    """Leave a session."""
    session = _session_manager.get_session(session_id)
    if session:
        return session.remove_participant(user_id)
    return False
