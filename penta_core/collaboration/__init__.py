"""
iDAW Collaboration Module.

Real-time multi-user session sharing for collaborative music production.
"""

from .websocket_server import (
    CollaborationServer,
    Session,
    Participant,
    MessageType,
)

from .intent_versioning import (
    IntentVersionControl,
    Commit,
    Branch,
    Tag,
    Change,
    ChangeType,
)

from .collab_ui import (
    CollaborationClient,
    CollaboratorInfo,
    render_collaboration_ui,
)

__all__ = [
    # WebSocket server
    "CollaborationServer",
    "Session",
    "Participant",
    "MessageType",
    # Intent versioning
    "IntentVersionControl",
    "Commit",
    "Branch",
    "Tag",
    "Change",
    "ChangeType",
    # Collaboration UI
    "CollaborationClient",
    "CollaboratorInfo",
    "render_collaboration_ui",
]
