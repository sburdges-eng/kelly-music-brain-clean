"""
WebSocket Session Sharing Server for iDAW Collaboration.

Implements the iDAW-Collab/1.0 protocol for real-time multi-user
music production sessions.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketServerProtocol = Any

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Protocol message types."""
    SESSION_JOIN = "session.join"
    SESSION_LEAVE = "session.leave"
    SESSION_SYNC = "session.sync"
    PRESENCE_UPDATE = "presence.update"
    PRESENCE_CURSOR = "presence.cursor"
    INTENT_UPDATE = "intent.update"
    INTENT_LOCK = "intent.lock"
    INTENT_UNLOCK = "intent.unlock"
    MIDI_NOTE = "midi.note"
    MIDI_CC = "midi.cc"
    ARRANGEMENT_OP = "arrangement.op"
    CHAT_MESSAGE = "chat.message"
    UNDO_REQUEST = "undo.request"
    UNDO_ACK = "undo.ack"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat.ack"
    ERROR = "error"
    AUTH = "auth"


@dataclass
class Participant:
    """A participant in a collaboration session."""
    client_id: str
    name: str
    color: str
    websocket: WebSocketServerProtocol
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    cursor: Optional[dict] = None
    active_locks: list = field(default_factory=list)
    permissions: dict = field(default_factory=lambda: {
        "can_edit_intent": True,
        "can_edit_arrangement": True,
        "can_record_midi": True,
        "can_invite": False,
        "can_kick": False,
        "is_owner": False,
    })


@dataclass
class Lock:
    """A lock on an intent field."""
    path: str
    holder: str
    expires_at: datetime


@dataclass
class Session:
    """A collaboration session."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    participants: dict = field(default_factory=dict)  # client_id -> Participant
    state: dict = field(default_factory=lambda: {
        "intent": {
            "core": {"event": "", "resistance": "", "longing": ""},
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
            "_meta": {
                "lastModified": 0,
                "modifiedBy": "",
                "version": 1,
            },
        },
        "arrangement": {
            "tracks": [],
            "sections": [],
            "markers": [],
        },
        "chat": [],
    })
    locks: dict = field(default_factory=dict)  # path -> Lock
    vector_clock: dict = field(default_factory=dict)  # client_id -> logical time
    message_history: list = field(default_factory=list)
    version: int = 1


class CollaborationServer:
    """
    WebSocket server for real-time collaboration sessions.

    Implements the iDAW-Collab/1.0 protocol with CRDT-based
    conflict resolution for intent fields.
    """

    PROTOCOL_VERSION = "1.0"
    MAX_PARTICIPANTS = 8
    HEARTBEAT_INTERVAL = 30  # seconds
    HEARTBEAT_TIMEOUT = 90  # seconds
    LOCK_DEFAULT_TIMEOUT = 60000  # ms
    MAX_MESSAGE_HISTORY = 1000

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        """
        Initialize the collaboration server.

        Args:
            host: Host address to bind to
            port: Port number to listen on
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required: pip install websockets")

        self.host = host
        self.port = port
        self.sessions: dict[str, Session] = {}
        self.client_sessions: dict[str, str] = {}  # client_id -> session_id
        self._running = False
        self._server = None

        # Message handlers
        self._handlers: dict[str, Callable] = {
            MessageType.SESSION_JOIN: self._handle_join,
            MessageType.SESSION_LEAVE: self._handle_leave,
            MessageType.PRESENCE_CURSOR: self._handle_cursor,
            MessageType.PRESENCE_UPDATE: self._handle_presence,
            MessageType.INTENT_UPDATE: self._handle_intent_update,
            MessageType.INTENT_LOCK: self._handle_intent_lock,
            MessageType.INTENT_UNLOCK: self._handle_intent_unlock,
            MessageType.MIDI_NOTE: self._handle_midi_note,
            MessageType.MIDI_CC: self._handle_midi_cc,
            MessageType.ARRANGEMENT_OP: self._handle_arrangement_op,
            MessageType.CHAT_MESSAGE: self._handle_chat,
            MessageType.UNDO_REQUEST: self._handle_undo,
            MessageType.HEARTBEAT: self._handle_heartbeat,
        }

    async def start(self):
        """Start the WebSocket server."""
        self._running = True
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=self.HEARTBEAT_INTERVAL,
            ping_timeout=self.HEARTBEAT_TIMEOUT,
        )
        logger.info(f"Collaboration server started on ws://{self.host}:{self.port}")

        # Start cleanup task
        asyncio.create_task(self._cleanup_task())

        await self._server.wait_closed()

    async def stop(self):
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Collaboration server stopped")

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new collaboration session.

        Args:
            session_id: Optional session ID, generated if not provided

        Returns:
            The session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        self.sessions[session_id] = Session(session_id=session_id)
        logger.info(f"Created session: {session_id}")
        return session_id

    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket connection."""
        client_id = str(uuid.uuid4())[:12]
        logger.info(f"New connection: {client_id} from {websocket.remote_address}")

        try:
            async for message in websocket:
                await self._process_message(websocket, client_id, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {client_id}")
        finally:
            await self._handle_disconnect(client_id)

    async def _process_message(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        raw_message: str
    ):
        """Process an incoming message."""
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type")

            if msg_type not in self._handlers:
                await self._send_error(
                    websocket,
                    "INVALID_OPERATION",
                    f"Unknown message type: {msg_type}"
                )
                return

            handler = self._handlers[msg_type]
            await handler(websocket, client_id, message)

        except json.JSONDecodeError:
            await self._send_error(websocket, "INVALID_OPERATION", "Invalid JSON")
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            await self._send_error(websocket, "SERVER_ERROR", str(e))

    async def _handle_join(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle session join request."""
        payload = message.get("payload", {})
        session_id = payload.get("sessionId")
        client_name = payload.get("clientName", f"User-{client_id[:4]}")
        client_color = payload.get("clientColor", "#4A90D9")

        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.create_session(session_id)

        session = self.sessions[session_id]

        # Check participant limit
        if len(session.participants) >= self.MAX_PARTICIPANTS:
            await self._send_error(websocket, "SESSION_FULL", "Session is full")
            return

        # Create participant
        is_owner = len(session.participants) == 0
        participant = Participant(
            client_id=client_id,
            name=client_name,
            color=client_color,
            websocket=websocket,
        )
        participant.permissions["is_owner"] = is_owner
        participant.permissions["can_invite"] = is_owner
        participant.permissions["can_kick"] = is_owner

        session.participants[client_id] = participant
        session.vector_clock[client_id] = 0
        self.client_sessions[client_id] = session_id

        logger.info(f"Client {client_id} ({client_name}) joined session {session_id}")

        # Send sync to joining client
        await self._send(websocket, {
            "v": self.PROTOCOL_VERSION,
            "type": MessageType.SESSION_SYNC,
            "id": str(uuid.uuid4()),
            "session": session_id,
            "sender": "server",
            "ts": self._get_timestamp(),
            "payload": {
                "sessionId": session_id,
                "clientId": client_id,
                "participants": [
                    {
                        "clientId": p.client_id,
                        "name": p.name,
                        "color": p.color,
                        "joinedAt": p.joined_at.isoformat(),
                        "cursor": p.cursor,
                        "activeLocks": p.active_locks,
                    }
                    for p in session.participants.values()
                ],
                "state": session.state,
                "version": session.version,
            },
        })

        # Notify other participants
        await self._broadcast(session, {
            "v": self.PROTOCOL_VERSION,
            "type": MessageType.PRESENCE_UPDATE,
            "id": str(uuid.uuid4()),
            "session": session_id,
            "sender": "server",
            "ts": self._get_timestamp(),
            "payload": {
                "event": "join",
                "clientId": client_id,
                "name": client_name,
                "color": client_color,
            },
        }, exclude=client_id)

    async def _handle_leave(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle session leave request."""
        await self._handle_disconnect(client_id, reason="user")

    async def _handle_disconnect(self, client_id: str, reason: str = "disconnect"):
        """Handle client disconnection."""
        session_id = self.client_sessions.get(client_id)
        if not session_id or session_id not in self.sessions:
            return

        session = self.sessions[session_id]

        if client_id in session.participants:
            # Release all locks held by this client
            locks_to_remove = [
                path for path, lock in session.locks.items()
                if lock.holder == client_id
            ]
            for path in locks_to_remove:
                del session.locks[path]

            # Remove participant
            del session.participants[client_id]
            del self.client_sessions[client_id]

            logger.info(f"Client {client_id} left session {session_id}")

            # Notify others
            await self._broadcast(session, {
                "v": self.PROTOCOL_VERSION,
                "type": MessageType.PRESENCE_UPDATE,
                "id": str(uuid.uuid4()),
                "session": session_id,
                "sender": "server",
                "ts": self._get_timestamp(),
                "payload": {
                    "event": "leave",
                    "clientId": client_id,
                    "reason": reason,
                },
            })

            # Clean up empty sessions
            if not session.participants:
                del self.sessions[session_id]
                logger.info(f"Session {session_id} closed (no participants)")

    async def _handle_cursor(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle cursor position update."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]
        participant = session.participants.get(client_id)
        if participant:
            participant.cursor = message.get("payload")

        # Broadcast to others
        await self._broadcast(session, message, exclude=client_id)

    async def _handle_presence(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle presence status update."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]
        await self._broadcast(session, message, exclude=client_id)

    async def _handle_intent_update(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle intent field update with CRDT merge."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]
        payload = message.get("payload", {})
        path = payload.get("path")
        value = payload.get("value")
        vclock = payload.get("vclock", {})

        # Check for lock
        if path in session.locks:
            lock = session.locks[path]
            if lock.holder != client_id and lock.expires_at > datetime.utcnow():
                await self._send_error(
                    websocket,
                    "LOCK_CONFLICT",
                    f"Field {path} is locked by another user"
                )
                return

        # Apply update using LWW (Last Writer Wins) with vector clock
        self._apply_intent_update(session, path, value, vclock, client_id)

        # Update version
        session.version += 1
        session.vector_clock[client_id] = session.vector_clock.get(client_id, 0) + 1

        # Add to history
        self._add_to_history(session, message)

        # Broadcast to all
        await self._broadcast(session, message)

    def _apply_intent_update(
        self,
        session: Session,
        path: str,
        value: Any,
        vclock: dict,
        client_id: str
    ):
        """Apply an intent update to the session state."""
        parts = path.split(".")
        target = session.state["intent"]

        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Apply the value
        final_key = parts[-1]
        target[final_key] = value

        # Update metadata
        session.state["intent"]["_meta"]["lastModified"] = self._get_timestamp()
        session.state["intent"]["_meta"]["modifiedBy"] = client_id
        session.state["intent"]["_meta"]["version"] = session.version

    async def _handle_intent_lock(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle intent lock request."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]
        payload = message.get("payload", {})
        path = payload.get("path")
        timeout = payload.get("timeout", self.LOCK_DEFAULT_TIMEOUT)

        # Check if already locked
        if path in session.locks:
            lock = session.locks[path]
            if lock.holder != client_id and lock.expires_at > datetime.utcnow():
                await self._send(websocket, {
                    "v": self.PROTOCOL_VERSION,
                    "type": MessageType.INTENT_LOCK,
                    "id": str(uuid.uuid4()),
                    "session": session_id,
                    "sender": "server",
                    "ts": self._get_timestamp(),
                    "payload": {
                        "path": path,
                        "granted": False,
                        "holder": lock.holder,
                        "expiresAt": lock.expires_at.isoformat(),
                    },
                })
                return

        # Grant lock
        expires_at = datetime.utcnow() + timedelta(milliseconds=timeout)
        session.locks[path] = Lock(path=path, holder=client_id, expires_at=expires_at)

        participant = session.participants.get(client_id)
        if participant:
            participant.active_locks.append(path)

        await self._send(websocket, {
            "v": self.PROTOCOL_VERSION,
            "type": MessageType.INTENT_LOCK,
            "id": str(uuid.uuid4()),
            "session": session_id,
            "sender": "server",
            "ts": self._get_timestamp(),
            "payload": {
                "path": path,
                "granted": True,
                "expiresAt": expires_at.isoformat(),
            },
        })

        # Notify others
        await self._broadcast(session, {
            "v": self.PROTOCOL_VERSION,
            "type": MessageType.INTENT_LOCK,
            "id": str(uuid.uuid4()),
            "session": session_id,
            "sender": client_id,
            "ts": self._get_timestamp(),
            "payload": {
                "path": path,
                "holder": client_id,
                "expiresAt": expires_at.isoformat(),
            },
        }, exclude=client_id)

    async def _handle_intent_unlock(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle intent unlock request."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]
        payload = message.get("payload", {})
        path = payload.get("path")

        if path in session.locks and session.locks[path].holder == client_id:
            del session.locks[path]

            participant = session.participants.get(client_id)
            if participant and path in participant.active_locks:
                participant.active_locks.remove(path)

            # Notify others
            await self._broadcast(session, {
                "v": self.PROTOCOL_VERSION,
                "type": MessageType.INTENT_UNLOCK,
                "id": str(uuid.uuid4()),
                "session": session_id,
                "sender": client_id,
                "ts": self._get_timestamp(),
                "payload": {"path": path},
            })

    async def _handle_midi_note(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle MIDI note event."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]

        # Broadcast to all (including sender for latency measurement)
        await self._broadcast(session, message)

    async def _handle_midi_cc(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle MIDI CC event."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]
        await self._broadcast(session, message)

    async def _handle_arrangement_op(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle arrangement operation."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]
        payload = message.get("payload", {})

        # Apply operation to state
        self._apply_arrangement_op(session, payload)

        # Update version
        session.version += 1

        # Add to history
        self._add_to_history(session, message)

        # Broadcast to all
        await self._broadcast(session, message)

    def _apply_arrangement_op(self, session: Session, payload: dict):
        """Apply an arrangement operation to the session state."""
        target = payload.get("target")
        operation = payload.get("operation", {})
        op_type = operation.get("type")

        arrangement = session.state["arrangement"]

        if target == "track":
            tracks = arrangement.get("tracks", [])
            if op_type == "add":
                track = operation.get("track")
                index = operation.get("index", len(tracks))
                tracks.insert(index, track)
            elif op_type == "remove":
                track_id = operation.get("trackId")
                tracks[:] = [t for t in tracks if t.get("id") != track_id]
            elif op_type == "rename":
                track_id = operation.get("trackId")
                for track in tracks:
                    if track.get("id") == track_id:
                        track["name"] = operation.get("name")
                        break
            arrangement["tracks"] = tracks

        elif target == "section":
            sections = arrangement.get("sections", [])
            if op_type == "add":
                sections.append(operation.get("section"))
            elif op_type == "remove":
                section_id = operation.get("sectionId")
                sections[:] = [s for s in sections if s.get("id") != section_id]
            arrangement["sections"] = sections

        elif target == "marker":
            markers = arrangement.get("markers", [])
            if op_type == "add":
                markers.append(operation.get("marker"))
            elif op_type == "remove":
                marker_id = operation.get("markerId")
                markers[:] = [m for m in markers if m.get("id") != marker_id]
            arrangement["markers"] = markers

    async def _handle_chat(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle chat message."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]

        # Store in chat history
        chat_entry = {
            "id": message.get("id"),
            "sender": client_id,
            "ts": self._get_timestamp(),
            "payload": message.get("payload"),
        }
        session.state["chat"].append(chat_entry)

        # Keep chat history bounded
        if len(session.state["chat"]) > 100:
            session.state["chat"] = session.state["chat"][-100:]

        # Broadcast to all
        await self._broadcast(session, message)

    async def _handle_undo(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle undo request."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]
        payload = message.get("payload", {})
        operation_id = payload.get("operationId")

        # Find operation in history
        for i, hist_msg in enumerate(reversed(session.message_history)):
            if hist_msg.get("id") == operation_id:
                if hist_msg.get("sender") != client_id:
                    await self._send_error(
                        websocket,
                        "PERMISSION_DENIED",
                        "Can only undo your own operations"
                    )
                    return

                # Create inverse operation (simplified)
                inverse_op = self._create_inverse_op(hist_msg)

                if inverse_op:
                    await self._broadcast(session, {
                        "v": self.PROTOCOL_VERSION,
                        "type": MessageType.UNDO_ACK,
                        "id": str(uuid.uuid4()),
                        "session": session_id,
                        "sender": "server",
                        "ts": self._get_timestamp(),
                        "payload": {
                            "operationId": operation_id,
                            "inverseOp": inverse_op,
                            "success": True,
                        },
                    })
                return

        await self._send_error(websocket, "INVALID_OPERATION", "Operation not found")

    def _create_inverse_op(self, message: dict) -> Optional[dict]:
        """Create an inverse operation for undo."""
        # Simplified inverse - real implementation would need
        # full CRDT inverse operations
        msg_type = message.get("type")

        if msg_type == MessageType.INTENT_UPDATE:
            # Would need to store previous value
            return None

        if msg_type == MessageType.ARRANGEMENT_OP:
            payload = message.get("payload", {})
            op = payload.get("operation", {})
            op_type = op.get("type")

            if op_type == "add":
                return {
                    "type": MessageType.ARRANGEMENT_OP,
                    "payload": {
                        "target": payload.get("target"),
                        "operation": {
                            "type": "remove",
                            "trackId": op.get("track", {}).get("id"),
                            "sectionId": op.get("section", {}).get("id"),
                            "markerId": op.get("marker", {}).get("id"),
                        },
                    },
                }

        return None

    async def _handle_heartbeat(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        message: dict
    ):
        """Handle heartbeat."""
        session_id = self.client_sessions.get(client_id)
        if not session_id:
            return

        session = self.sessions[session_id]
        participant = session.participants.get(client_id)
        if participant:
            participant.last_heartbeat = datetime.utcnow()

        await self._send(websocket, {
            "v": self.PROTOCOL_VERSION,
            "type": MessageType.HEARTBEAT_ACK,
            "id": str(uuid.uuid4()),
            "session": session_id,
            "sender": "server",
            "ts": self._get_timestamp(),
            "payload": {
                "serverTime": self._get_timestamp(),
                "queueDepth": len(session.message_history),
            },
        })

    async def _broadcast(
        self,
        session: Session,
        message: dict,
        exclude: Optional[str] = None
    ):
        """Broadcast a message to all participants in a session."""
        for client_id, participant in session.participants.items():
            if exclude and client_id == exclude:
                continue
            try:
                await self._send(participant.websocket, message)
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Failed to send to {client_id}, connection closed")

    async def _send(self, websocket: WebSocketServerProtocol, message: dict):
        """Send a message to a WebSocket."""
        await websocket.send(json.dumps(message))

    async def _send_error(
        self,
        websocket: WebSocketServerProtocol,
        code: str,
        message: str,
        recoverable: bool = True
    ):
        """Send an error message."""
        await self._send(websocket, {
            "v": self.PROTOCOL_VERSION,
            "type": MessageType.ERROR,
            "id": str(uuid.uuid4()),
            "session": "",
            "sender": "server",
            "ts": self._get_timestamp(),
            "payload": {
                "code": code,
                "message": message,
                "recoverable": recoverable,
            },
        })

    def _add_to_history(self, session: Session, message: dict):
        """Add a message to the session history."""
        session.message_history.append(message)
        if len(session.message_history) > self.MAX_MESSAGE_HISTORY:
            session.message_history = session.message_history[-self.MAX_MESSAGE_HISTORY:]

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(datetime.utcnow().timestamp() * 1000)

    async def _cleanup_task(self):
        """Periodic cleanup of expired locks and stale connections."""
        while self._running:
            await asyncio.sleep(30)

            now = datetime.utcnow()

            for session in list(self.sessions.values()):
                # Clean up expired locks
                expired_locks = [
                    path for path, lock in session.locks.items()
                    if lock.expires_at < now
                ]
                for path in expired_locks:
                    del session.locks[path]
                    logger.debug(f"Expired lock on {path} in session {session.session_id}")

                # Check for stale participants
                stale_timeout = now - timedelta(seconds=self.HEARTBEAT_TIMEOUT)
                stale_clients = [
                    client_id for client_id, p in session.participants.items()
                    if p.last_heartbeat < stale_timeout
                ]
                for client_id in stale_clients:
                    await self._handle_disconnect(client_id, reason="timeout")


async def main():
    """Run the collaboration server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    server = CollaborationServer(host="0.0.0.0", port=8765)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
