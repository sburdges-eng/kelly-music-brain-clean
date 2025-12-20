"""
WebSocket Communication - Real-time messaging for collaboration.

Provides WebSocket server and client for real-time session synchronization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
from enum import Enum
import asyncio
import json
import uuid


class MessageType(Enum):
    """Types of collaboration messages."""
    # Connection
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    HEARTBEAT = "heartbeat"

    # Session
    JOIN_SESSION = "join_session"
    LEAVE_SESSION = "leave_session"
    SESSION_UPDATE = "session_update"

    # Presence
    PRESENCE_UPDATE = "presence_update"
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"

    # Operations
    OPERATION = "operation"
    OPERATION_ACK = "operation_ack"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"

    # Chat/Comments
    CHAT_MESSAGE = "chat_message"
    COMMENT_ADD = "comment_add"
    COMMENT_RESOLVE = "comment_resolve"

    # Error
    ERROR = "error"


@dataclass
class Message:
    """A collaboration message."""
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    sender_id: Optional[str] = None
    session_id: Optional[str] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "payload": self.payload,
            "sender_id": self.sender_id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            type=MessageType(data["type"]),
            payload=data.get("payload", {}),
            sender_id=data.get("sender_id"),
            session_id=data.get("session_id"),
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class CollaborationServer:
    """
    WebSocket server for real-time collaboration.

    Handles multiple sessions and broadcasts updates to participants.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port

        # Connection tracking
        self._connections: Dict[str, Any] = {}  # connection_id -> websocket
        self._user_connections: Dict[str, str] = {}  # user_id -> connection_id
        self._session_connections: Dict[str, Set[str]] = {}  # session_id -> set of connection_ids

        # Message handlers
        self._handlers: Dict[MessageType, Callable] = {}

        # Server state
        self._running = False
        self._server = None

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message, Any], None],
    ) -> None:
        """Register a handler for a message type."""
        self._handlers[message_type] = handler

    async def start(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets

            self._running = True
            self._server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
            )
            print(f"Collaboration server started on ws://{self.host}:{self.port}")

        except ImportError:
            print("websockets not installed. Install with: pip install websockets")
            raise

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(self, websocket, path) -> None:
        """Handle a new WebSocket connection."""
        connection_id = str(uuid.uuid4())
        self._connections[connection_id] = websocket

        try:
            async for message_str in websocket:
                try:
                    message = Message.from_json(message_str)
                    await self._process_message(message, websocket, connection_id)
                except Exception as e:
                    error_msg = Message(
                        type=MessageType.ERROR,
                        payload={"error": str(e)},
                    )
                    await websocket.send(error_msg.to_json())

        finally:
            # Cleanup on disconnect
            await self._handle_disconnect(connection_id)
            del self._connections[connection_id]

    async def _process_message(
        self,
        message: Message,
        websocket,
        connection_id: str,
    ) -> None:
        """Process an incoming message."""
        # Track user connection
        if message.sender_id:
            self._user_connections[message.sender_id] = connection_id

        # Track session connection
        if message.session_id:
            if message.session_id not in self._session_connections:
                self._session_connections[message.session_id] = set()
            self._session_connections[message.session_id].add(connection_id)

        # Call registered handler
        if message.type in self._handlers:
            await self._handlers[message.type](message, websocket)

        # Broadcast to session if needed
        if message.type in [
            MessageType.OPERATION,
            MessageType.PRESENCE_UPDATE,
            MessageType.CURSOR_MOVE,
            MessageType.CHAT_MESSAGE,
            MessageType.COMMENT_ADD,
        ]:
            await self._broadcast_to_session(
                message.session_id,
                message,
                exclude_connection=connection_id,
            )

    async def _handle_disconnect(self, connection_id: str) -> None:
        """Handle connection disconnect."""
        # Find and remove user
        user_id = None
        for uid, cid in list(self._user_connections.items()):
            if cid == connection_id:
                user_id = uid
                del self._user_connections[uid]
                break

        # Remove from sessions
        for session_id, connections in self._session_connections.items():
            connections.discard(connection_id)

            # Broadcast disconnect to session
            if user_id:
                msg = Message(
                    type=MessageType.DISCONNECT,
                    payload={"user_id": user_id},
                    session_id=session_id,
                )
                await self._broadcast_to_session(session_id, msg)

    async def _broadcast_to_session(
        self,
        session_id: str,
        message: Message,
        exclude_connection: Optional[str] = None,
    ) -> None:
        """Broadcast a message to all session participants."""
        if session_id not in self._session_connections:
            return

        for conn_id in self._session_connections[session_id]:
            if conn_id == exclude_connection:
                continue

            if conn_id in self._connections:
                try:
                    await self._connections[conn_id].send(message.to_json())
                except Exception:
                    pass  # Connection may have closed

    async def send_to_user(self, user_id: str, message: Message) -> bool:
        """Send a message to a specific user."""
        if user_id in self._user_connections:
            conn_id = self._user_connections[user_id]
            if conn_id in self._connections:
                try:
                    await self._connections[conn_id].send(message.to_json())
                    return True
                except Exception:
                    pass
        return False


class CollaborationClient:
    """
    WebSocket client for connecting to collaboration server.

    Provides async interface for sending and receiving messages.
    """

    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self._websocket = None
        self._connected = False
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Event handlers
        self._handlers: Dict[MessageType, List[Callable]] = {}

        # User info
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None

    def on(self, message_type: MessageType, handler: Callable) -> None:
        """Register an event handler."""
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)

    async def connect(self, user_id: str) -> bool:
        """Connect to the collaboration server."""
        try:
            import websockets

            self._websocket = await websockets.connect(self.server_url)
            self._connected = True
            self.user_id = user_id

            # Start message receiver
            asyncio.create_task(self._receive_messages())

            # Send connect message
            await self.send(Message(
                type=MessageType.CONNECT,
                sender_id=user_id,
                payload={"user_id": user_id},
            ))

            return True

        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._websocket:
            await self.send(Message(
                type=MessageType.DISCONNECT,
                sender_id=self.user_id,
            ))
            await self._websocket.close()
            self._connected = False

    async def join_session(self, session_id: str, username: str) -> bool:
        """Join a collaborative session."""
        self.session_id = session_id

        await self.send(Message(
            type=MessageType.JOIN_SESSION,
            sender_id=self.user_id,
            session_id=session_id,
            payload={
                "user_id": self.user_id,
                "username": username,
            },
        ))

        return True

    async def leave_session(self) -> None:
        """Leave the current session."""
        if self.session_id:
            await self.send(Message(
                type=MessageType.LEAVE_SESSION,
                sender_id=self.user_id,
                session_id=self.session_id,
            ))
            self.session_id = None

    async def send(self, message: Message) -> None:
        """Send a message to the server."""
        if self._websocket and self._connected:
            message.sender_id = self.user_id
            message.session_id = self.session_id
            await self._websocket.send(message.to_json())

    async def send_operation(self, operation: Dict[str, Any]) -> None:
        """Send an operation for real-time sync."""
        await self.send(Message(
            type=MessageType.OPERATION,
            payload={"operation": operation},
        ))

    async def send_cursor(self, position: Dict[str, Any]) -> None:
        """Send cursor position update."""
        await self.send(Message(
            type=MessageType.CURSOR_MOVE,
            payload={"position": position},
        ))

    async def send_chat(self, text: str) -> None:
        """Send a chat message."""
        await self.send(Message(
            type=MessageType.CHAT_MESSAGE,
            payload={"text": text},
        ))

    async def request_sync(self) -> None:
        """Request full sync from server."""
        await self.send(Message(
            type=MessageType.SYNC_REQUEST,
        ))

    async def _receive_messages(self) -> None:
        """Receive and process messages from server."""
        try:
            async for message_str in self._websocket:
                message = Message.from_json(message_str)

                # Call handlers
                if message.type in self._handlers:
                    for handler in self._handlers[message.type]:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message)
                            else:
                                handler(message)
                        except Exception as e:
                            print(f"Handler error: {e}")

                # Queue for processing
                await self._message_queue.put(message)

        except Exception as e:
            print(f"Connection lost: {e}")
            self._connected = False

    async def get_message(self, timeout: float = None) -> Optional[Message]:
        """Get the next message from the queue."""
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected
