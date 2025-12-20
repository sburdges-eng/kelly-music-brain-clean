"""
Collaborative Editing UI for iDAW.

Streamlit-based real-time collaborative interface for
multi-user intent editing and session management.
"""

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional
from queue import Queue

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

logger = logging.getLogger(__name__)


@dataclass
class CollaboratorInfo:
    """Information about a collaborator."""
    client_id: str
    name: str
    color: str
    cursor: Optional[dict] = None
    status: str = "active"


class CollaborationClient:
    """
    WebSocket client for collaboration sessions.

    Handles connection, message sending/receiving, and state sync.
    """

    def __init__(self, server_url: str = "ws://localhost:8765"):
        """
        Initialize the collaboration client.

        Args:
            server_url: WebSocket server URL
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets required: pip install websockets")

        self.server_url = server_url
        self.websocket = None
        self.client_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.connected = False

        self.participants: dict[str, CollaboratorInfo] = {}
        self.state: dict = {}
        self.message_queue: Queue = Queue()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._handlers: dict[str, list[Callable]] = {}

    def connect(self, session_id: str, name: str, color: str = "#4A90D9"):
        """
        Connect to a collaboration session.

        Args:
            session_id: Session to join
            name: Display name
            color: Cursor/highlight color
        """
        self.session_id = session_id

        # Start async event loop in background thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Schedule connection
        asyncio.run_coroutine_threadsafe(
            self._connect(session_id, name, color),
            self._loop
        )

    def disconnect(self):
        """Disconnect from the session."""
        if self._loop and self.connected:
            asyncio.run_coroutine_threadsafe(
                self._disconnect(),
                self._loop
            )

    def _run_loop(self):
        """Run the async event loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self, session_id: str, name: str, color: str):
        """Async connection handler."""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.connected = True

            # Send join message
            await self._send({
                "v": "1.0",
                "type": "session.join",
                "id": self._gen_id(),
                "session": session_id,
                "sender": "",
                "ts": self._timestamp(),
                "payload": {
                    "sessionId": session_id,
                    "clientName": name,
                    "clientColor": color,
                    "capabilities": ["midi", "intent", "arrangement"],
                },
            })

            # Start receiving messages
            asyncio.create_task(self._receive_loop())

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False

    async def _disconnect(self):
        """Async disconnect handler."""
        if self.websocket:
            await self._send({
                "v": "1.0",
                "type": "session.leave",
                "id": self._gen_id(),
                "session": self.session_id,
                "sender": self.client_id or "",
                "ts": self._timestamp(),
                "payload": {"reason": "user"},
            })
            await self.websocket.close()
        self.connected = False

    async def _receive_loop(self):
        """Receive and process incoming messages."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                self._process_message(data)
        except websockets.exceptions.ConnectionClosed:
            self.connected = False

    def _process_message(self, message: dict):
        """Process an incoming message."""
        msg_type = message.get("type", "")

        if msg_type == "session.sync":
            # Initial state sync
            payload = message.get("payload", {})
            self.client_id = payload.get("clientId")
            self.state = payload.get("state", {})

            for p in payload.get("participants", []):
                self.participants[p["clientId"]] = CollaboratorInfo(
                    client_id=p["clientId"],
                    name=p["name"],
                    color=p["color"],
                )

        elif msg_type == "presence.update":
            payload = message.get("payload", {})
            event = payload.get("event")

            if event == "join":
                self.participants[payload["clientId"]] = CollaboratorInfo(
                    client_id=payload["clientId"],
                    name=payload["name"],
                    color=payload["color"],
                )
            elif event == "leave":
                client_id = payload.get("clientId")
                if client_id in self.participants:
                    del self.participants[client_id]

        elif msg_type == "presence.cursor":
            sender = message.get("sender")
            if sender in self.participants:
                self.participants[sender].cursor = message.get("payload")

        elif msg_type == "intent.update":
            # Apply intent update
            payload = message.get("payload", {})
            path = payload.get("path", "")
            value = payload.get("value")
            self._set_nested(self.state.get("intent", {}), path, value)

        elif msg_type == "chat.message":
            # Add to message queue for UI
            self.message_queue.put(message)

        # Call registered handlers
        for handler in self._handlers.get(msg_type, []):
            handler(message)

    def _set_nested(self, obj: dict, path: str, value: Any):
        """Set a nested value by path."""
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]
        obj[parts[-1]] = value

    async def _send(self, message: dict):
        """Send a message."""
        if self.websocket:
            await self.websocket.send(json.dumps(message))

    def send_intent_update(self, path: str, value: Any):
        """
        Send an intent field update.

        Args:
            path: Field path (e.g., "emotional.moodPrimary")
            value: New value
        """
        if self._loop and self.connected:
            asyncio.run_coroutine_threadsafe(
                self._send({
                    "v": "1.0",
                    "type": "intent.update",
                    "id": self._gen_id(),
                    "session": self.session_id,
                    "sender": self.client_id,
                    "ts": self._timestamp(),
                    "payload": {
                        "path": path,
                        "value": value,
                        "vclock": {},
                    },
                }),
                self._loop
            )

    def send_cursor_update(self, position: dict):
        """Send cursor position update."""
        if self._loop and self.connected:
            asyncio.run_coroutine_threadsafe(
                self._send({
                    "v": "1.0",
                    "type": "presence.cursor",
                    "id": self._gen_id(),
                    "session": self.session_id,
                    "sender": self.client_id,
                    "ts": self._timestamp(),
                    "payload": position,
                }),
                self._loop
            )

    def send_chat(self, text: str):
        """Send a chat message."""
        if self._loop and self.connected:
            asyncio.run_coroutine_threadsafe(
                self._send({
                    "v": "1.0",
                    "type": "chat.message",
                    "id": self._gen_id(),
                    "session": self.session_id,
                    "sender": self.client_id,
                    "ts": self._timestamp(),
                    "payload": {"text": text},
                }),
                self._loop
            )

    def on(self, event: str, handler: Callable):
        """Register an event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def _gen_id(self) -> str:
        """Generate a message ID."""
        import uuid
        return str(uuid.uuid4())[:12]

    def _timestamp(self) -> int:
        """Get current timestamp in ms."""
        return int(datetime.utcnow().timestamp() * 1000)


def render_collaboration_ui():
    """
    Render the collaborative editing UI in Streamlit.

    This is a complete collaborative intent editing interface
    with real-time sync, presence indicators, and chat.
    """
    if not HAS_STREAMLIT:
        raise ImportError("streamlit required: pip install streamlit")

    st.set_page_config(
        page_title="iDAW Collaboration",
        page_icon="🎵",
        layout="wide",
    )

    # Custom CSS for collaboration features
    st.markdown("""
    <style>
    .collaborator-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        margin: 2px;
        font-size: 12px;
        color: white;
    }
    .cursor-indicator {
        position: absolute;
        width: 2px;
        height: 20px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        50% { opacity: 0; }
    }
    .chat-message {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        background: #f0f0f0;
    }
    .chat-message.own {
        background: #e3f2fd;
        text-align: right;
    }
    .field-locked {
        opacity: 0.6;
        pointer-events: none;
    }
    .field-editing {
        border: 2px solid;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "collab_client" not in st.session_state:
        st.session_state.collab_client = None
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = f"User-{hash(id(st)) % 1000}"
    if "user_color" not in st.session_state:
        import random
        colors = ["#E91E63", "#9C27B0", "#673AB7", "#3F51B5",
                  "#2196F3", "#00BCD4", "#009688", "#4CAF50"]
        st.session_state.user_color = random.choice(colors)

    # Header
    st.title("🎵 iDAW Collaborative Intent Editor")

    # Connection panel
    with st.sidebar:
        st.header("Session")

        if not st.session_state.connected:
            session_id = st.text_input(
                "Session ID",
                value="demo",
                help="Enter a session ID to join or create"
            )

            col1, col2 = st.columns(2)
            with col1:
                user_name = st.text_input("Your Name", value=st.session_state.user_name)
            with col2:
                user_color = st.color_picker("Color", value=st.session_state.user_color)

            server_url = st.text_input(
                "Server URL",
                value="ws://localhost:8765",
                help="WebSocket server address"
            )

            if st.button("Join Session", type="primary", use_container_width=True):
                try:
                    client = CollaborationClient(server_url)
                    client.connect(session_id, user_name, user_color)
                    st.session_state.collab_client = client
                    st.session_state.user_name = user_name
                    st.session_state.user_color = user_color
                    st.session_state.connected = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        else:
            client = st.session_state.collab_client

            st.success(f"Connected to: {client.session_id}")
            st.caption(f"Your ID: {client.client_id}")

            if st.button("Leave Session", use_container_width=True):
                client.disconnect()
                st.session_state.connected = False
                st.session_state.collab_client = None
                st.rerun()

            # Participants
            st.subheader("Collaborators")
            for p in client.participants.values():
                badge_html = f"""
                <span class="collaborator-badge" style="background-color: {p.color}">
                    {p.name} {'(you)' if p.client_id == client.client_id else ''}
                </span>
                """
                st.markdown(badge_html, unsafe_allow_html=True)

    # Main content
    if st.session_state.connected:
        client = st.session_state.collab_client

        # Wait for state sync
        if not client.state:
            st.info("Syncing session state...")
            st.spinner()
            return

        intent = client.state.get("intent", {})

        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Core Intent",
            "💭 Emotional",
            "🎛️ Technical",
            "💬 Chat"
        ])

        with tab1:
            st.header("Core Intent (Phase 0)")
            st.caption("The emotional foundation of your song")

            core = intent.get("core", {})

            # Core event
            event = st.text_area(
                "Core Event",
                value=core.get("event", ""),
                help="What specific moment or experience triggered this song?",
                key="core_event"
            )
            if event != core.get("event", ""):
                client.send_intent_update("core.event", event)

            col1, col2 = st.columns(2)

            with col1:
                resistance = st.text_area(
                    "Core Resistance",
                    value=core.get("resistance", ""),
                    help="What are you afraid to say or feel?",
                    key="core_resistance"
                )
                if resistance != core.get("resistance", ""):
                    client.send_intent_update("core.resistance", resistance)

            with col2:
                longing = st.text_area(
                    "Core Longing",
                    value=core.get("longing", ""),
                    help="What do you wish could be true?",
                    key="core_longing"
                )
                if longing != core.get("longing", ""):
                    client.send_intent_update("core.longing", longing)

        with tab2:
            st.header("Emotional Intent (Phase 1)")
            st.caption("How should the listener feel?")

            emotional = intent.get("emotional", {})

            col1, col2 = st.columns(2)

            with col1:
                moods = [
                    "Joy", "Sadness", "Anger", "Fear", "Love",
                    "Nostalgia", "Hope", "Despair", "Serenity", "Tension"
                ]
                mood_primary = st.selectbox(
                    "Primary Mood",
                    options=moods,
                    index=moods.index(emotional.get("moodPrimary", "Joy"))
                    if emotional.get("moodPrimary") in moods else 0,
                    key="mood_primary"
                )
                if mood_primary != emotional.get("moodPrimary"):
                    client.send_intent_update("emotional.moodPrimary", mood_primary)

                vulnerability = st.slider(
                    "Vulnerability Scale",
                    min_value=1,
                    max_value=10,
                    value=emotional.get("vulnerabilityScale", 5),
                    help="How emotionally exposed is this song?",
                    key="vulnerability"
                )
                if vulnerability != emotional.get("vulnerabilityScale"):
                    client.send_intent_update("emotional.vulnerabilityScale", vulnerability)

            with col2:
                mood_secondary = st.multiselect(
                    "Secondary Moods",
                    options=moods,
                    default=emotional.get("moodSecondary", []),
                    key="mood_secondary"
                )
                if mood_secondary != emotional.get("moodSecondary", []):
                    client.send_intent_update("emotional.moodSecondary", mood_secondary)

                arcs = ["Standard", "Rising", "Falling", "Circular", "Fragmented"]
                narrative_arc = st.selectbox(
                    "Narrative Arc",
                    options=arcs,
                    index=arcs.index(emotional.get("narrativeArc", "Standard").title())
                    if emotional.get("narrativeArc", "").title() in arcs else 0,
                    key="narrative_arc"
                )
                if narrative_arc.lower() != emotional.get("narrativeArc"):
                    client.send_intent_update("emotional.narrativeArc", narrative_arc.lower())

        with tab3:
            st.header("Technical Intent (Phase 2)")
            st.caption("Musical parameters and constraints")

            technical = intent.get("technical", {})

            col1, col2, col3 = st.columns(3)

            with col1:
                genres = [
                    "Pop", "Rock", "Hip-Hop", "R&B", "Jazz",
                    "Electronic", "Classical", "Folk", "Country", "Metal"
                ]
                genre = st.selectbox(
                    "Genre",
                    options=genres,
                    index=genres.index(technical.get("genre", "Pop"))
                    if technical.get("genre") in genres else 0,
                    key="genre"
                )
                if genre != technical.get("genre"):
                    client.send_intent_update("technical.genre", genre)

            with col2:
                keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                key_modes = []
                for k in keys:
                    key_modes.extend([f"{k} Major", f"{k} Minor"])

                key_selection = st.selectbox(
                    "Key",
                    options=key_modes,
                    index=key_modes.index(technical.get("key", "C Major"))
                    if technical.get("key") in key_modes else 0,
                    key="key"
                )
                if key_selection != technical.get("key"):
                    client.send_intent_update("technical.key", key_selection)

            with col3:
                tempo = st.number_input(
                    "Tempo (BPM)",
                    min_value=40,
                    max_value=240,
                    value=technical.get("tempo", 120),
                    key="tempo"
                )
                if tempo != technical.get("tempo"):
                    client.send_intent_update("technical.tempo", tempo)

            col1, col2 = st.columns(2)

            with col1:
                time_signatures = ["4/4", "3/4", "6/8", "2/4", "5/4", "7/8"]
                time_sig = st.selectbox(
                    "Time Signature",
                    options=time_signatures,
                    index=time_signatures.index(technical.get("timeSignature", "4/4"))
                    if technical.get("timeSignature") in time_signatures else 0,
                    key="time_sig"
                )
                if time_sig != technical.get("timeSignature"):
                    client.send_intent_update("technical.timeSignature", time_sig)

            with col2:
                rules = [
                    "HARMONY_AvoidTonicResolution",
                    "HARMONY_ParallelFifths",
                    "RHYTHM_ConstantDisplacement",
                    "RHYTHM_PolymetricTension",
                    "ARRANGEMENT_BuriedVocals",
                    "PRODUCTION_PitchImperfection",
                ]
                rules_to_break = st.multiselect(
                    "Rules to Break",
                    options=rules,
                    default=technical.get("rulesToBreak", []),
                    help="Intentional rule violations for emotional effect",
                    key="rules_to_break"
                )
                if rules_to_break != technical.get("rulesToBreak", []):
                    client.send_intent_update("technical.rulesToBreak", rules_to_break)

        with tab4:
            st.header("Session Chat")

            # Display chat messages
            chat_container = st.container()
            with chat_container:
                # Check for new messages
                while not client.message_queue.empty():
                    msg = client.message_queue.get()
                    if msg.get("type") == "chat.message":
                        st.session_state.chat_messages.append({
                            "sender": msg.get("sender"),
                            "text": msg.get("payload", {}).get("text", ""),
                            "ts": msg.get("ts"),
                        })

                # Display messages
                for msg in st.session_state.chat_messages[-20:]:
                    sender = client.participants.get(msg["sender"])
                    sender_name = sender.name if sender else "Unknown"
                    is_own = msg["sender"] == client.client_id

                    css_class = "chat-message own" if is_own else "chat-message"
                    st.markdown(
                        f'<div class="{css_class}"><b>{sender_name}:</b> {msg["text"]}</div>',
                        unsafe_allow_html=True
                    )

            # Chat input
            chat_input = st.text_input(
                "Message",
                key="chat_input",
                placeholder="Type a message..."
            )
            if st.button("Send", key="send_chat"):
                if chat_input:
                    client.send_chat(chat_input)
                    st.session_state.chat_messages.append({
                        "sender": client.client_id,
                        "text": chat_input,
                        "ts": client._timestamp(),
                    })
                    st.rerun()

    else:
        # Not connected - show instructions
        st.info("👈 Enter a session ID and click 'Join Session' to start collaborating")

        st.markdown("""
        ## How it works

        1. **Create or join a session** - Enter any session ID to create a new session or join an existing one
        2. **Edit collaboratively** - Changes sync in real-time with other participants
        3. **See who's editing** - Collaborator badges show who's in the session
        4. **Chat with your team** - Built-in chat for discussing ideas

        ## Features

        - **Real-time sync** - Changes appear instantly for all collaborators
        - **Presence indicators** - See who's editing which fields
        - **Version history** - Track changes and revert if needed
        - **Conflict resolution** - Automatic merging of concurrent edits
        """)


def main():
    """Run the collaboration UI."""
    render_collaboration_ui()


if __name__ == "__main__":
    main()
