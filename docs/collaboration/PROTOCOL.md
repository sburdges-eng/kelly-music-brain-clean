# Real-Time Collaboration Protocol

> Specification for multi-user collaborative music production in iDAW.

## Overview

The iDAW Collaboration Protocol enables real-time multi-user sessions where multiple musicians can work on the same project simultaneously, with synchronized intents, MIDI, and arrangement changes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         iDAW Collaboration Server                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Session   │  │   CRDT      │  │   Intent    │  │   Presence  │     │
│  │   Manager   │  │   Engine    │  │   Sync      │  │   Service   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │                │                │                │             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    WebSocket Gateway                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │ Client  │          │ Client  │          │ Client  │
    │   A     │          │   B     │          │   C     │
    └─────────┘          └─────────┘          └─────────┘
```

## Protocol Version

```
iDAW-Collab/1.0
```

## Message Format

All messages use JSON over WebSocket with the following envelope:

```typescript
interface Message {
  // Protocol version
  v: "1.0";

  // Message type
  type: MessageType;

  // Unique message ID (UUIDv7 for ordering)
  id: string;

  // Session ID
  session: string;

  // Sender's client ID
  sender: string;

  // Logical timestamp (Lamport clock)
  ts: number;

  // Type-specific payload
  payload: unknown;
}

type MessageType =
  | "session.join"
  | "session.leave"
  | "session.sync"
  | "presence.update"
  | "presence.cursor"
  | "intent.update"
  | "intent.lock"
  | "intent.unlock"
  | "midi.note"
  | "midi.cc"
  | "arrangement.op"
  | "chat.message"
  | "undo.request"
  | "undo.ack";
```

## Session Management

### Join Session

```typescript
// Client → Server
interface SessionJoinRequest {
  type: "session.join";
  payload: {
    sessionId: string;
    clientName: string;
    clientColor: string;  // Hex color for cursor/selection
    capabilities: string[];  // ["midi", "intent", "arrangement"]
  };
}

// Server → Client
interface SessionJoinResponse {
  type: "session.sync";
  payload: {
    sessionId: string;
    clientId: string;  // Assigned by server
    participants: Participant[];
    state: SessionState;  // Full CRDT state
    version: number;
  };
}

interface Participant {
  clientId: string;
  name: string;
  color: string;
  joinedAt: number;
  cursor?: CursorPosition;
  activeLocks: string[];
}
```

### Leave Session

```typescript
// Client → Server (or automatic on disconnect)
interface SessionLeaveRequest {
  type: "session.leave";
  payload: {
    reason?: "user" | "timeout" | "error";
  };
}

// Server → All Clients
interface ParticipantLeft {
  type: "presence.update";
  payload: {
    event: "leave";
    clientId: string;
    reason: string;
  };
}
```

## Presence System

### Cursor Position

```typescript
// Client → Server → All Clients
interface CursorUpdate {
  type: "presence.cursor";
  payload: {
    // Position in arrangement
    bar: number;
    beat: number;
    tick: number;

    // Track selection
    trackId?: string;

    // Selection range (if any)
    selection?: {
      startBar: number;
      startBeat: number;
      endBar: number;
      endBeat: number;
      trackIds: string[];
    };
  };
}
```

### Activity Status

```typescript
interface PresenceUpdate {
  type: "presence.update";
  payload: {
    event: "status";
    status: "active" | "idle" | "away" | "recording";
    activity?: string;  // "Editing verse chords", "Recording drums"
  };
}
```

## Intent Synchronization

### Intent Update

```typescript
// CRDT-based intent field updates
interface IntentUpdate {
  type: "intent.update";
  payload: {
    // Path to the field (dot-notation)
    path: string;

    // CRDT operation
    op: CRDTOperation;

    // New value
    value: unknown;

    // Vector clock for this field
    vclock: VectorClock;
  };
}

type CRDTOperation =
  | { type: "set"; value: unknown }
  | { type: "increment"; delta: number }
  | { type: "append"; items: unknown[] }
  | { type: "remove"; indices: number[] }
  | { type: "insert"; index: number; items: unknown[] };

interface VectorClock {
  [clientId: string]: number;
}
```

### Intent Fields

```typescript
interface CollaborativeIntent {
  // Phase 0: Core (usually set by session creator)
  core: {
    event: string;
    resistance: string;
    longing: string;
  };

  // Phase 1: Emotional (collaborative)
  emotional: {
    moodPrimary: string;
    moodSecondary: string[];
    vulnerabilityScale: number;
    narrativeArc: string;
  };

  // Phase 2: Technical (per-track or global)
  technical: {
    genre: string;
    key: string;
    tempo: number;
    timeSignature: string;
    rulesToBreak: string[];
  };

  // Metadata
  _meta: {
    lastModified: number;
    modifiedBy: string;
    version: number;
  };
}
```

### Intent Locking

To prevent conflicts on complex edits:

```typescript
// Request lock on intent field
interface IntentLockRequest {
  type: "intent.lock";
  payload: {
    path: string;  // "emotional.moodPrimary" or "technical"
    timeout: number;  // Max lock duration in ms
  };
}

// Server response
interface IntentLockResponse {
  type: "intent.lock";
  payload: {
    path: string;
    granted: boolean;
    holder?: string;  // Client ID if lock denied
    expiresAt?: number;
  };
}

// Release lock
interface IntentUnlockRequest {
  type: "intent.unlock";
  payload: {
    path: string;
  };
}
```

## MIDI Collaboration

### Real-time Note Events

```typescript
interface MIDINoteMessage {
  type: "midi.note";
  payload: {
    trackId: string;
    event: "on" | "off";
    pitch: number;
    velocity: number;
    channel: number;

    // Position (if recording)
    position?: {
      bar: number;
      beat: number;
      tick: number;
    };
  };
}

interface MIDICCMessage {
  type: "midi.cc";
  payload: {
    trackId: string;
    controller: number;
    value: number;
    channel: number;
  };
}
```

### MIDI Region Operations

```typescript
interface MIDIRegionOp {
  type: "arrangement.op";
  payload: {
    target: "midi_region";
    operation:
      | { type: "create"; region: MIDIRegion }
      | { type: "delete"; regionId: string }
      | { type: "move"; regionId: string; newPosition: Position }
      | { type: "resize"; regionId: string; newLength: number }
      | { type: "duplicate"; regionId: string; newPosition: Position }
      | { type: "quantize"; regionId: string; grid: string }
      | { type: "transpose"; regionId: string; semitones: number };
  };
}
```

## Arrangement Operations

### Track Operations

```typescript
interface ArrangementOp {
  type: "arrangement.op";
  payload: {
    target: "track" | "section" | "marker";
    operation: TrackOp | SectionOp | MarkerOp;
  };
}

type TrackOp =
  | { type: "add"; track: Track; index: number }
  | { type: "remove"; trackId: string }
  | { type: "rename"; trackId: string; name: string }
  | { type: "reorder"; trackId: string; newIndex: number }
  | { type: "mute"; trackId: string; muted: boolean }
  | { type: "solo"; trackId: string; soloed: boolean }
  | { type: "color"; trackId: string; color: string };

type SectionOp =
  | { type: "add"; section: Section }
  | { type: "remove"; sectionId: string }
  | { type: "rename"; sectionId: string; name: string }
  | { type: "resize"; sectionId: string; bars: number };

type MarkerOp =
  | { type: "add"; marker: Marker }
  | { type: "remove"; markerId: string }
  | { type: "move"; markerId: string; position: Position };
```

## Conflict Resolution

### CRDT Strategy

iDAW uses operation-based CRDTs (CmRDTs) for conflict-free merging:

| Data Type | CRDT Type | Merge Strategy |
|-----------|-----------|----------------|
| Intent text fields | LWW-Register | Last-write wins with vector clock |
| Intent arrays | OR-Set | Add-wins semantics |
| Intent numbers | Counter | Sum of increments |
| Track list | RGA | Replicated growable array |
| MIDI notes | OR-Set | Add-wins, remove by note ID |
| Arrangement | LWW-Map | Per-region last-write wins |

### Conflict Notification

```typescript
interface ConflictNotification {
  type: "session.sync";
  payload: {
    event: "conflict";
    path: string;
    yourValue: unknown;
    mergedValue: unknown;
    contributors: string[];  // Client IDs
  };
}
```

## Undo/Redo

### Distributed Undo

```typescript
// Request undo of own operation
interface UndoRequest {
  type: "undo.request";
  payload: {
    operationId: string;  // Message ID to undo
  };
}

// Server acknowledges and broadcasts inverse op
interface UndoAck {
  type: "undo.ack";
  payload: {
    operationId: string;
    inverseOp: Message;
    success: boolean;
    reason?: string;  // If denied
  };
}
```

### Undo Rules

1. Users can only undo their own operations
2. Operations are undone in reverse chronological order
3. Conflicting undos are resolved by timestamp
4. Some operations are not undoable (e.g., join/leave)

## Chat System

```typescript
interface ChatMessage {
  type: "chat.message";
  payload: {
    text: string;
    replyTo?: string;  // Message ID
    mentions?: string[];  // Client IDs
    attachments?: Attachment[];
  };
}

interface Attachment {
  type: "midi_snippet" | "audio_preview" | "intent_reference";
  data: string;  // Base64 or reference ID
}
```

## Error Handling

```typescript
interface ErrorMessage {
  type: "error";
  payload: {
    code: ErrorCode;
    message: string;
    details?: unknown;
    recoverable: boolean;
  };
}

type ErrorCode =
  | "SESSION_NOT_FOUND"
  | "SESSION_FULL"
  | "PERMISSION_DENIED"
  | "LOCK_CONFLICT"
  | "INVALID_OPERATION"
  | "VERSION_MISMATCH"
  | "RATE_LIMITED"
  | "SERVER_ERROR";
```

## Security

### Authentication

```typescript
// Initial handshake includes auth token
interface AuthHandshake {
  type: "auth";
  payload: {
    token: string;  // JWT or session token
    clientVersion: string;
  };
}
```

### Permissions

```typescript
interface SessionPermissions {
  canEditIntent: boolean;
  canEditArrangement: boolean;
  canRecordMIDI: boolean;
  canInvite: boolean;
  canKick: boolean;
  isOwner: boolean;
}
```

## Transport

### WebSocket Connection

```
wss://collab.idaw.dev/session/{sessionId}
```

### Heartbeat

```typescript
// Client → Server (every 30s)
interface Heartbeat {
  type: "heartbeat";
  payload: {
    lastReceived: string;  // Last message ID received
  };
}

// Server → Client
interface HeartbeatAck {
  type: "heartbeat.ack";
  payload: {
    serverTime: number;
    queueDepth: number;
  };
}
```

### Reconnection

1. Client stores last message ID
2. On reconnect, sends `session.join` with `resumeFrom: lastMessageId`
3. Server replays missed messages or sends full state if too old

## Performance Targets

| Metric | Target |
|--------|--------|
| Message latency (p50) | < 50ms |
| Message latency (p99) | < 200ms |
| State sync (full) | < 500ms |
| Reconnection time | < 2s |
| Max participants | 8 |
| Max message rate | 100/s per client |

## Example Flow

```
Client A                    Server                    Client B
    │                          │                          │
    │──session.join───────────▶│                          │
    │◀──session.sync───────────│                          │
    │                          │                          │
    │                          │◀──session.join───────────│
    │◀──presence.update────────│──session.sync───────────▶│
    │                          │──presence.update────────▶│
    │                          │                          │
    │──intent.lock─────────────▶│                          │
    │◀──intent.lock (granted)──│                          │
    │                          │                          │
    │──intent.update──────────▶│──intent.update─────────▶│
    │                          │                          │
    │──intent.unlock──────────▶│                          │
    │                          │                          │
    │◀──midi.note──────────────│◀──midi.note─────────────│
    │                          │                          │
```

---

*"The audience doesn't hear 'CRDT conflict resolution.' They hear 'we wrote this together.'"*
