"""
Collaborative Editing - Real-time conflict-free editing.

Implements Operational Transformation (OT) for concurrent edits.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import uuid
import copy


class OperationType(Enum):
    """Types of edit operations."""
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    MOVE = "move"

    # Music-specific operations
    ADD_NOTE = "add_note"
    DELETE_NOTE = "delete_note"
    MODIFY_NOTE = "modify_note"
    ADD_CHORD = "add_chord"
    DELETE_CHORD = "delete_chord"
    MODIFY_CHORD = "modify_chord"
    SET_TEMPO = "set_tempo"
    SET_KEY = "set_key"
    SET_INTENT = "set_intent"


@dataclass
class Operation:
    """
    An atomic edit operation.

    Operations can be transformed and applied to resolve conflicts.
    """
    op_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: OperationType = OperationType.UPDATE
    path: str = ""  # JSON path to the modified field
    value: Any = None
    old_value: Any = None  # For undo

    # Position info (for ordered collections)
    position: Optional[int] = None
    length: Optional[int] = None

    # Metadata
    author_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 0  # Document version this operation was based on

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "op_id": self.op_id,
            "type": self.type.value,
            "path": self.path,
            "value": self.value,
            "old_value": self.old_value,
            "position": self.position,
            "length": self.length,
            "author_id": self.author_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Operation":
        """Create from dictionary."""
        return cls(
            op_id=data.get("op_id", str(uuid.uuid4())),
            type=OperationType(data["type"]),
            path=data.get("path", ""),
            value=data.get("value"),
            old_value=data.get("old_value"),
            position=data.get("position"),
            length=data.get("length"),
            author_id=data.get("author_id", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            version=data.get("version", 0),
        )

    def inverse(self) -> "Operation":
        """Create the inverse operation (for undo)."""
        if self.type == OperationType.INSERT:
            return Operation(
                type=OperationType.DELETE,
                path=self.path,
                position=self.position,
                length=len(str(self.value)) if self.value else 0,
                author_id=self.author_id,
            )
        elif self.type == OperationType.DELETE:
            return Operation(
                type=OperationType.INSERT,
                path=self.path,
                value=self.old_value,
                position=self.position,
                author_id=self.author_id,
            )
        elif self.type == OperationType.UPDATE:
            return Operation(
                type=OperationType.UPDATE,
                path=self.path,
                value=self.old_value,
                old_value=self.value,
                author_id=self.author_id,
            )
        else:
            # For music operations, swap value and old_value
            return Operation(
                type=self.type,
                path=self.path,
                value=self.old_value,
                old_value=self.value,
                position=self.position,
                author_id=self.author_id,
            )


def transform_operations(
    op1: Operation,
    op2: Operation,
    priority: str = "left",
) -> Tuple[Operation, Operation]:
    """
    Transform two concurrent operations.

    Given two operations that were created concurrently,
    transform them so they can be applied in sequence.

    Args:
        op1: First operation
        op2: Second operation
        priority: Which operation takes priority ("left" or "right")

    Returns:
        Tuple of transformed (op1', op2')
    """
    # If operations are on different paths, no transformation needed
    if op1.path != op2.path:
        return op1, op2

    # Clone operations
    t_op1 = copy.copy(op1)
    t_op2 = copy.copy(op2)

    # Handle positional operations
    if op1.position is not None and op2.position is not None:
        if op1.type == OperationType.INSERT and op2.type == OperationType.INSERT:
            # Two inserts at different positions
            if op1.position < op2.position:
                t_op2.position += len(str(op1.value)) if op1.value else 0
            elif op1.position > op2.position:
                t_op1.position += len(str(op2.value)) if op2.value else 0
            else:
                # Same position - use priority
                if priority == "left":
                    t_op2.position += len(str(op1.value)) if op1.value else 0
                else:
                    t_op1.position += len(str(op2.value)) if op2.value else 0

        elif op1.type == OperationType.DELETE and op2.type == OperationType.DELETE:
            # Two deletes
            if op1.position < op2.position:
                t_op2.position -= min(op1.length or 0, op2.position - op1.position)
            elif op1.position > op2.position:
                t_op1.position -= min(op2.length or 0, op1.position - op2.position)

        elif op1.type == OperationType.INSERT and op2.type == OperationType.DELETE:
            if op1.position <= op2.position:
                t_op2.position += len(str(op1.value)) if op1.value else 0
            else:
                t_op1.position -= min(op2.length or 0, op1.position - op2.position)

        elif op1.type == OperationType.DELETE and op2.type == OperationType.INSERT:
            if op2.position <= op1.position:
                t_op1.position += len(str(op2.value)) if op2.value else 0
            else:
                t_op2.position -= min(op1.length or 0, op2.position - op1.position)

    return t_op1, t_op2


class CollaborativeDocument:
    """
    A document that supports collaborative editing.

    Tracks operations and handles conflict resolution.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self._state: Dict[str, Any] = initial_state or {}
        self._version: int = 0
        self._history: List[Operation] = []
        self._pending: List[Operation] = []

        # Callbacks
        self._on_change: Optional[callable] = None

    @property
    def state(self) -> Dict[str, Any]:
        """Get current document state."""
        return copy.deepcopy(self._state)

    @property
    def version(self) -> int:
        """Get current document version."""
        return self._version

    def set_on_change(self, callback: callable) -> None:
        """Set change callback."""
        self._on_change = callback

    def apply(self, operation: Operation) -> bool:
        """
        Apply an operation to the document.

        Args:
            operation: Operation to apply

        Returns:
            True if successful
        """
        try:
            self._apply_operation(operation)
            self._history.append(operation)
            self._version += 1
            operation.version = self._version

            if self._on_change:
                self._on_change(operation)

            return True
        except Exception as e:
            print(f"Failed to apply operation: {e}")
            return False

    def apply_remote(self, operation: Operation) -> bool:
        """
        Apply a remote operation, transforming against pending local ops.

        Args:
            operation: Remote operation

        Returns:
            True if successful
        """
        # Transform against pending operations
        transformed = operation
        for pending in self._pending:
            _, transformed = transform_operations(pending, transformed)

        return self.apply(transformed)

    def queue_local(self, operation: Operation) -> None:
        """Queue a local operation for sending."""
        self._pending.append(operation)

    def acknowledge(self, op_id: str) -> None:
        """Acknowledge a sent operation."""
        self._pending = [op for op in self._pending if op.op_id != op_id]

    def _apply_operation(self, op: Operation) -> None:
        """Apply operation to internal state."""
        if op.type == OperationType.UPDATE:
            self._set_value(op.path, op.value)

        elif op.type == OperationType.INSERT:
            self._insert_value(op.path, op.position, op.value)

        elif op.type == OperationType.DELETE:
            self._delete_value(op.path, op.position, op.length)

        elif op.type == OperationType.ADD_NOTE:
            notes = self._get_value(op.path) or []
            notes.append(op.value)
            self._set_value(op.path, notes)

        elif op.type == OperationType.DELETE_NOTE:
            notes = self._get_value(op.path) or []
            if op.position is not None and 0 <= op.position < len(notes):
                notes.pop(op.position)
            self._set_value(op.path, notes)

        elif op.type == OperationType.MODIFY_NOTE:
            notes = self._get_value(op.path) or []
            if op.position is not None and 0 <= op.position < len(notes):
                notes[op.position] = op.value
            self._set_value(op.path, notes)

        elif op.type in [
            OperationType.SET_TEMPO,
            OperationType.SET_KEY,
            OperationType.SET_INTENT,
        ]:
            self._set_value(op.path, op.value)

    def _get_value(self, path: str) -> Any:
        """Get value at JSON path."""
        if not path:
            return self._state

        parts = path.split(".")
        current = self._state

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current

    def _set_value(self, path: str, value: Any) -> None:
        """Set value at JSON path."""
        if not path:
            if isinstance(value, dict):
                self._state = value
            return

        parts = path.split(".")
        current = self._state

        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def _insert_value(self, path: str, position: int, value: Any) -> None:
        """Insert value at position."""
        target = self._get_value(path)
        if isinstance(target, list):
            target.insert(position or 0, value)
        elif isinstance(target, str):
            pos = position or 0
            new_val = target[:pos] + str(value) + target[pos:]
            self._set_value(path, new_val)

    def _delete_value(self, path: str, position: int, length: int) -> None:
        """Delete value at position."""
        target = self._get_value(path)
        if isinstance(target, list) and position is not None:
            for _ in range(length or 1):
                if 0 <= position < len(target):
                    target.pop(position)
        elif isinstance(target, str):
            pos = position or 0
            ln = length or 1
            new_val = target[:pos] + target[pos + ln:]
            self._set_value(path, new_val)

    def undo(self) -> bool:
        """Undo the last operation."""
        if self._history:
            last_op = self._history.pop()
            inverse = last_op.inverse()
            self._apply_operation(inverse)
            return True
        return False

    def get_history(self, limit: int = 100) -> List[Operation]:
        """Get operation history."""
        return list(reversed(self._history[-limit:]))


def apply_operation(
    state: Dict[str, Any],
    operation: Operation,
) -> Dict[str, Any]:
    """Apply an operation to a state dictionary."""
    doc = CollaborativeDocument(state)
    doc.apply(operation)
    return doc.state


def create_operation(
    op_type: OperationType,
    path: str,
    value: Any = None,
    author_id: str = "",
) -> Operation:
    """Create a new operation."""
    return Operation(
        type=op_type,
        path=path,
        value=value,
        author_id=author_id,
    )
