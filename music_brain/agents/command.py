"""
Command Pattern with Undo/Redo History for UnifiedHub.

Implements the command pattern for all state-changing operations,
enabling full undo/redo functionality for session changes.

Usage:
    from music_brain.agents import UnifiedHub

    with UnifiedHub() as hub:
        # Commands are auto-recorded
        hub.set_tempo(120)
        hub.set_vowel("O")
        hub.note_on(60)

        # Undo last command
        hub.undo()

        # Redo
        hub.redo()

        # Check history
        print(hub.command_history.can_undo)
        print(hub.command_history.undo_stack_size)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from .unified_hub import UnifiedHub

logger = logging.getLogger(__name__)


# =============================================================================
# Command Types
# =============================================================================


class CommandCategory(Enum):
    """Categories of commands for filtering and grouping."""

    DAW = "daw"  # Transport, tempo, track operations
    VOICE = "voice"  # Voice synthesis parameters
    SESSION = "session"  # Session metadata changes
    AGENT = "agent"  # AI agent queries (usually not undoable)
    ML = "ml"  # ML pipeline operations
    PLUGIN = "plugin"  # Plugin operations
    CUSTOM = "custom"  # User-defined commands


@dataclass
class CommandResult:
    """Result of command execution."""

    success: bool
    message: str = ""
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    error: Optional[Exception] = None


@dataclass
class CommandMetadata:
    """Metadata about a command execution."""

    id: str
    name: str
    category: CommandCategory
    description: str
    executed_at: float
    undone_at: Optional[float] = None
    redone_at: Optional[float] = None
    execution_time_ms: float = 0.0
    is_compound: bool = False  # True if this is a batch of commands


# =============================================================================
# Command Interface
# =============================================================================


class Command(ABC):
    """
    Abstract base class for all undoable commands.

    Implements the Command pattern with:
    - execute(): Perform the action
    - undo(): Reverse the action
    - redo(): Re-perform (default: calls execute)

    Commands capture state before execution for reliable undo.
    """

    def __init__(
        self,
        hub: "UnifiedHub",
        category: CommandCategory = CommandCategory.CUSTOM,
        description: str = "",
    ):
        self._hub = hub
        self._category = category
        self._description = description
        self._id = f"{self.name}_{int(time.time() * 1000)}"
        self._executed = False
        self._previous_state: Dict[str, Any] = {}
        self._execution_time_ms: float = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Command name for display and logging."""
        ...

    @property
    def category(self) -> CommandCategory:
        return self._category

    @property
    def description(self) -> str:
        return self._description or f"{self.name} command"

    @property
    def id(self) -> str:
        return self._id

    @property
    def executed(self) -> bool:
        return self._executed

    def execute(self) -> CommandResult:
        """
        Execute the command.

        Automatically captures state before execution for undo.
        """
        if self._executed:
            return CommandResult(
                success=False, message=f"Command {self.name} already executed"
            )

        # Capture state before execution
        self._capture_state()

        start = time.perf_counter()
        try:
            result = self._do_execute()
            self._executed = True
            self._execution_time_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                f"Executed {self.name} in {self._execution_time_ms:.2f}ms"
            )
            return result
        except Exception as e:
            logger.error(f"Command {self.name} failed: {e}")
            return CommandResult(success=False, message=str(e), error=e)

    def undo(self) -> CommandResult:
        """Undo the command, restoring previous state."""
        if not self._executed:
            return CommandResult(
                success=False, message=f"Command {self.name} not executed"
            )

        try:
            result = self._do_undo()
            self._executed = False
            logger.debug(f"Undone {self.name}")
            return result
        except Exception as e:
            logger.error(f"Undo {self.name} failed: {e}")
            return CommandResult(success=False, message=str(e), error=e)

    def redo(self) -> CommandResult:
        """Redo the command (default: re-execute)."""
        if self._executed:
            return CommandResult(
                success=False, message=f"Command {self.name} already executed"
            )

        try:
            result = self._do_redo()
            self._executed = True
            logger.debug(f"Redone {self.name}")
            return result
        except Exception as e:
            logger.error(f"Redo {self.name} failed: {e}")
            return CommandResult(success=False, message=str(e), error=e)

    @abstractmethod
    def _do_execute(self) -> CommandResult:
        """Implement the actual execution logic."""
        ...

    @abstractmethod
    def _do_undo(self) -> CommandResult:
        """Implement the undo logic."""
        ...

    def _do_redo(self) -> CommandResult:
        """Default redo: re-execute. Override for custom behavior."""
        return self._do_execute()

    def _capture_state(self) -> None:
        """Capture relevant state before execution. Override as needed."""
        pass

    def get_metadata(self) -> CommandMetadata:
        """Get command metadata."""
        return CommandMetadata(
            id=self._id,
            name=self.name,
            category=self._category,
            description=self._description,
            executed_at=time.time() if self._executed else 0,
            execution_time_ms=self._execution_time_ms,
        )


# =============================================================================
# Concrete Commands
# =============================================================================


class SetTempoCommand(Command):
    """Command to change DAW tempo."""

    def __init__(self, hub: "UnifiedHub", new_tempo: float):
        super().__init__(
            hub, CommandCategory.DAW, f"Set tempo to {new_tempo} BPM"
        )
        self._new_tempo = new_tempo
        self._old_tempo: float = 120.0

    @property
    def name(self) -> str:
        return "SetTempo"

    def _capture_state(self) -> None:
        self._old_tempo = self._hub.daw_state.tempo

    def _do_execute(self) -> CommandResult:
        if self._hub._daw:
            self._hub._daw.set_tempo(self._new_tempo)
            self._hub._daw_state.tempo = self._new_tempo
            self._hub._session.tempo = self._new_tempo
            return CommandResult(success=True, data={"tempo": self._new_tempo})
        return CommandResult(success=False, message="DAW not connected")

    def _do_undo(self) -> CommandResult:
        if self._hub._daw:
            self._hub._daw.set_tempo(self._old_tempo)
            self._hub._daw_state.tempo = self._old_tempo
            self._hub._session.tempo = self._old_tempo
            return CommandResult(success=True, data={"tempo": self._old_tempo})
        return CommandResult(success=False, message="DAW not connected")


class SetVowelCommand(Command):
    """Command to change voice vowel."""

    def __init__(self, hub: "UnifiedHub", vowel: str, channel: int = 0):
        super().__init__(hub, CommandCategory.VOICE, f"Set vowel to {vowel}")
        self._new_vowel = vowel.upper()
        self._channel = channel
        self._old_vowel: str = "A"

    @property
    def name(self) -> str:
        return "SetVowel"

    def _capture_state(self) -> None:
        self._old_vowel = self._hub.voice_state.vowel

    def _do_execute(self) -> CommandResult:
        if self._hub._voice:
            self._hub._voice.set_vowel(self._new_vowel, self._channel)
            self._hub._voice_state.vowel = self._new_vowel
            return CommandResult(success=True, data={"vowel": self._new_vowel})
        return CommandResult(success=False, message="Voice not initialized")

    def _do_undo(self) -> CommandResult:
        if self._hub._voice:
            self._hub._voice.set_vowel(self._old_vowel, self._channel)
            self._hub._voice_state.vowel = self._old_vowel
            return CommandResult(success=True, data={"vowel": self._old_vowel})
        return CommandResult(success=False, message="Voice not initialized")


class SetBreathinessCommand(Command):
    """Command to change voice breathiness."""

    def __init__(self, hub: "UnifiedHub", amount: float, channel: int = 0):
        super().__init__(
            hub, CommandCategory.VOICE, f"Set breathiness to {amount:.2f}"
        )
        self._new_amount = max(0.0, min(1.0, amount))
        self._channel = channel
        self._old_amount: float = 0.0

    @property
    def name(self) -> str:
        return "SetBreathiness"

    def _capture_state(self) -> None:
        self._old_amount = self._hub.voice_state.breathiness

    def _do_execute(self) -> CommandResult:
        if self._hub._voice:
            self._hub._voice.set_breathiness(self._new_amount, self._channel)
            self._hub._voice_state.breathiness = self._new_amount
            return CommandResult(
                success=True, data={"breathiness": self._new_amount}
            )
        return CommandResult(success=False, message="Voice not initialized")

    def _do_undo(self) -> CommandResult:
        if self._hub._voice:
            self._hub._voice.set_breathiness(self._old_amount, self._channel)
            self._hub._voice_state.breathiness = self._old_amount
            return CommandResult(
                success=True, data={"breathiness": self._old_amount}
            )
        return CommandResult(success=False, message="Voice not initialized")


class SetVibratoCommand(Command):
    """Command to change voice vibrato settings."""

    def __init__(
        self, hub: "UnifiedHub", rate: float, depth: float, channel: int = 0
    ):
        super().__init__(
            hub,
            CommandCategory.VOICE,
            f"Set vibrato rate={rate:.2f} depth={depth:.2f}",
        )
        self._new_rate = rate
        self._new_depth = depth
        self._channel = channel
        self._old_rate: float = 0.0
        self._old_depth: float = 0.0

    @property
    def name(self) -> str:
        return "SetVibrato"

    def _capture_state(self) -> None:
        self._old_rate = self._hub.voice_state.vibrato_rate
        self._old_depth = self._hub.voice_state.vibrato_depth

    def _do_execute(self) -> CommandResult:
        if self._hub._voice:
            self._hub._voice.set_vibrato(
                self._new_rate, self._new_depth, self._channel
            )
            self._hub._voice_state.vibrato_rate = self._new_rate
            self._hub._voice_state.vibrato_depth = self._new_depth
            return CommandResult(
                success=True,
                data={
                    "vibrato_rate": self._new_rate,
                    "vibrato_depth": self._new_depth,
                },
            )
        return CommandResult(success=False, message="Voice not initialized")

    def _do_undo(self) -> CommandResult:
        if self._hub._voice:
            self._hub._voice.set_vibrato(
                self._old_rate, self._old_depth, self._channel
            )
            self._hub._voice_state.vibrato_rate = self._old_rate
            self._hub._voice_state.vibrato_depth = self._old_depth
            return CommandResult(
                success=True,
                data={
                    "vibrato_rate": self._old_rate,
                    "vibrato_depth": self._old_depth,
                },
            )
        return CommandResult(success=False, message="Voice not initialized")


class NoteOnCommand(Command):
    """Command to start a note."""

    def __init__(
        self,
        hub: "UnifiedHub",
        pitch: int,
        velocity: int = 100,
        channel: int = 0,
    ):
        super().__init__(hub, CommandCategory.VOICE, f"Note on: {pitch}")
        self._pitch = pitch
        self._velocity = velocity
        self._channel = channel
        self._was_active: bool = False
        self._old_pitch: int = 60

    @property
    def name(self) -> str:
        return "NoteOn"

    def _capture_state(self) -> None:
        self._was_active = self._hub.voice_state.active
        self._old_pitch = self._hub.voice_state.pitch

    def _do_execute(self) -> CommandResult:
        if self._hub._voice:
            self._hub._voice.note_on(
                self._pitch, self._velocity, self._channel
            )
            self._hub._voice_state.pitch = self._pitch
            self._hub._voice_state.velocity = self._velocity
            self._hub._voice_state.active = True
            return CommandResult(success=True, data={"pitch": self._pitch})
        return CommandResult(success=False, message="Voice not initialized")

    def _do_undo(self) -> CommandResult:
        if self._hub._voice:
            # Stop the note we started
            self._hub._voice.note_off(self._pitch, self._channel)
            self._hub._voice_state.active = self._was_active
            self._hub._voice_state.pitch = self._old_pitch
            return CommandResult(success=True, data={"note_off": self._pitch})
        return CommandResult(success=False, message="Voice not initialized")


class NoteOffCommand(Command):
    """Command to stop a note."""

    def __init__(
        self,
        hub: "UnifiedHub",
        pitch: Optional[int] = None,
        channel: int = 0,
    ):
        super().__init__(
            hub, CommandCategory.VOICE, f"Note off: {pitch or 'active'}"
        )
        self._pitch = pitch
        self._channel = channel
        self._was_active: bool = False
        self._old_pitch: int = 60
        self._old_velocity: int = 100

    @property
    def name(self) -> str:
        return "NoteOff"

    def _capture_state(self) -> None:
        self._was_active = self._hub.voice_state.active
        self._old_pitch = self._hub.voice_state.pitch
        self._old_velocity = self._hub.voice_state.velocity

    def _do_execute(self) -> CommandResult:
        if self._hub._voice:
            self._hub._voice.note_off(self._pitch, self._channel)
            self._hub._voice_state.active = False
            return CommandResult(
                success=True,
                data={"note_off": self._pitch or self._old_pitch},
            )
        return CommandResult(success=False, message="Voice not initialized")

    def _do_undo(self) -> CommandResult:
        if self._hub._voice and self._was_active:
            # Re-start the note that was playing
            self._hub._voice.note_on(
                self._old_pitch, self._old_velocity, self._channel
            )
            self._hub._voice_state.active = True
            self._hub._voice_state.pitch = self._old_pitch
            self._hub._voice_state.velocity = self._old_velocity
            return CommandResult(
                success=True, data={"note_on": self._old_pitch}
            )
        return CommandResult(success=True, message="No note was active")


class UpdateSessionCommand(Command):
    """Command to update session metadata."""

    def __init__(self, hub: "UnifiedHub", **updates: Any) -> None:
        keys = list(updates.keys())
        super().__init__(
            hub, CommandCategory.SESSION, f"Update session: {keys}"
        )
        self._updates = updates
        self._old_values: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "UpdateSession"

    def _capture_state(self) -> None:
        for key in self._updates:
            if hasattr(self._hub._session, key):
                self._old_values[key] = getattr(self._hub._session, key)

    def _do_execute(self) -> CommandResult:
        for key, value in self._updates.items():
            if hasattr(self._hub._session, key):
                setattr(self._hub._session, key, value)
        self._hub._session.updated_at = datetime.now().isoformat()
        return CommandResult(success=True, data=self._updates)

    def _do_undo(self) -> CommandResult:
        for key, value in self._old_values.items():
            setattr(self._hub._session, key, value)
        return CommandResult(success=True, data=self._old_values)


# =============================================================================
# Compound Command
# =============================================================================


class CompoundCommand(Command):
    """
    A command that groups multiple commands together.

    All sub-commands are executed/undone as a single unit.
    Undo happens in reverse order.
    """

    def __init__(
        self,
        hub: "UnifiedHub",
        commands: List[Command],
        description: str = "Compound command",
    ):
        super().__init__(hub, CommandCategory.CUSTOM, description)
        self._commands = commands
        self._executed_count = 0

    @property
    def name(self) -> str:
        return "Compound"

    def _do_execute(self) -> CommandResult:
        results = []
        for cmd in self._commands:
            result = cmd.execute()
            results.append(result)
            if result.success:
                self._executed_count += 1
            else:
                # Rollback on failure
                self._rollback()
                return CommandResult(
                    success=False,
                    message=(
                        f"Compound failed at {cmd.name}: {result.message}"
                    ),
                    data=results,
                )
        return CommandResult(
            success=True,
            message=f"Executed {len(self._commands)} commands",
            data=results,
        )

    def _do_undo(self) -> CommandResult:
        results = []
        # Undo in reverse order
        for cmd in reversed(self._commands[: self._executed_count]):
            result = cmd.undo()
            results.append(result)
        self._executed_count = 0
        return CommandResult(success=True, data=results)

    def _rollback(self) -> None:
        """Rollback partially executed commands."""
        for cmd in reversed(self._commands[: self._executed_count]):
            try:
                cmd.undo()
            except Exception as e:
                logger.error(f"Rollback failed for {cmd.name}: {e}")
        self._executed_count = 0


# =============================================================================
# Command History Manager
# =============================================================================


@dataclass
class HistoryStats:
    """Statistics about command history."""

    total_commands: int = 0
    total_undos: int = 0
    total_redos: int = 0
    undo_stack_size: int = 0
    redo_stack_size: int = 0
    commands_by_category: Dict[str, int] = field(default_factory=dict)
    avg_execution_time_ms: float = 0.0


class CommandHistory:
    """
    Manages undo/redo history for commands.

    Features:
    - Configurable max history size
    - Command grouping/batching
    - History persistence (save/load)
    - Statistics and introspection
    """

    def __init__(self, max_size: int = 100):
        self._max_size = max_size
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self._total_commands = 0
        self._total_undos = 0
        self._total_redos = 0
        self._execution_times: List[float] = []
        self._on_change: List[Callable[[str, Command], None]] = []

    def execute(self, command: Command) -> CommandResult:
        """Execute a command and add to history."""
        result = command.execute()

        if result.success:
            self._undo_stack.append(command)
            self._redo_stack.clear()  # Clear redo on new command
            self._total_commands += 1
            self._execution_times.append(command._execution_time_ms)

            # Trim history if needed
            while len(self._undo_stack) > self._max_size:
                self._undo_stack.pop(0)

            self._notify("execute", command)

        return result

    def undo(self) -> Optional[CommandResult]:
        """Undo the last command."""
        if not self._undo_stack:
            return None

        command = self._undo_stack.pop()
        result = command.undo()

        if result.success:
            self._redo_stack.append(command)
            self._total_undos += 1
            self._notify("undo", command)

        return result

    def redo(self) -> Optional[CommandResult]:
        """Redo the last undone command."""
        if not self._redo_stack:
            return None

        command = self._redo_stack.pop()
        result = command.redo()

        if result.success:
            self._undo_stack.append(command)
            self._total_redos += 1
            self._notify("redo", command)

        return result

    def clear(self) -> None:
        """Clear all history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._notify("clear", None)  # type: ignore

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_stack_size(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_stack_size(self) -> int:
        return len(self._redo_stack)

    def peek_undo(self) -> Optional[CommandMetadata]:
        """Get metadata of the next command to undo."""
        if self._undo_stack:
            return self._undo_stack[-1].get_metadata()
        return None

    def peek_redo(self) -> Optional[CommandMetadata]:
        """Get metadata of the next command to redo."""
        if self._redo_stack:
            return self._redo_stack[-1].get_metadata()
        return None

    def get_history(self, limit: int = 20) -> List[CommandMetadata]:
        """Get recent command history."""
        return [
            cmd.get_metadata()
            for cmd in reversed(self._undo_stack[-limit:])
        ]

    def get_stats(self) -> HistoryStats:
        """Get history statistics."""
        by_category: Dict[str, int] = {}
        for cmd in self._undo_stack:
            cat = cmd.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        avg_time = (
            sum(self._execution_times) / len(self._execution_times)
            if self._execution_times
            else 0.0
        )

        return HistoryStats(
            total_commands=self._total_commands,
            total_undos=self._total_undos,
            total_redos=self._total_redos,
            undo_stack_size=len(self._undo_stack),
            redo_stack_size=len(self._redo_stack),
            commands_by_category=by_category,
            avg_execution_time_ms=avg_time,
        )

    def on_change(self, callback: Callable[[str, Command], None]) -> None:
        """Register callback for history changes."""
        self._on_change.append(callback)

    def _notify(self, action: str, command: Command) -> None:
        """Notify listeners of history change."""
        for cb in self._on_change:
            try:
                cb(action, command)
            except Exception as e:
                logger.error(f"History callback error: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize history state (for session save)."""
        history = [
            cmd.get_metadata().__dict__
            for cmd in self._undo_stack[-20:]
        ]
        return {
            "undo_count": len(self._undo_stack),
            "redo_count": len(self._redo_stack),
            "history": history,
            "stats": asdict(self.get_stats()),
        }


# =============================================================================
# Command Factory
# =============================================================================


class CommandFactory:
    """Factory for creating commands with hub reference."""

    def __init__(self, hub: "UnifiedHub"):
        self._hub = hub

    def set_tempo(self, tempo: float) -> SetTempoCommand:
        return SetTempoCommand(self._hub, tempo)

    def set_vowel(self, vowel: str, channel: int = 0) -> SetVowelCommand:
        return SetVowelCommand(self._hub, vowel, channel)

    def set_breathiness(
        self, amount: float, channel: int = 0
    ) -> SetBreathinessCommand:
        return SetBreathinessCommand(self._hub, amount, channel)

    def set_vibrato(
        self, rate: float, depth: float, channel: int = 0
    ) -> SetVibratoCommand:
        return SetVibratoCommand(self._hub, rate, depth, channel)

    def note_on(
        self, pitch: int, velocity: int = 100, channel: int = 0
    ) -> NoteOnCommand:
        return NoteOnCommand(self._hub, pitch, velocity, channel)

    def note_off(
        self, pitch: Optional[int] = None, channel: int = 0
    ) -> NoteOffCommand:
        return NoteOffCommand(self._hub, pitch, channel)

    def update_session(self, **updates: Any) -> UpdateSessionCommand:
        return UpdateSessionCommand(self._hub, **updates)

    def compound(
        self, commands: List[Command], description: str = ""
    ) -> CompoundCommand:
        return CompoundCommand(self._hub, commands, description)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    "Command",
    "CommandResult",
    "CommandMetadata",
    "CommandCategory",
    # Concrete commands
    "SetTempoCommand",
    "SetVowelCommand",
    "SetBreathinessCommand",
    "SetVibratoCommand",
    "NoteOnCommand",
    "NoteOffCommand",
    "UpdateSessionCommand",
    "CompoundCommand",
    # History
    "CommandHistory",
    "HistoryStats",
    # Factory
    "CommandFactory",
]
