"""
FL Studio Integration - Utilities for working with FL Studio projects.

FL Studio uses .flp files which are proprietary binary format.
This module provides utilities for MIDI exchange and VST3 plugin support.

FL Studio Specifics:
- Default PPQ: 96 (standard) or 384 (HD)
- Patterns are organized in a pattern-based workflow
- Piano roll with per-note pitch automation
- VST3 support with wrapper for older VST2
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import json

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


# FL Studio PPQ constants
FL_STUDIO_PPQ = 96
FL_STUDIO_PPQ_HD = 384

# FL Studio color palette (RGBA)
FL_COLORS = {
    "drums": (255, 128, 0, 255),     # Orange
    "bass": (128, 0, 255, 255),       # Purple
    "lead": (255, 0, 128, 255),       # Pink
    "pad": (0, 128, 255, 255),        # Blue
    "fx": (0, 255, 128, 255),         # Cyan
    "vocal": (255, 255, 0, 255),      # Yellow
}


class FLPatternType(Enum):
    """FL Studio pattern types."""
    DRUMS = "drums"
    MELODY = "melody"
    BASS = "bass"
    CHORDS = "chords"
    FX = "fx"


@dataclass
class FLPattern:
    """
    Represents an FL Studio pattern.

    FL Studio organizes music by patterns in the Channel Rack,
    which are then arranged in the Playlist.
    """
    name: str
    pattern_type: FLPatternType = FLPatternType.MELODY
    color: Tuple[int, int, int, int] = (128, 128, 128, 255)
    length_bars: int = 4
    notes: List[Dict] = field(default_factory=list)

    # FL-specific properties
    swing: float = 0.0  # -100 to 100
    time_signature: Tuple[int, int] = (4, 4)

    def add_note(
        self,
        pitch: int,
        velocity: int = 100,
        start_tick: int = 0,
        duration_ticks: int = 96,
        pan: float = 0.0,
        fine_pitch: float = 0.0,
        mod_x: float = 0.5,
        mod_y: float = 0.5,
    ) -> None:
        """
        Add a note to the pattern.

        Args:
            pitch: MIDI pitch (0-127)
            velocity: Note velocity (0-127)
            start_tick: Start position in ticks
            duration_ticks: Note length in ticks
            pan: Per-note panning (-1.0 to 1.0)
            fine_pitch: Fine pitch adjustment (-1.0 to 1.0)
            mod_x: Mod X parameter (0.0 to 1.0)
            mod_y: Mod Y parameter (0.0 to 1.0)
        """
        self.notes.append({
            "pitch": pitch,
            "velocity": velocity,
            "start_tick": start_tick,
            "duration_ticks": duration_ticks,
            "pan": pan,
            "fine_pitch": fine_pitch,
            "mod_x": mod_x,
            "mod_y": mod_y,
        })

    def to_midi_track(self, ppq: int = FL_STUDIO_PPQ) -> List[Dict]:
        """Convert pattern notes to MIDI events."""
        events = []
        for note in self.notes:
            # Note on
            events.append({
                "type": "note_on",
                "tick": note["start_tick"],
                "pitch": note["pitch"],
                "velocity": note["velocity"],
            })
            # Note off
            events.append({
                "type": "note_off",
                "tick": note["start_tick"] + note["duration_ticks"],
                "pitch": note["pitch"],
                "velocity": 0,
            })

        # Sort by tick time
        events.sort(key=lambda e: e["tick"])
        return events


@dataclass
class FLProject:
    """
    Represents an FL Studio project for MIDI export.

    Not a full FL Studio project parser - provides structure
    for organizing patterns for MIDI export to FL Studio.
    """
    name: str = "Untitled"
    tempo_bpm: float = 120.0
    ppq: int = FL_STUDIO_PPQ
    time_signature: Tuple[int, int] = (4, 4)

    patterns: List[FLPattern] = field(default_factory=list)

    # Playlist arrangement (pattern index, bar position, length)
    arrangement: List[Tuple[int, int, int]] = field(default_factory=list)

    # Project metadata
    genre: str = ""
    artist: str = ""
    title: str = ""

    def add_pattern(self, pattern: FLPattern) -> int:
        """
        Add a pattern to the project.

        Returns:
            Pattern index
        """
        self.patterns.append(pattern)
        return len(self.patterns) - 1

    def arrange_pattern(
        self,
        pattern_index: int,
        bar_position: int,
        length_bars: Optional[int] = None,
    ) -> None:
        """
        Place a pattern in the playlist arrangement.

        Args:
            pattern_index: Index of pattern to place
            bar_position: Bar position in playlist
            length_bars: Override pattern length (optional)
        """
        if 0 <= pattern_index < len(self.patterns):
            length = length_bars or self.patterns[pattern_index].length_bars
            self.arrangement.append((pattern_index, bar_position, length))

    def export_midi(self, output_path: str) -> str:
        """
        Export project patterns to MIDI file for FL Studio import.

        Args:
            output_path: Output MIDI file path

        Returns:
            Path to exported file
        """
        if not MIDO_AVAILABLE:
            raise ImportError("mido package required")

        mid = mido.MidiFile(ticks_per_beat=self.ppq)

        # Create tempo/meta track
        meta_track = mido.MidiTrack()
        mid.tracks.append(meta_track)

        # Set tempo
        tempo_us = int(60_000_000 / self.tempo_bpm)
        meta_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))

        # Set time signature
        meta_track.append(mido.MetaMessage(
            'time_signature',
            numerator=self.time_signature[0],
            denominator=self.time_signature[1],
            time=0
        ))

        # Export each pattern as a track
        for i, pattern in enumerate(self.patterns):
            track = mido.MidiTrack()
            mid.tracks.append(track)

            # Track name
            track.append(mido.MetaMessage(
                'track_name',
                name=pattern.name,
                time=0
            ))

            # Convert pattern to events
            events = pattern.to_midi_track(self.ppq)

            # Convert to delta times
            current_tick = 0
            for event in events:
                delta = event["tick"] - current_tick
                current_tick = event["tick"]

                if event["type"] == "note_on":
                    track.append(mido.Message(
                        'note_on',
                        note=event["pitch"],
                        velocity=event["velocity"],
                        channel=i % 16,
                        time=delta
                    ))
                else:
                    track.append(mido.Message(
                        'note_off',
                        note=event["pitch"],
                        velocity=0,
                        channel=i % 16,
                        time=delta
                    ))

            # End of track
            track.append(mido.MetaMessage('end_of_track', time=0))

        output_path = Path(output_path)
        mid.save(str(output_path))

        return str(output_path)


def export_to_fl_studio(
    midi_path: str,
    output_path: Optional[str] = None,
    use_hd_ppq: bool = False,
) -> str:
    """
    Prepare a MIDI file for optimal FL Studio import.

    - Normalizes PPQ to FL Studio's 96 (or 384 for HD)
    - Ensures proper channel assignments

    Args:
        midi_path: Input MIDI file
        output_path: Output path (default: input_fl.mid)
        use_hd_ppq: Use 384 PPQ for higher resolution

    Returns:
        Path to prepared file
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido package required")

    target_ppq = FL_STUDIO_PPQ_HD if use_hd_ppq else FL_STUDIO_PPQ

    midi_path = Path(midi_path)
    mid = mido.MidiFile(str(midi_path))

    if output_path is None:
        output_path = f"{midi_path.stem}_fl.mid"

    # If PPQ matches, just copy
    if mid.ticks_per_beat == target_ppq:
        mid.save(output_path)
        return output_path

    # Create new MIDI with FL Studio PPQ
    new_mid = mido.MidiFile(ticks_per_beat=target_ppq)
    ppq_ratio = target_ppq / mid.ticks_per_beat

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)

        for msg in track:
            new_msg = msg.copy()

            if hasattr(msg, 'time'):
                new_msg = msg.copy(time=int(msg.time * ppq_ratio))

            new_track.append(new_msg)

    new_mid.save(output_path)
    return output_path


def import_from_fl_studio(midi_path: str) -> FLProject:
    """
    Import a MIDI file exported from FL Studio.

    Args:
        midi_path: Path to MIDI file from FL Studio

    Returns:
        FLProject with imported data
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido package required")

    midi_path = Path(midi_path)
    mid = mido.MidiFile(str(midi_path))

    # Detect tempo
    tempo_bpm = 120.0
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_bpm = 60_000_000 / msg.tempo
                break

    project = FLProject(
        name=midi_path.stem,
        tempo_bpm=tempo_bpm,
        ppq=mid.ticks_per_beat,
    )

    # Convert each track to a pattern
    for i, track in enumerate(mid.tracks):
        track_name = f"Pattern {i+1}"
        notes = []
        current_tick = 0
        active_notes = {}

        for msg in track:
            current_tick += msg.time

            if msg.type == 'track_name':
                track_name = msg.name
            elif msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active_notes[key] = (current_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes:
                    start_tick, velocity = active_notes.pop(key)
                    notes.append({
                        "pitch": msg.note,
                        "velocity": velocity,
                        "start_tick": start_tick,
                        "duration_ticks": current_tick - start_tick,
                        "pan": 0.0,
                        "fine_pitch": 0.0,
                        "mod_x": 0.5,
                        "mod_y": 0.5,
                    })

        if notes:
            # Calculate pattern length
            max_tick = max(n["start_tick"] + n["duration_ticks"] for n in notes)
            length_bars = max(1, int(max_tick / (project.ppq * 4)) + 1)

            pattern = FLPattern(
                name=track_name,
                length_bars=length_bars,
                notes=notes,
            )
            project.add_pattern(pattern)

    return project


def create_fl_template(
    name: str,
    tempo: float = 120.0,
    bars: int = 4,
    style: str = "hip_hop",
) -> FLProject:
    """
    Create an FL Studio project template.

    Args:
        name: Project name
        tempo: Tempo in BPM
        bars: Pattern length in bars
        style: Production style (hip_hop, edm, trap, etc.)

    Returns:
        FLProject template
    """
    project = FLProject(
        name=name,
        tempo_bpm=tempo,
        genre=style,
    )

    # Create template patterns based on style
    style_configs = {
        "hip_hop": [
            ("Drums", FLPatternType.DRUMS, FL_COLORS["drums"]),
            ("808 Bass", FLPatternType.BASS, FL_COLORS["bass"]),
            ("Melody", FLPatternType.MELODY, FL_COLORS["lead"]),
            ("Hi-Hats", FLPatternType.DRUMS, FL_COLORS["fx"]),
        ],
        "edm": [
            ("Kick", FLPatternType.DRUMS, FL_COLORS["drums"]),
            ("Bass Drop", FLPatternType.BASS, FL_COLORS["bass"]),
            ("Lead Synth", FLPatternType.MELODY, FL_COLORS["lead"]),
            ("Pad", FLPatternType.CHORDS, FL_COLORS["pad"]),
            ("FX Riser", FLPatternType.FX, FL_COLORS["fx"]),
        ],
        "trap": [
            ("Trap Kit", FLPatternType.DRUMS, FL_COLORS["drums"]),
            ("808", FLPatternType.BASS, FL_COLORS["bass"]),
            ("Lead", FLPatternType.MELODY, FL_COLORS["lead"]),
            ("Hi-Hat Roll", FLPatternType.DRUMS, FL_COLORS["fx"]),
            ("Vox Chop", FLPatternType.FX, FL_COLORS["vocal"]),
        ],
        "lofi": [
            ("Dusty Drums", FLPatternType.DRUMS, FL_COLORS["drums"]),
            ("Mellow Bass", FLPatternType.BASS, FL_COLORS["bass"]),
            ("Keys", FLPatternType.CHORDS, FL_COLORS["pad"]),
            ("Lead", FLPatternType.MELODY, FL_COLORS["lead"]),
        ],
    }

    patterns = style_configs.get(style, style_configs["hip_hop"])

    for pattern_name, pattern_type, color in patterns:
        pattern = FLPattern(
            name=pattern_name,
            pattern_type=pattern_type,
            color=color,
            length_bars=bars,
        )
        project.add_pattern(pattern)

    return project


# VST3 Plugin Support for FL Studio
@dataclass
class FLVSTConfig:
    """Configuration for FL Studio VST3 plugin compatibility."""
    plugin_name: str
    manufacturer: str = "iDAW"
    version: str = "1.0.0"

    # FL-specific settings
    use_bridged_mode: bool = False  # 32-bit bridge
    enable_sidechain: bool = True
    midi_input: bool = True
    midi_output: bool = True

    # GUI settings
    resizable_ui: bool = True
    default_width: int = 800
    default_height: int = 600


def get_fl_vst3_info() -> Dict[str, Any]:
    """
    Get FL Studio VST3 compatibility information.

    Returns:
        Dict with FL Studio VST3 requirements
    """
    return {
        "format": "VST3",
        "recommended_ppq": FL_STUDIO_PPQ,
        "supported_sample_rates": [44100, 48000, 88200, 96000],
        "plugin_scan_paths": [
            "C:\\Program Files\\Common Files\\VST3",  # Windows
            "C:\\Program Files (x86)\\Common Files\\VST3",  # Windows 32-bit
            "/Library/Audio/Plug-Ins/VST3",  # macOS
            "~/.vst3",  # Linux
        ],
        "fl_studio_versions": ["FL Studio 20", "FL Studio 21"],
        "requirements": {
            "minimum_version": "20.0",
            "automation_support": True,
            "preset_support": True,
            "sidechain_routing": True,
        },
    }
