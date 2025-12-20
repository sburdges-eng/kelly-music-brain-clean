"""
Reaper Integration - Utilities for working with Reaper projects.

Reaper uses .rpp (Reaper Project) files which are plain text and human-readable.
This module provides utilities for MIDI exchange, OSC control, and ReaScript integration.

Reaper Specifics:
- Default PPQ: 960 (high resolution)
- Highly customizable and scriptable
- Excellent OSC support for remote control
- Supports VST2, VST3, AU, LV2, CLAP formats
- ReaScript (Lua, EEL2, Python) for automation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from pathlib import Path
from enum import Enum
import json
import socket
import struct

from music_brain.utils.path_utils import safe_path

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


# Reaper PPQ constant
REAPER_PPQ = 960

# Reaper OSC defaults
REAPER_OSC_LISTEN_PORT = 8000
REAPER_OSC_SEND_PORT = 9000


class ReaperTrackType(Enum):
    """Reaper track types."""
    AUDIO = "audio"
    MIDI = "midi"
    FOLDER = "folder"
    MASTER = "master"


class ReaperAction(Enum):
    """Common Reaper actions by ID."""
    PLAY = 1007
    STOP = 1016
    PAUSE = 1008
    RECORD = 1013
    REWIND = 1012
    FAST_FORWARD = 1014
    GOTO_START = 40042
    GOTO_END = 40043
    TOGGLE_LOOP = 1068
    TOGGLE_METRONOME = 40364
    UNDO = 40029
    REDO = 40030
    SAVE_PROJECT = 40026
    RENDER_PROJECT = 41824


@dataclass
class ReaperTrack:
    """
    Represents a Reaper track.

    Reaper tracks are highly flexible and support
    extensive routing and FX chains.
    """
    name: str
    track_type: ReaperTrackType = ReaperTrackType.MIDI
    index: int = 0

    # Track state
    armed: bool = False
    solo: bool = False
    mute: bool = False
    phase_inverted: bool = False
    pan: float = 0.0  # -1.0 to 1.0
    volume_db: float = 0.0  # dB
    pan_law: float = -3.0  # dB

    # Routing
    parent_track: Optional[int] = None
    receives: List[Tuple[int, float]] = field(default_factory=list)
    sends: List[Tuple[int, float]] = field(default_factory=list)

    # FX chain
    fx_chain: List[str] = field(default_factory=list)
    fx_enabled: List[bool] = field(default_factory=list)

    # MIDI data
    notes: List[Dict] = field(default_factory=list)
    items: List[Dict] = field(default_factory=list)

    def add_note(
        self,
        pitch: int,
        velocity: int = 100,
        start_tick: int = 0,
        duration_ticks: int = 960,
        channel: int = 0,
    ) -> None:
        """Add a MIDI note to the track."""
        self.notes.append({
            "pitch": pitch,
            "velocity": velocity,
            "start_tick": start_tick,
            "duration_ticks": duration_ticks,
            "channel": channel,
        })

    def add_item(
        self,
        name: str,
        position: float,
        length: float,
        source_file: Optional[str] = None,
    ) -> None:
        """Add a media item to the track."""
        self.items.append({
            "name": name,
            "position": position,
            "length": length,
            "source_file": source_file,
        })


@dataclass
class ReaperProject:
    """
    Represents a Reaper project.

    Can export to MIDI and generate basic .rpp structure.
    """
    name: str = "Untitled"
    tempo_bpm: float = 120.0
    ppq: int = REAPER_PPQ
    time_signature: Tuple[int, int] = (4, 4)
    sample_rate: int = 48000

    # Project structure
    tracks: List[ReaperTrack] = field(default_factory=list)

    # Tempo/meter maps
    tempo_map: List[Tuple[float, float]] = field(default_factory=list)  # (position, bpm)
    marker_map: List[Tuple[float, str]] = field(default_factory=list)  # (position, name)
    region_map: List[Tuple[float, float, str]] = field(default_factory=list)  # (start, end, name)

    # Project settings
    auto_crossfade: bool = True
    loop_enabled: bool = False
    loop_start: float = 0.0
    loop_end: float = 16.0

    def add_track(self, track: ReaperTrack) -> int:
        """Add a track to the project."""
        track.index = len(self.tracks)
        self.tracks.append(track)
        return track.index

    def add_marker(self, position: float, name: str) -> None:
        """Add a marker at the specified position (in beats)."""
        self.marker_map.append((position, name))
        self.marker_map.sort(key=lambda x: x[0])

    def add_region(self, start: float, end: float, name: str) -> None:
        """Add a region between start and end positions."""
        self.region_map.append((start, end, name))
        self.region_map.sort(key=lambda x: x[0])

    def export_midi(self, output_path: str) -> str:
        """
        Export project MIDI tracks to a MIDI file.

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

        # Initial tempo
        tempo_us = int(60_000_000 / self.tempo_bpm)
        meta_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))

        # Initial time signature
        meta_track.append(mido.MetaMessage(
            'time_signature',
            numerator=self.time_signature[0],
            denominator=self.time_signature[1],
            time=0
        ))

        # Add markers
        for position, name in self.marker_map:
            tick = int(position * self.ppq)
            meta_track.append(mido.MetaMessage(
                'marker',
                text=name,
                time=tick
            ))

        # Export MIDI tracks
        for track in self.tracks:
            if track.track_type == ReaperTrackType.MIDI and track.notes:
                midi_track = mido.MidiTrack()
                mid.tracks.append(midi_track)

                # Track name
                midi_track.append(mido.MetaMessage(
                    'track_name',
                    name=track.name,
                    time=0
                ))

                # Build note events
                events = []
                for note in track.notes:
                    events.append((
                        note["start_tick"],
                        "note_on",
                        note["pitch"],
                        note["velocity"],
                        note.get("channel", 0),
                    ))
                    events.append((
                        note["start_tick"] + note["duration_ticks"],
                        "note_off",
                        note["pitch"],
                        0,
                        note.get("channel", 0),
                    ))

                events.sort(key=lambda e: e[0])

                # Convert to delta times
                current_tick = 0
                for tick, msg_type, pitch, vel, ch in events:
                    delta = tick - current_tick
                    current_tick = tick

                    if msg_type == "note_on":
                        midi_track.append(mido.Message(
                            'note_on',
                            note=pitch,
                            velocity=vel,
                            channel=ch,
                            time=delta
                        ))
                    else:
                        midi_track.append(mido.Message(
                            'note_off',
                            note=pitch,
                            velocity=0,
                            channel=ch,
                            time=delta
                        ))

                # End of track
                midi_track.append(mido.MetaMessage('end_of_track', time=0))

        output_safe = safe_path(output_path)
        mid.save(str(output_safe))

        return str(output_safe)

    def export_rpp(self, output_path: str) -> str:
        """
        Export project to Reaper .rpp format.

        Args:
            output_path: Output .rpp file path

        Returns:
            Path to exported file
        """
        lines = []

        # Header
        lines.append("<REAPER_PROJECT 0.1 \"7.0\" 1725000000")
        lines.append(f"  TEMPO {self.tempo_bpm} {self.time_signature[0]} {self.time_signature[1]}")
        lines.append(f"  SAMPLERATE {self.sample_rate} 0 0")
        lines.append(f"  PROJECT_NAME \"{self.name}\"")

        # Add markers
        for i, (position, name) in enumerate(self.marker_map):
            lines.append(f"  MARKER {i} {position} \"{name}\" 0")

        # Add regions
        for i, (start, end, name) in enumerate(self.region_map):
            lines.append(f"  MARKER {i + len(self.marker_map)} {start} \"{name}\" 1 0 1 R 0 0 1 {end}")

        # Add tracks
        for track in self.tracks:
            lines.append("  <TRACK")
            lines.append(f"    NAME \"{track.name}\"")
            lines.append(f"    VOLPAN {10 ** (track.volume_db / 20):.6f} {track.pan:.6f} -1 -1 1")
            lines.append(f"    MUTESOLO {1 if track.mute else 0} {1 if track.solo else 0} 0")

            # Add FX chain if present
            if track.fx_chain:
                lines.append("    <FXCHAIN")
                for fx_name in track.fx_chain:
                    lines.append(f"      <VST \"{fx_name}\"")
                    lines.append("      >")
                lines.append("    >")

            lines.append("  >")

        lines.append(">")

        # Write file
        output_safe = safe_path(output_path)
        with open(str(output_safe), 'w') as f:
            f.write('\n'.join(lines))

        return str(output_safe)


def export_to_reaper(
    midi_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Prepare a MIDI file for optimal Reaper import.

    Args:
        midi_path: Input MIDI file
        output_path: Output path (default: input_reaper.mid)

    Returns:
        Path to prepared file
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido package required")

    midi_safe = safe_path(midi_path)
    mid = mido.MidiFile(str(midi_safe))

    if output_path is None:
        output_path = f"{midi_safe.to_path().stem}_reaper.mid"

    # If PPQ matches, just copy
    if mid.ticks_per_beat == REAPER_PPQ:
        mid.save(output_path)
        return output_path

    # Create new MIDI with Reaper PPQ
    new_mid = mido.MidiFile(ticks_per_beat=REAPER_PPQ)
    ppq_ratio = REAPER_PPQ / mid.ticks_per_beat

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


def import_from_reaper(midi_path: str) -> ReaperProject:
    """
    Import a MIDI file exported from Reaper.

    Args:
        midi_path: Path to MIDI file from Reaper

    Returns:
        ReaperProject with imported data
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido package required")

    midi_safe = safe_path(midi_path)
    mid = mido.MidiFile(str(midi_safe))

    # Detect tempo and time signature
    tempo_bpm = 120.0
    time_sig = (4, 4)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_bpm = 60_000_000 / msg.tempo
            elif msg.type == 'time_signature':
                time_sig = (msg.numerator, msg.denominator)

    project = ReaperProject(
        name=midi_path.stem,
        tempo_bpm=tempo_bpm,
        ppq=mid.ticks_per_beat,
        time_signature=time_sig,
    )

    # Import markers
    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            if msg.type == 'marker':
                position = current_tick / project.ppq
                project.add_marker(position, msg.text)

    # Convert each track
    for i, track in enumerate(mid.tracks):
        track_name = f"Track {i+1}"
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
                        "channel": msg.channel,
                    })

        if notes:
            reaper_track = ReaperTrack(
                name=track_name,
                track_type=ReaperTrackType.MIDI,
                notes=notes,
            )
            project.add_track(reaper_track)

    return project


def create_reaper_template(
    name: str,
    tempo: float = 120.0,
    style: str = "music_production",
) -> ReaperProject:
    """
    Create a Reaper project template.

    Args:
        name: Project name
        tempo: Tempo in BPM
        style: Production style

    Returns:
        ReaperProject template
    """
    project = ReaperProject(
        name=name,
        tempo_bpm=tempo,
    )

    # Template configurations
    templates = {
        "music_production": [
            ("Drums", ReaperTrackType.MIDI),
            ("Bass", ReaperTrackType.MIDI),
            ("Keys", ReaperTrackType.MIDI),
            ("Guitar", ReaperTrackType.AUDIO),
            ("Lead Vocal", ReaperTrackType.AUDIO),
            ("Backing Vocals", ReaperTrackType.FOLDER),
            ("BGV 1", ReaperTrackType.AUDIO),
            ("BGV 2", ReaperTrackType.AUDIO),
        ],
        "podcast": [
            ("Host", ReaperTrackType.AUDIO),
            ("Guest", ReaperTrackType.AUDIO),
            ("Music Bed", ReaperTrackType.AUDIO),
            ("SFX", ReaperTrackType.AUDIO),
        ],
        "sound_design": [
            ("Source Audio", ReaperTrackType.AUDIO),
            ("Processing 1", ReaperTrackType.AUDIO),
            ("Processing 2", ReaperTrackType.AUDIO),
            ("Render", ReaperTrackType.AUDIO),
        ],
    }

    track_config = templates.get(style, templates["music_production"])

    for track_name, track_type in track_config:
        track = ReaperTrack(
            name=track_name,
            track_type=track_type,
        )
        project.add_track(track)

    return project


# OSC Control for Reaper
class ReaperOSC:
    """
    OSC controller for Reaper.

    Reaper has excellent OSC support for remote control.
    Enable OSC in Reaper: Preferences > Control/OSC/Web
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        send_port: int = REAPER_OSC_SEND_PORT,
        listen_port: int = REAPER_OSC_LISTEN_PORT,
    ):
        """
        Initialize OSC controller.

        Args:
            host: Reaper host address
            send_port: Port to send OSC messages to Reaper
            listen_port: Port to receive OSC messages from Reaper
        """
        self.host = host
        self.send_port = send_port
        self.listen_port = listen_port
        self._socket: Optional[socket.socket] = None

    def connect(self) -> bool:
        """Establish UDP socket for OSC communication."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        """Close the OSC socket."""
        if self._socket:
            self._socket.close()
            self._socket = None

    def _send_osc(self, address: str, *args) -> bool:
        """
        Send an OSC message to Reaper.

        Args:
            address: OSC address (e.g., "/action", "/track/1/volume")
            *args: OSC arguments

        Returns:
            True if sent successfully
        """
        if not self._socket:
            return False

        try:
            # Simple OSC message encoding
            # Note: For production, use python-osc or oscpy
            msg = self._encode_osc_message(address, args)
            self._socket.sendto(msg, (self.host, self.send_port))
            return True
        except Exception:
            return False

    def _encode_osc_message(self, address: str, args: tuple) -> bytes:
        """Encode an OSC message (simplified)."""
        # Pad address to 4-byte boundary
        address_bytes = address.encode('utf-8') + b'\x00'
        while len(address_bytes) % 4 != 0:
            address_bytes += b'\x00'

        # Type tag string
        type_tags = ','
        for arg in args:
            if isinstance(arg, int):
                type_tags += 'i'
            elif isinstance(arg, float):
                type_tags += 'f'
            elif isinstance(arg, str):
                type_tags += 's'

        type_tag_bytes = type_tags.encode('utf-8') + b'\x00'
        while len(type_tag_bytes) % 4 != 0:
            type_tag_bytes += b'\x00'

        # Encode arguments
        arg_bytes = b''
        for arg in args:
            if isinstance(arg, int):
                arg_bytes += struct.pack('>i', arg)
            elif isinstance(arg, float):
                arg_bytes += struct.pack('>f', arg)
            elif isinstance(arg, str):
                arg_str = arg.encode('utf-8') + b'\x00'
                while len(arg_str) % 4 != 0:
                    arg_str += b'\x00'
                arg_bytes += arg_str

        return address_bytes + type_tag_bytes + arg_bytes

    # Transport controls
    def play(self) -> bool:
        """Start playback."""
        return self._send_osc("/action", ReaperAction.PLAY.value)

    def stop(self) -> bool:
        """Stop playback."""
        return self._send_osc("/action", ReaperAction.STOP.value)

    def pause(self) -> bool:
        """Pause playback."""
        return self._send_osc("/action", ReaperAction.PAUSE.value)

    def record(self) -> bool:
        """Start recording."""
        return self._send_osc("/action", ReaperAction.RECORD.value)

    def goto_start(self) -> bool:
        """Go to project start."""
        return self._send_osc("/action", ReaperAction.GOTO_START.value)

    def toggle_loop(self) -> bool:
        """Toggle loop mode."""
        return self._send_osc("/action", ReaperAction.TOGGLE_LOOP.value)

    def toggle_metronome(self) -> bool:
        """Toggle metronome."""
        return self._send_osc("/action", ReaperAction.TOGGLE_METRONOME.value)

    # Track controls
    def set_track_volume(self, track_index: int, volume_db: float) -> bool:
        """Set track volume in dB."""
        # Reaper uses 0-1 for volume, convert from dB
        vol_linear = 10 ** (volume_db / 20)
        return self._send_osc(f"/track/{track_index}/volume", vol_linear)

    def set_track_pan(self, track_index: int, pan: float) -> bool:
        """Set track pan (-1 to 1)."""
        return self._send_osc(f"/track/{track_index}/pan", pan)

    def set_track_mute(self, track_index: int, mute: bool) -> bool:
        """Set track mute state."""
        return self._send_osc(f"/track/{track_index}/mute", 1 if mute else 0)

    def set_track_solo(self, track_index: int, solo: bool) -> bool:
        """Set track solo state."""
        return self._send_osc(f"/track/{track_index}/solo", 1 if solo else 0)

    # Project controls
    def set_tempo(self, bpm: float) -> bool:
        """Set project tempo."""
        return self._send_osc("/tempo/raw", bpm)

    def set_time_position(self, seconds: float) -> bool:
        """Set playhead position in seconds."""
        return self._send_osc("/time", seconds)

    def set_beat_position(self, beats: float) -> bool:
        """Set playhead position in beats."""
        return self._send_osc("/beat", beats)

    def trigger_action(self, action_id: int) -> bool:
        """Trigger a Reaper action by ID."""
        return self._send_osc("/action", action_id)


# ReaScript integration
def generate_reascript_lua(
    script_name: str,
    actions: List[Dict[str, Any]],
) -> str:
    """
    Generate a ReaScript Lua script.

    Args:
        script_name: Name of the script
        actions: List of action definitions

    Returns:
        Lua script content
    """
    lines = [
        f"-- {script_name}",
        "-- Generated by iDAW",
        "",
        "function main()",
    ]

    for action in actions:
        action_type = action.get("type", "")

        if action_type == "insert_track":
            lines.append("  reaper.InsertTrackAtIndex(0, true)")

        elif action_type == "set_track_name":
            track_idx = action.get("track", 0)
            name = action.get("name", "Track")
            lines.append(f'  local track = reaper.GetTrack(0, {track_idx})')
            lines.append(f'  reaper.GetSetMediaTrackInfo_String(track, "P_NAME", "{name}", true)')

        elif action_type == "insert_midi_note":
            track_idx = action.get("track", 0)
            pitch = action.get("pitch", 60)
            start = action.get("start", 0)
            end = action.get("end", 1)
            velocity = action.get("velocity", 100)
            lines.extend([
                f'  local track = reaper.GetTrack(0, {track_idx})',
                '  local item = reaper.GetTrackMediaItem(track, 0)',
                '  if item then',
                '    local take = reaper.GetActiveTake(item)',
                '    if take and reaper.TakeIsMIDI(take) then',
                f'      reaper.MIDI_InsertNote(take, false, false, {start}, {end}, 0, {pitch}, {velocity}, false)',
                '    end',
                '  end',
            ])

        elif action_type == "set_tempo":
            bpm = action.get("bpm", 120)
            lines.append(f'  reaper.SetCurrentBPM(0, {bpm}, true)')

        elif action_type == "run_action":
            action_id = action.get("id", 0)
            lines.append(f'  reaper.Main_OnCommand({action_id}, 0)')

    lines.extend([
        "end",
        "",
        "main()",
    ])

    return '\n'.join(lines)


def get_reaper_plugin_info() -> Dict[str, Any]:
    """
    Get plugin format information for Reaper.

    Returns:
        Dict with Reaper plugin requirements
    """
    return {
        "supported_formats": ["VST2", "VST3", "AU", "LV2", "CLAP", "JS"],
        "recommended_format": "VST3",
        "recommended_ppq": REAPER_PPQ,
        "supported_sample_rates": [44100, 48000, 88200, 96000, 176400, 192000],
        "plugin_scan_paths": {
            "windows": [
                "C:\\Program Files\\VSTPlugins",
                "C:\\Program Files\\Common Files\\VST3",
                "C:\\Program Files\\CLAP",
            ],
            "macos": [
                "/Library/Audio/Plug-Ins/VST",
                "/Library/Audio/Plug-Ins/VST3",
                "/Library/Audio/Plug-Ins/Components",
                "/Library/Audio/Plug-Ins/CLAP",
            ],
            "linux": [
                "~/.vst",
                "~/.vst3",
                "/usr/lib/vst",
                "/usr/lib/vst3",
                "/usr/lib/lv2",
                "~/.clap",
            ],
        },
        "osc_support": True,
        "reascript_support": ["Lua", "EEL2", "Python"],
        "extensions": [
            "SWS/S&M Extension",
            "ReaPack",
            "js_ReaScriptAPI",
        ],
    }
