"""
Pro Tools Integration - Utilities for working with Pro Tools projects.

Pro Tools uses .ptx (Pro Tools session) files which are proprietary.
This module provides utilities for MIDI exchange and AAX plugin support.

Pro Tools Specifics:
- Default PPQ: 960 (high resolution)
- Industry standard for professional studios
- AAX format required for plugins
- Supports MIDI and audio simultaneously
- Session-based workflow with strict timeline
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import json

from music_brain.utils.path_utils import safe_path

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


# Pro Tools PPQ constant
PRO_TOOLS_PPQ = 960

# Pro Tools track types
class PTTrackType(Enum):
    """Pro Tools track types."""
    AUDIO = "audio"
    MIDI = "midi"
    INSTRUMENT = "instrument"
    AUX = "aux"
    MASTER = "master"
    VCA = "vca"


# Standard Pro Tools I/O configurations
PT_IO_CONFIGS = {
    "stereo": {
        "inputs": [("In 1-2", 2)],
        "outputs": [("Out 1-2", 2)],
        "buses": [("Bus 1-2", 2), ("Reverb", 2), ("Delay", 2)],
    },
    "surround_51": {
        "inputs": [("In 1-2", 2), ("In 3-4", 2), ("In 5-6", 2)],
        "outputs": [("5.1 Out", 6)],
        "buses": [("Film Mix", 6), ("Music Stem", 6), ("Dialog Stem", 2)],
    },
    "dolby_atmos": {
        "inputs": [("In 1-2", 2)],
        "outputs": [("Atmos 7.1.4", 12)],
        "buses": [("Objects", 128)],
    },
}


@dataclass
class PTTrack:
    """
    Represents a Pro Tools track.

    Pro Tools tracks have extensive routing options
    and support for high-resolution timing.
    """
    name: str
    track_type: PTTrackType = PTTrackType.MIDI
    channel: int = 1

    # Routing
    input_bus: str = "In 1-2"
    output_bus: str = "Out 1-2"

    # Track state
    armed: bool = False
    solo: bool = False
    mute: bool = False
    pan: float = 0.0  # -1.0 to 1.0
    volume_db: float = 0.0  # dB

    # MIDI data
    notes: List[Dict] = field(default_factory=list)
    clips: List[Dict] = field(default_factory=list)

    # Plugins (AAX)
    inserts: List[str] = field(default_factory=list)
    sends: List[Tuple[str, float]] = field(default_factory=list)

    def add_note(
        self,
        pitch: int,
        velocity: int = 100,
        start_tick: int = 0,
        duration_ticks: int = 960,
    ) -> None:
        """Add a MIDI note to the track."""
        self.notes.append({
            "pitch": pitch,
            "velocity": velocity,
            "start_tick": start_tick,
            "duration_ticks": duration_ticks,
        })

    def add_clip(
        self,
        name: str,
        start_tick: int,
        end_tick: int,
        offset: int = 0,
    ) -> None:
        """Add a clip reference to the track."""
        self.clips.append({
            "name": name,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "offset": offset,
        })


@dataclass
class PTSession:
    """
    Represents a Pro Tools session for MIDI/project export.

    Not a full Pro Tools session parser - provides structure
    for organizing tracks for MIDI export to Pro Tools.
    """
    name: str = "Untitled Session"
    tempo_bpm: float = 120.0
    ppq: int = PRO_TOOLS_PPQ
    time_signature: Tuple[int, int] = (4, 4)
    sample_rate: int = 48000

    # Session configuration
    bit_depth: int = 24
    io_config: str = "stereo"

    # Tracks
    tracks: List[PTTrack] = field(default_factory=list)

    # Tempo/meter events
    tempo_map: List[Tuple[int, float]] = field(default_factory=list)
    meter_map: List[Tuple[int, int, int]] = field(default_factory=list)

    # Session metadata
    artist: str = ""
    album: str = ""
    session_notes: str = ""

    def add_track(self, track: PTTrack) -> int:
        """
        Add a track to the session.

        Returns:
            Track index
        """
        self.tracks.append(track)
        return len(self.tracks) - 1

    def add_tempo_change(self, tick: int, bpm: float) -> None:
        """Add a tempo change at the specified tick position."""
        self.tempo_map.append((tick, bpm))
        self.tempo_map.sort(key=lambda x: x[0])

    def add_meter_change(
        self,
        tick: int,
        numerator: int,
        denominator: int,
    ) -> None:
        """Add a meter (time signature) change."""
        self.meter_map.append((tick, numerator, denominator))
        self.meter_map.sort(key=lambda x: x[0])

    def export_midi(self, output_path: str) -> str:
        """
        Export session MIDI tracks to a MIDI file.

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

        # Add tempo changes
        current_tick = 0
        for tick, bpm in self.tempo_map:
            delta = tick - current_tick
            tempo_us = int(60_000_000 / bpm)
            meta_track.append(mido.MetaMessage(
                'set_tempo',
                tempo=tempo_us,
                time=delta
            ))
            current_tick = tick

        # Export MIDI tracks
        for track in self.tracks:
            if track.track_type in [PTTrackType.MIDI, PTTrackType.INSTRUMENT]:
                if not track.notes:
                    continue

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
                    ))
                    events.append((
                        note["start_tick"] + note["duration_ticks"],
                        "note_off",
                        note["pitch"],
                        0,
                    ))

                events.sort(key=lambda e: e[0])

                # Convert to delta times
                current_tick = 0
                for tick, msg_type, pitch, vel in events:
                    delta = tick - current_tick
                    current_tick = tick

                    if msg_type == "note_on":
                        midi_track.append(mido.Message(
                            'note_on',
                            note=pitch,
                            velocity=vel,
                            channel=track.channel - 1,
                            time=delta
                        ))
                    else:
                        midi_track.append(mido.Message(
                            'note_off',
                            note=pitch,
                            velocity=0,
                            channel=track.channel - 1,
                            time=delta
                        ))

                # End of track
                midi_track.append(mido.MetaMessage('end_of_track', time=0))

        output_safe = safe_path(output_path)
        mid.save(str(output_safe))

        return str(output_safe)


def export_to_pro_tools(
    midi_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Prepare a MIDI file for optimal Pro Tools import.

    - Normalizes PPQ to Pro Tools' 960
    - Ensures proper timing resolution

    Args:
        midi_path: Input MIDI file
        output_path: Output path (default: input_pt.mid)

    Returns:
        Path to prepared file
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido package required")

    midi_safe = safe_path(midi_path)
    mid = mido.MidiFile(str(midi_safe))

    if output_path is None:
        output_path = f"{midi_safe.to_path().stem}_pt.mid"

    # If PPQ matches, just copy
    if mid.ticks_per_beat == PRO_TOOLS_PPQ:
        mid.save(output_path)
        return output_path

    # Create new MIDI with Pro Tools PPQ
    new_mid = mido.MidiFile(ticks_per_beat=PRO_TOOLS_PPQ)
    ppq_ratio = PRO_TOOLS_PPQ / mid.ticks_per_beat

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


def import_from_pro_tools(midi_path: str) -> PTSession:
    """
    Import a MIDI file exported from Pro Tools.

    Args:
        midi_path: Path to MIDI file from Pro Tools

    Returns:
        PTSession with imported data
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido package required")

    midi_path = Path(midi_path)
    mid = mido.MidiFile(str(midi_path))

    # Detect tempo
    tempo_bpm = 120.0
    time_sig = (4, 4)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_bpm = 60_000_000 / msg.tempo
            elif msg.type == 'time_signature':
                time_sig = (msg.numerator, msg.denominator)

    session = PTSession(
        name=midi_path.stem,
        tempo_bpm=tempo_bpm,
        ppq=mid.ticks_per_beat,
        time_signature=time_sig,
    )

    # Convert each track
    for i, track in enumerate(mid.tracks):
        track_name = f"Track {i+1}"
        notes = []
        current_tick = 0
        active_notes = {}
        channel = 0

        for msg in track:
            current_tick += msg.time

            if msg.type == 'track_name':
                track_name = msg.name
            elif msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active_notes[key] = (current_tick, msg.velocity)
                channel = msg.channel
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes:
                    start_tick, velocity = active_notes.pop(key)
                    notes.append({
                        "pitch": msg.note,
                        "velocity": velocity,
                        "start_tick": start_tick,
                        "duration_ticks": current_tick - start_tick,
                    })

        if notes:
            pt_track = PTTrack(
                name=track_name,
                track_type=PTTrackType.MIDI,
                channel=channel + 1,
                notes=notes,
            )
            session.add_track(pt_track)

    return session


def create_pt_template(
    name: str,
    tempo: float = 120.0,
    sample_rate: int = 48000,
    style: str = "music_production",
) -> PTSession:
    """
    Create a Pro Tools session template.

    Args:
        name: Session name
        tempo: Tempo in BPM
        sample_rate: Sample rate (44100, 48000, 96000)
        style: Production style

    Returns:
        PTSession template
    """
    session = PTSession(
        name=name,
        tempo_bpm=tempo,
        sample_rate=sample_rate,
    )

    # Template configurations
    templates = {
        "music_production": [
            ("Drums", PTTrackType.INSTRUMENT),
            ("Bass", PTTrackType.INSTRUMENT),
            ("Keys", PTTrackType.INSTRUMENT),
            ("Guitar", PTTrackType.INSTRUMENT),
            ("Lead Vocal", PTTrackType.AUDIO),
            ("BGV 1", PTTrackType.AUDIO),
            ("BGV 2", PTTrackType.AUDIO),
            ("Reverb", PTTrackType.AUX),
            ("Delay", PTTrackType.AUX),
            ("Mix Bus", PTTrackType.MASTER),
        ],
        "film_scoring": [
            ("Strings", PTTrackType.INSTRUMENT),
            ("Brass", PTTrackType.INSTRUMENT),
            ("Woodwinds", PTTrackType.INSTRUMENT),
            ("Percussion", PTTrackType.INSTRUMENT),
            ("Piano", PTTrackType.INSTRUMENT),
            ("Choir", PTTrackType.INSTRUMENT),
            ("Synth", PTTrackType.INSTRUMENT),
            ("Perc Aux", PTTrackType.AUX),
            ("String Verb", PTTrackType.AUX),
            ("Orchestral Mix", PTTrackType.MASTER),
        ],
        "podcast": [
            ("Host", PTTrackType.AUDIO),
            ("Guest 1", PTTrackType.AUDIO),
            ("Guest 2", PTTrackType.AUDIO),
            ("Music Bed", PTTrackType.AUDIO),
            ("SFX", PTTrackType.AUDIO),
            ("Vocal Processing", PTTrackType.AUX),
            ("Master", PTTrackType.MASTER),
        ],
    }

    track_config = templates.get(style, templates["music_production"])

    for i, (track_name, track_type) in enumerate(track_config):
        track = PTTrack(
            name=track_name,
            track_type=track_type,
            channel=(i % 16) + 1,
        )
        session.add_track(track)

    return session


# AAX Plugin Support
@dataclass
class AAXPluginConfig:
    """Configuration for AAX plugin compatibility."""
    plugin_name: str
    manufacturer: str = "iDAW"
    version: str = "1.0.0"
    plugin_id: str = ""

    # AAX-specific settings
    supports_mono: bool = True
    supports_stereo: bool = True
    supports_surround: bool = False
    supports_atmos: bool = False

    # DSP support (for HDX systems)
    aax_native: bool = True
    aax_dsp: bool = False

    # MIDI capabilities
    midi_input: bool = True
    midi_output: bool = False

    # GUI settings
    resizable_ui: bool = True
    default_width: int = 800
    default_height: int = 600


def get_aax_plugin_info() -> Dict[str, Any]:
    """
    Get AAX plugin format requirements for Pro Tools.

    Returns:
        Dict with AAX format requirements
    """
    return {
        "format": "AAX",
        "required_sdk": "AAX SDK 2.4.0+",
        "signing_required": True,
        "ilok_required": "Yes (for distribution)",
        "supported_sample_rates": [44100, 48000, 88200, 96000, 176400, 192000],
        "recommended_ppq": PRO_TOOLS_PPQ,
        "plugin_scan_paths": [
            "C:\\Program Files\\Common Files\\Avid\\Audio\\Plug-Ins",  # Windows
            "/Library/Application Support/Avid/Audio/Plug-Ins",  # macOS
        ],
        "pro_tools_versions": [
            "Pro Tools 2021",
            "Pro Tools 2022",
            "Pro Tools 2023",
            "Pro Tools 2024",
        ],
        "requirements": {
            "developer_agreement": True,
            "code_signing": True,
            "notarization_macos": True,
            "pace_ilok": "Required for copy protection",
        },
        "formats": {
            "AAX Native": "CPU-based processing",
            "AAX DSP": "HDX hardware processing",
            "AAX AudioSuite": "Offline processing",
        },
    }


def create_aax_manifest(config: AAXPluginConfig) -> Dict[str, Any]:
    """
    Create an AAX plugin manifest for Pro Tools.

    Args:
        config: AAX plugin configuration

    Returns:
        Dict representing the manifest
    """
    manifest = {
        "name": config.plugin_name,
        "manufacturer": config.manufacturer,
        "version": config.version,
        "plugin_id": config.plugin_id or f"com.{config.manufacturer.lower()}.{config.plugin_name.lower().replace(' ', '')}",
        "category": "Effect",
        "audio_configs": [],
        "features": {
            "midi_input": config.midi_input,
            "midi_output": config.midi_output,
            "sidechain": True,
            "automation": True,
        },
        "processing": {
            "aax_native": config.aax_native,
            "aax_dsp": config.aax_dsp,
        },
        "ui": {
            "resizable": config.resizable_ui,
            "default_size": [config.default_width, config.default_height],
        },
    }

    # Add audio configurations
    if config.supports_mono:
        manifest["audio_configs"].append({"in": 1, "out": 1, "name": "Mono"})
    if config.supports_stereo:
        manifest["audio_configs"].append({"in": 2, "out": 2, "name": "Stereo"})
    if config.supports_surround:
        manifest["audio_configs"].append({"in": 6, "out": 6, "name": "5.1"})
    if config.supports_atmos:
        manifest["audio_configs"].append({"in": 12, "out": 12, "name": "7.1.4"})

    return manifest
