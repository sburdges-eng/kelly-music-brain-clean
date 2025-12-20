"""
Utilities - MIDI I/O, instrument mappings, PPQ handling, and cross-platform paths.

Common utilities for working with MIDI data across DAWs and cross-platform file operations.
"""

from music_brain.utils.midi_io import load_midi, save_midi, get_midi_info
from music_brain.utils.instruments import (
    GM_DRUMS,
    GM_INSTRUMENTS,
    get_instrument_name,
    get_drum_name,
    is_drum_channel,
)
from music_brain.utils.ppq import (
    normalize_ppq,
    scale_ticks,
    ticks_to_beats,
    beats_to_ticks,
)
from music_brain.utils.path_utils import (
    CrossPlatformPath,
    safe_path,
    safe_filename,
    ensure_path_exists,
    get_home_dir,
    get_temp_dir,
    PathError,
)

__all__ = [
    # MIDI I/O
    "load_midi",
    "save_midi",
    "get_midi_info",
    # Instruments
    "GM_DRUMS",
    "GM_INSTRUMENTS",
    "get_instrument_name",
    "get_drum_name",
    "is_drum_channel",
    # PPQ
    "normalize_ppq",
    "scale_ticks",
    "ticks_to_beats",
    "beats_to_ticks",
    # Cross-platform paths (P0-004)
    "CrossPlatformPath",
    "safe_path",
    "safe_filename",
    "ensure_path_exists",
    "get_home_dir",
    "get_temp_dir",
    "PathError",
]
