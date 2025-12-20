"""
Bass Line Generator - Create bass lines from chord progressions.

Generates bass lines with:
- Root note patterns
- Walking bass
- Rhythmic pocket synchronization
- Genre-specific patterns
"""

from dataclasses import dataclass
from enum import Enum

try:
    import mido

    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


class BassPattern(Enum):
    """Bass line pattern types."""

    ROOT_ONLY = "root_only"  # Just root notes
    ROOT_FIFTH = "root_fifth"  # Root and fifth
    WALKING = "walking"  # Walking bass (chromatic approach)
    PEDAL = "pedal"  # Pedal tone (sustained root)
    OCTAVE_JUMP = "octave_jump"  # Root with octave jumps
    ARPEGGIO = "arpeggio"  # Chord tone arpeggiation
    SYNCOPATED = "syncopated"  # Off-beat rhythms
    FUNK = "funk"  # Funk-style slap bass patterns


@dataclass
class BassNote:
    """Single bass note."""

    pitch: int  # MIDI note number
    start_tick: int
    duration_ticks: int
    velocity: int = 80

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pitch": self.pitch,
            "start_tick": self.start_tick,
            "duration_ticks": self.duration_ticks,
            "velocity": self.velocity,
        }


@dataclass
class BassLine:
    """Complete bass line."""

    notes: list[BassNote]
    pattern: BassPattern
    octave: int = 2  # Bass typically in octave 2-3

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "notes": [n.to_dict() for n in self.notes],
            "pattern": self.pattern.value,
            "octave": self.octave,
        }


# =================================================================
# CHORD PARSING
# =================================================================


def parse_chord_root(chord_name: str) -> int:
    """
    Parse chord name to get MIDI root note.

    Args:
        chord_name: Chord name (e.g., "C", "F#m", "Bbmaj7")

    Returns:
        MIDI note number of root in octave 2
    """
    # Note name to pitch class
    note_map = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }

    # Handle enharmonics
    chord_name = chord_name.replace("Db", "C#").replace("Eb", "D#")
    chord_name = chord_name.replace("Gb", "F#").replace("Ab", "G#")
    chord_name = chord_name.replace("Bb", "A#")

    # Extract root note
    if len(chord_name) >= 2 and chord_name[1] in ["#", "b"]:
        root = chord_name[0]
        modifier = chord_name[1]
    else:
        root = chord_name[0]
        modifier = None

    if root not in note_map:
        # Default to C if parsing fails
        return 36  # C2

    pitch_class = note_map[root]

    if modifier == "#":
        pitch_class += 1
    elif modifier == "b":
        pitch_class -= 1

    # Map to octave 2 (standard bass range)
    midi_note = 24 + (pitch_class % 12)  # C2 = 36

    return midi_note


def get_chord_tones(chord_name: str) -> list[int]:
    """
    Get chord tones from chord name.

    Returns list of MIDI notes (relative to root).
    """
    root = parse_chord_root(chord_name)

    # Determine quality from chord name
    name_lower = chord_name.lower()

    if "dim" in name_lower:
        intervals = [0, 3, 6]  # Diminished
    elif "m7b5" in name_lower:
        intervals = [0, 3, 6, 10]  # Half-diminished
    elif "aug" in name_lower:
        intervals = [0, 4, 8]  # Augmented
    elif "maj7" in name_lower:
        intervals = [0, 4, 7, 11]  # Major 7th
    elif "m7" in name_lower or "min7" in name_lower:
        intervals = [0, 3, 7, 10]  # Minor 7th
    elif "7" in name_lower:
        intervals = [0, 4, 7, 10]  # Dominant 7th
    elif "m" in name_lower or "min" in name_lower:
        intervals = [0, 3, 7]  # Minor triad
    else:
        intervals = [0, 4, 7]  # Major triad (default)

    # Convert intervals to MIDI notes
    return [root + interval for interval in intervals]


# =================================================================
# PATTERN GENERATORS
# =================================================================


def generate_root_only(
    chord_name: str,
    bars: int,
    ppq: int,
    time_sig: tuple[int, int] = (4, 4),
) -> list[BassNote]:
    """Generate root-note-only bass line."""
    root = parse_chord_root(chord_name)
    notes = []

    ticks_per_bar = ppq * time_sig[0]
    beats_per_bar = time_sig[0]

    # Place root on each beat
    for bar in range(bars):
        for beat in range(beats_per_bar):
            start = bar * ticks_per_bar + beat * ppq
            notes.append(
                BassNote(
                    pitch=root,
                    start_tick=start,
                    duration_ticks=ppq,  # Quarter note
                    velocity=85,
                )
            )

    return notes


def generate_root_fifth(
    chord_name: str,
    bars: int,
    ppq: int,
    time_sig: tuple[int, int] = (4, 4),
) -> list[BassNote]:
    """Generate root-and-fifth bass pattern."""
    root = parse_chord_root(chord_name)
    fifth = root + 7
    notes = []

    ticks_per_bar = ppq * time_sig[0]
    beats_per_bar = time_sig[0]

    for bar in range(bars):
        for beat in range(beats_per_bar):
            start = bar * ticks_per_bar + beat * ppq
            # Alternate root and fifth
            pitch = root if beat % 2 == 0 else fifth
            notes.append(
                BassNote(
                    pitch=pitch,
                    start_tick=start,
                    duration_ticks=ppq,
                    velocity=80,
                )
            )

    return notes


def generate_walking_bass(
    chord_names: list[str],
    ppq: int,
    time_sig: tuple[int, int] = (4, 4),
) -> list[BassNote]:
    """
    Generate walking bass line through chord progression.

    Uses chromatic approach notes between chord tones.
    """
    notes = []
    ticks_per_bar = ppq * time_sig[0]
    beats_per_bar = time_sig[0]

    for bar_idx, chord_name in enumerate(chord_names):
        chord_tones = get_chord_tones(chord_name)
        root = chord_tones[0]

        # Generate walking pattern for this bar
        for beat in range(beats_per_bar):
            start = bar_idx * ticks_per_bar + beat * ppq

            if beat == 0:
                # Start with root
                pitch = root
            elif beat == beats_per_bar - 1 and bar_idx < len(chord_names) - 1:
                # Last beat: approach next chord chromatically
                next_root = parse_chord_root(chord_names[bar_idx + 1])
                # Approach from half-step below
                pitch = next_root - 1
            else:
                # Use chord tones
                pitch = chord_tones[beat % len(chord_tones)]

            notes.append(
                BassNote(
                    pitch=pitch,
                    start_tick=start,
                    duration_ticks=ppq,
                    velocity=75,
                )
            )

    return notes


def generate_pedal_tone(
    chord_names: list[str],
    ppq: int,
    time_sig: tuple[int, int] = (4, 4),
) -> list[BassNote]:
    """Generate pedal tone bass (sustained root throughout)."""
    # Use root of first chord as pedal
    root = parse_chord_root(chord_names[0])
    notes = []

    ticks_per_bar = ppq * time_sig[0]
    total_duration = len(chord_names) * ticks_per_bar

    # Single long note
    notes.append(
        BassNote(
            pitch=root,
            start_tick=0,
            duration_ticks=total_duration,
            velocity=70,
        )
    )

    return notes


def generate_funk_bass(
    chord_name: str,
    bars: int,
    ppq: int,
    time_sig: tuple[int, int] = (4, 4),
) -> list[BassNote]:
    """Generate funk-style syncopated bass pattern."""
    root = parse_chord_root(chord_name)
    fifth = root + 7
    notes = []

    ticks_per_bar = ppq * time_sig[0]
    sixteenth = ppq // 4

    # Funk pattern: emphasis on 1, and of 2, and of 4
    for bar in range(bars):
        bar_start = bar * ticks_per_bar

        # Beat 1 (downbeat)
        notes.append(
            BassNote(
                pitch=root,
                start_tick=bar_start,
                duration_ticks=sixteenth * 2,
                velocity=95,
            )
        )

        # And of 2
        notes.append(
            BassNote(
                pitch=fifth,
                start_tick=bar_start + ppq * 2 + sixteenth * 2,
                duration_ticks=sixteenth,
                velocity=75,
            )
        )

        # Beat 4
        notes.append(
            BassNote(
                pitch=root,
                start_tick=bar_start + ppq * 3,
                duration_ticks=sixteenth,
                velocity=85,
            )
        )

        # And of 4
        notes.append(
            BassNote(
                pitch=fifth,
                start_tick=bar_start + ppq * 3 + sixteenth * 2,
                duration_ticks=sixteenth,
                velocity=80,
            )
        )

    return notes


# =================================================================
# MAIN GENERATOR
# =================================================================


def generate_bass_line(
    chord_progression: list[str],
    pattern: BassPattern = BassPattern.ROOT_FIFTH,
    ppq: int = 480,
    time_signature: tuple[int, int] = (4, 4),
    octave: int = 2,
) -> BassLine:
    """
    Generate bass line from chord progression.

    Args:
        chord_progression: List of chord names
        pattern: Bass pattern type
        ppq: Pulses per quarter note
        time_signature: Time signature (numerator, denominator)
        octave: Bass octave (default 2)

    Returns:
        BassLine with generated notes
    """
    if not chord_progression:
        return BassLine(notes=[], pattern=pattern, octave=octave)

    # Generate notes based on pattern
    if pattern == BassPattern.ROOT_ONLY:
        notes = []
        for chord in chord_progression:
            notes.extend(generate_root_only(chord, 1, ppq, time_signature))

    elif pattern == BassPattern.ROOT_FIFTH:
        notes = []
        for chord in chord_progression:
            notes.extend(generate_root_fifth(chord, 1, ppq, time_signature))

    elif pattern == BassPattern.WALKING:
        notes = generate_walking_bass(chord_progression, ppq, time_signature)

    elif pattern == BassPattern.PEDAL:
        notes = generate_pedal_tone(chord_progression, ppq, time_signature)

    elif pattern == BassPattern.FUNK:
        notes = []
        for chord in chord_progression:
            notes.extend(generate_funk_bass(chord, 1, ppq, time_signature))

    else:
        # Default to root-fifth
        notes = []
        for chord in chord_progression:
            notes.extend(generate_root_fifth(chord, 1, ppq, time_signature))

    # Adjust octave if needed
    if octave != 2:
        octave_shift = (octave - 2) * 12
        for note in notes:
            note.pitch += octave_shift

    return BassLine(notes=notes, pattern=pattern, octave=octave)


def bass_line_to_midi(
    bass_line: BassLine,
    output_path: str,
    ppq: int = 480,
    tempo_bpm: float = 120.0,
) -> None:
    """
    Export bass line to MIDI file.

    Args:
        bass_line: BassLine to export
        output_path: Output MIDI file path
        ppq: Pulses per quarter note
        tempo_bpm: Tempo in BPM
    """
    if not MIDO_AVAILABLE:
        raise ImportError(
            "mido required for MIDI export. Install with: pip install mido"
        )

    mid = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Add tempo
    tempo_microseconds = int(60_000_000 / tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_microseconds, time=0))

    # Add track name
    track.append(mido.MetaMessage("track_name", name="Bass", time=0))

    # Convert notes to MIDI messages
    events = []
    for note in bass_line.notes:
        events.append((note.start_tick, "note_on", note.pitch, note.velocity))
        events.append(
            (note.start_tick + note.duration_ticks, "note_off", note.pitch, 0)
        )

    # Sort by time
    events.sort(key=lambda x: x[0])

    # Convert to delta times
    current_tick = 0
    for tick, msg_type, pitch, velocity in events:
        delta = tick - current_tick

        if msg_type == "note_on":
            track.append(
                mido.Message("note_on", note=pitch, velocity=velocity, time=delta)
            )
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))

        current_tick = tick

    # Save file
    mid.save(output_path)


def suggest_bass_pattern(genre: str, energy_level: float = 0.5) -> BassPattern:
    """
    Suggest bass pattern for genre and energy level.

    Args:
        genre: Musical genre
        energy_level: Energy/intensity (0.0-1.0)

    Returns:
        Recommended bass pattern
    """
    genre_lower = genre.lower()

    if "funk" in genre_lower or "disco" in genre_lower:
        return BassPattern.FUNK

    elif "jazz" in genre_lower or "swing" in genre_lower:
        return BassPattern.WALKING

    elif "rock" in genre_lower or "punk" in genre_lower:
        if energy_level > 0.7:
            return BassPattern.ROOT_ONLY
        else:
            return BassPattern.ROOT_FIFTH

    elif "edm" in genre_lower or "electronic" in genre_lower:
        if energy_level > 0.7:
            return BassPattern.ROOT_ONLY
        else:
            return BassPattern.SYNCOPATED

    elif "ambient" in genre_lower or "drone" in genre_lower:
        return BassPattern.PEDAL

    # Default based on energy
    elif energy_level > 0.7:
        return BassPattern.ROOT_ONLY
    elif energy_level < 0.3:
        return BassPattern.PEDAL
    else:
        return BassPattern.ROOT_FIFTH
