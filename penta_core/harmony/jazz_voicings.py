"""
Jazz Voicings - Advanced jazz chord voicing generation.

Provides:
- Close and open voicings
- Drop 2, Drop 3, Drop 2+4 voicings
- Rootless voicings (A and B)
- Shell voicings
- Upper structure voicings
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class VoicingStyle(Enum):
    """Jazz voicing styles."""
    CLOSE = "close"
    DROP_2 = "drop_2"
    DROP_3 = "drop_3"
    DROP_2_4 = "drop_2_4"
    ROOTLESS_A = "rootless_a"
    ROOTLESS_B = "rootless_b"
    SHELL = "shell"
    UPPER_STRUCTURE = "upper_structure"
    QUARTAL = "quartal"
    CLUSTER = "cluster"


# Chord quality templates (intervals from root in semitones)
CHORD_TEMPLATES = {
    "maj7": [0, 4, 7, 11],
    "dom7": [0, 4, 7, 10],
    "min7": [0, 3, 7, 10],
    "min7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "maj9": [0, 4, 7, 11, 14],
    "dom9": [0, 4, 7, 10, 14],
    "min9": [0, 3, 7, 10, 14],
    "dom13": [0, 4, 7, 10, 14, 21],
    "maj13": [0, 4, 7, 11, 14, 21],
    "7b9": [0, 4, 7, 10, 13],
    "7#9": [0, 4, 7, 10, 15],
    "7#11": [0, 4, 7, 10, 18],
    "7alt": [0, 4, 8, 10, 13, 15],  # b5, #5, b9, #9
    "6/9": [0, 4, 7, 9, 14],
    "sus4": [0, 5, 7],
    "7sus4": [0, 5, 7, 10],
}

# Note names
NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def note_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    note_upper = note[0].upper()
    if len(note) > 1:
        if note[1] == '#':
            offset = 1
        elif note[1] == 'b':
            offset = -1
        else:
            offset = 0
    else:
        offset = 0

    base_notes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    return (octave + 1) * 12 + base_notes.get(note_upper, 0) + offset


def midi_to_note(midi: int) -> Tuple[str, int]:
    """Convert MIDI number to note name and octave."""
    octave = (midi // 12) - 1
    note_idx = midi % 12
    return NOTES[note_idx], octave


@dataclass
class JazzVoicing:
    """A jazz chord voicing."""
    root: str
    quality: str
    style: VoicingStyle
    notes: List[int]  # MIDI note numbers
    bass_note: Optional[int] = None  # For slash chords

    def get_note_names(self) -> List[str]:
        """Get note names for the voicing."""
        return [midi_to_note(n)[0] for n in self.notes]

    def transpose(self, semitones: int) -> "JazzVoicing":
        """Transpose the voicing."""
        return JazzVoicing(
            root=NOTES[(NOTES.index(self.root) + semitones) % 12],
            quality=self.quality,
            style=self.style,
            notes=[n + semitones for n in self.notes],
            bass_note=self.bass_note + semitones if self.bass_note else None,
        )

    def invert(self, inversion: int = 1) -> "JazzVoicing":
        """Invert the voicing."""
        notes = list(self.notes)
        for _ in range(inversion):
            notes.append(notes.pop(0) + 12)
        return JazzVoicing(
            root=self.root,
            quality=self.quality,
            style=self.style,
            notes=notes,
            bass_note=self.bass_note,
        )


def generate_jazz_voicing(
    root: str,
    quality: str,
    style: VoicingStyle = VoicingStyle.CLOSE,
    octave: int = 4,
    extensions: Optional[List[int]] = None,
) -> JazzVoicing:
    """
    Generate a jazz voicing.

    Args:
        root: Root note (e.g., "C", "Db")
        quality: Chord quality (e.g., "maj7", "dom7")
        style: Voicing style
        octave: Base octave
        extensions: Additional extensions (9, 11, 13)

    Returns:
        Jazz voicing
    """
    # Get base intervals
    intervals = list(CHORD_TEMPLATES.get(quality, CHORD_TEMPLATES["maj7"]))

    # Add extensions
    if extensions:
        for ext in extensions:
            if ext == 9:
                intervals.append(14)
            elif ext == 11:
                intervals.append(17)
            elif ext == 13:
                intervals.append(21)
            elif ext == -9:  # b9
                intervals.append(13)
            elif ext == 99:  # #9
                intervals.append(15)
            elif ext == -11:  # #11
                intervals.append(18)
            elif ext == -13:  # b13
                intervals.append(20)

    # Calculate base MIDI note
    root_midi = note_to_midi(root, octave)

    # Generate notes based on style
    if style == VoicingStyle.CLOSE:
        notes = [root_midi + i for i in intervals]

    elif style == VoicingStyle.DROP_2:
        notes = get_drop_voicing(root_midi, intervals, drop=2)

    elif style == VoicingStyle.DROP_3:
        notes = get_drop_voicing(root_midi, intervals, drop=3)

    elif style == VoicingStyle.DROP_2_4:
        notes = get_drop_voicing(root_midi, intervals, drop=[2, 4])

    elif style == VoicingStyle.ROOTLESS_A:
        notes = get_rootless_voicing(root_midi, intervals, voicing_type="A")

    elif style == VoicingStyle.ROOTLESS_B:
        notes = get_rootless_voicing(root_midi, intervals, voicing_type="B")

    elif style == VoicingStyle.SHELL:
        # Shell voicing: root, 3rd, 7th only
        shell_intervals = [0]
        if 4 in intervals:  # Major 3rd
            shell_intervals.append(4)
        elif 3 in intervals:  # Minor 3rd
            shell_intervals.append(3)
        if 11 in intervals:  # Major 7th
            shell_intervals.append(11)
        elif 10 in intervals:  # Minor 7th
            shell_intervals.append(10)
        notes = [root_midi + i for i in shell_intervals]

    elif style == VoicingStyle.QUARTAL:
        # Quartal voicing: stacked 4ths
        notes = [root_midi, root_midi + 5, root_midi + 10, root_midi + 15]

    elif style == VoicingStyle.UPPER_STRUCTURE:
        # Upper structure: triad over 7th chord
        notes = get_upper_structure_voicing(root_midi, quality)

    else:
        notes = [root_midi + i for i in intervals]

    return JazzVoicing(
        root=root,
        quality=quality,
        style=style,
        notes=sorted(notes),
    )


def get_drop_voicing(
    root_midi: int,
    intervals: List[int],
    drop: int | List[int] = 2,
) -> List[int]:
    """
    Create a drop voicing.

    Drop 2 drops the second highest note down an octave.
    Drop 3 drops the third highest note.
    Drop 2+4 drops both.
    """
    # Start with close position
    notes = [root_midi + i for i in intervals]
    notes = sorted(notes)

    if isinstance(drop, int):
        drops = [drop]
    else:
        drops = list(drop)

    for d in sorted(drops, reverse=True):
        if len(notes) >= d:
            idx = len(notes) - d
            notes[idx] -= 12

    return sorted(notes)


def get_rootless_voicing(
    root_midi: int,
    intervals: List[int],
    voicing_type: str = "A",
) -> List[int]:
    """
    Create a rootless voicing.

    Type A: 3-5-7-9 (or 3-5-6-9 for maj6)
    Type B: 7-9-3-5 (inverted)
    """
    # Remove root
    intervals = [i for i in intervals if i != 0]

    # Ensure we have enough notes
    if len(intervals) < 3:
        return [root_midi + i for i in intervals]

    notes = [root_midi + i for i in intervals[:4]]

    if voicing_type == "B":
        # Invert: put 7th at bottom
        notes = sorted(notes)
        # Rotate so 7th is at bottom
        while notes[0] % 12 not in [10, 11]:  # 7th
            notes.append(notes.pop(0) + 12)

    return sorted(notes)


def get_upper_structure_voicing(
    root_midi: int,
    quality: str,
) -> List[int]:
    """
    Create an upper structure voicing.

    Upper structures place a triad over a dominant 7th chord.
    """
    # Base: 7th chord shell
    shell = [root_midi, root_midi + 4, root_midi + 10]  # 1, 3, b7

    # Upper structure triads by quality
    upper_triads = {
        "dom7": [14, 18, 21],  # D major over C7 = 9, #11, 13
        "7#11": [13, 17, 20],  # Db major over C7 = b9, 11, b13
        "7alt": [13, 17, 20],
        "7b9": [13, 16, 20],
        "7#9": [15, 19, 22],
    }

    upper = upper_triads.get(quality, [14, 17, 21])

    return shell + [root_midi + i for i in upper]


def voice_lead_progression(
    chords: List[Tuple[str, str]],
    style: VoicingStyle = VoicingStyle.DROP_2,
    starting_octave: int = 4,
) -> List[JazzVoicing]:
    """
    Voice lead a chord progression.

    Minimizes voice movement between chords.

    Args:
        chords: List of (root, quality) tuples
        style: Voicing style
        starting_octave: Octave for first chord

    Returns:
        List of voice-led voicings
    """
    if not chords:
        return []

    voicings = []
    prev_voicing = None

    for root, quality in chords:
        voicing = generate_jazz_voicing(root, quality, style, starting_octave)

        if prev_voicing:
            voicing = _minimize_voice_movement(prev_voicing, voicing)

        voicings.append(voicing)
        prev_voicing = voicing

    return voicings


def _minimize_voice_movement(
    prev: JazzVoicing,
    curr: JazzVoicing,
) -> JazzVoicing:
    """Adjust current voicing to minimize movement from previous."""
    prev_notes = prev.notes
    curr_notes = list(curr.notes)

    # Try different inversions/octave shifts
    best_notes = curr_notes
    best_movement = sum(abs(c - p) for c, p in zip(curr_notes, prev_notes) if len(prev_notes) >= len(curr_notes))

    for octave_shift in range(-12, 13, 12):
        for inversion, _ in enumerate(curr_notes):
            test_notes = curr_notes[:]
            # Apply inversion
            for _ in range(inversion):
                test_notes.append(test_notes.pop(0) + 12)
            # Apply octave shift
            test_notes = [n + octave_shift for n in test_notes]

            # Calculate movement
            if len(test_notes) <= len(prev_notes):
                movement = sum(abs(c - p) for c, p in zip(test_notes, prev_notes[:len(test_notes)]))
                if movement < best_movement:
                    best_movement = movement
                    best_notes = test_notes

    return JazzVoicing(
        root=curr.root,
        quality=curr.quality,
        style=curr.style,
        notes=sorted(best_notes),
        bass_note=curr.bass_note,
    )


def get_common_progressions() -> Dict[str, List[Tuple[str, str]]]:
    """Get common jazz chord progressions."""
    return {
        "ii-V-I major": [("D", "min7"), ("G", "dom7"), ("C", "maj7")],
        "ii-V-I minor": [("D", "min7b5"), ("G", "7b9"), ("C", "min7")],
        "I-VI-ii-V": [("C", "maj7"), ("A", "min7"), ("D", "min7"), ("G", "dom7")],
        "iii-VI-ii-V": [("E", "min7"), ("A", "dom7"), ("D", "min7"), ("G", "dom7")],
        "Coltrane changes": [("C", "maj7"), ("Eb", "dom7"), ("Ab", "maj7"), ("B", "dom7"), ("E", "maj7"), ("G", "dom7")],
        "rhythm changes A": [("Bb", "maj7"), ("G", "min7"), ("C", "min7"), ("F", "dom7")],
        "blues": [("C", "dom7"), ("F", "dom7"), ("C", "dom7"), ("C", "dom7"),
                  ("F", "dom7"), ("F", "dom7"), ("C", "dom7"), ("A", "dom7"),
                  ("D", "min7"), ("G", "dom7"), ("C", "dom7"), ("G", "dom7")],
    }
