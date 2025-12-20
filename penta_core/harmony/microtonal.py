"""
Microtonal Support - Non-12-TET tuning systems.

Provides:
- 24-TET (quarter-tones)
- Just intonation
- Pythagorean tuning
- Custom EDO (Equal Division of the Octave)
- Cents-based pitch representation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math


class TuningSystem(Enum):
    """Common tuning systems."""
    EQUAL_12 = "12tet"      # Standard 12-TET
    EQUAL_19 = "19tet"      # 19-TET
    EQUAL_24 = "24tet"      # Quarter-tones
    EQUAL_31 = "31tet"      # 31-TET (close to 1/4 comma meantone)
    EQUAL_53 = "53tet"      # Very close to just intonation
    JUST_5_LIMIT = "just5"  # 5-limit just intonation
    JUST_7_LIMIT = "just7"  # 7-limit just intonation
    PYTHAGOREAN = "pyth"    # Pythagorean tuning
    MEANTONE = "meantone"   # Quarter-comma meantone


# Just intonation ratios (5-limit)
JUST_RATIOS_5_LIMIT = {
    "unison": (1, 1),
    "minor_second": (16, 15),
    "major_second": (9, 8),
    "minor_third": (6, 5),
    "major_third": (5, 4),
    "perfect_fourth": (4, 3),
    "tritone": (45, 32),
    "perfect_fifth": (3, 2),
    "minor_sixth": (8, 5),
    "major_sixth": (5, 3),
    "minor_seventh": (9, 5),
    "major_seventh": (15, 8),
    "octave": (2, 1),
}

# Extended 7-limit ratios
JUST_RATIOS_7_LIMIT = {
    **JUST_RATIOS_5_LIMIT,
    "septimal_minor_third": (7, 6),
    "septimal_major_third": (9, 7),
    "harmonic_seventh": (7, 4),
    "septimal_tritone": (7, 5),
}


@dataclass
class MicrotonalPitch:
    """
    A pitch in a microtonal system.

    Can be represented as:
    - MIDI note + cents deviation
    - Frequency in Hz
    - Ratio from reference pitch
    """
    midi_note: int = 60  # Base MIDI note (C4)
    cents_deviation: float = 0.0  # Cents from 12-TET
    frequency: Optional[float] = None  # Hz

    @property
    def total_cents(self) -> float:
        """Total cents from C0."""
        return (self.midi_note - 12) * 100 + self.cents_deviation

    def get_frequency(self, a4_hz: float = 440.0) -> float:
        """Get frequency in Hz."""
        if self.frequency:
            return self.frequency

        # A4 = MIDI 69
        cents_from_a4 = (self.midi_note - 69) * 100 + self.cents_deviation
        return a4_hz * (2 ** (cents_from_a4 / 1200))

    def transpose_cents(self, cents: float) -> "MicrotonalPitch":
        """Transpose by cents."""
        new_cents = self.cents_deviation + cents

        # Normalize to within -50 to +50 cents
        midi_adjustment = int(new_cents / 100)
        if new_cents < -50:
            midi_adjustment -= 1
        elif new_cents > 50:
            midi_adjustment += 1

        return MicrotonalPitch(
            midi_note=self.midi_note + midi_adjustment,
            cents_deviation=new_cents - (midi_adjustment * 100),
        )

    def transpose_ratio(self, ratio: Tuple[int, int]) -> "MicrotonalPitch":
        """Transpose by a frequency ratio."""
        cents = ratio_to_cents(ratio)
        return self.transpose_cents(cents)

    def __str__(self) -> str:
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (self.midi_note // 12) - 1
        note_idx = self.midi_note % 12
        note_name = note_names[note_idx]

        if abs(self.cents_deviation) < 1:
            return f"{note_name}{octave}"
        else:
            sign = "+" if self.cents_deviation > 0 else ""
            return f"{note_name}{octave}{sign}{self.cents_deviation:.0f}c"


def cents_to_ratio(cents: float) -> float:
    """Convert cents to frequency ratio."""
    return 2 ** (cents / 1200)


def ratio_to_cents(ratio: Tuple[int, int]) -> float:
    """Convert frequency ratio to cents."""
    return 1200 * math.log2(ratio[0] / ratio[1])


def hz_to_midi_cents(hz: float, a4_hz: float = 440.0) -> Tuple[int, float]:
    """
    Convert frequency to MIDI note and cents deviation.

    Args:
        hz: Frequency in Hz
        a4_hz: Reference A4 frequency

    Returns:
        (midi_note, cents_deviation)
    """
    midi_exact = 69 + 12 * math.log2(hz / a4_hz)
    midi_note = round(midi_exact)
    cents = (midi_exact - midi_note) * 100
    return midi_note, cents


def equal_temperament(
    edo: int,
    base_note: int = 60,
    num_notes: int = 12,
) -> List[MicrotonalPitch]:
    """
    Generate pitches in an equal division of the octave.

    Args:
        edo: Equal divisions per octave (12, 19, 24, 31, etc.)
        base_note: Starting MIDI note
        num_notes: Number of pitches to generate

    Returns:
        List of pitches
    """
    step_cents = 1200 / edo
    pitches = []

    for i in range(num_notes):
        total_cents = i * step_cents
        midi_offset = int(total_cents / 100)
        cents_dev = total_cents - (midi_offset * 100)

        # Normalize cents to -50 to +50
        if cents_dev > 50:
            midi_offset += 1
            cents_dev -= 100

        pitches.append(MicrotonalPitch(
            midi_note=base_note + midi_offset,
            cents_deviation=cents_dev,
        ))

    return pitches


def just_intonation(
    root_note: int = 60,
    limit: int = 5,
) -> Dict[str, MicrotonalPitch]:
    """
    Generate just intonation pitches.

    Args:
        root_note: Root MIDI note
        limit: 5-limit or 7-limit

    Returns:
        Dict mapping interval names to pitches
    """
    ratios = JUST_RATIOS_7_LIMIT if limit >= 7 else JUST_RATIOS_5_LIMIT
    root = MicrotonalPitch(midi_note=root_note)

    pitches = {}
    for name, ratio in ratios.items():
        cents = ratio_to_cents(ratio)
        pitches[name] = root.transpose_cents(cents)

    return pitches


def pythagorean_scale(
    root_note: int = 60,
    num_fifths: int = 6,
) -> List[MicrotonalPitch]:
    """
    Generate Pythagorean scale based on stacked perfect fifths.

    Args:
        root_note: Root MIDI note
        num_fifths: Number of fifths to stack in each direction

    Returns:
        Scale pitches sorted by pitch
    """
    root = MicrotonalPitch(midi_note=root_note)
    fifth_cents = ratio_to_cents((3, 2))

    pitches = [root]

    # Stack fifths upward
    current = root
    for _ in range(num_fifths):
        current = current.transpose_cents(fifth_cents)
        # Bring back within octave
        while current.total_cents > root.total_cents + 1200:
            current = current.transpose_cents(-1200)
        pitches.append(current)

    # Stack fifths downward
    current = root
    for _ in range(num_fifths):
        current = current.transpose_cents(-fifth_cents)
        # Bring within octave above root
        while current.total_cents < root.total_cents:
            current = current.transpose_cents(1200)
        pitches.append(current)

    # Sort by pitch
    pitches.sort(key=lambda p: p.total_cents)

    # Remove duplicates
    unique = []
    for p in pitches:
        if not unique or abs(p.total_cents - unique[-1].total_cents) > 5:
            unique.append(p)

    return unique


def meantone_temperament(
    root_note: int = 60,
    comma_fraction: float = 0.25,
) -> List[MicrotonalPitch]:
    """
    Generate meantone temperament scale.

    Args:
        root_note: Root MIDI note
        comma_fraction: Fraction of syntonic comma to temper (0.25 = quarter-comma)

    Returns:
        12-note scale in meantone temperament
    """
    # Syntonic comma in cents
    syntonic_comma = ratio_to_cents((81, 80))  # ~21.5 cents

    # Tempered fifth
    perfect_fifth_cents = ratio_to_cents((3, 2))
    tempered_fifth = perfect_fifth_cents - (syntonic_comma * comma_fraction)

    root = MicrotonalPitch(midi_note=root_note)
    pitches = [root]

    # Generate scale by stacking tempered fifths
    # C-G-D-A-E-B-F#-C#-G#-D#-A#-E#(F)
    current = root
    for i in range(11):
        current = current.transpose_cents(tempered_fifth)

        # Bring back within octave
        while current.total_cents > root.total_cents + 1200:
            current = current.transpose_cents(-1200)

        pitches.append(current)

    # Sort by pitch
    pitches.sort(key=lambda p: p.total_cents)

    return pitches


def get_comma_info() -> Dict[str, Dict]:
    """Get information about common musical commas."""
    return {
        "syntonic_comma": {
            "ratio": (81, 80),
            "cents": ratio_to_cents((81, 80)),
            "description": "Difference between four perfect fifths and two octaves + major third",
        },
        "pythagorean_comma": {
            "ratio": (531441, 524288),
            "cents": ratio_to_cents((531441, 524288)),
            "description": "Difference between 12 perfect fifths and 7 octaves",
        },
        "septimal_comma": {
            "ratio": (64, 63),
            "cents": ratio_to_cents((64, 63)),
            "description": "Difference between 7:4 and 16:9",
        },
        "diesis": {
            "ratio": (128, 125),
            "cents": ratio_to_cents((128, 125)),
            "description": "Difference between three major thirds and an octave",
        },
    }


def compare_tunings(
    note_count: int = 12,
    reference_note: int = 60,
) -> Dict[str, List[float]]:
    """
    Compare cents values across different tuning systems.

    Args:
        note_count: Number of notes to compare
        reference_note: Starting MIDI note

    Returns:
        Dict mapping tuning name to list of cents from unison
    """
    results = {}

    # 12-TET reference
    results["12-TET"] = [i * 100 for i in range(note_count)]

    # 24-TET
    results["24-TET"] = [i * 50 for i in range(min(note_count * 2, 24))][:note_count]

    # Just intonation (selecting first 12 intervals)
    ji_pitches = just_intonation(reference_note)
    ji_names = ["unison", "minor_second", "major_second", "minor_third", "major_third",
                "perfect_fourth", "tritone", "perfect_fifth", "minor_sixth",
                "major_sixth", "minor_seventh", "major_seventh"]
    results["Just (5-limit)"] = [
        ji_pitches[name].cents_deviation + (ji_pitches[name].midi_note - reference_note) * 100
        for name in ji_names[:note_count]
    ]

    # Pythagorean
    pyth = pythagorean_scale(reference_note)
    results["Pythagorean"] = [p.total_cents - reference_note * 100 for p in pyth[:note_count]]

    # Quarter-comma meantone
    meantone = meantone_temperament(reference_note)
    results["Meantone (1/4)"] = [p.total_cents - reference_note * 100 for p in meantone[:note_count]]

    return results
