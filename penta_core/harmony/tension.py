"""
Tension Analysis - Harmonic tension and release analysis.

Analyzes and plans tension curves for emotional impact.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class TensionLevel(Enum):
    """Levels of harmonic tension."""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


# Chord tension scores (relative tension values)
CHORD_TENSIONS = {
    # Triads
    "maj": 1.0,
    "min": 1.5,
    "dim": 3.0,
    "aug": 3.5,
    # Seventh chords
    "maj7": 1.5,
    "min7": 2.0,
    "dom7": 2.5,
    "min7b5": 3.5,
    "dim7": 4.0,
    # Extended chords
    "dom9": 2.5,
    "min9": 2.0,
    "maj9": 1.5,
    # Altered dominants
    "7b9": 4.0,
    "7#9": 4.0,
    "7#11": 3.5,
    "7alt": 4.5,
    "7b5": 3.5,
    "7#5": 3.5,
    # Suspensions
    "sus2": 2.0,
    "sus4": 2.0,
    "7sus4": 2.5,
}


# Interval tension (for vertical analysis)
INTERVAL_TENSIONS = {
    0: 0.0,   # Unison
    1: 4.0,   # m2
    2: 3.0,   # M2
    3: 1.0,   # m3
    4: 1.0,   # M3
    5: 2.0,   # P4
    6: 4.5,   # Tritone
    7: 0.5,   # P5
    8: 1.5,   # m6
    9: 1.5,   # M6
    10: 2.5,  # m7
    11: 2.0,  # M7
}


@dataclass
class TensionAnalysis:
    """Result of tension analysis."""
    chord_tensions: List[float]
    average_tension: float
    tension_curve: List[float]  # Normalized 0-1
    peak_tension_index: int
    release_points: List[int]

    def get_tension_level(self, index: int) -> TensionLevel:
        """Get tension level at index."""
        if index >= len(self.chord_tensions):
            return TensionLevel.MODERATE

        t = self.chord_tensions[index]
        if t < 1.5:
            return TensionLevel.VERY_LOW
        elif t < 2.0:
            return TensionLevel.LOW
        elif t < 3.0:
            return TensionLevel.MODERATE
        elif t < 4.0:
            return TensionLevel.HIGH
        else:
            return TensionLevel.VERY_HIGH


def analyze_tension(
    chords: List[Tuple[str, str]],
    include_voice_leading: bool = True,
) -> TensionAnalysis:
    """
    Analyze harmonic tension in a chord progression.

    Args:
        chords: List of (root, quality) tuples
        include_voice_leading: Include voice leading tension

    Returns:
        Tension analysis
    """
    if not chords:
        return TensionAnalysis(
            chord_tensions=[],
            average_tension=0.0,
            tension_curve=[],
            peak_tension_index=0,
            release_points=[],
        )

    tensions = []

    for i, (root, quality) in enumerate(chords):
        # Base chord tension
        base_tension = CHORD_TENSIONS.get(quality, 2.0)

        # Voice leading tension (if previous chord exists)
        if include_voice_leading and i > 0:
            prev_root, prev_quality = chords[i - 1]
            vl_tension = _calculate_voice_leading_tension(
                prev_root, prev_quality, root, quality
            )
            base_tension += vl_tension * 0.3

        tensions.append(base_tension)

    # Find peak and release points
    max_tension = max(tensions) if tensions else 0
    min_tension = min(tensions) if tensions else 0
    tension_range = max_tension - min_tension if max_tension > min_tension else 1

    peak_index = tensions.index(max(tensions)) if tensions else 0

    # Find release points (significant drops in tension)
    release_points = []
    for i in range(1, len(tensions)):
        if tensions[i] < tensions[i - 1] - 0.5:
            release_points.append(i)

    # Normalize curve
    curve = [(t - min_tension) / tension_range for t in tensions]

    return TensionAnalysis(
        chord_tensions=tensions,
        average_tension=sum(tensions) / len(tensions) if tensions else 0,
        tension_curve=curve,
        peak_tension_index=peak_index,
        release_points=release_points,
    )


def _calculate_voice_leading_tension(
    prev_root: str,
    prev_quality: str,
    curr_root: str,
    curr_quality: str,
) -> float:
    """Calculate tension from voice leading between chords."""
    notes = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    prev_idx = notes.index(prev_root) if prev_root in notes else 0
    curr_idx = notes.index(curr_root) if curr_root in notes else 0

    # Root movement interval
    interval = abs(curr_idx - prev_idx)
    if interval > 6:
        interval = 12 - interval

    # Certain movements are more "expected"
    expected_movements = {
        5: 0.0,   # Down a fifth (very expected)
        7: 0.0,   # Up a fifth
        2: 0.5,   # Whole step
        1: 1.0,   # Half step (chromatic)
        3: 0.5,   # Minor third
        4: 0.5,   # Major third
        6: 2.0,   # Tritone (unexpected)
    }

    return expected_movements.get(interval, 1.0)


def plan_tension_curve(
    target_shape: str,
    length: int,
    peak_position: float = 0.7,
) -> List[float]:
    """
    Plan a tension curve for a piece.

    Args:
        target_shape: Curve shape ("rising", "arc", "wave", "climax")
        length: Number of points
        peak_position: Position of peak (0-1)

    Returns:
        List of target tension values (0-1)
    """
    import math

    curve = []
    peak_idx = int(length * peak_position)

    if target_shape == "rising":
        # Gradually rising tension
        for i in range(length):
            curve.append(i / (length - 1) if length > 1 else 0.5)

    elif target_shape == "arc":
        # Rise to peak, then fall
        for i in range(length):
            if i <= peak_idx:
                curve.append(i / peak_idx if peak_idx > 0 else 1)
            else:
                remaining = length - peak_idx
                curve.append(1 - (i - peak_idx) / remaining)

    elif target_shape == "wave":
        # Sinusoidal wave
        for i in range(length):
            curve.append(0.5 + 0.5 * math.sin(2 * math.pi * i / (length - 1) - math.pi / 2))

    elif target_shape == "climax":
        # Low tension building to climax at end
        for i in range(length):
            # Exponential rise
            curve.append((i / (length - 1)) ** 2 if length > 1 else 0.5)

    elif target_shape == "falling":
        # High tension resolving
        for i in range(length):
            curve.append(1 - (i / (length - 1)) if length > 1 else 0.5)

    else:
        # Default: moderate, slight arc
        for i in range(length):
            base = 0.5
            arc = 0.2 * math.sin(math.pi * i / (length - 1)) if length > 1 else 0
            curve.append(base + arc)

    return curve


def suggest_tension_chords(
    target_tension: TensionLevel,
    key: str = "C",
    mode: str = "major",
) -> List[Tuple[str, str]]:
    """
    Suggest chords for a target tension level.

    Args:
        target_tension: Desired tension level
        key: Key center
        mode: major or minor

    Returns:
        List of suggested (root, quality) tuples
    """
    notes = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    key_idx = notes.index(key) if key in notes else 0

    # Major scale degrees
    if mode == "major":
        degrees = [0, 2, 4, 5, 7, 9, 11]
        qualities = ["maj7", "min7", "min7", "maj7", "dom7", "min7", "min7b5"]
    else:
        degrees = [0, 2, 3, 5, 7, 8, 10]
        qualities = ["min7", "min7b5", "maj7", "min7", "min7", "maj7", "dom7"]

    suggestions = {
        TensionLevel.VERY_LOW: [
            (notes[(key_idx + degrees[0]) % 12], qualities[0]),  # I
        ],
        TensionLevel.LOW: [
            (notes[(key_idx + degrees[3]) % 12], qualities[3]),  # IV
            (notes[(key_idx + degrees[5]) % 12], qualities[5]),  # vi
        ],
        TensionLevel.MODERATE: [
            (notes[(key_idx + degrees[1]) % 12], qualities[1]),  # ii
            (notes[(key_idx + degrees[2]) % 12], qualities[2]),  # iii
        ],
        TensionLevel.HIGH: [
            (notes[(key_idx + degrees[4]) % 12], qualities[4]),  # V
            (notes[(key_idx + degrees[4]) % 12], "7b9"),  # V7b9
        ],
        TensionLevel.VERY_HIGH: [
            (notes[(key_idx + degrees[6]) % 12], qualities[6]),  # vii
            (notes[(key_idx + degrees[4]) % 12], "7alt"),  # Valt
            (notes[(key_idx + 6) % 12], "dim7"),  # Tritone substitution
        ],
    }

    return suggestions.get(target_tension, [])


def analyze_resolution(
    from_chord: Tuple[str, str],
    to_chord: Tuple[str, str],
) -> Dict:
    """
    Analyze the resolution strength between two chords.
    """
    from_tension = CHORD_TENSIONS.get(from_chord[1], 2.0)
    to_tension = CHORD_TENSIONS.get(to_chord[1], 2.0)

    tension_drop = from_tension - to_tension

    notes = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    from_idx = notes.index(from_chord[0]) if from_chord[0] in notes else 0
    to_idx = notes.index(to_chord[0]) if to_chord[0] in notes else 0
    interval = (to_idx - from_idx) % 12

    # Strong resolutions
    is_authentic = interval == 7 and "dom" in from_chord[1]  # V-I
    is_plagal = interval == 7 and "maj" in from_chord[1]  # IV-I (going up)
    is_deceptive = interval == 2 and "dom" in from_chord[1]  # V-vi

    return {
        "tension_drop": tension_drop,
        "is_strong_resolution": tension_drop > 1.0,
        "is_authentic": is_authentic,
        "is_plagal": is_plagal,
        "is_deceptive": is_deceptive,
        "resolution_strength": min(tension_drop + (1.0 if is_authentic else 0), 5.0),
    }
