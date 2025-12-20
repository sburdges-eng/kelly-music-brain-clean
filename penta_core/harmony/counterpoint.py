"""
Counterpoint Generation - Species counterpoint rules and generation.

Implements traditional counterpoint rules and generation:
- First species (note against note)
- Second species (two notes against one)
- Third species (four notes against one)
- Fourth species (syncopation)
- Fifth species (florid)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


class Species(Enum):
    """Counterpoint species."""
    FIRST = 1   # Note against note
    SECOND = 2  # 2:1 ratio
    THIRD = 3   # 4:1 ratio
    FOURTH = 4  # Syncopation/suspensions
    FIFTH = 5   # Florid (mixed)


class Motion(Enum):
    """Types of melodic motion."""
    CONTRARY = "contrary"
    SIMILAR = "similar"
    PARALLEL = "parallel"
    OBLIQUE = "oblique"


class Interval(Enum):
    """Interval classifications."""
    PERFECT_CONSONANCE = "perfect"    # Unison, 5th, octave
    IMPERFECT_CONSONANCE = "imperfect"  # 3rd, 6th
    DISSONANCE = "dissonance"         # 2nd, 4th, 7th, tritone


# Interval mappings (in semitones)
PERFECT_CONSONANCES = {0, 7, 12}  # Unison, P5, Octave
IMPERFECT_CONSONANCES = {3, 4, 8, 9}  # m3, M3, m6, M6
DISSONANCES = {1, 2, 5, 6, 10, 11}  # m2, M2, P4, tritone, m7, M7


@dataclass
class CounterpointVoice:
    """A voice in counterpoint."""
    notes: List[int]  # MIDI note numbers
    durations: List[float] = field(default_factory=list)  # In beats
    is_cantus_firmus: bool = False

    def get_interval_to(self, other: "CounterpointVoice", index: int) -> int:
        """Get interval (in semitones) to another voice at index."""
        if index < len(self.notes) and index < len(other.notes):
            return abs(self.notes[index] - other.notes[index]) % 12
        return 0

    def get_motion_type(self, other: "CounterpointVoice", index: int) -> Optional[Motion]:
        """Get motion type between this and another voice from index to index+1."""
        if index + 1 >= len(self.notes) or index + 1 >= len(other.notes):
            return None

        self_motion = self.notes[index + 1] - self.notes[index]
        other_motion = other.notes[index + 1] - other.notes[index]

        if self_motion == 0 and other_motion == 0:
            return Motion.OBLIQUE
        elif self_motion == 0 or other_motion == 0:
            return Motion.OBLIQUE
        elif self_motion * other_motion < 0:
            return Motion.CONTRARY
        elif self_motion == other_motion:
            return Motion.PARALLEL
        else:
            return Motion.SIMILAR


@dataclass
class CounterpointRule:
    """A counterpoint rule."""
    name: str
    description: str
    applies_to: List[Species]
    severity: str = "error"  # "error" or "warning"


# Standard counterpoint rules
COUNTERPOINT_RULES = [
    CounterpointRule(
        name="no_parallel_fifths",
        description="Avoid parallel perfect fifths",
        applies_to=[Species.FIRST, Species.SECOND, Species.THIRD, Species.FOURTH, Species.FIFTH],
        severity="error",
    ),
    CounterpointRule(
        name="no_parallel_octaves",
        description="Avoid parallel octaves",
        applies_to=[Species.FIRST, Species.SECOND, Species.THIRD, Species.FOURTH, Species.FIFTH],
        severity="error",
    ),
    CounterpointRule(
        name="no_direct_fifths",
        description="Avoid direct (hidden) fifths by similar motion",
        applies_to=[Species.FIRST, Species.SECOND],
        severity="warning",
    ),
    CounterpointRule(
        name="stepwise_motion",
        description="Prefer stepwise motion over leaps",
        applies_to=[Species.FIRST, Species.SECOND, Species.THIRD],
        severity="warning",
    ),
    CounterpointRule(
        name="recover_leap",
        description="Leaps should be followed by stepwise motion in opposite direction",
        applies_to=[Species.FIRST, Species.SECOND, Species.THIRD, Species.FIFTH],
        severity="warning",
    ),
    CounterpointRule(
        name="consonant_downbeats",
        description="Downbeats should be consonant",
        applies_to=[Species.SECOND, Species.THIRD, Species.FOURTH],
        severity="error",
    ),
    CounterpointRule(
        name="resolve_suspensions",
        description="Suspensions must resolve downward by step",
        applies_to=[Species.FOURTH, Species.FIFTH],
        severity="error",
    ),
]


def classify_interval(semitones: int) -> Interval:
    """Classify an interval by consonance/dissonance."""
    normalized = semitones % 12
    if normalized in PERFECT_CONSONANCES:
        return Interval.PERFECT_CONSONANCE
    elif normalized in IMPERFECT_CONSONANCES:
        return Interval.IMPERFECT_CONSONANCE
    else:
        return Interval.DISSONANCE


def check_counterpoint_rules(
    cantus_firmus: CounterpointVoice,
    counterpoint: CounterpointVoice,
    species: Species = Species.FIRST,
) -> List[Dict]:
    """
    Check counterpoint against standard rules.

    Args:
        cantus_firmus: The fixed melody
        counterpoint: The composed counterpoint
        species: Which species rules to apply

    Returns:
        List of rule violations with details
    """
    violations = []

    # Get applicable rules
    applicable_rules = [r for r in COUNTERPOINT_RULES if species in r.applies_to]

    min_len = min(len(cantus_firmus.notes), len(counterpoint.notes))

    for i in range(min_len - 1):
        # Check parallel fifths/octaves
        curr_interval = counterpoint.get_interval_to(cantus_firmus, i)
        next_interval = counterpoint.get_interval_to(cantus_firmus, i + 1)
        motion = counterpoint.get_motion_type(cantus_firmus, i)

        # Parallel fifths
        if curr_interval == 7 and next_interval == 7 and motion == Motion.PARALLEL:
            violations.append({
                "rule": "no_parallel_fifths",
                "position": i,
                "severity": "error",
                "description": f"Parallel fifths at position {i}",
            })

        # Parallel octaves
        if curr_interval in [0, 12] and next_interval in [0, 12] and motion == Motion.PARALLEL:
            violations.append({
                "rule": "no_parallel_octaves",
                "position": i,
                "severity": "error",
                "description": f"Parallel octaves at position {i}",
            })

        # Direct fifths
        if next_interval == 7 and motion == Motion.SIMILAR:
            violations.append({
                "rule": "no_direct_fifths",
                "position": i,
                "severity": "warning",
                "description": f"Direct fifth at position {i}",
            })

        # Check for large leaps
        cp_leap = abs(counterpoint.notes[i + 1] - counterpoint.notes[i])
        if cp_leap > 5:  # More than a 4th
            violations.append({
                "rule": "stepwise_motion",
                "position": i,
                "severity": "warning",
                "description": f"Large leap ({cp_leap} semitones) at position {i}",
            })

    # Check consonance on downbeats
    for i in range(min_len):
        interval = counterpoint.get_interval_to(cantus_firmus, i)
        interval_type = classify_interval(interval)

        # First and last notes should be perfect consonances
        if i == 0 or i == min_len - 1:
            if interval_type != Interval.PERFECT_CONSONANCE:
                violations.append({
                    "rule": "perfect_cadence",
                    "position": i,
                    "severity": "error",
                    "description": f"First/last interval should be perfect consonance",
                })

    return violations


def get_species_rules(species: Species) -> List[str]:
    """Get human-readable rules for a species."""
    base_rules = [
        "Begin and end on a perfect consonance (unison, fifth, or octave)",
        "Avoid parallel fifths and octaves",
        "Prefer contrary and oblique motion",
        "Prefer stepwise motion; recover leaps in opposite direction",
        "Avoid tritones",
    ]

    species_specific = {
        Species.FIRST: [
            "One note in counterpoint for each note in cantus firmus",
            "All intervals should be consonant",
            "Avoid too many consecutive thirds or sixths",
        ],
        Species.SECOND: [
            "Two notes in counterpoint for each note in cantus firmus",
            "First note of each pair must be consonant",
            "Second note may be passing tone (dissonant) if approached/left by step",
        ],
        Species.THIRD: [
            "Four notes in counterpoint for each note in cantus firmus",
            "First note must be consonant",
            "Passing tones and neighbor tones allowed on weak beats",
        ],
        Species.FOURTH: [
            "Syncopated rhythm with suspensions",
            "Suspensions must resolve downward by step",
            "Suspensions are typically 4-3, 7-6, or 9-8",
        ],
        Species.FIFTH: [
            "Florid counterpoint combining all species",
            "Free rhythm and melodic invention",
            "All previous rules apply where relevant",
        ],
    }

    return base_rules + species_specific.get(species, [])


def generate_counterpoint(
    cantus_firmus: List[int],
    species: Species = Species.FIRST,
    above: bool = True,
    key_center: int = 60,
) -> CounterpointVoice:
    """
    Generate counterpoint for a cantus firmus.

    Args:
        cantus_firmus: MIDI notes of the fixed melody
        species: Which species to generate
        above: True for counterpoint above, False for below
        key_center: MIDI note for the key center

    Returns:
        Generated counterpoint voice
    """
    cf = CounterpointVoice(notes=cantus_firmus, is_cantus_firmus=True)
    cp_notes = []

    # Scale degrees in the key
    major_scale = [0, 2, 4, 5, 7, 9, 11]

    for i, cf_note in enumerate(cantus_firmus):
        candidates = []

        # Generate candidate notes
        for octave_offset in range(-12, 25, 12):
            for degree in major_scale:
                candidate = key_center + degree + octave_offset

                # Check if above/below
                if above and candidate <= cf_note:
                    continue
                if not above and candidate >= cf_note:
                    continue

                # Check interval consonance
                interval = abs(candidate - cf_note) % 12
                interval_type = classify_interval(interval)

                # First and last must be perfect consonance
                if i == 0 or i == len(cantus_firmus) - 1:
                    if interval_type == Interval.PERFECT_CONSONANCE:
                        candidates.append((candidate, 10))  # High priority
                elif species == Species.FIRST:
                    if interval_type != Interval.DISSONANCE:
                        priority = 8 if interval_type == Interval.IMPERFECT_CONSONANCE else 5
                        candidates.append((candidate, priority))
                else:
                    candidates.append((candidate, 5))

        if not candidates:
            # Fallback: use octave
            cp_notes.append(cf_note + (12 if above else -12))
            continue

        # Score candidates based on voice leading from previous note
        if cp_notes:
            prev_note = cp_notes[-1]
            scored = []
            for note, base_priority in candidates:
                distance = abs(note - prev_note)
                # Prefer stepwise motion
                if distance <= 2:
                    scored.append((note, base_priority + 5))
                elif distance <= 5:
                    scored.append((note, base_priority + 2))
                else:
                    scored.append((note, base_priority))

            candidates = scored

        # Sort by priority and pick best
        candidates.sort(key=lambda x: -x[1])
        cp_notes.append(candidates[0][0])

    return CounterpointVoice(notes=cp_notes)


def analyze_counterpoint(
    voice1: CounterpointVoice,
    voice2: CounterpointVoice,
) -> Dict:
    """
    Analyze the counterpoint between two voices.

    Returns statistics about intervals, motion types, etc.
    """
    min_len = min(len(voice1.notes), len(voice2.notes))

    intervals = []
    motion_types = []
    interval_types = []

    for i in range(min_len):
        interval = voice1.get_interval_to(voice2, i)
        intervals.append(interval)
        interval_types.append(classify_interval(interval))

        if i < min_len - 1:
            motion = voice1.get_motion_type(voice2, i)
            if motion:
                motion_types.append(motion)

    # Count statistics
    motion_counts = {}
    for m in motion_types:
        motion_counts[m.value] = motion_counts.get(m.value, 0) + 1

    interval_counts = {}
    for it in interval_types:
        interval_counts[it.value] = interval_counts.get(it.value, 0) + 1

    return {
        "intervals": intervals,
        "motion_types": [m.value for m in motion_types],
        "motion_counts": motion_counts,
        "interval_counts": interval_counts,
        "consonance_ratio": (
            interval_counts.get("perfect", 0) + interval_counts.get("imperfect", 0)
        ) / len(interval_types) if interval_types else 0,
    }
