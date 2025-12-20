"""
Polyrhythm Detection and Generation.

Provides:
- Detection of polyrhythmic patterns in MIDI
- Generation of common polyrhythms (3:2, 4:3, 5:4, etc.)
- LCM-based duration calculation
- Accent pattern extraction
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math


@dataclass
class PolyrhythmPattern:
    """
    A polyrhythmic pattern with multiple concurrent rhythmic layers.

    Attributes:
        ratios: List of beat divisions (e.g., [3, 2] for 3:2)
        duration_beats: Duration in beats
        accents: Accent patterns for each layer
    """
    ratios: List[int]
    duration_beats: float = 1.0
    accents: List[List[float]] = field(default_factory=list)
    name: Optional[str] = None

    def __post_init__(self):
        # Generate default accents if not provided
        if not self.accents:
            self.accents = [
                [1.0 if i == 0 else 0.7 for i in range(r)]
                for r in self.ratios
            ]

    @property
    def lcm_subdivisions(self) -> int:
        """Get LCM of all ratios for common subdivision."""
        return calculate_lcm(*self.ratios)

    def get_beat_positions(self, layer: int = 0) -> List[float]:
        """
        Get beat positions for a specific layer.

        Args:
            layer: Layer index (0-based)

        Returns:
            List of positions in beats
        """
        if layer >= len(self.ratios):
            return []

        divisions = self.ratios[layer]
        step = self.duration_beats / divisions
        return [i * step for i in range(divisions)]

    def get_all_positions(self) -> Dict[int, List[float]]:
        """Get beat positions for all layers."""
        return {
            i: self.get_beat_positions(i)
            for i, _ in enumerate(self.ratios)
        }

    def get_coincident_points(self) -> List[float]:
        """Find positions where multiple layers align."""
        all_positions = []
        for layer, _ in enumerate(self.ratios):
            all_positions.extend(self.get_beat_positions(layer))

        # Find positions that appear multiple times
        position_counts: Dict[float, int] = {}
        for pos in all_positions:
            rounded = round(pos, 6)
            position_counts[rounded] = position_counts.get(rounded, 0) + 1

        return sorted([
            pos for pos, count in position_counts.items()
            if count > 1
        ])


@dataclass
class Polyrhythm:
    """
    Represents a polyrhythm with timing and note information.
    """
    pattern: PolyrhythmPattern
    tempo_bpm: float = 120.0
    start_time: float = 0.0

    # Note assignments per layer
    notes: List[List[int]] = field(default_factory=list)
    velocities: List[List[int]] = field(default_factory=list)

    def get_duration_seconds(self) -> float:
        """Get total duration in seconds."""
        beats = self.pattern.duration_beats
        return (beats / self.tempo_bpm) * 60.0

    def to_midi_events(self) -> List[Dict]:
        """
        Convert to MIDI note events.

        Returns:
            List of event dicts with time, note, velocity, duration
        """
        events = []
        beat_duration = 60.0 / self.tempo_bpm

        for layer_idx, positions in self.pattern.get_all_positions().items():
            layer_notes = self.notes[layer_idx] if layer_idx < len(self.notes) else [60]
            layer_velocities = self.velocities[layer_idx] if layer_idx < len(self.velocities) else [100]
            layer_accents = self.pattern.accents[layer_idx] if layer_idx < len(self.pattern.accents) else [1.0]

            for i, pos in enumerate(positions):
                note = layer_notes[i % len(layer_notes)]
                base_velocity = layer_velocities[i % len(layer_velocities)]
                accent = layer_accents[i % len(layer_accents)]

                events.append({
                    "time": self.start_time + (pos * beat_duration),
                    "note": note,
                    "velocity": int(base_velocity * accent),
                    "duration": beat_duration * 0.5,  # Default to half beat
                    "layer": layer_idx,
                })

        # Sort by time
        events.sort(key=lambda e: e["time"])
        return events


def calculate_lcm(*numbers: int) -> int:
    """
    Calculate Least Common Multiple of numbers.

    Args:
        *numbers: Variable number of integers

    Returns:
        LCM of all numbers
    """
    if not numbers:
        return 1

    result = numbers[0]
    for num in numbers[1:]:
        result = (result * num) // math.gcd(result, num)
    return result


def calculate_lcm_duration(ratios: List[int], base_duration: float = 1.0) -> float:
    """
    Calculate duration needed for polyrhythm to complete one full cycle.

    Args:
        ratios: List of beat divisions
        base_duration: Base duration in beats

    Returns:
        Duration in beats for one complete cycle
    """
    lcm = calculate_lcm(*ratios)
    return base_duration * lcm / min(ratios)


def detect_polyrhythm(
    events: List[Dict],
    tolerance_ms: float = 20.0,
    min_events_per_layer: int = 3,
) -> Optional[PolyrhythmPattern]:
    """
    Detect polyrhythmic patterns in MIDI events.

    Args:
        events: List of MIDI events with 'time' key (in seconds)
        tolerance_ms: Timing tolerance in milliseconds
        min_events_per_layer: Minimum events to identify a layer

    Returns:
        Detected polyrhythm pattern or None
    """
    if len(events) < 2:
        return None

    # Sort by time
    sorted_events = sorted(events, key=lambda e: e.get("time", 0))

    # Calculate inter-onset intervals
    times = [e.get("time", 0) for e in sorted_events]
    # More efficient: use zip to create pairs
    intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]

    if not intervals:
        return None

    # Cluster intervals to find layers
    tolerance_sec = tolerance_ms / 1000.0
    interval_clusters: Dict[float, List[float]] = {}

    for interval in intervals:
        found_cluster = False
        for center in interval_clusters:
            if abs(interval - center) < tolerance_sec:
                interval_clusters[center].append(interval)
                found_cluster = True
                break

        if not found_cluster:
            interval_clusters[interval] = [interval]

    # Filter clusters by minimum count
    valid_clusters = {
        k: v for k, v in interval_clusters.items()
        if len(v) >= min_events_per_layer - 1
    }

    if len(valid_clusters) < 2:
        return None  # No polyrhythm detected

    # Calculate average intervals
    avg_intervals = sorted([
        sum(v) / len(v) for v in valid_clusters.values()
    ])

    # Try to find integer ratios
    base_interval = avg_intervals[0]
    ratios = []

    for interval in avg_intervals:
        ratio = base_interval / interval
        rounded_ratio = round(ratio)
        if abs(ratio - rounded_ratio) < 0.15 and rounded_ratio > 0:
            ratios.append(rounded_ratio)

    if len(ratios) >= 2:
        return PolyrhythmPattern(
            ratios=ratios,
            duration_beats=1.0,
        )

    return None


def generate_polyrhythm(
    ratios: List[int],
    tempo_bpm: float = 120.0,
    duration_beats: float = 4.0,
    notes: Optional[List[List[int]]] = None,
) -> Polyrhythm:
    """
    Generate a polyrhythm with specified parameters.

    Args:
        ratios: List of beat divisions (e.g., [3, 2])
        tempo_bpm: Tempo in BPM
        duration_beats: Duration in beats
        notes: Optional note assignments per layer

    Returns:
        Generated Polyrhythm object
    """
    pattern = PolyrhythmPattern(
        ratios=ratios,
        duration_beats=duration_beats,
    )

    # Default notes if not provided (percussion-like)
    if notes is None:
        notes = [
            [42],  # Hi-hat for first layer
            [36],  # Kick for second layer
        ]
        if len(ratios) > 2:
            notes.extend([[38]] * (len(ratios) - 2))  # Snare for others

    # Default velocities
    velocities = [[100] for _ in ratios]

    return Polyrhythm(
        pattern=pattern,
        tempo_bpm=tempo_bpm,
        notes=notes,
        velocities=velocities,
    )


def get_common_polyrhythms() -> Dict[str, PolyrhythmPattern]:
    """
    Get dictionary of common polyrhythms.

    Returns:
        Dict mapping name to PolyrhythmPattern
    """
    return {
        # Simple polyrhythms
        "3:2 (hemiola)": PolyrhythmPattern(
            ratios=[3, 2],
            name="Hemiola",
            accents=[[1.0, 0.6, 0.6], [1.0, 0.7]],
        ),
        "4:3": PolyrhythmPattern(
            ratios=[4, 3],
            name="4 against 3",
            accents=[[1.0, 0.5, 0.7, 0.5], [1.0, 0.6, 0.6]],
        ),
        "5:4": PolyrhythmPattern(
            ratios=[5, 4],
            name="5 against 4",
        ),
        "5:3": PolyrhythmPattern(
            ratios=[5, 3],
            name="5 against 3",
        ),
        "7:4": PolyrhythmPattern(
            ratios=[7, 4],
            name="7 against 4",
        ),

        # African/Cuban rhythms
        "6:4 (afro-cuban)": PolyrhythmPattern(
            ratios=[6, 4],
            name="Afro-Cuban 6/4",
            accents=[[1.0, 0.5, 0.7, 0.5, 0.7, 0.5], [1.0, 0.5, 0.8, 0.5]],
        ),

        # Complex polyrhythms
        "3:4:5": PolyrhythmPattern(
            ratios=[3, 4, 5],
            name="Triple polyrhythm",
        ),
        "2:3:4": PolyrhythmPattern(
            ratios=[2, 3, 4],
            name="2-3-4 layers",
        ),

        # Indian classical
        "7:3 (tisra-misra)": PolyrhythmPattern(
            ratios=[7, 3],
            name="Tisra-Misra",
        ),
    }


def create_polyrhythmic_grid(
    ratios: List[int],
    subdivisions: int = 48,
) -> List[List[bool]]:
    """
    Create a grid showing polyrhythm alignment.

    Args:
        ratios: List of beat divisions
        subdivisions: Grid resolution

    Returns:
        2D list where each row is a layer and columns are subdivisions
    """
    lcm = calculate_lcm(*ratios)
    grid = []

    for ratio in ratios:
        row = [False] * subdivisions
        step = subdivisions // ratio
        for i in range(ratio):
            pos = i * step
            if pos < subdivisions:
                row[pos] = True
        grid.append(row)

    return grid


def polyrhythm_to_notation(pattern: PolyrhythmPattern) -> str:
    """
    Convert polyrhythm to text notation.

    Args:
        pattern: PolyrhythmPattern to notate

    Returns:
        Text representation of the polyrhythm
    """
    if len(pattern.ratios) == 2:
        return f"{pattern.ratios[0]}:{pattern.ratios[1]}"
    else:
        return ":".join(str(r) for r in pattern.ratios)
