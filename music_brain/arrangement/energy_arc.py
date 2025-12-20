"""
Energy Arc Calculator - Map emotional intent to energy curves over time.

Creates energy progression curves based on:
- Narrative arc (climb, slow reveal, repetitive, etc.)
- Emotional journey
- Genre conventions
"""

import math
from dataclasses import dataclass
from enum import Enum


class NarrativeArc(Enum):
    """Narrative progression types from intent schema."""

    CLIMB_TO_CLIMAX = "climb-to-climax"  # Gradual build to peak
    SLOW_REVEAL = "slow-reveal"  # Subtle, delayed intensity
    REPETITIVE_DESPAIR = "repetitive-despair"  # Circular, no resolution
    SUDDEN_BREAK = "sudden-break"  # Calm → intense shift
    QUIET_ACCEPTANCE = "quiet-acceptance"  # Gentle, no climax
    EXPLOSIVE_START = "explosive-start"  # High energy → fade
    WAVE_PATTERN = "wave-pattern"  # Multiple peaks and valleys


@dataclass
class EnergyArc:
    """Energy curve over the course of a song."""

    narrative_arc: NarrativeArc
    energy_curve: list[float]  # Energy value per section (0.0-1.0)
    peak_position: float = 0.7  # Where peak occurs (0.0-1.0)
    intensity_range: tuple = (0.2, 0.9)  # Min/max energy

    def get_energy_at_position(self, position: float) -> float:
        """
        Get energy at a specific position in the song.

        Args:
            position: Position from 0.0 (start) to 1.0 (end)

        Returns:
            Energy level at that position
        """
        if not self.energy_curve:
            return 0.5

        # Interpolate between points
        index = position * (len(self.energy_curve) - 1)
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, len(self.energy_curve) - 1)

        if lower_idx == upper_idx:
            return self.energy_curve[lower_idx]

        # Linear interpolation
        frac = index - lower_idx
        lower_val = self.energy_curve[lower_idx]
        upper_val = self.energy_curve[upper_idx]

        return lower_val + (upper_val - lower_val) * frac

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "narrative_arc": self.narrative_arc.value,
            "energy_curve": self.energy_curve,
            "peak_position": self.peak_position,
            "intensity_range": self.intensity_range,
        }


def calculate_energy_curve(
    narrative_arc: NarrativeArc,
    num_sections: int,
    base_intensity: float = 0.6,
    peak_position: float | None = None,
) -> EnergyArc:
    """
    Calculate energy curve for a given narrative arc.

    Args:
        narrative_arc: Type of narrative progression
        num_sections: Number of sections in arrangement
        base_intensity: Overall intensity level (0.0-1.0)
        peak_position: Where peak should occur (None = use default)

    Returns:
        EnergyArc with calculated curve
    """
    # Determine intensity range based on base intensity
    min_energy = max(0.1, base_intensity - 0.3)
    max_energy = min(1.0, base_intensity + 0.3)

    # Use default peak position if not specified
    if peak_position is None:
        peak_position = _get_default_peak_position(narrative_arc)

    # Calculate curve based on arc type
    if narrative_arc == NarrativeArc.CLIMB_TO_CLIMAX:
        energy_curve = _climb_to_climax(
            num_sections, min_energy, max_energy, peak_position
        )

    elif narrative_arc == NarrativeArc.SLOW_REVEAL:
        energy_curve = _slow_reveal(num_sections, min_energy, max_energy)

    elif narrative_arc == NarrativeArc.REPETITIVE_DESPAIR:
        energy_curve = _repetitive_despair(num_sections, base_intensity)

    elif narrative_arc == NarrativeArc.SUDDEN_BREAK:
        energy_curve = _sudden_break(num_sections, min_energy, max_energy)

    elif narrative_arc == NarrativeArc.QUIET_ACCEPTANCE:
        energy_curve = _quiet_acceptance(num_sections, min_energy)

    elif narrative_arc == NarrativeArc.EXPLOSIVE_START:
        energy_curve = _explosive_start(num_sections, min_energy, max_energy)

    elif narrative_arc == NarrativeArc.WAVE_PATTERN:
        energy_curve = _wave_pattern(num_sections, min_energy, max_energy)

    else:
        # Default linear growth
        energy_curve = _linear_growth(num_sections, min_energy, max_energy)

    return EnergyArc(
        narrative_arc=narrative_arc,
        energy_curve=energy_curve,
        peak_position=peak_position,
        intensity_range=(min_energy, max_energy),
    )


# =================================================================
# ARC SHAPE FUNCTIONS
# =================================================================


def _get_default_peak_position(arc: NarrativeArc) -> float:
    """Get default peak position for each arc type."""
    defaults = {
        NarrativeArc.CLIMB_TO_CLIMAX: 0.75,
        NarrativeArc.SLOW_REVEAL: 0.85,
        NarrativeArc.REPETITIVE_DESPAIR: 0.5,
        NarrativeArc.SUDDEN_BREAK: 0.6,
        NarrativeArc.QUIET_ACCEPTANCE: 0.3,
        NarrativeArc.EXPLOSIVE_START: 0.1,
        NarrativeArc.WAVE_PATTERN: 0.7,
    }
    return defaults.get(arc, 0.7)


def _climb_to_climax(
    num_sections: int,
    min_energy: float,
    max_energy: float,
    peak_pos: float,
) -> list[float]:
    """Gradual build to peak, then slight decline."""
    curve = []
    peak_idx = int(num_sections * peak_pos)

    for i in range(num_sections):
        if i < peak_idx:
            # Build to peak
            progress = i / max(peak_idx, 1)
            # Use exponential curve for more dramatic build
            energy = min_energy + (max_energy - min_energy) * (progress**1.5)
        else:
            # Slight decline after peak
            progress = (i - peak_idx) / max(num_sections - peak_idx, 1)
            energy = max_energy - (max_energy - min_energy) * 0.2 * progress

        curve.append(energy)

    return curve


def _slow_reveal(
    num_sections: int, min_energy: float, max_energy: float
) -> list[float]:
    """Stay low for most of song, late intensity surge."""
    curve = []
    reveal_point = int(num_sections * 0.7)

    for i in range(num_sections):
        if i < reveal_point:
            # Stay relatively flat and low
            progress = i / max(reveal_point, 1)
            energy = min_energy + (max_energy - min_energy) * 0.2 * progress
        else:
            # Rapid rise to full intensity
            progress = (i - reveal_point) / max(num_sections - reveal_point, 1)
            energy = min_energy + (max_energy - min_energy) * (
                0.2 + 0.8 * (progress**0.5)
            )

        curve.append(energy)

    return curve


def _repetitive_despair(num_sections: int, base_intensity: float) -> list[float]:
    """Circular, no real progression or resolution."""
    curve = []
    # Small oscillation around base intensity
    for i in range(num_sections):
        # Sine wave with small amplitude
        phase = (i / num_sections) * 2 * math.pi
        variation = 0.1 * math.sin(phase * 2)  # Two cycles
        energy = base_intensity + variation
        curve.append(max(0.2, min(0.8, energy)))

    return curve


def _sudden_break(
    num_sections: int, min_energy: float, max_energy: float
) -> list[float]:
    """Calm opening, sudden shift to intensity."""
    curve = []
    break_point = int(num_sections * 0.4)

    for i in range(num_sections):
        if i < break_point:
            # Stay calm
            energy = min_energy
        else:
            # Sudden jump to high energy
            energy = max_energy * 0.95

        curve.append(energy)

    return curve


def _quiet_acceptance(num_sections: int, min_energy: float) -> list[float]:
    """Gentle throughout, no dramatic peaks."""
    curve = []
    # Gradual very slight increase, stay mostly low
    for i in range(num_sections):
        progress = i / max(num_sections - 1, 1)
        energy = min_energy + 0.15 * progress
        curve.append(min(0.5, energy))

    return curve


def _explosive_start(
    num_sections: int, min_energy: float, max_energy: float
) -> list[float]:
    """Start intense, gradually fade out."""
    curve = []
    for i in range(num_sections):
        progress = i / max(num_sections - 1, 1)
        # Exponential decay
        energy = max_energy - (max_energy - min_energy) * (progress**0.7)
        curve.append(energy)

    return curve


def _wave_pattern(
    num_sections: int, min_energy: float, max_energy: float
) -> list[float]:
    """Multiple peaks and valleys."""
    curve = []
    num_waves = 2  # Two major waves

    for i in range(num_sections):
        # Sine wave with upward trend
        phase = (i / num_sections) * num_waves * 2 * math.pi
        wave = (math.sin(phase) + 1) / 2  # Normalize to 0-1

        # Add upward trend
        trend = i / max(num_sections - 1, 1) * 0.3

        energy = min_energy + (max_energy - min_energy) * (wave * 0.7 + trend)
        curve.append(energy)

    return curve


def _linear_growth(
    num_sections: int, min_energy: float, max_energy: float
) -> list[float]:
    """Simple linear growth."""
    curve = []
    for i in range(num_sections):
        progress = i / max(num_sections - 1, 1)
        energy = min_energy + (max_energy - min_energy) * progress
        curve.append(energy)

    return curve


# =================================================================
# UTILITY FUNCTIONS
# =================================================================


def smooth_energy_curve(curve: list[float], smoothing: float = 0.3) -> list[float]:
    """
    Smooth energy curve to avoid sudden jumps.

    Args:
        curve: Energy curve to smooth
        smoothing: Smoothing factor (0.0 = no smoothing, 1.0 = max smoothing)

    Returns:
        Smoothed curve
    """
    if len(curve) < 3 or smoothing <= 0:
        return curve

    smoothed = [curve[0]]  # Keep first value

    for i in range(1, len(curve) - 1):
        # Simple moving average
        avg = (curve[i - 1] + curve[i] + curve[i + 1]) / 3
        # Blend original and averaged
        smoothed_val = curve[i] * (1 - smoothing) + avg * smoothing
        smoothed.append(smoothed_val)

    smoothed.append(curve[-1])  # Keep last value

    return smoothed


def map_emotion_to_arc(
    primary_emotion: str,
    secondary_tension: float = 0.0,
) -> NarrativeArc:
    """
    Map emotional state to narrative arc.

    Args:
        primary_emotion: Primary emotion (grief, anxiety, etc.)
        secondary_tension: Internal conflict level (0.0-1.0)

    Returns:
        Recommended narrative arc
    """
    emotion_arcs = {
        "grief": NarrativeArc.SLOW_REVEAL,
        "anxiety": NarrativeArc.WAVE_PATTERN,
        "anger": NarrativeArc.EXPLOSIVE_START,
        "nostalgia": NarrativeArc.QUIET_ACCEPTANCE,
        "hope": NarrativeArc.CLIMB_TO_CLIMAX,
        "despair": NarrativeArc.REPETITIVE_DESPAIR,
        "triumph": NarrativeArc.CLIMB_TO_CLIMAX,
        "fear": NarrativeArc.SUDDEN_BREAK,
        "joy": NarrativeArc.WAVE_PATTERN,
        "calm": NarrativeArc.QUIET_ACCEPTANCE,
    }

    # Get base arc from emotion
    base_arc = emotion_arcs.get(primary_emotion.lower(), NarrativeArc.CLIMB_TO_CLIMAX)

    # Modify based on secondary tension
    if secondary_tension > 0.7:
        # High tension -> more dramatic arcs
        if base_arc == NarrativeArc.QUIET_ACCEPTANCE:
            return NarrativeArc.SUDDEN_BREAK
        elif base_arc == NarrativeArc.SLOW_REVEAL:
            return NarrativeArc.CLIMB_TO_CLIMAX

    return base_arc
