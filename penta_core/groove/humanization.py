"""
Humanization - Add human feel to quantized MIDI.

Provides:
- Timing randomization with musical constraints
- Velocity variation patterns
- Artist-style humanization presets
- Genre-specific humanization
- Custom preset creation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import random
import math


class HumanizationStyle(Enum):
    """Predefined humanization styles."""
    TIGHT = "tight"          # Minimal variation, studio precision
    LOOSE = "loose"          # Relaxed, laid-back feel
    DRUNK = "drunk"          # Heavy timing variation (J Dilla style)
    MECHANICAL = "mechanical"  # Very tight, almost robotic
    LIVE = "live"            # Natural live performance variation
    SHUFFLE = "shuffle"      # Shuffle/triplet feel
    SWING = "swing"          # Jazz swing feel


@dataclass
class HumanizationPreset:
    """
    Configuration for humanization parameters.
    """
    name: str

    # Timing parameters (in milliseconds)
    timing_random_ms: float = 5.0       # Random timing variation
    timing_drift_ms: float = 0.0        # Gradual timing drift
    timing_ahead_bias: float = 0.0      # -1 to 1 (behind to ahead)

    # Velocity parameters (0.0-1.0 scale)
    velocity_random: float = 0.05       # Random velocity variation
    velocity_curve: float = 0.0         # Accent curve (-1 to 1)
    velocity_dynamic_range: float = 1.0 # Dynamic range compression/expansion

    # Swing/shuffle
    swing_amount: float = 0.0           # Swing ratio (0.0-1.0)
    swing_grid: int = 8                 # Grid for swing (8 = 8th notes)

    # Note-specific adjustments
    accent_beats: List[int] = field(default_factory=list)  # Beats to accent
    ghost_notes: bool = False           # Add ghost notes
    ghost_velocity: float = 0.3         # Velocity of ghost notes

    # Advanced
    humanize_duration: bool = True      # Vary note durations
    duration_random: float = 0.1        # Duration variation (0.0-1.0)
    correlation: float = 0.3            # Correlation between consecutive notes

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timing_random_ms": self.timing_random_ms,
            "timing_drift_ms": self.timing_drift_ms,
            "timing_ahead_bias": self.timing_ahead_bias,
            "velocity_random": self.velocity_random,
            "velocity_curve": self.velocity_curve,
            "velocity_dynamic_range": self.velocity_dynamic_range,
            "swing_amount": self.swing_amount,
            "swing_grid": self.swing_grid,
            "accent_beats": self.accent_beats,
            "ghost_notes": self.ghost_notes,
            "ghost_velocity": self.ghost_velocity,
            "humanize_duration": self.humanize_duration,
            "duration_random": self.duration_random,
            "correlation": self.correlation,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HumanizationPreset":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Custom"),
            timing_random_ms=data.get("timing_random_ms", 5.0),
            timing_drift_ms=data.get("timing_drift_ms", 0.0),
            timing_ahead_bias=data.get("timing_ahead_bias", 0.0),
            velocity_random=data.get("velocity_random", 0.05),
            velocity_curve=data.get("velocity_curve", 0.0),
            velocity_dynamic_range=data.get("velocity_dynamic_range", 1.0),
            swing_amount=data.get("swing_amount", 0.0),
            swing_grid=data.get("swing_grid", 8),
            accent_beats=data.get("accent_beats", []),
            ghost_notes=data.get("ghost_notes", False),
            ghost_velocity=data.get("ghost_velocity", 0.3),
            humanize_duration=data.get("humanize_duration", True),
            duration_random=data.get("duration_random", 0.1),
            correlation=data.get("correlation", 0.3),
        )


def humanize_midi(
    events: List[Dict],
    preset: HumanizationPreset,
    tempo_bpm: float = 120.0,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Apply humanization to MIDI events.

    Args:
        events: List of MIDI events with 'time', 'velocity', 'duration'
        preset: HumanizationPreset to apply
        tempo_bpm: Tempo in BPM
        seed: Random seed for reproducibility

    Returns:
        Humanized events
    """
    if not events:
        return []

    if seed is not None:
        random.seed(seed)

    beat_duration = 60.0 / tempo_bpm
    result = []
    prev_timing_offset = 0.0

    for i, event in enumerate(events):
        new_event = event.copy()
        time = event.get("time", 0)
        velocity = event.get("velocity", 100)
        duration = event.get("duration", beat_duration / 2)

        # Calculate beat position
        beat_number = int(time / beat_duration)
        beat_fraction = (time / beat_duration) % 1.0

        # --- Timing Humanization ---

        # Random timing variation with correlation
        if preset.correlation > 0 and i > 0:
            timing_offset = (
                prev_timing_offset * preset.correlation +
                random.gauss(0, preset.timing_random_ms) * (1 - preset.correlation)
            )
        else:
            timing_offset = random.gauss(0, preset.timing_random_ms)

        # Apply timing drift
        timing_offset += preset.timing_drift_ms * (i / max(len(events), 1))

        # Apply ahead/behind bias
        timing_offset += preset.timing_ahead_bias * preset.timing_random_ms

        # Apply swing
        if preset.swing_amount > 0:
            swing_offset = _calculate_swing_offset(
                beat_fraction, preset.swing_amount, preset.swing_grid, beat_duration
            )
            timing_offset += swing_offset * 1000  # Convert to ms

        new_event["time"] = time + (timing_offset / 1000.0)
        prev_timing_offset = timing_offset

        # --- Velocity Humanization ---

        # Random velocity variation
        velocity_offset = random.gauss(0, preset.velocity_random * 127)

        # Apply velocity curve (accent certain positions)
        if preset.velocity_curve != 0:
            curve_factor = math.sin(beat_fraction * math.pi * 2) * preset.velocity_curve
            velocity_offset += curve_factor * 20

        # Apply accent beats
        if preset.accent_beats and (beat_number % len(preset.accent_beats)) in preset.accent_beats:
            velocity_offset += 15

        # Apply dynamic range
        avg_velocity = 80
        velocity_diff = velocity - avg_velocity
        velocity = avg_velocity + (velocity_diff * preset.velocity_dynamic_range)
        velocity += velocity_offset

        new_event["velocity"] = int(max(1, min(127, velocity)))

        # --- Duration Humanization ---

        if preset.humanize_duration:
            duration_variation = 1.0 + random.gauss(0, preset.duration_random)
            new_event["duration"] = duration * max(0.5, min(1.5, duration_variation))

        result.append(new_event)

    # Add ghost notes if enabled
    if preset.ghost_notes:
        ghost_events = _generate_ghost_notes(result, preset, tempo_bpm)
        result.extend(ghost_events)
        result.sort(key=lambda e: e.get("time", 0))

    return result


def _calculate_swing_offset(
    beat_fraction: float,
    swing_amount: float,
    swing_grid: int,
    beat_duration: float,
) -> float:
    """
    Calculate swing timing offset.

    Args:
        beat_fraction: Position within beat (0.0-1.0)
        swing_amount: Swing amount (0.0-1.0)
        swing_grid: Grid resolution for swing
        beat_duration: Beat duration in seconds

    Returns:
        Offset in seconds
    """
    # Swing affects off-beat positions
    grid_position = beat_fraction * swing_grid
    is_off_beat = (int(grid_position) % 2) == 1

    if is_off_beat:
        # Maximum swing offset is half grid step
        max_offset = beat_duration / swing_grid / 2
        return swing_amount * max_offset

    return 0.0


def _generate_ghost_notes(
    events: List[Dict],
    preset: HumanizationPreset,
    tempo_bpm: float,
) -> List[Dict]:
    """Generate ghost notes between main events."""
    ghost_events = []
    beat_duration = 60.0 / tempo_bpm
    grid_step = beat_duration / 16  # 16th note grid

    for i, event in enumerate(events):
        # 20% chance to add ghost note before main note
        if random.random() < 0.2:
            ghost_time = event["time"] - grid_step
            if ghost_time > 0:
                ghost_events.append({
                    "time": ghost_time + random.gauss(0, 0.005),
                    "note": event.get("note", 60),
                    "velocity": int(event.get("velocity", 100) * preset.ghost_velocity),
                    "duration": grid_step * 0.5,
                    "is_ghost": True,
                })

    return ghost_events


def get_style_preset(style: HumanizationStyle) -> HumanizationPreset:
    """
    Get preset for a humanization style.

    Args:
        style: HumanizationStyle enum value

    Returns:
        Corresponding HumanizationPreset
    """
    presets = {
        HumanizationStyle.TIGHT: HumanizationPreset(
            name="Tight",
            timing_random_ms=3.0,
            velocity_random=0.03,
            correlation=0.5,
        ),
        HumanizationStyle.LOOSE: HumanizationPreset(
            name="Loose",
            timing_random_ms=12.0,
            timing_ahead_bias=-0.3,
            velocity_random=0.1,
            correlation=0.2,
        ),
        HumanizationStyle.DRUNK: HumanizationPreset(
            name="Drunk",
            timing_random_ms=25.0,
            velocity_random=0.15,
            correlation=0.1,
            humanize_duration=True,
            duration_random=0.2,
        ),
        HumanizationStyle.MECHANICAL: HumanizationPreset(
            name="Mechanical",
            timing_random_ms=1.0,
            velocity_random=0.01,
            correlation=0.8,
        ),
        HumanizationStyle.LIVE: HumanizationPreset(
            name="Live",
            timing_random_ms=8.0,
            timing_drift_ms=2.0,
            velocity_random=0.08,
            ghost_notes=True,
            ghost_velocity=0.25,
        ),
        HumanizationStyle.SHUFFLE: HumanizationPreset(
            name="Shuffle",
            timing_random_ms=5.0,
            swing_amount=0.67,  # Triplet feel
            swing_grid=8,
            velocity_random=0.06,
        ),
        HumanizationStyle.SWING: HumanizationPreset(
            name="Swing",
            timing_random_ms=6.0,
            swing_amount=0.55,
            swing_grid=8,
            velocity_random=0.08,
            accent_beats=[0, 2],  # Accent 1 and 3
        ),
    }

    return presets.get(style, presets[HumanizationStyle.TIGHT])


def get_artist_preset(artist: str) -> Optional[HumanizationPreset]:
    """
    Get humanization preset based on famous drummer/artist style.

    Args:
        artist: Artist or drummer name

    Returns:
        HumanizationPreset or None if not found
    """
    artist_presets = {
        "john_bonham": HumanizationPreset(
            name="John Bonham",
            timing_random_ms=10.0,
            timing_ahead_bias=-0.4,  # Behind the beat
            velocity_random=0.12,
            velocity_dynamic_range=1.3,
            ghost_notes=True,
            ghost_velocity=0.35,
        ),
        "questlove": HumanizationPreset(
            name="Questlove",
            timing_random_ms=15.0,
            timing_ahead_bias=-0.5,  # Very laid back
            swing_amount=0.3,
            velocity_random=0.1,
            correlation=0.4,
        ),
        "steve_gadd": HumanizationPreset(
            name="Steve Gadd",
            timing_random_ms=4.0,
            velocity_random=0.05,
            ghost_notes=True,
            ghost_velocity=0.2,
            correlation=0.6,
        ),
        "j_dilla": HumanizationPreset(
            name="J Dilla",
            timing_random_ms=30.0,
            velocity_random=0.15,
            correlation=0.15,
            humanize_duration=True,
            duration_random=0.25,
        ),
        "bernard_purdie": HumanizationPreset(
            name="Bernard Purdie",
            timing_random_ms=7.0,
            swing_amount=0.4,
            velocity_random=0.08,
            ghost_notes=True,
            ghost_velocity=0.3,
        ),
        "clyde_stubblefield": HumanizationPreset(
            name="Clyde Stubblefield",
            timing_random_ms=6.0,
            velocity_random=0.1,
            velocity_curve=0.2,
            accent_beats=[0],
        ),
        "travis_barker": HumanizationPreset(
            name="Travis Barker",
            timing_random_ms=3.0,
            timing_ahead_bias=0.2,  # Slightly ahead
            velocity_random=0.08,
            velocity_dynamic_range=1.4,
        ),
        "chad_smith": HumanizationPreset(
            name="Chad Smith",
            timing_random_ms=7.0,
            velocity_random=0.1,
            ghost_notes=True,
            ghost_velocity=0.25,
            correlation=0.4,
        ),
    }

    key = artist.lower().replace(" ", "_").replace("-", "_")
    return artist_presets.get(key)


def get_genre_preset(genre: str) -> Optional[HumanizationPreset]:
    """
    Get humanization preset for a specific genre.

    Args:
        genre: Genre name

    Returns:
        HumanizationPreset or None if not found
    """
    genre_presets = {
        "rock": HumanizationPreset(
            name="Rock",
            timing_random_ms=8.0,
            velocity_random=0.1,
            velocity_dynamic_range=1.2,
            accent_beats=[0, 2],
        ),
        "jazz": HumanizationPreset(
            name="Jazz",
            timing_random_ms=10.0,
            swing_amount=0.5,
            velocity_random=0.12,
            ghost_notes=True,
            ghost_velocity=0.25,
        ),
        "funk": HumanizationPreset(
            name="Funk",
            timing_random_ms=5.0,
            velocity_random=0.08,
            velocity_curve=0.15,
            ghost_notes=True,
            ghost_velocity=0.35,
        ),
        "hip_hop": HumanizationPreset(
            name="Hip Hop",
            timing_random_ms=15.0,
            timing_ahead_bias=-0.3,
            velocity_random=0.1,
            correlation=0.3,
        ),
        "edm": HumanizationPreset(
            name="EDM",
            timing_random_ms=2.0,
            velocity_random=0.03,
            correlation=0.7,
        ),
        "metal": HumanizationPreset(
            name="Metal",
            timing_random_ms=4.0,
            timing_ahead_bias=0.1,
            velocity_random=0.05,
            velocity_dynamic_range=0.9,
        ),
        "r_and_b": HumanizationPreset(
            name="R&B",
            timing_random_ms=12.0,
            timing_ahead_bias=-0.4,
            swing_amount=0.2,
            velocity_random=0.1,
        ),
        "country": HumanizationPreset(
            name="Country",
            timing_random_ms=6.0,
            velocity_random=0.07,
            accent_beats=[0, 2],
        ),
        "latin": HumanizationPreset(
            name="Latin",
            timing_random_ms=5.0,
            velocity_random=0.1,
            velocity_curve=0.2,
        ),
        "reggae": HumanizationPreset(
            name="Reggae",
            timing_random_ms=10.0,
            timing_ahead_bias=-0.5,
            velocity_random=0.08,
            accent_beats=[1, 3],  # Accent offbeats
        ),
    }

    key = genre.lower().replace(" ", "_").replace("-", "_").replace("&", "and")
    return genre_presets.get(key)


def create_custom_preset(
    name: str,
    base_style: Optional[HumanizationStyle] = None,
    **overrides,
) -> HumanizationPreset:
    """
    Create a custom humanization preset.

    Args:
        name: Preset name
        base_style: Optional style to use as base
        **overrides: Parameter overrides

    Returns:
        Custom HumanizationPreset
    """
    if base_style:
        preset = get_style_preset(base_style)
        preset_dict = preset.to_dict()
    else:
        preset_dict = HumanizationPreset(name="Base").to_dict()

    preset_dict["name"] = name
    preset_dict.update(overrides)

    return HumanizationPreset.from_dict(preset_dict)


def list_available_presets() -> Dict[str, List[str]]:
    """
    List all available humanization presets.

    Returns:
        Dict with categories and preset names
    """
    return {
        "styles": [s.value for s in HumanizationStyle],
        "artists": [
            "john_bonham", "questlove", "steve_gadd", "j_dilla",
            "bernard_purdie", "clyde_stubblefield", "travis_barker", "chad_smith"
        ],
        "genres": [
            "rock", "jazz", "funk", "hip_hop", "edm", "metal",
            "r_and_b", "country", "latin", "reggae"
        ],
    }
