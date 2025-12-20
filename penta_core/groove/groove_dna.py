"""
Groove DNA - Extract and apply timing fingerprints.

Provides:
- Groove extraction from MIDI/audio
- Timing deviation patterns (the "feel")
- Velocity curves and dynamics
- Groove comparison and matching
- Application to quantized material
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math


@dataclass
class GrooveDNA:
    """
    Timing and velocity fingerprint of a groove.

    The DNA captures micro-timing deviations and velocity patterns
    that give a groove its characteristic "feel".
    """
    # Timing deviations from grid (in ms, per beat position)
    timing_deviations: List[float] = field(default_factory=list)

    # Velocity pattern (0.0-1.0 scale)
    velocity_pattern: List[float] = field(default_factory=list)

    # Grid resolution (subdivisions per beat)
    grid_resolution: int = 16

    # Swing amount (0.0-1.0)
    swing_amount: float = 0.0

    # Tempo and time signature
    tempo_bpm: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)

    # Metadata
    name: str = ""
    source: str = ""  # Original source (file, artist, etc.)
    genre: str = ""
    tags: List[str] = field(default_factory=list)

    def get_deviation_at(self, position: float) -> float:
        """
        Get timing deviation at a specific beat position.

        Args:
            position: Position in beats

        Returns:
            Deviation in milliseconds
        """
        if not self.timing_deviations:
            return 0.0

        # Convert to grid position
        grid_pos = int((position % 1.0) * self.grid_resolution)
        grid_pos = grid_pos % len(self.timing_deviations)

        return self.timing_deviations[grid_pos]

    def get_velocity_at(self, position: float) -> float:
        """
        Get velocity multiplier at a specific beat position.

        Args:
            position: Position in beats

        Returns:
            Velocity multiplier (0.0-1.0)
        """
        if not self.velocity_pattern:
            return 1.0

        grid_pos = int((position % 1.0) * self.grid_resolution)
        grid_pos = grid_pos % len(self.velocity_pattern)

        return self.velocity_pattern[grid_pos]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timing_deviations": self.timing_deviations,
            "velocity_pattern": self.velocity_pattern,
            "grid_resolution": self.grid_resolution,
            "swing_amount": self.swing_amount,
            "tempo_bpm": self.tempo_bpm,
            "time_signature": list(self.time_signature),
            "name": self.name,
            "source": self.source,
            "genre": self.genre,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GrooveDNA":
        """Create from dictionary."""
        return cls(
            timing_deviations=data.get("timing_deviations", []),
            velocity_pattern=data.get("velocity_pattern", []),
            grid_resolution=data.get("grid_resolution", 16),
            swing_amount=data.get("swing_amount", 0.0),
            tempo_bpm=data.get("tempo_bpm", 120.0),
            time_signature=tuple(data.get("time_signature", [4, 4])),
            name=data.get("name", ""),
            source=data.get("source", ""),
            genre=data.get("genre", ""),
            tags=data.get("tags", []),
        )


def extract_groove_dna(
    events: List[Dict],
    grid_resolution: int = 16,
    tempo_bpm: float = 120.0,
) -> GrooveDNA:
    """
    Extract groove DNA from MIDI events.

    Args:
        events: List of MIDI events with 'time' and 'velocity' keys
        grid_resolution: Grid subdivisions per beat
        tempo_bpm: Tempo in BPM

    Returns:
        Extracted GrooveDNA
    """
    if not events:
        return GrooveDNA(grid_resolution=grid_resolution, tempo_bpm=tempo_bpm)

    # Sort by time
    sorted_events = sorted(events, key=lambda e: e.get("time", 0))

    # Calculate beat duration
    beat_duration = 60.0 / tempo_bpm
    grid_duration = beat_duration / grid_resolution

    # Collect deviations and velocities per grid position
    deviations_per_pos: Dict[int, List[float]] = {i: [] for i in range(grid_resolution)}
    velocities_per_pos: Dict[int, List[float]] = {i: [] for i in range(grid_resolution)}

    for event in sorted_events:
        time = event.get("time", 0)
        velocity = event.get("velocity", 100) / 127.0

        # Find nearest grid position
        beat_pos = (time / beat_duration) % 1.0
        grid_pos = int(beat_pos * grid_resolution)
        nearest_grid_time = grid_pos * grid_duration

        # Calculate deviation in ms
        actual_time_in_beat = (time % beat_duration)
        deviation_ms = (actual_time_in_beat - nearest_grid_time) * 1000

        grid_pos = grid_pos % grid_resolution
        deviations_per_pos[grid_pos].append(deviation_ms)
        velocities_per_pos[grid_pos].append(velocity)

    # Average deviations and velocities
    timing_deviations = []
    velocity_pattern = []

    for i in range(grid_resolution):
        devs = deviations_per_pos[i]
        vels = velocities_per_pos[i]

        timing_deviations.append(
            sum(devs) / len(devs) if devs else 0.0
        )
        velocity_pattern.append(
            sum(vels) / len(vels) if vels else 0.8
        )

    # Detect swing
    swing_amount = _detect_swing(timing_deviations, grid_resolution)

    return GrooveDNA(
        timing_deviations=timing_deviations,
        velocity_pattern=velocity_pattern,
        grid_resolution=grid_resolution,
        swing_amount=swing_amount,
        tempo_bpm=tempo_bpm,
    )


def _detect_swing(deviations: List[float], resolution: int) -> float:
    """
    Detect swing amount from timing deviations.

    Swing typically delays every other 8th/16th note.
    """
    if resolution < 4 or len(deviations) < 4:
        return 0.0

    # Check 8th note positions (every 2 grid positions for 16th note grid)
    step = resolution // 8
    if step < 1:
        return 0.0

    # Compare on-beat vs off-beat deviations
    on_beat_avg = 0.0
    off_beat_avg = 0.0
    on_count = 0
    off_count = 0

    for i in range(0, len(deviations), step):
        if (i // step) % 2 == 0:
            on_beat_avg += deviations[i]
            on_count += 1
        else:
            off_beat_avg += deviations[i]
            off_count += 1

    if on_count > 0:
        on_beat_avg /= on_count
    if off_count > 0:
        off_beat_avg /= off_count

    # Swing = positive delay on off-beats
    swing_deviation = off_beat_avg - on_beat_avg

    # Convert to 0-1 scale (assuming max swing is ~50ms delay)
    swing_amount = max(0.0, min(1.0, swing_deviation / 50.0))

    return swing_amount


def compare_grooves(groove_a: GrooveDNA, groove_b: GrooveDNA) -> float:
    """
    Compare two grooves and return similarity score.

    Args:
        groove_a: First groove
        groove_b: Second groove

    Returns:
        Similarity score (0.0-1.0, where 1.0 is identical)
    """
    # Normalize to same resolution
    res = min(groove_a.grid_resolution, groove_b.grid_resolution)

    # Resample if needed
    devs_a = _resample_pattern(groove_a.timing_deviations, res)
    devs_b = _resample_pattern(groove_b.timing_deviations, res)
    vels_a = _resample_pattern(groove_a.velocity_pattern, res)
    vels_b = _resample_pattern(groove_b.velocity_pattern, res)

    # Calculate timing similarity (correlation)
    timing_sim = _correlation(devs_a, devs_b)

    # Calculate velocity similarity
    velocity_sim = _correlation(vels_a, vels_b)

    # Swing similarity
    swing_sim = 1.0 - abs(groove_a.swing_amount - groove_b.swing_amount)

    # Weighted average
    return timing_sim * 0.5 + velocity_sim * 0.35 + swing_sim * 0.15


def _resample_pattern(pattern: List[float], target_length: int) -> List[float]:
    """Resample a pattern to target length."""
    if not pattern:
        return [0.0] * target_length

    if len(pattern) == target_length:
        return pattern.copy()

    result = []
    ratio = len(pattern) / target_length

    for i in range(target_length):
        source_idx = i * ratio
        lower_idx = int(source_idx)
        upper_idx = min(lower_idx + 1, len(pattern) - 1)
        frac = source_idx - lower_idx

        # Linear interpolation
        value = pattern[lower_idx] * (1 - frac) + pattern[upper_idx] * frac
        result.append(value)

    return result


def _correlation(a: List[float], b: List[float]) -> float:
    """Calculate correlation coefficient between two lists."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0

    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n

    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)

    if var_a == 0 or var_b == 0:
        return 1.0 if var_a == var_b else 0.0

    covariance = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    correlation = covariance / (math.sqrt(var_a) * math.sqrt(var_b))

    # Normalize to 0-1 range
    return (correlation + 1) / 2


def apply_groove_dna(
    events: List[Dict],
    groove: GrooveDNA,
    strength: float = 1.0,
    apply_timing: bool = True,
    apply_velocity: bool = True,
) -> List[Dict]:
    """
    Apply groove DNA to MIDI events.

    Args:
        events: List of MIDI events
        groove: GrooveDNA to apply
        strength: Application strength (0.0-1.0)
        apply_timing: Whether to apply timing deviations
        apply_velocity: Whether to apply velocity pattern

    Returns:
        Modified events list
    """
    if not events:
        return []

    beat_duration = 60.0 / groove.tempo_bpm
    result = []

    for event in events:
        new_event = event.copy()
        time = event.get("time", 0)
        velocity = event.get("velocity", 100)

        # Calculate beat position
        beat_pos = (time / beat_duration) % 1.0

        # Apply timing deviation
        if apply_timing:
            deviation_ms = groove.get_deviation_at(beat_pos)
            deviation_sec = (deviation_ms / 1000.0) * strength
            new_event["time"] = time + deviation_sec

        # Apply velocity pattern
        if apply_velocity:
            velocity_mult = groove.get_velocity_at(beat_pos)
            # Blend original velocity with groove velocity
            blended = velocity * (1 - strength) + (velocity * velocity_mult) * strength
            new_event["velocity"] = int(max(1, min(127, blended)))

        result.append(new_event)

    return result


def get_artist_groove_dna(artist: str) -> Optional[GrooveDNA]:
    """
    Get pre-defined groove DNA for famous artists/drummers.

    Args:
        artist: Artist or drummer name

    Returns:
        GrooveDNA or None if not found
    """
    artist_grooves = {
        "james_brown": GrooveDNA(
            name="James Brown Funk",
            timing_deviations=[0, -3, 2, -2, 0, 4, -1, 3,
                              0, -2, 3, -1, 0, 5, -2, 2],
            velocity_pattern=[1.0, 0.5, 0.7, 0.5, 0.9, 0.5, 0.6, 0.5,
                             0.95, 0.5, 0.7, 0.5, 0.85, 0.5, 0.65, 0.5],
            grid_resolution=16,
            swing_amount=0.1,
            genre="funk",
            tags=["tight", "syncopated", "pocket"],
        ),
        "john_bonham": GrooveDNA(
            name="John Bonham Rock",
            timing_deviations=[0, 0, -5, 0, 0, 0, 8, 0,
                              0, 0, -3, 0, 0, 0, 10, 0],
            velocity_pattern=[1.0, 0.6, 0.8, 0.6, 1.0, 0.6, 0.75, 0.6,
                             0.95, 0.6, 0.8, 0.6, 0.9, 0.6, 0.7, 0.6],
            grid_resolution=16,
            swing_amount=0.15,
            genre="rock",
            tags=["powerful", "behind-beat", "heavy"],
        ),
        "questlove": GrooveDNA(
            name="Questlove Neo-Soul",
            timing_deviations=[0, 5, -2, 6, 0, 8, -3, 5,
                              0, 6, -1, 7, 0, 9, -2, 4],
            velocity_pattern=[0.9, 0.55, 0.7, 0.5, 0.85, 0.5, 0.65, 0.55,
                             0.9, 0.5, 0.7, 0.55, 0.85, 0.55, 0.6, 0.5],
            grid_resolution=16,
            swing_amount=0.25,
            genre="neo-soul",
            tags=["laid-back", "swung", "organic"],
        ),
        "steve_gadd": GrooveDNA(
            name="Steve Gadd Studio",
            timing_deviations=[0, 0, 0, 0, 0, 2, 0, 1,
                              0, 0, 0, 0, 0, 3, 0, 1],
            velocity_pattern=[1.0, 0.65, 0.8, 0.6, 0.95, 0.6, 0.75, 0.6,
                             0.95, 0.6, 0.8, 0.65, 0.9, 0.6, 0.7, 0.6],
            grid_resolution=16,
            swing_amount=0.08,
            genre="studio",
            tags=["precise", "musical", "adaptable"],
        ),
        "clyde_stubblefield": GrooveDNA(
            name="Clyde Stubblefield Funky Drummer",
            timing_deviations=[0, -2, 4, -3, 0, 6, -2, 5,
                              0, -1, 5, -2, 0, 7, -3, 4],
            velocity_pattern=[1.0, 0.45, 0.75, 0.5, 0.9, 0.5, 0.7, 0.45,
                             0.95, 0.5, 0.75, 0.45, 0.85, 0.5, 0.65, 0.5],
            grid_resolution=16,
            swing_amount=0.12,
            genre="funk",
            tags=["iconic", "sampled", "deep-pocket"],
        ),
        "bernard_purdie": GrooveDNA(
            name="Bernard Purdie Shuffle",
            timing_deviations=[0, 8, 2, 10, 0, 12, 3, 9,
                              0, 7, 2, 11, 0, 13, 2, 8],
            velocity_pattern=[1.0, 0.5, 0.7, 0.55, 0.9, 0.5, 0.65, 0.5,
                             0.95, 0.5, 0.7, 0.5, 0.85, 0.55, 0.6, 0.5],
            grid_resolution=16,
            swing_amount=0.35,
            genre="soul",
            tags=["shuffle", "triplet-feel", "classic"],
        ),
        "j_dilla": GrooveDNA(
            name="J Dilla Drunk Beat",
            timing_deviations=[0, 15, -8, 20, 5, 18, -5, 12,
                              3, 22, -10, 15, 8, 25, -3, 10],
            velocity_pattern=[0.85, 0.5, 0.7, 0.55, 0.8, 0.45, 0.65, 0.5,
                             0.9, 0.5, 0.6, 0.5, 0.75, 0.55, 0.7, 0.45],
            grid_resolution=16,
            swing_amount=0.0,  # Not swing, just loose
            genre="hip-hop",
            tags=["drunk", "loose", "humanized"],
        ),
    }

    key = artist.lower().replace(" ", "_").replace("-", "_")
    return artist_grooves.get(key)


def blend_grooves(
    groove_a: GrooveDNA,
    groove_b: GrooveDNA,
    blend_amount: float = 0.5,
) -> GrooveDNA:
    """
    Blend two grooves together.

    Args:
        groove_a: First groove
        groove_b: Second groove
        blend_amount: Blend ratio (0.0 = all A, 1.0 = all B)

    Returns:
        Blended GrooveDNA
    """
    # Use higher resolution
    res = max(groove_a.grid_resolution, groove_b.grid_resolution)

    devs_a = _resample_pattern(groove_a.timing_deviations, res)
    devs_b = _resample_pattern(groove_b.timing_deviations, res)
    vels_a = _resample_pattern(groove_a.velocity_pattern, res)
    vels_b = _resample_pattern(groove_b.velocity_pattern, res)

    # Blend patterns
    timing_deviations = [
        devs_a[i] * (1 - blend_amount) + devs_b[i] * blend_amount
        for i in range(res)
    ]
    velocity_pattern = [
        vels_a[i] * (1 - blend_amount) + vels_b[i] * blend_amount
        for i in range(res)
    ]

    swing = groove_a.swing_amount * (1 - blend_amount) + groove_b.swing_amount * blend_amount
    tempo = groove_a.tempo_bpm * (1 - blend_amount) + groove_b.tempo_bpm * blend_amount

    return GrooveDNA(
        timing_deviations=timing_deviations,
        velocity_pattern=velocity_pattern,
        grid_resolution=res,
        swing_amount=swing,
        tempo_bpm=tempo,
        name=f"{groove_a.name} + {groove_b.name}",
    )
