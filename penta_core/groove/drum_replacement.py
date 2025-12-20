"""
Drum Replacement - Replace drum samples while preserving timing.

Provides:
- Drum hit detection and classification
- Sample mapping and replacement
- Timing preservation from original performance
- Velocity and dynamics matching
- Multi-layer sample triggering
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pathlib import Path


class DrumType(Enum):
    """Standard drum kit pieces."""
    KICK = "kick"
    SNARE = "snare"
    HIHAT_CLOSED = "hihat_closed"
    HIHAT_OPEN = "hihat_open"
    HIHAT_PEDAL = "hihat_pedal"
    TOM_HIGH = "tom_high"
    TOM_MID = "tom_mid"
    TOM_LOW = "tom_low"
    CRASH = "crash"
    RIDE = "ride"
    RIDE_BELL = "ride_bell"
    CLAP = "clap"
    RIM = "rim"
    COWBELL = "cowbell"
    PERCUSSION = "percussion"


# Standard General MIDI drum mapping
GM_DRUM_MAP: Dict[int, DrumType] = {
    35: DrumType.KICK,        # Acoustic Bass Drum
    36: DrumType.KICK,        # Bass Drum 1
    37: DrumType.RIM,         # Side Stick
    38: DrumType.SNARE,       # Acoustic Snare
    39: DrumType.CLAP,        # Hand Clap
    40: DrumType.SNARE,       # Electric Snare
    41: DrumType.TOM_LOW,     # Low Floor Tom
    42: DrumType.HIHAT_CLOSED,  # Closed Hi-Hat
    43: DrumType.TOM_LOW,     # High Floor Tom
    44: DrumType.HIHAT_PEDAL,   # Pedal Hi-Hat
    45: DrumType.TOM_MID,     # Low Tom
    46: DrumType.HIHAT_OPEN,    # Open Hi-Hat
    47: DrumType.TOM_MID,     # Low-Mid Tom
    48: DrumType.TOM_HIGH,    # Hi-Mid Tom
    49: DrumType.CRASH,       # Crash Cymbal 1
    50: DrumType.TOM_HIGH,    # High Tom
    51: DrumType.RIDE,        # Ride Cymbal 1
    52: DrumType.CRASH,       # Chinese Cymbal
    53: DrumType.RIDE_BELL,   # Ride Bell
    54: DrumType.PERCUSSION,  # Tambourine
    55: DrumType.CRASH,       # Splash Cymbal
    56: DrumType.COWBELL,     # Cowbell
    57: DrumType.CRASH,       # Crash Cymbal 2
    59: DrumType.RIDE,        # Ride Cymbal 2
}


@dataclass
class DrumSample:
    """
    A drum sample with metadata.
    """
    path: str
    drum_type: DrumType
    velocity_min: int = 1    # Minimum velocity to trigger
    velocity_max: int = 127  # Maximum velocity to trigger

    # Audio properties
    pitch: float = 0.0       # Pitch offset in semitones
    gain_db: float = 0.0     # Volume adjustment
    pan: float = 0.0         # Stereo position (-1 to 1)

    # Articulation
    articulation: str = "normal"  # normal, rimshot, sidestick, etc.

    def matches_velocity(self, velocity: int) -> bool:
        """Check if sample matches the given velocity."""
        return self.velocity_min <= velocity <= self.velocity_max


@dataclass
class DrumKit:
    """
    A collection of drum samples forming a kit.
    """
    name: str
    samples: Dict[DrumType, List[DrumSample]] = field(default_factory=dict)
    description: str = ""
    genre: str = ""

    def add_sample(self, sample: DrumSample) -> None:
        """Add a sample to the kit."""
        if sample.drum_type not in self.samples:
            self.samples[sample.drum_type] = []
        self.samples[sample.drum_type].append(sample)

    def get_sample(
        self,
        drum_type: DrumType,
        velocity: int = 100,
    ) -> Optional[DrumSample]:
        """
        Get appropriate sample for drum type and velocity.

        Uses velocity layers if available.
        """
        if drum_type not in self.samples:
            return None

        samples = self.samples[drum_type]

        # Find samples matching velocity
        matching = [s for s in samples if s.matches_velocity(velocity)]

        if matching:
            # Return random sample from matching velocity layer
            import random
            return random.choice(matching)

        # Fallback to any sample
        return samples[0] if samples else None


@dataclass
class DrumHit:
    """
    A detected drum hit.
    """
    time: float
    drum_type: DrumType
    velocity: int
    original_note: int
    duration: float = 0.1

    # Timing analysis
    timing_deviation_ms: float = 0.0
    grid_position: Optional[float] = None


@dataclass
class DrumReplacement:
    """
    Result of a drum replacement operation.
    """
    original_hits: List[DrumHit]
    replaced_hits: List[Dict]
    source_kit: Optional[str] = None
    target_kit: str = ""
    preserve_timing: bool = True
    preserve_velocity: bool = True

    def get_midi_events(self) -> List[Dict]:
        """Get replaced hits as MIDI events."""
        return self.replaced_hits


def map_drum_hits(
    events: List[Dict],
    drum_map: Optional[Dict[int, DrumType]] = None,
) -> List[DrumHit]:
    """
    Map MIDI events to drum hits.

    Args:
        events: List of MIDI events with 'note', 'time', 'velocity'
        drum_map: Custom note-to-drum mapping (defaults to GM)

    Returns:
        List of DrumHit objects
    """
    if drum_map is None:
        drum_map = GM_DRUM_MAP

    hits = []

    for event in events:
        note = event.get("note", 0)
        time = event.get("time", 0)
        velocity = event.get("velocity", 100)
        duration = event.get("duration", 0.1)

        # Map note to drum type
        drum_type = drum_map.get(note)
        if drum_type is None:
            # Try to infer from note number
            drum_type = _infer_drum_type(note)

        if drum_type:
            hits.append(DrumHit(
                time=time,
                drum_type=drum_type,
                velocity=velocity,
                original_note=note,
                duration=duration,
            ))

    return sorted(hits, key=lambda h: h.time)


def _infer_drum_type(note: int) -> Optional[DrumType]:
    """Infer drum type from MIDI note number."""
    if note < 35:
        return None
    elif note <= 36:
        return DrumType.KICK
    elif note <= 40:
        return DrumType.SNARE
    elif note <= 44:
        return DrumType.HIHAT_CLOSED
    elif note <= 46:
        return DrumType.HIHAT_OPEN
    elif note <= 50:
        return DrumType.TOM_MID
    elif note <= 53:
        return DrumType.RIDE
    elif note <= 57:
        return DrumType.CRASH
    else:
        return DrumType.PERCUSSION


def replace_drums(
    events: List[Dict],
    target_kit: DrumKit,
    preserve_timing: bool = True,
    preserve_velocity: bool = True,
    velocity_offset: int = 0,
    humanize_velocity: float = 0.0,
) -> DrumReplacement:
    """
    Replace drum sounds while preserving performance.

    Args:
        events: Original MIDI drum events
        target_kit: Target drum kit
        preserve_timing: Keep original timing deviations
        preserve_velocity: Keep original velocity dynamics
        velocity_offset: Global velocity adjustment
        humanize_velocity: Add velocity randomization (0.0-1.0)

    Returns:
        DrumReplacement result
    """
    import random

    # Map events to drum hits
    original_hits = map_drum_hits(events)

    replaced_hits = []

    for hit in original_hits:
        # Find replacement sample
        sample = target_kit.get_sample(hit.drum_type, hit.velocity)

        if sample is None:
            # No replacement available, skip or use original
            replaced_hits.append({
                "time": hit.time,
                "note": hit.original_note,
                "velocity": hit.velocity,
                "duration": hit.duration,
                "sample_path": None,
            })
            continue

        # Calculate velocity
        if preserve_velocity:
            velocity = hit.velocity + velocity_offset
        else:
            velocity = 100 + velocity_offset

        # Add humanization
        if humanize_velocity > 0:
            variation = random.gauss(0, humanize_velocity * 15)
            velocity = int(velocity + variation)

        velocity = max(1, min(127, velocity))

        replaced_hits.append({
            "time": hit.time,
            "note": hit.original_note,
            "velocity": velocity,
            "duration": hit.duration,
            "sample_path": sample.path,
            "drum_type": hit.drum_type.value,
            "pitch": sample.pitch,
            "gain_db": sample.gain_db,
            "pan": sample.pan,
        })

    return DrumReplacement(
        original_hits=original_hits,
        replaced_hits=replaced_hits,
        target_kit=target_kit.name,
        preserve_timing=preserve_timing,
        preserve_velocity=preserve_velocity,
    )


def preserve_timing(
    source_events: List[Dict],
    target_events: List[Dict],
    tempo_bpm: float = 120.0,
) -> List[Dict]:
    """
    Transfer timing from source performance to target.

    Args:
        source_events: Events with timing to preserve
        target_events: Events to apply timing to
        tempo_bpm: Tempo for grid calculation

    Returns:
        Target events with source timing applied
    """
    if not source_events or not target_events:
        return target_events

    beat_duration = 60.0 / tempo_bpm
    grid_step = beat_duration / 16  # 16th note grid

    # Calculate timing deviations from source
    source_deviations = {}
    for event in source_events:
        time = event.get("time", 0)
        grid_pos = round(time / grid_step)
        expected = grid_pos * grid_step
        deviation = time - expected
        source_deviations[grid_pos] = deviation

    # Apply deviations to target
    result = []
    for event in target_events:
        new_event = event.copy()
        time = event.get("time", 0)

        # Find nearest grid position
        grid_pos = round(time / grid_step)

        # Apply deviation if available
        if grid_pos in source_deviations:
            new_time = (grid_pos * grid_step) + source_deviations[grid_pos]
            new_event["time"] = new_time

        result.append(new_event)

    return result


def get_drum_samples(kit_name: str) -> Optional[DrumKit]:
    """
    Get a predefined drum kit by name.

    Args:
        kit_name: Name of the drum kit

    Returns:
        DrumKit or None if not found
    """
    # Note: In real implementation, these would load actual sample paths
    kits = {
        "acoustic_rock": DrumKit(
            name="Acoustic Rock",
            description="Natural acoustic rock kit",
            genre="rock",
            samples={
                DrumType.KICK: [
                    DrumSample("kick_soft.wav", DrumType.KICK, 1, 60),
                    DrumSample("kick_med.wav", DrumType.KICK, 61, 100),
                    DrumSample("kick_hard.wav", DrumType.KICK, 101, 127),
                ],
                DrumType.SNARE: [
                    DrumSample("snare_soft.wav", DrumType.SNARE, 1, 60),
                    DrumSample("snare_med.wav", DrumType.SNARE, 61, 100),
                    DrumSample("snare_hard.wav", DrumType.SNARE, 101, 127),
                ],
                DrumType.HIHAT_CLOSED: [
                    DrumSample("hihat_closed.wav", DrumType.HIHAT_CLOSED),
                ],
                DrumType.HIHAT_OPEN: [
                    DrumSample("hihat_open.wav", DrumType.HIHAT_OPEN),
                ],
            },
        ),
        "electronic_808": DrumKit(
            name="Electronic 808",
            description="Classic 808 sounds",
            genre="hip_hop",
            samples={
                DrumType.KICK: [
                    DrumSample("808_kick.wav", DrumType.KICK),
                ],
                DrumType.SNARE: [
                    DrumSample("808_snare.wav", DrumType.SNARE),
                ],
                DrumType.HIHAT_CLOSED: [
                    DrumSample("808_hihat.wav", DrumType.HIHAT_CLOSED),
                ],
                DrumType.CLAP: [
                    DrumSample("808_clap.wav", DrumType.CLAP),
                ],
            },
        ),
        "jazz_brush": DrumKit(
            name="Jazz Brush",
            description="Jazz kit with brush sounds",
            genre="jazz",
            samples={
                DrumType.KICK: [
                    DrumSample("jazz_kick.wav", DrumType.KICK),
                ],
                DrumType.SNARE: [
                    DrumSample("brush_swirl.wav", DrumType.SNARE, 1, 80),
                    DrumSample("brush_hit.wav", DrumType.SNARE, 81, 127),
                ],
                DrumType.HIHAT_CLOSED: [
                    DrumSample("jazz_hihat.wav", DrumType.HIHAT_CLOSED),
                ],
                DrumType.RIDE: [
                    DrumSample("jazz_ride.wav", DrumType.RIDE),
                ],
            },
        ),
        "metal_double": DrumKit(
            name="Metal Double",
            description="Heavy metal kit with double bass",
            genre="metal",
            samples={
                DrumType.KICK: [
                    DrumSample("metal_kick.wav", DrumType.KICK),
                ],
                DrumType.SNARE: [
                    DrumSample("metal_snare.wav", DrumType.SNARE),
                ],
                DrumType.HIHAT_CLOSED: [
                    DrumSample("metal_hihat.wav", DrumType.HIHAT_CLOSED),
                ],
                DrumType.TOM_HIGH: [
                    DrumSample("metal_tom_hi.wav", DrumType.TOM_HIGH),
                ],
                DrumType.TOM_MID: [
                    DrumSample("metal_tom_mid.wav", DrumType.TOM_MID),
                ],
                DrumType.TOM_LOW: [
                    DrumSample("metal_tom_low.wav", DrumType.TOM_LOW),
                ],
                DrumType.CRASH: [
                    DrumSample("metal_crash.wav", DrumType.CRASH),
                ],
            },
        ),
        "funk_tight": DrumKit(
            name="Funk Tight",
            description="Tight funky drum sounds",
            genre="funk",
            samples={
                DrumType.KICK: [
                    DrumSample("funk_kick.wav", DrumType.KICK),
                ],
                DrumType.SNARE: [
                    DrumSample("funk_snare_ghost.wav", DrumType.SNARE, 1, 50),
                    DrumSample("funk_snare.wav", DrumType.SNARE, 51, 127),
                ],
                DrumType.HIHAT_CLOSED: [
                    DrumSample("funk_hihat_closed.wav", DrumType.HIHAT_CLOSED),
                ],
                DrumType.HIHAT_OPEN: [
                    DrumSample("funk_hihat_open.wav", DrumType.HIHAT_OPEN),
                ],
            },
        ),
    }

    key = kit_name.lower().replace(" ", "_").replace("-", "_")
    return kits.get(key)


def list_available_kits() -> List[str]:
    """List all available drum kit names."""
    return [
        "acoustic_rock",
        "electronic_808",
        "jazz_brush",
        "metal_double",
        "funk_tight",
    ]


def create_custom_kit(
    name: str,
    samples_dict: Dict[str, List[str]],
) -> DrumKit:
    """
    Create a custom drum kit from sample paths.

    Args:
        name: Kit name
        samples_dict: Dict mapping drum type names to sample path lists

    Returns:
        Custom DrumKit
    """
    kit = DrumKit(name=name)

    for drum_name, paths in samples_dict.items():
        try:
            drum_type = DrumType(drum_name)
        except ValueError:
            continue

        for path in paths:
            kit.add_sample(DrumSample(
                path=path,
                drum_type=drum_type,
            ))

    return kit


def analyze_drum_pattern(
    events: List[Dict],
    tempo_bpm: float = 120.0,
    bars: int = 1,
) -> Dict:
    """
    Analyze drum pattern structure.

    Args:
        events: Drum MIDI events
        tempo_bpm: Tempo in BPM
        bars: Number of bars to analyze

    Returns:
        Analysis dict with pattern info
    """
    hits = map_drum_hits(events)
    beat_duration = 60.0 / tempo_bpm
    bar_duration = beat_duration * 4

    # Count hits per drum type
    hit_counts: Dict[DrumType, int] = {}
    for hit in hits:
        hit_counts[hit.drum_type] = hit_counts.get(hit.drum_type, 0) + 1

    # Find kick and snare pattern
    kick_positions = []
    snare_positions = []

    for hit in hits:
        beat = (hit.time / beat_duration) % 4
        if hit.drum_type == DrumType.KICK:
            kick_positions.append(round(beat, 2))
        elif hit.drum_type == DrumType.SNARE:
            snare_positions.append(round(beat, 2))

    # Calculate density
    total_duration = bar_duration * bars
    density = len(hits) / total_duration if total_duration > 0 else 0

    return {
        "total_hits": len(hits),
        "hit_counts": {k.value: v for k, v in hit_counts.items()},
        "kick_pattern": sorted(set(kick_positions)),
        "snare_pattern": sorted(set(snare_positions)),
        "density_per_second": density,
        "tempo_bpm": tempo_bpm,
    }
