"""
Genre-specific timing pockets and microtiming characteristics.

"Pocket" refers to the specific timing feel that defines a genre's groove.
Based on research by Honing, Keil, and Butterfield on microtiming.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum, auto


class SwingType(Enum):
    """Types of swing/shuffle feel."""
    STRAIGHT = auto()  # 50/50 eighth notes
    LIGHT_SWING = auto()  # 54/46 (bebop)
    MEDIUM_SWING = auto()  # 58/42 (big band)
    HARD_SWING = auto()  # 62/38 (New Orleans)
    SHUFFLE = auto()  # 67/33 (blues shuffle)
    HALF_TIME_SHUFFLE = auto()  # 67/33 with half-time feel
    DILLA_SWING = auto()  # 60-64/40-36 with laid-back feel


@dataclass
class TimingPocket:
    """
    Genre-specific timing characteristics.
    
    Attributes:
        swing_ratio: Ratio of first/second note in swing pair (0.5 = straight)
        kick_offset_ms: Milliseconds before (-) or after (+) the beat
        snare_offset_ms: Snare drum placement relative to beat
        hihat_offset_ms: Hi-hat placement (often ahead of beat)
        bass_offset_ms: Bass note placement
        humanization_variance: Random timing variation (±ms)
        push_pull_tendency: Overall tendency to rush (+) or drag (-)
    """
    swing_ratio: float  # 0.5 = straight, 0.67 = triplet shuffle
    kick_offset_ms: float = 0.0
    snare_offset_ms: float = 0.0
    hihat_offset_ms: float = 0.0
    bass_offset_ms: float = 0.0
    humanization_variance: float = 5.0  # ±5ms default
    push_pull_tendency: float = 0.0  # -10 to +10ms


# Genre timing pockets (based on empirical research)
GENRE_POCKETS: Dict[str, TimingPocket] = {
    # Hip-hop producers
    "dilla": TimingPocket(
        swing_ratio=0.62,  # J Dilla's signature "drunk" feel
        kick_offset_ms=20,  # Kicks slightly late (laid-back)
        snare_offset_ms=-12,  # Snares ahead (creates tension)
        hihat_offset_ms=-8,  # Hi-hats push forward
        bass_offset_ms=18,  # Bass follows kick
        humanization_variance=15.0,  # High variation
        push_pull_tendency=10.0,  # Overall dragging feel
    ),
    
    "madlib": TimingPocket(
        swing_ratio=0.58,
        kick_offset_ms=5,
        snare_offset_ms=-5,
        hihat_offset_ms=-10,
        bass_offset_ms=3,
        humanization_variance=12.0,
        push_pull_tendency=3.0,
    ),
    
    "alchemist": TimingPocket(
        swing_ratio=0.54,  # More subtle swing
        kick_offset_ms=-5,  # Kicks slightly early (urgent)
        snare_offset_ms=8,  # Snares late (heavy)
        hihat_offset_ms=-15,  # Hi-hats very forward
        bass_offset_ms=-3,
        humanization_variance=8.0,
        push_pull_tendency=-2.0,
    ),
    
    # Jazz styles
    "bebop": TimingPocket(
        swing_ratio=0.54,  # Light swing
        kick_offset_ms=-2,
        snare_offset_ms=3,
        hihat_offset_ms=-5,  # Ride cymbal ahead
        bass_offset_ms=-3,  # Walking bass slightly ahead
        humanization_variance=10.0,
        push_pull_tendency=-5.0,  # Tendency to rush
    ),
    
    "new_orleans": TimingPocket(
        swing_ratio=0.62,  # Hard swing
        kick_offset_ms=10,  # Laid-back kick
        snare_offset_ms=5,
        hihat_offset_ms=-8,
        bass_offset_ms=8,
        humanization_variance=12.0,
        push_pull_tendency=8.0,  # Dragging behind
    ),
    
    # Electronic
    "techno": TimingPocket(
        swing_ratio=0.50,  # Straight quantization
        kick_offset_ms=0,
        snare_offset_ms=0,
        hihat_offset_ms=0,
        bass_offset_ms=0,
        humanization_variance=0.0,  # No humanization
        push_pull_tendency=0.0,
    ),
    
    "house": TimingPocket(
        swing_ratio=0.50,
        kick_offset_ms=0,
        snare_offset_ms=0,
        hihat_offset_ms=-3,  # Hi-hats slightly ahead
        bass_offset_ms=0,
        humanization_variance=2.0,  # Minimal variation
        push_pull_tendency=0.0,
    ),
    
    "dnb": TimingPocket(
        swing_ratio=0.50,
        kick_offset_ms=-5,  # Kicks early for impact
        snare_offset_ms=-8,  # Snares very forward
        hihat_offset_ms=-10,
        bass_offset_ms=3,  # Bass slightly late
        humanization_variance=3.0,
        push_pull_tendency=-3.0,
    ),
    
    # Rock/funk
    "funk": TimingPocket(
        swing_ratio=0.52,  # Slight swing
        kick_offset_ms=5,
        snare_offset_ms=-10,  # Snare on top of beat
        hihat_offset_ms=-8,
        bass_offset_ms=-5,  # Bass locks with snare
        humanization_variance=8.0,
        push_pull_tendency=-3.0,
    ),
    
    "reggae": TimingPocket(
        swing_ratio=0.50,
        kick_offset_ms=20,  # Very laid-back kick
        snare_offset_ms=15,  # Late snare (one-drop)
        hihat_offset_ms=-5,
        bass_offset_ms=18,
        humanization_variance=10.0,
        push_pull_tendency=15.0,  # Heavy dragging
    ),
    
    # Blues/soul
    "shuffle": TimingPocket(
        swing_ratio=0.67,  # Triplet shuffle
        kick_offset_ms=8,
        snare_offset_ms=5,
        hihat_offset_ms=-3,
        bass_offset_ms=6,
        humanization_variance=12.0,
        push_pull_tendency=5.0,
    ),
    
    "motown": TimingPocket(
        swing_ratio=0.51,  # Very subtle swing
        kick_offset_ms=-2,
        snare_offset_ms=-5,  # Snare drives forward
        hihat_offset_ms=-8,
        bass_offset_ms=-3,  # Bass locks tight
        humanization_variance=6.0,
        push_pull_tendency=-2.0,
    ),
}


def get_genre_pocket(genre: str) -> Optional[TimingPocket]:
    """
    Get timing pocket for a specific genre or producer.
    
    Args:
        genre: Genre name or producer style (e.g., "dilla", "bebop", "techno")
    
    Returns:
        TimingPocket with microtiming characteristics, or None if not found
    
    Example:
        >>> pocket = get_genre_pocket("dilla")
        >>> print(pocket.swing_ratio)
        0.62
        >>> print(pocket.kick_offset_ms)
        20
    """
    return GENRE_POCKETS.get(genre.lower())


def apply_pocket_to_midi(
    midi_notes: list,
    pocket: TimingPocket,
    instrument_type: str = "kick"
) -> list:
    """
    Apply timing pocket to MIDI note data.
    
    Args:
        midi_notes: List of (pitch, time_ms, velocity) tuples
        pocket: TimingPocket to apply
        instrument_type: "kick", "snare", "hihat", or "bass"
    
    Returns:
        Modified MIDI notes with pocket timing applied
    """
    offset_map = {
        "kick": pocket.kick_offset_ms,
        "snare": pocket.snare_offset_ms,
        "hihat": pocket.hihat_offset_ms,
        "bass": pocket.bass_offset_ms,
    }
    
    offset = offset_map.get(instrument_type, 0.0)
    
    # Apply offset and humanization
    import random
    result = []
    for pitch, time_ms, velocity in midi_notes:
        variance = random.uniform(
            -pocket.humanization_variance,
            pocket.humanization_variance
        )
        new_time = time_ms + offset + variance + pocket.push_pull_tendency
        result.append((pitch, new_time, velocity))
    
    return result
