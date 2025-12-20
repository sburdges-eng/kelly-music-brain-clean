"""String Engine - Generates orchestral string arrangements."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random

TICKS_PER_BEAT = 480


class StringArticulation(Enum):
    LEGATO = "legato"
    STACCATO = "staccato"
    PIZZICATO = "pizzicato"
    TREMOLO = "tremolo"
    SPICCATO = "spiccato"


@dataclass
class StringConfig:
    emotion: str = "neutral"
    chord_progression: List[str] = field(default_factory=list)
    key: str = "C"
    bars: int = 4
    tempo_bpm: int = 120


@dataclass
class StringNote:
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int
    articulation: StringArticulation = StringArticulation.LEGATO


@dataclass
class StringOutput:
    notes: List[StringNote]
    articulation: StringArticulation
    emotion: str


EMOTION_PROFILES = {
    "grief": {"articulation": StringArticulation.LEGATO, "velocity_range": (40, 65)},
    "sadness": {"articulation": StringArticulation.LEGATO, "velocity_range": (45, 70)},
    "anger": {"articulation": StringArticulation.TREMOLO, "velocity_range": (80, 110)},
    "anxiety": {"articulation": StringArticulation.TREMOLO, "velocity_range": (55, 85)},
    "joy": {"articulation": StringArticulation.SPICCATO, "velocity_range": (70, 100)},
    "hope": {"articulation": StringArticulation.LEGATO, "velocity_range": (55, 85)},
}


class StringEngine:
    def __init__(self, seed: Optional[int] = None):
        if seed: random.seed(seed)
    
    def generate(self, config: StringConfig) -> StringOutput:
        profile = EMOTION_PROFILES.get(config.emotion.lower(), EMOTION_PROFILES["hope"])
        notes = []
        ticks_per_bar = TICKS_PER_BEAT * 4
        
        for bar in range(config.bars):
            bar_start = bar * ticks_per_bar
            vel = random.randint(*profile["velocity_range"])
            for pitch in [48, 55, 60, 64]:
                notes.append(StringNote(pitch, bar_start, ticks_per_bar - 10, vel, profile["articulation"]))
        
        return StringOutput(notes, profile["articulation"], config.emotion)
