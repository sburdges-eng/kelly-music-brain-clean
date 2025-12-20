"""Pad Engine - Generates atmospheric pad textures."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
import math

TICKS_PER_BEAT = 480


class PadTexture(Enum):
    SUSTAINED = "sustained"
    EVOLVING = "evolving"
    PULSING = "pulsing"
    SWELLING = "swelling"
    TREMOLO = "tremolo"


@dataclass
class PadConfig:
    emotion: str = "neutral"
    chord_progression: List[str] = field(default_factory=list)
    key: str = "C"
    bars: int = 4
    tempo_bpm: int = 120
    texture_override: Optional[PadTexture] = None


@dataclass
class PadNote:
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int


@dataclass
class PadOutput:
    notes: List[PadNote]
    texture: PadTexture
    emotion: str


EMOTION_PROFILES = {
    "grief": {"texture": PadTexture.SUSTAINED, "velocity_range": (35, 55), "density": 0.8},
    "sadness": {"texture": PadTexture.SWELLING, "velocity_range": (40, 65), "density": 0.7},
    "anger": {"texture": PadTexture.TREMOLO, "velocity_range": (70, 100), "density": 0.9},
    "anxiety": {"texture": PadTexture.PULSING, "velocity_range": (50, 80), "density": 0.85},
    "joy": {"texture": PadTexture.EVOLVING, "velocity_range": (60, 90), "density": 0.75},
    "hope": {"texture": PadTexture.SWELLING, "velocity_range": (50, 80), "density": 0.7},
    "serenity": {"texture": PadTexture.SUSTAINED, "velocity_range": (30, 50), "density": 0.6},
    "emptiness": {"texture": PadTexture.SUSTAINED, "velocity_range": (20, 40), "density": 0.4},
}


class PadEngine:
    def __init__(self, seed: Optional[int] = None):
        if seed: random.seed(seed)
    
    def generate(self, config: PadConfig) -> PadOutput:
        profile = EMOTION_PROFILES.get(config.emotion.lower(), EMOTION_PROFILES["hope"])
        texture = config.texture_override or profile["texture"]
        notes = []
        ticks_per_bar = TICKS_PER_BEAT * 4
        
        for bar in range(config.bars):
            bar_start = bar * ticks_per_bar
            vel = random.randint(*profile["velocity_range"])
            # Simple pad voicing
            for pitch in [60, 64, 67]:  # C major triad
                notes.append(PadNote(pitch, bar_start, ticks_per_bar - 10, vel))
        
        return PadOutput(notes, texture, config.emotion)
