"""Fill Engine - Generates transitional fills."""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import random

TICKS_PER_BEAT = 480


class FillType(Enum):
    DRUM = "drum"
    MELODIC = "melodic"
    HARMONIC = "harmonic"


@dataclass
class FillConfig:
    emotion: str = "neutral"
    fill_type: FillType = FillType.DRUM
    duration_beats: int = 2


@dataclass
class FillNote:
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int


@dataclass
class FillOutput:
    notes: List[FillNote]
    fill_type: FillType


class FillEngine:
    def __init__(self, seed: Optional[int] = None):
        if seed: random.seed(seed)
    
    def generate(self, config: FillConfig) -> FillOutput:
        notes = []
        sixteenth = TICKS_PER_BEAT // 4
        num_notes = config.duration_beats * 4
        
        for i in range(num_notes):
            notes.append(FillNote(
                pitch=random.choice([38, 45, 47, 50]) if config.fill_type == FillType.DRUM else random.randint(60, 84),
                start_tick=i * sixteenth,
                duration_ticks=sixteenth - 10,
                velocity=random.randint(70, 100),
            ))
        
        return FillOutput(notes, config.fill_type)
