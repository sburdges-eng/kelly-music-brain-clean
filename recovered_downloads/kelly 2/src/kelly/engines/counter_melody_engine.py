"""Counter Melody Engine - Generates complementary melodic lines."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import random

TICKS_PER_BEAT = 480


class CounterTechnique(Enum):
    THIRDS = "thirds"
    SIXTHS = "sixths"
    CONTRARY = "contrary"
    IMITATION = "imitation"
    FILL = "fill"


@dataclass
class CounterNote:
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int
    interval_from_source: int = 0


@dataclass
class CounterMelodyOutput:
    notes: List[CounterNote]
    technique: CounterTechnique
    emotion: str


class CounterMelodyEngine:
    def __init__(self, seed: Optional[int] = None):
        if seed: random.seed(seed)
    
    def generate(
        self,
        main_melody: List[Dict],
        emotion: str = "neutral",
        technique: Optional[CounterTechnique] = None
    ) -> CounterMelodyOutput:
        technique = technique or CounterTechnique.THIRDS
        notes = []
        
        interval = 4 if technique == CounterTechnique.THIRDS else 9
        if technique == CounterTechnique.CONTRARY:
            interval = -interval
        
        for note in main_melody:
            counter_pitch = note.get("pitch", 60) + interval
            notes.append(CounterNote(
                pitch=counter_pitch,
                start_tick=note.get("start_tick", 0),
                duration_ticks=note.get("duration_ticks", TICKS_PER_BEAT),
                velocity=int(note.get("velocity", 70) * 0.8),
                interval_from_source=interval,
            ))
        
        return CounterMelodyOutput(notes, technique, emotion)
