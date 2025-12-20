"""Transition Engine - Generates musical transitions between sections."""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
import random

TICKS_PER_BEAT = 480


class TransitionType(Enum):
    FILL = "fill"
    BREAKDOWN = "breakdown"
    BUILD = "build"
    DROP = "drop"
    CROSSFADE = "crossfade"
    RISER = "riser"
    SWEEP = "sweep"


@dataclass
class TransitionConfig:
    from_section: str = "verse"
    to_section: str = "chorus"
    duration_bars: int = 1
    transition_type: Optional[TransitionType] = None
    emotion: str = "neutral"


@dataclass
class TransitionOutput:
    transition_type: TransitionType
    notes: List[Dict]
    automation: List[Dict]


SECTION_TRANSITIONS = {
    ("verse", "chorus"): TransitionType.BUILD,
    ("chorus", "verse"): TransitionType.BREAKDOWN,
    ("verse", "bridge"): TransitionType.CROSSFADE,
    ("bridge", "chorus"): TransitionType.RISER,
    ("intro", "verse"): TransitionType.BUILD,
    ("chorus", "outro"): TransitionType.SWEEP,
}


class TransitionEngine:
    def __init__(self, seed: Optional[int] = None):
        if seed: random.seed(seed)
    
    def generate(self, config: TransitionConfig) -> TransitionOutput:
        transition_type = config.transition_type
        if not transition_type:
            key = (config.from_section.lower(), config.to_section.lower())
            transition_type = SECTION_TRANSITIONS.get(key, TransitionType.FILL)
        
        duration_ticks = config.duration_bars * TICKS_PER_BEAT * 4
        notes = []
        automation = []
        
        if transition_type == TransitionType.BUILD:
            # Rising velocity
            for i in range(8):
                tick = int(i * duration_ticks / 8)
                automation.append({"tick": tick, "param": "velocity", "value": 60 + i * 8})
        
        elif transition_type == TransitionType.BREAKDOWN:
            # Dropping elements
            automation.append({"tick": 0, "param": "filter_cutoff", "value": 127})
            automation.append({"tick": duration_ticks // 2, "param": "filter_cutoff", "value": 40})
        
        elif transition_type == TransitionType.RISER:
            # Sweep up
            for i in range(16):
                tick = int(i * duration_ticks / 16)
                automation.append({"tick": tick, "param": "pitch_bend", "value": i * 512})
        
        return TransitionOutput(transition_type, notes, automation)
