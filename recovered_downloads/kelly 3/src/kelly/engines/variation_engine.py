"""Variation Engine - Creates variations of musical material."""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import random

TICKS_PER_BEAT = 480


class VariationType(Enum):
    RHYTHMIC = "rhythmic"
    MELODIC = "melodic"
    HARMONIC = "harmonic"
    DYNAMIC = "dynamic"
    ORNAMENTATION = "ornamentation"


@dataclass
class VariationConfig:
    variation_type: VariationType = VariationType.MELODIC
    intensity: float = 0.5
    preserve_rhythm: bool = True
    preserve_contour: bool = True


class VariationEngine:
    def __init__(self, seed: Optional[int] = None):
        if seed: random.seed(seed)
    
    def create_variation(
        self,
        notes: List[Dict],
        config: Optional[VariationConfig] = None
    ) -> List[Dict]:
        config = config or VariationConfig()
        result = []
        
        for note in notes:
            new_note = note.copy()
            
            if config.variation_type == VariationType.MELODIC:
                if random.random() < config.intensity:
                    new_note["pitch"] = note.get("pitch", 60) + random.choice([-2, -1, 1, 2])
            
            elif config.variation_type == VariationType.RHYTHMIC:
                if random.random() < config.intensity and not config.preserve_rhythm:
                    shift = random.choice([-TICKS_PER_BEAT//4, TICKS_PER_BEAT//4])
                    new_note["start_tick"] = max(0, note.get("start_tick", 0) + shift)
            
            elif config.variation_type == VariationType.DYNAMIC:
                vel_change = int((random.random() - 0.5) * 20 * config.intensity)
                new_note["velocity"] = max(1, min(127, note.get("velocity", 80) + vel_change))
            
            elif config.variation_type == VariationType.ORNAMENTATION:
                # Add grace note
                if random.random() < config.intensity * 0.3:
                    grace = note.copy()
                    grace["start_tick"] = note.get("start_tick", 0) - TICKS_PER_BEAT // 8
                    grace["duration_ticks"] = TICKS_PER_BEAT // 8
                    grace["pitch"] = note.get("pitch", 60) + random.choice([-1, 1, 2])
                    grace["velocity"] = int(note.get("velocity", 80) * 0.6)
                    result.append(grace)
            
            result.append(new_note)
        
        return result
