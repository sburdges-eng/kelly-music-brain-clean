"""Dynamics Engine - Controls dynamic expression curves."""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
import math

TICKS_PER_BEAT = 480


class DynamicShape(Enum):
    FLAT = "flat"
    CRESCENDO = "crescendo"
    DECRESCENDO = "decrescendo"
    SWELL = "swell"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    WAVE = "wave"


class DynamicMarking(Enum):
    PPP = 20
    PP = 35
    P = 50
    MP = 65
    MF = 80
    F = 95
    FF = 110
    FFF = 125


@dataclass
class DynamicPoint:
    tick: int
    velocity: int
    expression: int = 100


@dataclass
class DynamicCurve:
    points: List[DynamicPoint]
    shape: DynamicShape


EMOTION_PROFILES = {
    "grief": {"base": DynamicMarking.P, "shape": DynamicShape.SWELL, "range": (35, 70)},
    "sadness": {"base": DynamicMarking.MP, "shape": DynamicShape.DECRESCENDO, "range": (40, 75)},
    "anger": {"base": DynamicMarking.FF, "shape": DynamicShape.CRESCENDO, "range": (85, 127)},
    "anxiety": {"base": DynamicMarking.MF, "shape": DynamicShape.WAVE, "range": (55, 95)},
    "joy": {"base": DynamicMarking.F, "shape": DynamicShape.SWELL, "range": (70, 110)},
    "hope": {"base": DynamicMarking.MF, "shape": DynamicShape.CRESCENDO, "range": (55, 90)},
    "emptiness": {"base": DynamicMarking.PP, "shape": DynamicShape.FADE_OUT, "range": (20, 50)},
}


class DynamicsEngine:
    def __init__(self):
        self.profiles = EMOTION_PROFILES
    
    def generate_curve(
        self,
        emotion: str,
        duration_ticks: int,
        num_points: int = 16
    ) -> DynamicCurve:
        profile = self.profiles.get(emotion.lower(), self.profiles["hope"])
        shape = profile["shape"]
        vel_min, vel_max = profile["range"]
        
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            tick = int(t * duration_ticks)
            
            if shape == DynamicShape.FLAT:
                vel = (vel_min + vel_max) // 2
            elif shape == DynamicShape.CRESCENDO:
                vel = int(vel_min + (vel_max - vel_min) * t)
            elif shape == DynamicShape.DECRESCENDO:
                vel = int(vel_max - (vel_max - vel_min) * t)
            elif shape == DynamicShape.SWELL:
                vel = int(vel_min + (vel_max - vel_min) * math.sin(t * math.pi))
            elif shape == DynamicShape.WAVE:
                vel = int((vel_min + vel_max) / 2 + (vel_max - vel_min) / 2 * math.sin(t * math.pi * 2))
            elif shape == DynamicShape.FADE_OUT:
                vel = int(vel_max * (1 - t ** 2))
            else:
                vel = (vel_min + vel_max) // 2
            
            points.append(DynamicPoint(tick, max(1, min(127, vel))))
        
        return DynamicCurve(points, shape)
    
    def apply_to_notes(self, notes: List[Dict], curve: DynamicCurve) -> List[Dict]:
        """Apply dynamic curve to notes."""
        if not curve.points:
            return notes
        
        result = []
        for note in notes:
            tick = note.get("start_tick", 0)
            # Find nearest curve point
            nearest = min(curve.points, key=lambda p: abs(p.tick - tick))
            new_note = note.copy()
            new_note["velocity"] = nearest.velocity
            result.append(new_note)
        
        return result
