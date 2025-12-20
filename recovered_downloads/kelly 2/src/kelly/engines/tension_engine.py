"""Tension Engine - Controls musical tension and release."""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import math


class TensionShape(Enum):
    LINEAR_BUILD = "linear_build"
    LINEAR_RELEASE = "linear_release"
    EXPONENTIAL = "exponential"
    WAVE = "wave"
    PLATEAU = "plateau"
    SPIKE = "spike"


@dataclass
class TensionPoint:
    position: float  # 0-1
    tension: float   # 0-1
    dissonance: float = 0.5
    density: float = 0.5


@dataclass
class TensionCurve:
    points: List[TensionPoint]
    shape: TensionShape
    peak_position: float = 0.75


EMOTION_TENSION = {
    "grief": {"base": 0.6, "shape": TensionShape.WAVE, "peak": 0.5},
    "sadness": {"base": 0.4, "shape": TensionShape.LINEAR_RELEASE, "peak": 0.3},
    "anger": {"base": 0.8, "shape": TensionShape.PLATEAU, "peak": 0.6},
    "anxiety": {"base": 0.7, "shape": TensionShape.WAVE, "peak": 0.8},
    "joy": {"base": 0.3, "shape": TensionShape.SPIKE, "peak": 0.9},
    "hope": {"base": 0.4, "shape": TensionShape.LINEAR_BUILD, "peak": 0.85},
    "serenity": {"base": 0.2, "shape": TensionShape.PLATEAU, "peak": 0.5},
    "emptiness": {"base": 0.3, "shape": TensionShape.LINEAR_RELEASE, "peak": 0.2},
}


class TensionEngine:
    def __init__(self):
        self.profiles = EMOTION_TENSION
    
    def generate_curve(
        self,
        emotion: str,
        num_points: int = 16
    ) -> TensionCurve:
        profile = self.profiles.get(emotion.lower(), {"base": 0.5, "shape": TensionShape.WAVE, "peak": 0.5})
        shape = profile["shape"]
        base = profile["base"]
        peak_pos = profile["peak"]
        
        points = []
        for i in range(num_points):
            pos = i / (num_points - 1)
            
            if shape == TensionShape.LINEAR_BUILD:
                tension = base * 0.5 + pos * base
            elif shape == TensionShape.LINEAR_RELEASE:
                tension = base - pos * base * 0.5
            elif shape == TensionShape.EXPONENTIAL:
                tension = base * (pos ** 2)
            elif shape == TensionShape.WAVE:
                tension = base + (1 - base) * 0.3 * math.sin(pos * math.pi * 2)
            elif shape == TensionShape.PLATEAU:
                tension = base if 0.2 < pos < 0.8 else base * 0.6
            elif shape == TensionShape.SPIKE:
                dist = abs(pos - peak_pos)
                tension = max(0.2, 1 - dist * 2)
            else:
                tension = base
            
            points.append(TensionPoint(
                position=pos,
                tension=max(0, min(1, tension)),
                dissonance=tension * 0.7,
                density=0.3 + tension * 0.5,
            ))
        
        return TensionCurve(points, shape, peak_pos)
    
    def get_tension_at(self, curve: TensionCurve, position: float) -> TensionPoint:
        """Get interpolated tension at position."""
        if not curve.points:
            return TensionPoint(position, 0.5)
        
        # Find surrounding points
        for i, point in enumerate(curve.points):
            if point.position >= position:
                if i == 0:
                    return point
                prev = curve.points[i - 1]
                t = (position - prev.position) / (point.position - prev.position)
                return TensionPoint(
                    position=position,
                    tension=prev.tension + (point.tension - prev.tension) * t,
                    dissonance=prev.dissonance + (point.dissonance - prev.dissonance) * t,
                    density=prev.density + (point.density - prev.density) * t,
                )
        
        return curve.points[-1]
