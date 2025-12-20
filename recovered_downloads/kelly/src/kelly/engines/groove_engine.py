"""Groove Engine - Applies humanization and timing feel to MIDI.

Handles swing, push/pull timing, velocity curves, and ghost notes
to make MIDI feel more human and emotionally expressive.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import random
import math


TICKS_PER_BEAT = 480
GHOST_NOTE_PROBABILITY = 0.15


class GrooveFeel(Enum):
    """Types of rhythmic feel."""
    STRAIGHT = "straight"
    SWING = "swing"
    SHUFFLE = "shuffle"
    BEHIND = "behind_the_beat"
    AHEAD = "ahead_of_beat"
    RUBATO = "rubato"
    MECHANICAL = "mechanical"
    LAID_BACK = "laid_back"
    DRIVING = "driving"


@dataclass
class GrooveSettings:
    """Settings for groove application."""
    complexity: float = 0.5
    vulnerability: float = 0.5
    timing_sigma_override: Optional[float] = None
    dropout_prob_override: Optional[float] = None
    velocity_range_override: Optional[Tuple[int, int]] = None
    kick_timing_mult: float = 0.5
    snare_timing_mult: float = 0.7
    hihat_timing_mult: float = 1.2
    enable_ghost_notes: bool = True
    ghost_note_probability: float = GHOST_NOTE_PROBABILITY
    ghost_note_velocity_mult: float = 0.4
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity,
            "vulnerability": self.vulnerability,
            "timing_sigma_override": self.timing_sigma_override,
            "dropout_prob_override": self.dropout_prob_override,
            "velocity_range_override": self.velocity_range_override,
            "kick_timing_mult": self.kick_timing_mult,
            "snare_timing_mult": self.snare_timing_mult,
            "hihat_timing_mult": self.hihat_timing_mult,
            "enable_ghost_notes": self.enable_ghost_notes,
            "ghost_note_probability": self.ghost_note_probability,
            "ghost_note_velocity_mult": self.ghost_note_velocity_mult,
        }


EMOTION_GROOVE_PROFILES = {
    "grief": {
        "feel": GrooveFeel.BEHIND,
        "swing_amount": 0.1,
        "timing_variance": 20,
        "velocity_variance": 15,
        "dropout_prob": 0.15,
        "ghost_prob": 0.05,
    },
    "sadness": {
        "feel": GrooveFeel.LAID_BACK,
        "swing_amount": 0.15,
        "timing_variance": 15,
        "velocity_variance": 12,
        "dropout_prob": 0.1,
        "ghost_prob": 0.08,
    },
    "anger": {
        "feel": GrooveFeel.DRIVING,
        "swing_amount": 0.0,
        "timing_variance": 5,
        "velocity_variance": 20,
        "dropout_prob": 0.02,
        "ghost_prob": 0.2,
    },
    "anxiety": {
        "feel": GrooveFeel.AHEAD,
        "swing_amount": 0.05,
        "timing_variance": 25,
        "velocity_variance": 18,
        "dropout_prob": 0.08,
        "ghost_prob": 0.12,
    },
    "joy": {
        "feel": GrooveFeel.SWING,
        "swing_amount": 0.2,
        "timing_variance": 10,
        "velocity_variance": 10,
        "dropout_prob": 0.05,
        "ghost_prob": 0.15,
    },
    "serenity": {
        "feel": GrooveFeel.RUBATO,
        "swing_amount": 0.1,
        "timing_variance": 30,
        "velocity_variance": 8,
        "dropout_prob": 0.2,
        "ghost_prob": 0.03,
    },
    "hope": {
        "feel": GrooveFeel.STRAIGHT,
        "swing_amount": 0.08,
        "timing_variance": 12,
        "velocity_variance": 10,
        "dropout_prob": 0.05,
        "ghost_prob": 0.1,
    },
    "defiance": {
        "feel": GrooveFeel.AHEAD,
        "swing_amount": 0.0,
        "timing_variance": 8,
        "velocity_variance": 15,
        "dropout_prob": 0.03,
        "ghost_prob": 0.18,
    },
    "nostalgia": {
        "feel": GrooveFeel.SWING,
        "swing_amount": 0.18,
        "timing_variance": 18,
        "velocity_variance": 12,
        "dropout_prob": 0.1,
        "ghost_prob": 0.08,
    },
    "emptiness": {
        "feel": GrooveFeel.RUBATO,
        "swing_amount": 0.05,
        "timing_variance": 35,
        "velocity_variance": 5,
        "dropout_prob": 0.3,
        "ghost_prob": 0.02,
    },
}


@dataclass
class GroovedNote:
    """A note with groove applied."""
    original_tick: int
    grooved_tick: int
    original_velocity: int
    grooved_velocity: int
    pitch: int
    duration: int
    is_ghost: bool = False


class GrooveEngine:
    """Applies groove and humanization to MIDI data.
    
    Usage:
        engine = GrooveEngine()
        grooved = engine.apply_groove(notes, emotion="grief")
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.profiles = EMOTION_GROOVE_PROFILES
        if seed is not None:
            random.seed(seed)
    
    def get_profile(self, emotion: str) -> Dict[str, Any]:
        """Get groove profile for emotion."""
        return self.profiles.get(emotion.lower(), self.profiles["hope"])
    
    def apply_groove(
        self,
        notes: List[Dict],
        emotion: str = "neutral",
        settings: Optional[GrooveSettings] = None,
    ) -> List[GroovedNote]:
        """Apply groove to a list of notes."""
        settings = settings or GrooveSettings()
        profile = self.get_profile(emotion)
        
        grooved_notes = []
        
        for note in notes:
            # Base timing adjustment
            timing_offset = self._calculate_timing_offset(
                note.get("tick", 0),
                profile["feel"],
                profile["swing_amount"],
                profile["timing_variance"],
            )
            
            # Velocity adjustment
            velocity_offset = random.gauss(0, profile["velocity_variance"])
            
            # Dropout check
            if random.random() < profile["dropout_prob"] * settings.complexity:
                continue
            
            grooved = GroovedNote(
                original_tick=note.get("tick", 0),
                grooved_tick=note.get("tick", 0) + int(timing_offset),
                original_velocity=note.get("velocity", 80),
                grooved_velocity=max(1, min(127, note.get("velocity", 80) + int(velocity_offset))),
                pitch=note.get("pitch", 60),
                duration=note.get("duration", TICKS_PER_BEAT),
            )
            grooved_notes.append(grooved)
            
            # Ghost note insertion
            if settings.enable_ghost_notes and random.random() < profile["ghost_prob"]:
                ghost_tick = grooved.grooved_tick - TICKS_PER_BEAT // 4
                if ghost_tick >= 0:
                    ghost = GroovedNote(
                        original_tick=ghost_tick,
                        grooved_tick=ghost_tick + random.randint(-10, 10),
                        original_velocity=grooved.grooved_velocity,
                        grooved_velocity=int(grooved.grooved_velocity * settings.ghost_note_velocity_mult),
                        pitch=note.get("pitch", 60),
                        duration=TICKS_PER_BEAT // 8,
                        is_ghost=True,
                    )
                    grooved_notes.append(ghost)
        
        return sorted(grooved_notes, key=lambda n: n.grooved_tick)
    
    def _calculate_timing_offset(
        self,
        tick: int,
        feel: GrooveFeel,
        swing: float,
        variance: float,
    ) -> float:
        """Calculate timing offset for a note."""
        offset = 0.0
        
        # Beat position
        beat_pos = tick % TICKS_PER_BEAT
        eighth_pos = tick % (TICKS_PER_BEAT // 2)
        
        # Swing on off-beats
        if feel == GrooveFeel.SWING and eighth_pos > TICKS_PER_BEAT // 4:
            offset += swing * TICKS_PER_BEAT * 0.33
        
        # Behind the beat
        elif feel == GrooveFeel.BEHIND or feel == GrooveFeel.LAID_BACK:
            offset += random.uniform(5, 25)
        
        # Ahead of beat
        elif feel == GrooveFeel.AHEAD or feel == GrooveFeel.DRIVING:
            offset -= random.uniform(5, 20)
        
        # Rubato - variable timing
        elif feel == GrooveFeel.RUBATO:
            offset += random.gauss(0, variance * 1.5)
        
        # Add general variance
        offset += random.gauss(0, variance)
        
        return offset
    
    def apply_velocity_curve(
        self,
        notes: List[GroovedNote],
        curve_type: str = "accent_1"
    ) -> List[GroovedNote]:
        """Apply velocity accent patterns."""
        patterns = {
            "accent_1": [1.0, 0.7, 0.85, 0.7],  # Strong 1
            "accent_13": [1.0, 0.7, 0.9, 0.7],  # Strong 1 and 3
            "backbeat": [0.7, 1.0, 0.7, 1.0],   # Strong 2 and 4
            "shuffle": [1.0, 0.5, 0.8, 0.5, 0.9, 0.5, 0.8, 0.5],
        }
        
        pattern = patterns.get(curve_type, patterns["accent_1"])
        
        for note in notes:
            beat_idx = (note.grooved_tick // (TICKS_PER_BEAT // 2)) % len(pattern)
            note.grooved_velocity = int(note.grooved_velocity * pattern[beat_idx])
            note.grooved_velocity = max(1, min(127, note.grooved_velocity))
        
        return notes


def apply_emotion_groove(
    notes: List[Dict],
    emotion: str,
    seed: Optional[int] = None
) -> List[Dict]:
    """Quick helper to apply emotion-based groove."""
    engine = GrooveEngine(seed=seed)
    grooved = engine.apply_groove(notes, emotion=emotion)
    
    return [{
        "tick": n.grooved_tick,
        "velocity": n.grooved_velocity,
        "pitch": n.pitch,
        "duration": n.duration,
        "is_ghost": n.is_ghost,
    } for n in grooved]
