"""Rhythm Engine - Generates emotion-driven drum patterns.

Creates drum patterns that express emotional intent through
density, accent patterns, and ghost notes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random

TICKS_PER_BEAT = 480

# GM Drum Map
GM_KICK = 36
GM_SNARE = 38
GM_CLOSED_HH = 42
GM_OPEN_HH = 46
GM_CRASH = 49
GM_RIDE = 51
GM_TOM_LOW = 45
GM_TOM_MID = 47
GM_TOM_HIGH = 50


class DrumPattern(Enum):
    """Drum pattern archetypes."""
    FOUR_ON_FLOOR = "four_on_floor"
    BACKBEAT = "backbeat"
    HALFTIME = "halftime"
    BREAKBEAT = "breakbeat"
    LINEAR = "linear"
    SPARSE = "sparse"
    TRIBAL = "tribal"
    SHUFFLE = "shuffle"


@dataclass
class RhythmConfig:
    """Configuration for rhythm generation."""
    emotion: str = "neutral"
    bars: int = 4
    tempo_bpm: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    pattern_override: Optional[DrumPattern] = None
    include_hihats: bool = True
    include_fills: bool = True
    seed: int = -1


@dataclass
class DrumHit:
    """A single drum hit."""
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int
    is_ghost: bool = False


@dataclass
class RhythmOutput:
    """Complete rhythm generation output."""
    hits: List[DrumHit]
    pattern: DrumPattern
    emotion: str
    bars: int


EMOTION_PROFILES = {
    "grief": {
        "pattern": DrumPattern.SPARSE,
        "kick_density": 0.3,
        "snare_density": 0.2,
        "hihat_density": 0.4,
        "velocity_range": (35, 60),
        "ghost_probability": 0.05,
        "fill_probability": 0.1,
    },
    "sadness": {
        "pattern": DrumPattern.HALFTIME,
        "kick_density": 0.4,
        "snare_density": 0.25,
        "hihat_density": 0.5,
        "velocity_range": (40, 70),
        "ghost_probability": 0.1,
        "fill_probability": 0.15,
    },
    "anger": {
        "pattern": DrumPattern.FOUR_ON_FLOOR,
        "kick_density": 0.9,
        "snare_density": 0.7,
        "hihat_density": 1.0,
        "velocity_range": (90, 127),
        "ghost_probability": 0.25,
        "fill_probability": 0.3,
    },
    "anxiety": {
        "pattern": DrumPattern.BREAKBEAT,
        "kick_density": 0.6,
        "snare_density": 0.5,
        "hihat_density": 0.8,
        "velocity_range": (60, 95),
        "ghost_probability": 0.2,
        "fill_probability": 0.25,
    },
    "joy": {
        "pattern": DrumPattern.BACKBEAT,
        "kick_density": 0.7,
        "snare_density": 0.5,
        "hihat_density": 0.9,
        "velocity_range": (70, 100),
        "ghost_probability": 0.15,
        "fill_probability": 0.2,
    },
    "hope": {
        "pattern": DrumPattern.BACKBEAT,
        "kick_density": 0.6,
        "snare_density": 0.4,
        "hihat_density": 0.7,
        "velocity_range": (55, 85),
        "ghost_probability": 0.1,
        "fill_probability": 0.15,
    },
    "defiance": {
        "pattern": DrumPattern.BREAKBEAT,
        "kick_density": 0.8,
        "snare_density": 0.6,
        "hihat_density": 0.85,
        "velocity_range": (80, 115),
        "ghost_probability": 0.2,
        "fill_probability": 0.25,
    },
    "emptiness": {
        "pattern": DrumPattern.SPARSE,
        "kick_density": 0.2,
        "snare_density": 0.1,
        "hihat_density": 0.3,
        "velocity_range": (25, 50),
        "ghost_probability": 0.02,
        "fill_probability": 0.05,
    },
    "nostalgia": {
        "pattern": DrumPattern.SHUFFLE,
        "kick_density": 0.5,
        "snare_density": 0.4,
        "hihat_density": 0.6,
        "velocity_range": (50, 80),
        "ghost_probability": 0.12,
        "fill_probability": 0.15,
    },
}


class RhythmEngine:
    """Generates emotion-driven drum patterns."""
    
    def __init__(self, seed: Optional[int] = None):
        self.profiles = EMOTION_PROFILES
        if seed is not None:
            random.seed(seed)
    
    def generate(self, config: RhythmConfig) -> RhythmOutput:
        """Generate drum pattern from config."""
        if config.seed >= 0:
            random.seed(config.seed)
        
        emotion = config.emotion.lower()
        profile = self.profiles.get(emotion, self.profiles["hope"])
        pattern = config.pattern_override or profile["pattern"]
        
        hits = []
        ticks_per_bar = TICKS_PER_BEAT * config.time_signature[0]
        
        for bar in range(config.bars):
            bar_start = bar * ticks_per_bar
            is_fill_bar = config.include_fills and bar == config.bars - 1
            
            if is_fill_bar and random.random() < profile["fill_probability"]:
                hits.extend(self._generate_fill(bar_start, ticks_per_bar, profile))
            else:
                hits.extend(self._generate_bar(
                    bar_start, ticks_per_bar, pattern, profile, config
                ))
        
        return RhythmOutput(
            hits=hits,
            pattern=pattern,
            emotion=emotion,
            bars=config.bars,
        )
    
    def _generate_bar(
        self,
        start: int,
        bar_ticks: int,
        pattern: DrumPattern,
        profile: Dict,
        config: RhythmConfig
    ) -> List[DrumHit]:
        """Generate one bar of drums."""
        hits = []
        vel_min, vel_max = profile["velocity_range"]
        sixteenth = TICKS_PER_BEAT // 4
        
        # Kick pattern
        kick_positions = self._get_kick_positions(pattern, bar_ticks)
        for pos in kick_positions:
            if random.random() < profile["kick_density"]:
                hits.append(DrumHit(
                    pitch=GM_KICK,
                    start_tick=start + pos,
                    duration_ticks=sixteenth,
                    velocity=random.randint(vel_min + 10, vel_max),
                ))
        
        # Snare pattern
        snare_positions = self._get_snare_positions(pattern, bar_ticks)
        for pos in snare_positions:
            if random.random() < profile["snare_density"]:
                hits.append(DrumHit(
                    pitch=GM_SNARE,
                    start_tick=start + pos,
                    duration_ticks=sixteenth,
                    velocity=random.randint(vel_min, vel_max),
                ))
                # Ghost note
                if random.random() < profile["ghost_probability"]:
                    ghost_pos = pos - sixteenth if pos > 0 else pos + sixteenth
                    hits.append(DrumHit(
                        pitch=GM_SNARE,
                        start_tick=start + ghost_pos,
                        duration_ticks=sixteenth // 2,
                        velocity=random.randint(vel_min // 2, vel_min),
                        is_ghost=True,
                    ))
        
        # Hi-hats
        if config.include_hihats:
            hihat_positions = self._get_hihat_positions(pattern, bar_ticks)
            for i, pos in enumerate(hihat_positions):
                if random.random() < profile["hihat_density"]:
                    is_open = pattern == DrumPattern.SHUFFLE and i % 3 == 2
                    hits.append(DrumHit(
                        pitch=GM_OPEN_HH if is_open else GM_CLOSED_HH,
                        start_tick=start + pos,
                        duration_ticks=sixteenth // 2,
                        velocity=random.randint(vel_min - 10, vel_max - 20),
                    ))
        
        return hits
    
    def _get_kick_positions(self, pattern: DrumPattern, bar_ticks: int) -> List[int]:
        """Get kick drum positions for pattern."""
        beat = TICKS_PER_BEAT
        
        if pattern == DrumPattern.FOUR_ON_FLOOR:
            return [0, beat, beat * 2, beat * 3]
        elif pattern == DrumPattern.BACKBEAT:
            return [0, beat * 2]
        elif pattern == DrumPattern.HALFTIME:
            return [0]
        elif pattern == DrumPattern.BREAKBEAT:
            return [0, int(beat * 1.5), beat * 2, int(beat * 3.5)]
        elif pattern == DrumPattern.SPARSE:
            return [0]
        elif pattern == DrumPattern.SHUFFLE:
            return [0, beat * 2]
        else:
            return [0, beat * 2]
    
    def _get_snare_positions(self, pattern: DrumPattern, bar_ticks: int) -> List[int]:
        """Get snare drum positions for pattern."""
        beat = TICKS_PER_BEAT
        
        if pattern == DrumPattern.FOUR_ON_FLOOR:
            return [beat, beat * 3]
        elif pattern == DrumPattern.BACKBEAT:
            return [beat, beat * 3]
        elif pattern == DrumPattern.HALFTIME:
            return [beat * 2]
        elif pattern == DrumPattern.BREAKBEAT:
            return [beat, int(beat * 2.75)]
        elif pattern == DrumPattern.SPARSE:
            return [beat * 2]
        elif pattern == DrumPattern.SHUFFLE:
            return [beat, beat * 3]
        else:
            return [beat, beat * 3]
    
    def _get_hihat_positions(self, pattern: DrumPattern, bar_ticks: int) -> List[int]:
        """Get hi-hat positions for pattern."""
        eighth = TICKS_PER_BEAT // 2
        sixteenth = TICKS_PER_BEAT // 4
        
        if pattern in [DrumPattern.FOUR_ON_FLOOR, DrumPattern.BACKBEAT]:
            return [i * eighth for i in range(8)]
        elif pattern == DrumPattern.HALFTIME:
            return [i * eighth for i in range(8)]
        elif pattern == DrumPattern.BREAKBEAT:
            return [i * sixteenth for i in range(16)]
        elif pattern == DrumPattern.SPARSE:
            return [0, TICKS_PER_BEAT * 2]
        elif pattern == DrumPattern.SHUFFLE:
            triplet = TICKS_PER_BEAT // 3
            return [i * triplet for i in range(12)]
        else:
            return [i * eighth for i in range(8)]
    
    def _generate_fill(
        self,
        start: int,
        bar_ticks: int,
        profile: Dict
    ) -> List[DrumHit]:
        """Generate a drum fill."""
        hits = []
        vel_min, vel_max = profile["velocity_range"]
        sixteenth = TICKS_PER_BEAT // 4
        
        # Fill in last half of bar
        fill_start = start + bar_ticks // 2
        
        toms = [GM_TOM_HIGH, GM_TOM_MID, GM_TOM_LOW]
        for i in range(8):
            pos = fill_start + i * sixteenth
            pitch = random.choice(toms + [GM_SNARE])
            hits.append(DrumHit(
                pitch=pitch,
                start_tick=pos,
                duration_ticks=sixteenth,
                velocity=random.randint(vel_min, vel_max),
            ))
        
        # Crash at end
        hits.append(DrumHit(
            pitch=GM_CRASH,
            start_tick=start + bar_ticks,
            duration_ticks=TICKS_PER_BEAT,
            velocity=vel_max,
        ))
        
        return hits


def generate_rhythm(
    emotion: str,
    bars: int = 4,
    tempo: int = 120
) -> RhythmOutput:
    """Quick rhythm generation helper."""
    engine = RhythmEngine()
    config = RhythmConfig(emotion=emotion, bars=bars, tempo_bpm=tempo)
    return engine.generate(config)
