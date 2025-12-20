"""Melody Engine - Generates emotion-driven melodic lines.

Creates melodies that express emotional intent through contour,
interval choices, rhythm density, and articulation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
import math

TICKS_PER_BEAT = 480
CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class MelodyContour(Enum):
    """Melodic shape/contour types."""
    ASCENDING = "ascending"
    DESCENDING = "descending"
    ARCH = "arch"
    INVERSE_ARCH = "inverse_arch"
    WAVE = "wave"
    STATIC = "static"
    CLIMAX_EARLY = "climax_early"
    CLIMAX_LATE = "climax_late"


class ArticulationType(Enum):
    """Note articulation."""
    LEGATO = "legato"
    STACCATO = "staccato"
    TENUTO = "tenuto"
    MARCATO = "marcato"
    BREATH = "breath"


class RhythmDensity(Enum):
    """Rhythmic density levels."""
    SPARSE = "sparse"
    MODERATE = "moderate"
    DENSE = "dense"
    FRANTIC = "frantic"


@dataclass
class MelodyConfig:
    """Configuration for melody generation."""
    emotion: str = "neutral"
    key: str = "C"
    mode: str = "minor"
    bars: int = 4
    tempo_bpm: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    register: Tuple[int, int] = (60, 84)
    contour_override: Optional[MelodyContour] = None
    density_override: Optional[RhythmDensity] = None
    seed: int = -1


@dataclass
class MelodyNote:
    """A single melody note."""
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int
    articulation: ArticulationType = ArticulationType.LEGATO


@dataclass 
class MelodyOutput:
    """Complete melody generation output."""
    notes: List[MelodyNote]
    contour: MelodyContour
    emotion: str
    key: str
    mode: str
    bars: int


SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}

EMOTION_PROFILES = {
    "grief": {
        "contour": MelodyContour.DESCENDING,
        "density": RhythmDensity.SPARSE,
        "interval_tendency": "stepwise",
        "velocity_range": (40, 70),
        "rest_probability": 0.4,
        "articulation": ArticulationType.LEGATO,
        "sustain_ratio": 0.9,
    },
    "sadness": {
        "contour": MelodyContour.INVERSE_ARCH,
        "density": RhythmDensity.SPARSE,
        "interval_tendency": "stepwise",
        "velocity_range": (45, 75),
        "rest_probability": 0.35,
        "articulation": ArticulationType.LEGATO,
        "sustain_ratio": 0.85,
    },
    "anger": {
        "contour": MelodyContour.ARCH,
        "density": RhythmDensity.DENSE,
        "interval_tendency": "angular",
        "velocity_range": (80, 120),
        "rest_probability": 0.1,
        "articulation": ArticulationType.MARCATO,
        "sustain_ratio": 0.5,
    },
    "anxiety": {
        "contour": MelodyContour.WAVE,
        "density": RhythmDensity.DENSE,
        "interval_tendency": "mixed",
        "velocity_range": (55, 90),
        "rest_probability": 0.15,
        "articulation": ArticulationType.STACCATO,
        "sustain_ratio": 0.6,
    },
    "joy": {
        "contour": MelodyContour.ASCENDING,
        "density": RhythmDensity.MODERATE,
        "interval_tendency": "leaps",
        "velocity_range": (70, 100),
        "rest_probability": 0.15,
        "articulation": ArticulationType.STACCATO,
        "sustain_ratio": 0.7,
    },
    "hope": {
        "contour": MelodyContour.CLIMAX_LATE,
        "density": RhythmDensity.MODERATE,
        "interval_tendency": "stepwise",
        "velocity_range": (55, 90),
        "rest_probability": 0.2,
        "articulation": ArticulationType.LEGATO,
        "sustain_ratio": 0.8,
    },
    "serenity": {
        "contour": MelodyContour.WAVE,
        "density": RhythmDensity.SPARSE,
        "interval_tendency": "stepwise",
        "velocity_range": (35, 65),
        "rest_probability": 0.4,
        "articulation": ArticulationType.LEGATO,
        "sustain_ratio": 0.95,
    },
    "defiance": {
        "contour": MelodyContour.ARCH,
        "density": RhythmDensity.MODERATE,
        "interval_tendency": "angular",
        "velocity_range": (75, 110),
        "rest_probability": 0.1,
        "articulation": ArticulationType.MARCATO,
        "sustain_ratio": 0.6,
    },
    "nostalgia": {
        "contour": MelodyContour.INVERSE_ARCH,
        "density": RhythmDensity.SPARSE,
        "interval_tendency": "stepwise",
        "velocity_range": (50, 80),
        "rest_probability": 0.3,
        "articulation": ArticulationType.TENUTO,
        "sustain_ratio": 0.85,
    },
    "emptiness": {
        "contour": MelodyContour.STATIC,
        "density": RhythmDensity.SPARSE,
        "interval_tendency": "minimal",
        "velocity_range": (25, 50),
        "rest_probability": 0.5,
        "articulation": ArticulationType.LEGATO,
        "sustain_ratio": 0.95,
    },
}


def note_to_midi(note: str, octave: int = 4) -> int:
    """Convert note name to MIDI number."""
    note = note.upper().replace('♯', '#')
    base = note[0]
    idx = CHROMATIC.index(base) if base in CHROMATIC else 0
    if '#' in note:
        idx = (idx + 1) % 12
    return 12 + (octave * 12) + idx


def get_scale(root: str, mode: str, octave: int = 4) -> List[int]:
    """Get scale pitches."""
    root_midi = note_to_midi(root, octave)
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["minor"])
    return [root_midi + i for i in intervals]


class MelodyEngine:
    """Generates emotion-driven melodic lines."""
    
    def __init__(self, seed: Optional[int] = None):
        self.profiles = EMOTION_PROFILES
        if seed is not None:
            random.seed(seed)
    
    def generate(self, config: MelodyConfig) -> MelodyOutput:
        """Generate melody from config."""
        if config.seed >= 0:
            random.seed(config.seed)
        
        emotion = config.emotion.lower()
        profile = self.profiles.get(emotion, self.profiles["hope"])
        
        contour = config.contour_override or profile["contour"]
        density = config.density_override or profile["density"]
        
        scale = get_scale(config.key, config.mode, 5)
        scale_extended = scale + [p + 12 for p in scale]
        
        notes = self._generate_melody(
            config, profile, contour, density, scale_extended
        )
        
        return MelodyOutput(
            notes=notes,
            contour=contour,
            emotion=emotion,
            key=config.key,
            mode=config.mode,
            bars=config.bars,
        )
    
    def _generate_melody(
        self,
        config: MelodyConfig,
        profile: Dict,
        contour: MelodyContour,
        density: RhythmDensity,
        scale: List[int]
    ) -> List[MelodyNote]:
        """Generate the actual melody notes."""
        notes = []
        ticks_per_bar = TICKS_PER_BEAT * config.time_signature[0]
        total_ticks = config.bars * ticks_per_bar
        
        # Determine note count from density
        if density == RhythmDensity.SPARSE:
            notes_per_bar = random.randint(2, 4)
        elif density == RhythmDensity.MODERATE:
            notes_per_bar = random.randint(4, 8)
        elif density == RhythmDensity.DENSE:
            notes_per_bar = random.randint(8, 12)
        else:
            notes_per_bar = random.randint(12, 16)
        
        target_notes = notes_per_bar * config.bars
        
        # Generate contour curve
        contour_values = self._generate_contour(contour, target_notes)
        
        # Generate rhythm
        rhythm = self._generate_rhythm(density, config.bars, ticks_per_bar, profile)
        
        vel_min, vel_max = profile["velocity_range"]
        art = profile["articulation"]
        sustain = profile["sustain_ratio"]
        
        # Create notes
        scale_len = len(scale)
        for i, (start, duration) in enumerate(rhythm[:target_notes]):
            if random.random() < profile["rest_probability"]:
                continue
            
            # Map contour to scale index
            contour_val = contour_values[i] if i < len(contour_values) else 0.5
            scale_idx = int(contour_val * (scale_len - 1))
            
            # Apply interval tendency
            if profile["interval_tendency"] == "stepwise" and notes:
                last_idx = scale.index(notes[-1].pitch) if notes[-1].pitch in scale else scale_idx
                step = random.choice([-1, 0, 1])
                scale_idx = max(0, min(scale_len - 1, last_idx + step))
            elif profile["interval_tendency"] == "angular" and notes:
                last_idx = scale.index(notes[-1].pitch) if notes[-1].pitch in scale else scale_idx
                step = random.choice([-3, -2, 2, 3, 4])
                scale_idx = max(0, min(scale_len - 1, last_idx + step))
            
            pitch = scale[scale_idx]
            
            # Constrain to register
            while pitch < config.register[0]:
                pitch += 12
            while pitch > config.register[1]:
                pitch -= 12
            
            velocity = random.randint(vel_min, vel_max)
            
            notes.append(MelodyNote(
                pitch=pitch,
                start_tick=start,
                duration_ticks=int(duration * sustain),
                velocity=velocity,
                articulation=art,
            ))
        
        return notes
    
    def _generate_contour(self, contour: MelodyContour, length: int) -> List[float]:
        """Generate contour values 0-1."""
        values = []
        
        for i in range(length):
            t = i / max(1, length - 1)
            
            if contour == MelodyContour.ASCENDING:
                val = t
            elif contour == MelodyContour.DESCENDING:
                val = 1 - t
            elif contour == MelodyContour.ARCH:
                val = math.sin(t * math.pi)
            elif contour == MelodyContour.INVERSE_ARCH:
                val = 1 - math.sin(t * math.pi)
            elif contour == MelodyContour.WAVE:
                val = 0.5 + 0.4 * math.sin(t * math.pi * 2)
            elif contour == MelodyContour.STATIC:
                val = 0.5 + random.uniform(-0.1, 0.1)
            elif contour == MelodyContour.CLIMAX_EARLY:
                val = 1 - (t - 0.3) ** 2 if t < 0.6 else 0.3 * (1 - t)
            elif contour == MelodyContour.CLIMAX_LATE:
                val = t ** 2 if t < 0.7 else 1 - (t - 0.7) * 2
            else:
                val = 0.5
            
            values.append(max(0, min(1, val)))
        
        return values
    
    def _generate_rhythm(
        self,
        density: RhythmDensity,
        bars: int,
        ticks_per_bar: int,
        profile: Dict
    ) -> List[Tuple[int, int]]:
        """Generate rhythm as list of (start_tick, duration)."""
        rhythm = []
        total_ticks = bars * ticks_per_bar
        
        if density == RhythmDensity.SPARSE:
            durations = [TICKS_PER_BEAT, TICKS_PER_BEAT * 2]
            weights = [0.4, 0.6]
        elif density == RhythmDensity.MODERATE:
            durations = [TICKS_PER_BEAT // 2, TICKS_PER_BEAT, TICKS_PER_BEAT * 2]
            weights = [0.3, 0.5, 0.2]
        elif density == RhythmDensity.DENSE:
            durations = [TICKS_PER_BEAT // 4, TICKS_PER_BEAT // 2, TICKS_PER_BEAT]
            weights = [0.4, 0.4, 0.2]
        else:
            durations = [TICKS_PER_BEAT // 4, TICKS_PER_BEAT // 2]
            weights = [0.7, 0.3]
        
        current = 0
        while current < total_ticks:
            dur = random.choices(durations, weights=weights)[0]
            if current + dur > total_ticks:
                dur = total_ticks - current
            if dur > 0:
                rhythm.append((current, dur))
            current += dur
        
        return rhythm


def generate_melody(
    emotion: str,
    key: str = "C",
    mode: str = "minor",
    bars: int = 4,
    tempo: int = 120
) -> MelodyOutput:
    """Quick melody generation helper."""
    engine = MelodyEngine()
    config = MelodyConfig(
        emotion=emotion,
        key=key,
        mode=mode,
        bars=bars,
        tempo_bpm=tempo,
    )
    return engine.generate(config)
