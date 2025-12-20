"""
Kelly Melody Engine - Emotion-driven melodic line generation.

Wired to EmotionThesaurus for automatic emotion → melody mapping.

Philosophy: Melodies should reflect emotional contour, not just scale 
correctness. A grief melody descends and lingers. A hope melody reaches 
upward with hesitation. A rage melody attacks with angular intervals.

Usage:
    from kelly_melody_engine import MelodyEngine
    
    engine = MelodyEngine()
    
    # From emotion word
    melody = engine.from_emotion("devastated", bars=4, key="F")
    
    # From thesaurus node
    melody = engine.from_node(node, bars=4, key="F")
    
    # With chord context
    melody = engine.generate(
        emotion="grief",
        key="F",
        mode="phrygian",
        chord_tones=[65, 68, 72],  # Current chord
        bars=4
    )
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import random
import math

import mido

# Import thesaurus
try:
    from emotion_thesaurus import (
        EmotionThesaurus, EmotionNode, MusicalAttributes, Mode,
        EmotionCategory, get_thesaurus
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from emotion_thesaurus import (
        EmotionThesaurus, EmotionNode, MusicalAttributes, Mode,
        EmotionCategory, get_thesaurus
    )


# =============================================================================
# CONSTANTS
# =============================================================================

TICKS_PER_BEAT = 480
CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALE_INTERVALS = {
    Mode.IONIAN: [0, 2, 4, 5, 7, 9, 11],
    Mode.DORIAN: [0, 2, 3, 5, 7, 9, 10],
    Mode.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
    Mode.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
    Mode.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
    Mode.AEOLIAN: [0, 2, 3, 5, 7, 8, 10],
    Mode.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
}

# Additional scales for extended palette
EXTENDED_SCALES = {
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "diminished": [0, 2, 3, 5, 6, 8, 9, 11],
}


# =============================================================================
# ENUMS
# =============================================================================

class ContourType(Enum):
    """Melodic contour shapes."""
    ASCENDING = "ascending"
    DESCENDING = "descending"
    ARCH = "arch"
    INVERSE_ARCH = "inverse_arch"
    STATIC = "static"
    WAVE = "wave"
    SPIRAL_DOWN = "spiral_down"
    SPIRAL_UP = "spiral_up"
    JAGGED = "jagged"
    COLLAPSE = "collapse"
    REACH_AND_FALL = "reach_and_fall"
    QUESTION = "question"  # Ends on unresolved note


class RhythmDensity(Enum):
    """Note density per bar."""
    SPARSE = "sparse"       # 2-4 notes/bar
    MODERATE = "moderate"   # 4-8 notes/bar
    DENSE = "dense"         # 8-12 notes/bar
    FRANTIC = "frantic"     # 12-16 notes/bar


class ArticulationType(Enum):
    """Note articulation."""
    LEGATO = "legato"
    STACCATO = "staccato"
    TENUTO = "tenuto"
    MARCATO = "marcato"
    BREATH = "breath"


# =============================================================================
# EMOTION → MELODY PROFILE MAPPING
# =============================================================================

@dataclass
class MelodyProfile:
    """Complete melodic behavior profile."""
    contours: List[ContourType]
    density: RhythmDensity
    articulation: ArticulationType
    interval_weights: Dict[int, float]
    rest_probability: float
    octave_range: Tuple[int, int]
    velocity_range: Tuple[int, int]
    resolution_tendency: float
    chromatic_tendency: float
    repetition_tendency: float
    syncopation: float = 0.0
    grace_note_probability: float = 0.0
    trill_probability: float = 0.0


# Category-based default profiles
CATEGORY_MELODY_PROFILES: Dict[EmotionCategory, MelodyProfile] = {
    EmotionCategory.JOY: MelodyProfile(
        contours=[ContourType.ARCH, ContourType.WAVE, ContourType.ASCENDING],
        density=RhythmDensity.MODERATE,
        articulation=ArticulationType.STACCATO,
        interval_weights={2: 0.2, 3: 0.25, 4: 0.2, 5: 0.2, 7: 0.15},
        rest_probability=0.15,
        octave_range=(0, 2),
        velocity_range=(70, 110),
        resolution_tendency=0.7,
        chromatic_tendency=0.05,
        repetition_tendency=0.2,
        syncopation=0.3,
        grace_note_probability=0.1,
    ),
    EmotionCategory.SADNESS: MelodyProfile(
        contours=[ContourType.DESCENDING, ContourType.SPIRAL_DOWN, ContourType.COLLAPSE],
        density=RhythmDensity.SPARSE,
        articulation=ArticulationType.LEGATO,
        interval_weights={1: 0.3, 2: 0.25, 3: 0.2, 4: 0.1, 5: 0.1, 7: 0.05},
        rest_probability=0.3,
        octave_range=(0, 1),
        velocity_range=(40, 80),
        resolution_tendency=0.3,
        chromatic_tendency=0.2,
        repetition_tendency=0.4,
        syncopation=0.0,
    ),
    EmotionCategory.ANGER: MelodyProfile(
        contours=[ContourType.JAGGED, ContourType.ASCENDING, ContourType.ARCH],
        density=RhythmDensity.DENSE,
        articulation=ArticulationType.MARCATO,
        interval_weights={1: 0.1, 4: 0.2, 5: 0.2, 6: 0.15, 7: 0.2, 8: 0.15},
        rest_probability=0.1,
        octave_range=(-1, 2),
        velocity_range=(80, 127),
        resolution_tendency=0.2,
        chromatic_tendency=0.3,
        repetition_tendency=0.5,
        syncopation=0.4,
    ),
    EmotionCategory.FEAR: MelodyProfile(
        contours=[ContourType.JAGGED, ContourType.COLLAPSE, ContourType.WAVE],
        density=RhythmDensity.MODERATE,
        articulation=ArticulationType.STACCATO,
        interval_weights={1: 0.3, 2: 0.2, 6: 0.2, 7: 0.15, 11: 0.15},
        rest_probability=0.35,
        octave_range=(0, 2),
        velocity_range=(30, 100),
        resolution_tendency=0.1,
        chromatic_tendency=0.4,
        repetition_tendency=0.3,
        syncopation=0.2,
        trill_probability=0.15,
    ),
    EmotionCategory.SURPRISE: MelodyProfile(
        contours=[ContourType.JAGGED, ContourType.ARCH, ContourType.QUESTION],
        density=RhythmDensity.MODERATE,
        articulation=ArticulationType.STACCATO,
        interval_weights={3: 0.15, 4: 0.2, 5: 0.2, 7: 0.25, 8: 0.2},
        rest_probability=0.25,
        octave_range=(0, 2),
        velocity_range=(60, 120),
        resolution_tendency=0.3,
        chromatic_tendency=0.2,
        repetition_tendency=0.1,
        syncopation=0.5,
        grace_note_probability=0.2,
    ),
    EmotionCategory.DISGUST: MelodyProfile(
        contours=[ContourType.DESCENDING, ContourType.STATIC, ContourType.COLLAPSE],
        density=RhythmDensity.SPARSE,
        articulation=ArticulationType.STACCATO,
        interval_weights={1: 0.3, 2: 0.25, 6: 0.2, 11: 0.15, 1: 0.1},
        rest_probability=0.3,
        octave_range=(-1, 1),
        velocity_range=(50, 85),
        resolution_tendency=0.1,
        chromatic_tendency=0.35,
        repetition_tendency=0.4,
        syncopation=0.1,
    ),
    EmotionCategory.TRUST: MelodyProfile(
        contours=[ContourType.ARCH, ContourType.WAVE, ContourType.ASCENDING],
        density=RhythmDensity.MODERATE,
        articulation=ArticulationType.TENUTO,
        interval_weights={2: 0.25, 3: 0.3, 4: 0.2, 5: 0.15, 7: 0.1},
        rest_probability=0.2,
        octave_range=(0, 1),
        velocity_range=(55, 85),
        resolution_tendency=0.7,
        chromatic_tendency=0.05,
        repetition_tendency=0.25,
        syncopation=0.1,
    ),
    EmotionCategory.ANTICIPATION: MelodyProfile(
        contours=[ContourType.SPIRAL_UP, ContourType.ASCENDING, ContourType.QUESTION],
        density=RhythmDensity.MODERATE,
        articulation=ArticulationType.TENUTO,
        interval_weights={2: 0.2, 3: 0.2, 4: 0.25, 5: 0.2, 7: 0.15},
        rest_probability=0.15,
        octave_range=(0, 2),
        velocity_range=(60, 100),
        resolution_tendency=0.3,
        chromatic_tendency=0.15,
        repetition_tendency=0.3,
        syncopation=0.35,
    ),
}

# Specific emotion overrides (sub-emotions)
SPECIFIC_EMOTION_PROFILES: Dict[str, MelodyProfile] = {
    "grief": MelodyProfile(
        contours=[ContourType.DESCENDING, ContourType.SPIRAL_DOWN, ContourType.COLLAPSE],
        density=RhythmDensity.SPARSE,
        articulation=ArticulationType.LEGATO,
        interval_weights={1: 0.35, 2: 0.25, 3: 0.2, 5: 0.1, 7: 0.1},
        rest_probability=0.35,
        octave_range=(-1, 1),
        velocity_range=(35, 70),
        resolution_tendency=0.15,
        chromatic_tendency=0.25,
        repetition_tendency=0.5,
    ),
    "rage": MelodyProfile(
        contours=[ContourType.JAGGED, ContourType.ASCENDING],
        density=RhythmDensity.FRANTIC,
        articulation=ArticulationType.MARCATO,
        interval_weights={4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2, 8: 0.2},
        rest_probability=0.05,
        octave_range=(-1, 2),
        velocity_range=(95, 127),
        resolution_tendency=0.1,
        chromatic_tendency=0.4,
        repetition_tendency=0.6,
        syncopation=0.5,
    ),
    "longing": MelodyProfile(
        contours=[ContourType.REACH_AND_FALL, ContourType.SPIRAL_UP, ContourType.QUESTION],
        density=RhythmDensity.SPARSE,
        articulation=ArticulationType.LEGATO,
        interval_weights={2: 0.15, 4: 0.25, 5: 0.25, 7: 0.2, 9: 0.15},
        rest_probability=0.25,
        octave_range=(0, 2),
        velocity_range=(45, 85),
        resolution_tendency=0.2,
        chromatic_tendency=0.15,
        repetition_tendency=0.35,
    ),
    "anxiety": MelodyProfile(
        contours=[ContourType.WAVE, ContourType.STATIC, ContourType.JAGGED],
        density=RhythmDensity.DENSE,
        articulation=ArticulationType.STACCATO,
        interval_weights={1: 0.4, 2: 0.3, 3: 0.15, 4: 0.1, 5: 0.05},
        rest_probability=0.1,
        octave_range=(0, 1),
        velocity_range=(50, 90),
        resolution_tendency=0.1,
        chromatic_tendency=0.5,
        repetition_tendency=0.6,
        trill_probability=0.2,
    ),
    "tenderness": MelodyProfile(
        contours=[ContourType.WAVE, ContourType.ARCH, ContourType.STATIC],
        density=RhythmDensity.SPARSE,
        articulation=ArticulationType.BREATH,
        interval_weights={1: 0.1, 2: 0.35, 3: 0.3, 4: 0.15, 5: 0.1},
        rest_probability=0.3,
        octave_range=(0, 1),
        velocity_range=(35, 70),
        resolution_tendency=0.5,
        chromatic_tendency=0.1,
        repetition_tendency=0.25,
    ),
    "nostalgia": MelodyProfile(
        contours=[ContourType.ARCH, ContourType.WAVE, ContourType.INVERSE_ARCH],
        density=RhythmDensity.MODERATE,
        articulation=ArticulationType.TENUTO,
        interval_weights={2: 0.25, 3: 0.3, 4: 0.2, 5: 0.15, 7: 0.1},
        rest_probability=0.2,
        octave_range=(0, 1),
        velocity_range=(50, 80),
        resolution_tendency=0.5,
        chromatic_tendency=0.1,
        repetition_tendency=0.4,
    ),
    "defiance": MelodyProfile(
        contours=[ContourType.ASCENDING, ContourType.JAGGED, ContourType.ARCH],
        density=RhythmDensity.MODERATE,
        articulation=ArticulationType.MARCATO,
        interval_weights={3: 0.2, 4: 0.25, 5: 0.25, 7: 0.2, 8: 0.1},
        rest_probability=0.15,
        octave_range=(0, 2),
        velocity_range=(75, 115),
        resolution_tendency=0.4,
        chromatic_tendency=0.15,
        repetition_tendency=0.3,
        syncopation=0.4,
    ),
    "euphoria": MelodyProfile(
        contours=[ContourType.ASCENDING, ContourType.SPIRAL_UP, ContourType.ARCH],
        density=RhythmDensity.DENSE,
        articulation=ArticulationType.STACCATO,
        interval_weights={3: 0.2, 4: 0.2, 5: 0.25, 7: 0.2, 12: 0.15},
        rest_probability=0.1,
        octave_range=(0, 2),
        velocity_range=(80, 120),
        resolution_tendency=0.6,
        chromatic_tendency=0.1,
        repetition_tendency=0.15,
        syncopation=0.4,
        grace_note_probability=0.15,
    ),
    "despair": MelodyProfile(
        contours=[ContourType.COLLAPSE, ContourType.DESCENDING, ContourType.STATIC],
        density=RhythmDensity.SPARSE,
        articulation=ArticulationType.LEGATO,
        interval_weights={1: 0.4, 2: 0.3, 3: 0.15, 5: 0.1, 7: 0.05},
        rest_probability=0.4,
        octave_range=(-1, 0),
        velocity_range=(25, 60),
        resolution_tendency=0.05,
        chromatic_tendency=0.3,
        repetition_tendency=0.6,
    ),
    "terror": MelodyProfile(
        contours=[ContourType.COLLAPSE, ContourType.JAGGED],
        density=RhythmDensity.FRANTIC,
        articulation=ArticulationType.STACCATO,
        interval_weights={1: 0.2, 6: 0.25, 7: 0.2, 11: 0.2, 12: 0.15},
        rest_probability=0.2,
        octave_range=(-1, 2),
        velocity_range=(40, 127),
        resolution_tendency=0.0,
        chromatic_tendency=0.5,
        repetition_tendency=0.3,
        trill_probability=0.25,
    ),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MelodyNote:
    """A single melody note."""
    pitch: int
    start_ticks: int
    duration_ticks: int
    velocity: int
    is_chromatic: bool = False
    is_grace_note: bool = False
    articulation: ArticulationType = ArticulationType.TENUTO


@dataclass
class MelodyOutput:
    """Generated melody output."""
    notes: List[MelodyNote]
    key: str
    mode: Mode
    bars: int
    tempo: int
    emotion: str
    contour_used: ContourType
    profile_used: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def key_to_midi(key: str, octave: int = 4) -> int:
    """Convert key name to MIDI note number."""
    key_clean = key.replace('b', '').replace('#', '').upper()
    base = CHROMATIC.index(key_clean) if key_clean in CHROMATIC else 0
    if 'b' in key:
        base -= 1
    elif '#' in key:
        base += 1
    return base + (octave + 1) * 12


def get_scale_pitches(
    key: str,
    mode: Mode,
    base_octave: int = 4,
    octave_range: Tuple[int, int] = (0, 1)
) -> List[int]:
    """Get all MIDI pitches in scale across octave range."""
    root = key_to_midi(key, base_octave)
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS[Mode.AEOLIAN])
    
    pitches = []
    for oct_offset in range(octave_range[0], octave_range[1] + 1):
        for interval in intervals:
            pitch = root + oct_offset * 12 + interval
            if 21 <= pitch <= 108:  # Piano range
                pitches.append(pitch)
    
    return sorted(set(pitches))


def weighted_choice(weights: Dict[int, float]) -> int:
    """Choose from weighted options."""
    items = list(weights.keys())
    probs = list(weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]
    return random.choices(items, weights=probs)[0]


def generate_contour_targets(
    contour: ContourType,
    num_points: int
) -> List[float]:
    """Generate target positions (0.0 to 1.0) for contour."""
    targets = []
    
    for i in range(num_points):
        t = i / max(1, num_points - 1)
        
        if contour == ContourType.ASCENDING:
            target = t * 0.7 + 0.15
        elif contour == ContourType.DESCENDING:
            target = (1 - t) * 0.7 + 0.15
        elif contour == ContourType.ARCH:
            target = -4 * (t - 0.5) ** 2 + 1
            target = target * 0.6 + 0.2
        elif contour == ContourType.INVERSE_ARCH:
            target = 4 * (t - 0.5) ** 2
            target = target * 0.6 + 0.2
        elif contour == ContourType.STATIC:
            target = 0.5 + random.uniform(-0.1, 0.1)
        elif contour == ContourType.WAVE:
            target = 0.5 + 0.3 * math.sin(t * math.pi * 2)
        elif contour == ContourType.SPIRAL_DOWN:
            base = (1 - t) * 0.6 + 0.2
            wave = 0.15 * math.sin(t * math.pi * 4)
            target = base + wave
        elif contour == ContourType.SPIRAL_UP:
            base = t * 0.6 + 0.2
            wave = 0.15 * math.sin(t * math.pi * 4)
            target = base + wave
        elif contour == ContourType.JAGGED:
            target = random.uniform(0.2, 0.8)
        elif contour == ContourType.COLLAPSE:
            if t < 0.7:
                target = 0.6 + random.uniform(-0.1, 0.1)
            else:
                collapse_t = (t - 0.7) / 0.3
                target = 0.6 - collapse_t * 0.5
        elif contour == ContourType.REACH_AND_FALL:
            if t < 0.4:
                target = 0.3 + t * 1.5
            else:
                target = 0.9 - (t - 0.4) * 0.8
        elif contour == ContourType.QUESTION:
            if t < 0.8:
                target = 0.4 + 0.2 * math.sin(t * math.pi * 2)
            else:
                target = 0.7  # End high (unresolved)
        else:
            target = 0.5
        
        targets.append(max(0.0, min(1.0, target)))
    
    return targets


def get_rhythm_pattern(
    density: RhythmDensity,
    bars: int,
    rest_probability: float,
    syncopation: float = 0.0
) -> List[Tuple[int, int]]:
    """Generate rhythm pattern as (start_tick, duration_ticks)."""
    ticks_per_bar = TICKS_PER_BEAT * 4
    total_ticks = ticks_per_bar * bars
    
    notes_per_bar = {
        RhythmDensity.SPARSE: random.randint(2, 4),
        RhythmDensity.MODERATE: random.randint(4, 8),
        RhythmDensity.DENSE: random.randint(8, 12),
        RhythmDensity.FRANTIC: random.randint(12, 16),
    }[density]
    
    rhythm = []
    current_tick = 0
    
    # Duration options
    durations = [
        TICKS_PER_BEAT // 4,   # 16th
        TICKS_PER_BEAT // 2,   # 8th
        TICKS_PER_BEAT,        # Quarter
        TICKS_PER_BEAT * 2,    # Half
    ]
    
    weights = {
        RhythmDensity.SPARSE: [0.1, 0.2, 0.4, 0.3],
        RhythmDensity.MODERATE: [0.2, 0.4, 0.3, 0.1],
        RhythmDensity.DENSE: [0.4, 0.4, 0.15, 0.05],
        RhythmDensity.FRANTIC: [0.6, 0.3, 0.1, 0.0],
    }[density]
    
    while current_tick < total_ticks:
        # Rest?
        if random.random() < rest_probability and rhythm:
            skip = random.choice([TICKS_PER_BEAT // 4, TICKS_PER_BEAT // 2, TICKS_PER_BEAT])
            current_tick += skip
            continue
        
        # Syncopation offset
        offset = 0
        if syncopation > 0 and random.random() < syncopation:
            offset = random.choice([-TICKS_PER_BEAT // 4, TICKS_PER_BEAT // 4])
        
        start = max(0, current_tick + offset)
        duration = random.choices(durations, weights=weights)[0]
        
        if start + duration > total_ticks:
            duration = total_ticks - start
        
        if duration > 0:
            rhythm.append((start, duration))
        
        current_tick += duration
    
    return rhythm


# =============================================================================
# MELODY ENGINE
# =============================================================================

class MelodyEngine:
    """
    Emotion-driven melody generator.
    
    Wired to EmotionThesaurus for automatic emotion mapping.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize with emotion thesaurus."""
        self.thesaurus = EmotionThesaurus(data_dir)
    
    def from_emotion(
        self,
        emotion: str,
        bars: int = 4,
        key: str = "C",
        octave: int = 4,
        chord_tones: Optional[List[int]] = None,
    ) -> MelodyOutput:
        """
        Generate melody from emotion word.
        
        Args:
            emotion: Any emotion word ("devastated", "joyful", etc.)
            bars: Number of bars
            key: Musical key
            octave: Base octave
            chord_tones: Optional chord context
        """
        node = self.thesaurus.lookup(emotion)
        if node:
            return self.from_node(node, bars, key, octave, chord_tones)
        
        # Fallback to string matching
        return self.generate(
            emotion=emotion,
            key=key,
            mode=Mode.AEOLIAN,
            bars=bars,
            octave=octave,
            chord_tones=chord_tones,
        )
    
    def from_node(
        self,
        node: EmotionNode,
        bars: int = 4,
        key: str = "C",
        octave: int = 4,
        chord_tones: Optional[List[int]] = None,
    ) -> MelodyOutput:
        """Generate melody from EmotionNode."""
        return self.generate(
            emotion=node.name,
            key=key,
            mode=node.musical.mode,
            bars=bars,
            octave=octave,
            chord_tones=chord_tones,
            params=node.musical,
            category=node.category,
        )
    
    def from_attributes(
        self,
        params: MusicalAttributes,
        bars: int = 4,
        key: str = "C",
        octave: int = 4,
        chord_tones: Optional[List[int]] = None,
    ) -> MelodyOutput:
        """Generate melody from MusicalAttributes directly."""
        return self.generate(
            emotion="custom",
            key=key,
            mode=params.mode,
            bars=bars,
            octave=octave,
            chord_tones=chord_tones,
            params=params,
        )
    
    def generate(
        self,
        emotion: str = "neutral",
        key: str = "C",
        mode: Mode = Mode.AEOLIAN,
        bars: int = 4,
        octave: int = 4,
        chord_tones: Optional[List[int]] = None,
        params: Optional[MusicalAttributes] = None,
        category: Optional[EmotionCategory] = None,
    ) -> MelodyOutput:
        """
        Full generation with all parameters.
        """
        # Get profile
        profile = self._get_profile(emotion, category, params)
        
        # Adjust profile based on MusicalAttributes if provided
        if params:
            profile = self._adjust_profile_from_params(profile, params)
        
        # Get scale
        scale_pitches = get_scale_pitches(key, mode, octave, profile.octave_range)
        if not scale_pitches:
            scale_pitches = list(range(48, 84))
        
        # Select contour
        contour = random.choice(profile.contours)
        
        # Generate rhythm
        rhythm = get_rhythm_pattern(
            profile.density,
            bars,
            profile.rest_probability,
            profile.syncopation,
        )
        
        if not rhythm:
            rhythm = [(0, TICKS_PER_BEAT)]
        
        # Generate contour targets
        targets = generate_contour_targets(contour, len(rhythm))
        
        # Generate notes
        notes = self._generate_notes(
            rhythm=rhythm,
            targets=targets,
            scale_pitches=scale_pitches,
            profile=profile,
            chord_tones=chord_tones,
        )
        
        return MelodyOutput(
            notes=notes,
            key=key,
            mode=mode,
            bars=bars,
            tempo=params.tempo_base if params else 100,
            emotion=emotion,
            contour_used=contour,
            profile_used=emotion,
        )
    
    def _get_profile(
        self,
        emotion: str,
        category: Optional[EmotionCategory],
        params: Optional[MusicalAttributes],
    ) -> MelodyProfile:
        """Get melody profile for emotion."""
        # Check specific overrides first
        emotion_lower = emotion.lower()
        for key in SPECIFIC_EMOTION_PROFILES:
            if key in emotion_lower:
                return SPECIFIC_EMOTION_PROFILES[key]
        
        # Use category default
        if category and category in CATEGORY_MELODY_PROFILES:
            return CATEGORY_MELODY_PROFILES[category]
        
        # Neutral default
        return MelodyProfile(
            contours=[ContourType.WAVE, ContourType.ARCH],
            density=RhythmDensity.MODERATE,
            articulation=ArticulationType.TENUTO,
            interval_weights={2: 0.25, 3: 0.25, 4: 0.2, 5: 0.2, 7: 0.1},
            rest_probability=0.2,
            octave_range=(0, 1),
            velocity_range=(60, 90),
            resolution_tendency=0.5,
            chromatic_tendency=0.1,
            repetition_tendency=0.2,
        )
    
    def _adjust_profile_from_params(
        self,
        profile: MelodyProfile,
        params: MusicalAttributes,
    ) -> MelodyProfile:
        """Adjust profile based on MusicalAttributes."""
        # Create a modified copy
        return MelodyProfile(
            contours=profile.contours,
            density=profile.density,
            articulation=profile.articulation,
            interval_weights=profile.interval_weights,
            rest_probability=profile.rest_probability * (1 - params.note_density),
            octave_range=profile.octave_range,
            velocity_range=(
                max(1, params.velocity_range[0]),
                min(127, params.velocity_range[1]),
            ),
            resolution_tendency=profile.resolution_tendency,
            chromatic_tendency=profile.chromatic_tendency + params.dissonance_level * 0.2,
            repetition_tendency=profile.repetition_tendency,
            syncopation=params.syncopation,
            grace_note_probability=profile.grace_note_probability,
            trill_probability=profile.trill_probability,
        )
    
    def _generate_notes(
        self,
        rhythm: List[Tuple[int, int]],
        targets: List[float],
        scale_pitches: List[int],
        profile: MelodyProfile,
        chord_tones: Optional[List[int]],
    ) -> List[MelodyNote]:
        """Generate actual notes."""
        notes = []
        prev_pitch = scale_pitches[len(scale_pitches) // 2]
        
        for (start, dur), target in zip(rhythm, targets):
            # Target pitch from contour
            target_idx = int(target * (len(scale_pitches) - 1))
            target_pitch = scale_pitches[target_idx]
            
            # Choose interval
            interval = weighted_choice(profile.interval_weights)
            
            # Direction toward target
            direction = 1 if target_pitch > prev_pitch else -1
            if target_pitch == prev_pitch:
                direction = random.choice([-1, 1])
            
            # Calculate pitch
            new_pitch = prev_pitch + (interval * direction)
            
            # Chromatic or diatonic
            is_chromatic = False
            if random.random() < profile.chromatic_tendency:
                is_chromatic = new_pitch not in scale_pitches
            else:
                # Snap to scale
                if new_pitch not in scale_pitches:
                    distances = [(abs(p - new_pitch), p) for p in scale_pitches]
                    new_pitch = min(distances)[1]
            
            # Clamp
            new_pitch = max(scale_pitches[0], min(scale_pitches[-1], new_pitch))
            
            # Repetition tendency
            if random.random() < profile.repetition_tendency and notes:
                new_pitch = prev_pitch
            
            # Chord tone emphasis
            if chord_tones and random.random() < 0.4:
                chord_matches = [p for p in scale_pitches 
                               if any(ct % 12 == p % 12 for ct in chord_tones)]
                if chord_matches:
                    new_pitch = min(chord_matches, key=lambda p: abs(p - prev_pitch))
            
            # Velocity
            vel_min, vel_max = profile.velocity_range
            velocity = random.randint(vel_min, vel_max)
            
            # Articulation adjusts duration
            actual_dur = dur
            if profile.articulation == ArticulationType.STACCATO:
                actual_dur = int(dur * 0.5)
            elif profile.articulation == ArticulationType.LEGATO:
                actual_dur = int(dur * 0.95)
            elif profile.articulation == ArticulationType.BREATH:
                actual_dur = int(dur * 0.8)
            
            # Grace note?
            if profile.grace_note_probability > 0 and random.random() < profile.grace_note_probability:
                grace_pitch = new_pitch + random.choice([-1, 1, -2, 2])
                notes.append(MelodyNote(
                    pitch=grace_pitch,
                    start_ticks=max(0, start - TICKS_PER_BEAT // 8),
                    duration_ticks=TICKS_PER_BEAT // 8,
                    velocity=velocity - 15,
                    is_grace_note=True,
                    articulation=profile.articulation,
                ))
            
            notes.append(MelodyNote(
                pitch=new_pitch,
                start_ticks=start,
                duration_ticks=actual_dur,
                velocity=velocity,
                is_chromatic=is_chromatic,
                articulation=profile.articulation,
            ))
            
            prev_pitch = new_pitch
        
        return notes
    
    def to_midi(
        self,
        output: MelodyOutput,
        channel: int = 0,
    ) -> mido.MidiFile:
        """Convert MelodyOutput to MIDI file."""
        mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(output.tempo)))
        track.append(mido.MetaMessage('track_name', name=f"Melody: {output.emotion}"))
        
        # Create events
        events = []
        for note in output.notes:
            events.append(('on', note.start_ticks, note.pitch, note.velocity))
            events.append(('off', note.start_ticks + note.duration_ticks, note.pitch, 0))
        
        events.sort(key=lambda x: (x[1], 0 if x[0] == 'off' else 1))
        
        current_time = 0
        for event_type, abs_time, pitch, vel in events:
            delta = abs_time - current_time
            msg_type = 'note_on' if event_type == 'on' else 'note_off'
            track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta, channel=channel))
            current_time = abs_time
        
        return mid


# =============================================================================
# CLI / TEST
# =============================================================================

def main():
    import sys
    
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    
    engine = MelodyEngine(data_dir)
    
    print("=== KELLY MELODY ENGINE TEST ===\n")
    
    test_emotions = ["devastated", "furious", "joyful", "anxious", "longing"]
    
    for emotion in test_emotions:
        print(f"Generating melody for '{emotion}'...")
        
        melody = engine.from_emotion(emotion, bars=4, key="F")
        
        print(f"  Contour: {melody.contour_used.value}")
        print(f"  Notes: {len(melody.notes)}")
        print(f"  Pitch range: {min(n.pitch for n in melody.notes)}-{max(n.pitch for n in melody.notes)}")
        
        midi = engine.to_midi(melody)
        output_path = output_dir / f"melody_{emotion}.mid"
        midi.save(str(output_path))
        print(f"  Saved: {output_path}\n")
    
    print("=== MELODY WITH CHORD CONTEXT ===")
    melody = engine.from_emotion(
        "grief",
        bars=4,
        key="F",
        chord_tones=[65, 68, 72],  # F minor
    )
    print(f"Grief melody with Fm chord context: {len(melody.notes)} notes")
    
    midi = engine.to_midi(melody)
    output_path = output_dir / "melody_grief_with_chord.mid"
    midi.save(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
