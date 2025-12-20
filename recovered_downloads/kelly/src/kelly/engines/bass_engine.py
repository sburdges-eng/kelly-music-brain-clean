"""Bass Engine - Generates emotion-driven bass lines.

Bass isn't just chord roots. A grief bass line breathes and sighs.
A defiant bass line pushes against the beat. An anxious bass line never settles.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
import math

TICKS_PER_BEAT = 480
CHROMATIC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class BassPattern(Enum):
    """Bass line pattern archetypes."""
    ROOT_ONLY = "root_only"
    ROOT_FIFTH = "root_fifth"
    WALKING = "walking"
    PEDAL = "pedal"
    ARPEGGIATED = "arpeggiated"
    SYNCOPATED = "syncopated"
    DRIVING = "driving"
    PULSING = "pulsing"
    BREATHING = "breathing"
    DESCENDING = "descending"
    CLIMBING = "climbing"
    GHOST = "ghost"


class BassArticulation(Enum):
    """Bass note articulation."""
    SUSTAINED = "sustained"
    STACCATO = "staccato"
    MUTED = "muted"
    SLIDE = "slide"
    DEAD = "dead"


@dataclass
class BassConfig:
    """Configuration for bass generation."""
    emotion: str = "neutral"
    chord_progression: List[str] = field(default_factory=list)
    key: str = "C"
    bars: int = 4
    tempo_bpm: int = 120
    pattern_override: Optional[BassPattern] = None
    octave: int = 2
    seed: int = -1


@dataclass
class BassNote:
    """A single bass note."""
    pitch: int
    start_tick: int
    duration_ticks: int
    velocity: int
    articulation: BassArticulation = BassArticulation.SUSTAINED
    technique: str = "finger"


@dataclass
class BassOutput:
    """Complete bass generation output."""
    notes: List[BassNote]
    pattern_used: BassPattern
    emotion: str
    key: str
    bars: int


EMOTION_PROFILES = {
    "grief": {
        "pattern": BassPattern.BREATHING,
        "velocity_range": (40, 65),
        "sustain_ratio": 0.9,
        "rest_probability": 0.3,
        "octave_preference": 2,
        "articulation": BassArticulation.SUSTAINED,
    },
    "sadness": {
        "pattern": BassPattern.ROOT_FIFTH,
        "velocity_range": (45, 70),
        "sustain_ratio": 0.85,
        "rest_probability": 0.2,
        "octave_preference": 2,
        "articulation": BassArticulation.SUSTAINED,
    },
    "anger": {
        "pattern": BassPattern.DRIVING,
        "velocity_range": (80, 115),
        "sustain_ratio": 0.5,
        "rest_probability": 0.05,
        "octave_preference": 1,
        "articulation": BassArticulation.STACCATO,
    },
    "anxiety": {
        "pattern": BassPattern.SYNCOPATED,
        "velocity_range": (55, 85),
        "sustain_ratio": 0.6,
        "rest_probability": 0.15,
        "octave_preference": 2,
        "articulation": BassArticulation.STACCATO,
    },
    "joy": {
        "pattern": BassPattern.WALKING,
        "velocity_range": (70, 95),
        "sustain_ratio": 0.7,
        "rest_probability": 0.1,
        "octave_preference": 2,
        "articulation": BassArticulation.SUSTAINED,
    },
    "hope": {
        "pattern": BassPattern.CLIMBING,
        "velocity_range": (60, 85),
        "sustain_ratio": 0.75,
        "rest_probability": 0.1,
        "octave_preference": 2,
        "articulation": BassArticulation.SUSTAINED,
    },
    "defiance": {
        "pattern": BassPattern.SYNCOPATED,
        "velocity_range": (75, 105),
        "sustain_ratio": 0.55,
        "rest_probability": 0.08,
        "octave_preference": 1,
        "articulation": BassArticulation.MUTED,
    },
    "emptiness": {
        "pattern": BassPattern.PEDAL,
        "velocity_range": (30, 50),
        "sustain_ratio": 0.95,
        "rest_probability": 0.4,
        "octave_preference": 2,
        "articulation": BassArticulation.SUSTAINED,
    },
    "nostalgia": {
        "pattern": BassPattern.ROOT_FIFTH,
        "velocity_range": (50, 75),
        "sustain_ratio": 0.8,
        "rest_probability": 0.15,
        "octave_preference": 2,
        "articulation": BassArticulation.SUSTAINED,
    },
}


def note_to_midi(note: str, octave: int = 2) -> int:
    """Convert note name to MIDI number."""
    note = note.upper().replace('♯', '#').replace('♭', 'b')
    if 'B' in note and note != 'B' and len(note) > 1:
        flat_map = {'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#'}
        note = flat_map.get(note, note)
    base = note.replace('#', '').replace('m', '').replace('M', '')[:1]
    idx = CHROMATIC.index(base) if base in CHROMATIC else 0
    if '#' in note:
        idx = (idx + 1) % 12
    return 12 + (octave * 12) + idx


def parse_chord(chord: str) -> Tuple[str, str]:
    """Parse chord into root and quality."""
    chord = chord.strip()
    if not chord:
        return "C", "maj"
    
    root = chord[0].upper()
    rest = chord[1:] if len(chord) > 1 else ""
    
    if rest.startswith('#') or rest.startswith('b'):
        root += rest[0]
        rest = rest[1:]
    
    quality = "min" if 'm' in rest.lower() and 'maj' not in rest.lower() else "maj"
    return root, quality


class BassEngine:
    """Generates emotion-driven bass lines."""
    
    def __init__(self, seed: Optional[int] = None):
        self.profiles = EMOTION_PROFILES
        if seed is not None:
            random.seed(seed)
    
    def generate(self, config: BassConfig) -> BassOutput:
        """Generate bass line from config."""
        if config.seed >= 0:
            random.seed(config.seed)
        
        emotion = config.emotion.lower()
        profile = self.profiles.get(emotion, self.profiles["hope"])
        
        pattern = config.pattern_override or profile["pattern"]
        notes = []
        
        ticks_per_bar = TICKS_PER_BEAT * 4
        
        for bar_idx in range(config.bars):
            chord_idx = bar_idx % len(config.chord_progression) if config.chord_progression else 0
            chord = config.chord_progression[chord_idx] if config.chord_progression else config.key
            root, quality = parse_chord(chord)
            root_midi = note_to_midi(root, profile["octave_preference"])
            
            bar_start = bar_idx * ticks_per_bar
            bar_notes = self._generate_pattern(
                pattern, root_midi, quality, bar_start, ticks_per_bar, profile
            )
            notes.extend(bar_notes)
        
        return BassOutput(
            notes=notes,
            pattern_used=pattern,
            emotion=emotion,
            key=config.key,
            bars=config.bars,
        )
    
    def _generate_pattern(
        self,
        pattern: BassPattern,
        root: int,
        quality: str,
        start_tick: int,
        bar_ticks: int,
        profile: Dict
    ) -> List[BassNote]:
        """Generate notes for a specific pattern."""
        vel_min, vel_max = profile["velocity_range"]
        sustain = profile["sustain_ratio"]
        rest_prob = profile["rest_probability"]
        art = profile["articulation"]
        
        fifth = root + 7
        third = root + (3 if quality == "min" else 4)
        
        notes = []
        
        if pattern == BassPattern.ROOT_ONLY:
            if random.random() > rest_prob:
                notes.append(BassNote(
                    pitch=root,
                    start_tick=start_tick,
                    duration_ticks=int(bar_ticks * sustain),
                    velocity=random.randint(vel_min, vel_max),
                    articulation=art,
                ))
        
        elif pattern == BassPattern.ROOT_FIFTH:
            half = bar_ticks // 2
            if random.random() > rest_prob:
                notes.append(BassNote(
                    pitch=root,
                    start_tick=start_tick,
                    duration_ticks=int(half * sustain),
                    velocity=random.randint(vel_min, vel_max),
                    articulation=art,
                ))
            if random.random() > rest_prob:
                notes.append(BassNote(
                    pitch=fifth,
                    start_tick=start_tick + half,
                    duration_ticks=int(half * sustain),
                    velocity=random.randint(vel_min, vel_max) - 5,
                    articulation=art,
                ))
        
        elif pattern == BassPattern.WALKING:
            eighth = TICKS_PER_BEAT // 2
            scale = [root, root + 2, third, root + 5, fifth, root + 9, root + 10, root + 12]
            for i in range(bar_ticks // eighth):
                if random.random() > rest_prob:
                    pitch = random.choice(scale[:4]) if i < 4 else random.choice(scale[4:])
                    notes.append(BassNote(
                        pitch=pitch,
                        start_tick=start_tick + i * eighth,
                        duration_ticks=int(eighth * 0.9),
                        velocity=random.randint(vel_min, vel_max),
                        articulation=art,
                    ))
        
        elif pattern == BassPattern.DRIVING:
            eighth = TICKS_PER_BEAT // 2
            for i in range(bar_ticks // eighth):
                vel = vel_max if i % 2 == 0 else vel_min + (vel_max - vel_min) // 2
                notes.append(BassNote(
                    pitch=root if i % 4 < 3 else fifth,
                    start_tick=start_tick + i * eighth,
                    duration_ticks=int(eighth * sustain),
                    velocity=vel,
                    articulation=art,
                ))
        
        elif pattern == BassPattern.BREATHING:
            notes.append(BassNote(
                pitch=root,
                start_tick=start_tick,
                duration_ticks=int(bar_ticks * 0.6),
                velocity=random.randint(vel_min, vel_max),
                articulation=BassArticulation.SUSTAINED,
            ))
            if random.random() > rest_prob:
                notes.append(BassNote(
                    pitch=fifth,
                    start_tick=start_tick + int(bar_ticks * 0.7),
                    duration_ticks=int(bar_ticks * 0.25),
                    velocity=random.randint(vel_min, vel_max) - 10,
                    articulation=BassArticulation.SUSTAINED,
                ))
        
        elif pattern == BassPattern.PEDAL:
            notes.append(BassNote(
                pitch=root,
                start_tick=start_tick,
                duration_ticks=bar_ticks - 10,
                velocity=random.randint(vel_min, vel_max),
                articulation=BassArticulation.SUSTAINED,
            ))
        
        elif pattern == BassPattern.SYNCOPATED:
            positions = [0, TICKS_PER_BEAT * 0.75, TICKS_PER_BEAT * 1.5, TICKS_PER_BEAT * 2.75]
            for pos in positions:
                if random.random() > rest_prob:
                    notes.append(BassNote(
                        pitch=random.choice([root, fifth, root]),
                        start_tick=start_tick + int(pos),
                        duration_ticks=int(TICKS_PER_BEAT * sustain),
                        velocity=random.randint(vel_min, vel_max),
                        articulation=art,
                    ))
        
        elif pattern == BassPattern.CLIMBING:
            scale = [root, root + 2, third, root + 5, fifth]
            quarter = TICKS_PER_BEAT
            for i, pitch in enumerate(scale[:4]):
                notes.append(BassNote(
                    pitch=pitch,
                    start_tick=start_tick + i * quarter,
                    duration_ticks=int(quarter * sustain),
                    velocity=vel_min + int((vel_max - vel_min) * i / 4),
                    articulation=art,
                ))
        
        return notes


def generate_bass(
    emotion: str,
    chords: List[str],
    key: str = "C",
    bars: int = 4,
    tempo: int = 120
) -> BassOutput:
    """Quick bass generation helper."""
    engine = BassEngine()
    config = BassConfig(
        emotion=emotion,
        chord_progression=chords,
        key=key,
        bars=bars,
        tempo_bpm=tempo,
    )
    return engine.generate(config)
