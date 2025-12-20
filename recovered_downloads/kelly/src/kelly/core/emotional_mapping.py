"""Emotional Mapping - Maps emotional states to musical parameters.

Core mapping between the psychological dimensions of emotion
(valence, arousal) and concrete musical attributes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class Valence(Enum):
    """Emotional valence (positive/negative)."""
    VERY_NEGATIVE = -1.0
    NEGATIVE = -0.5
    NEUTRAL = 0.0
    POSITIVE = 0.5
    VERY_POSITIVE = 1.0


class Arousal(Enum):
    """Emotional arousal (energy level)."""
    VERY_LOW = 0.0
    LOW = 0.25
    MODERATE = 0.5
    HIGH = 0.75
    VERY_HIGH = 1.0


class TimingFeel(Enum):
    """Rhythmic feel/groove."""
    STRAIGHT = "straight"
    SWING = "swing"
    SHUFFLE = "shuffle"
    BEHIND = "behind_the_beat"
    AHEAD = "ahead_of_beat"
    RUBATO = "rubato"
    MECHANICAL = "mechanical"
    HUMAN = "human"


class Mode(Enum):
    """Musical modes."""
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"


@dataclass
class EmotionalState:
    """Represents a complete emotional state."""
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    primary_emotion: str
    secondary_emotions: List[str] = field(default_factory=list)
    intensity: float = 0.5
    
    def quadrant(self) -> str:
        """Get the emotional quadrant."""
        if self.valence >= 0 and self.arousal >= 0.5:
            return "excited_positive"  # Joy, excitement
        elif self.valence >= 0 and self.arousal < 0.5:
            return "calm_positive"  # Serenity, contentment
        elif self.valence < 0 and self.arousal >= 0.5:
            return "excited_negative"  # Anger, fear
        else:
            return "calm_negative"  # Sadness, melancholy


@dataclass
class MusicalParameters:
    """Complete musical parameters derived from emotion."""
    # Tempo
    tempo_suggested: int
    tempo_min: int
    tempo_max: int
    
    # Harmony
    mode: Mode
    harmonic_complexity: float  # 0-1
    dissonance: float  # 0-1
    
    # Rhythm
    timing_feel: TimingFeel
    swing_amount: float  # 0-1
    rhythmic_density: float  # 0-1
    
    # Dynamics
    velocity_min: int
    velocity_max: int
    dynamic_variance: float  # 0-1
    
    # Texture
    density_suggested: float  # 0-1
    space_probability: float  # 0-1, chance of rests
    register_low: int  # MIDI note
    register_high: int  # MIDI note
    
    # Articulation
    articulation: str
    sustain_ratio: float  # 0-1, note length vs grid
    
    # Expression
    humanize_timing: float  # 0-1
    humanize_velocity: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tempo": {
                "suggested": self.tempo_suggested,
                "min": self.tempo_min,
                "max": self.tempo_max,
            },
            "harmony": {
                "mode": self.mode.value,
                "complexity": self.harmonic_complexity,
                "dissonance": self.dissonance,
            },
            "rhythm": {
                "timing_feel": self.timing_feel.value,
                "swing": self.swing_amount,
                "density": self.rhythmic_density,
            },
            "dynamics": {
                "velocity_min": self.velocity_min,
                "velocity_max": self.velocity_max,
                "variance": self.dynamic_variance,
            },
            "texture": {
                "density": self.density_suggested,
                "space_probability": self.space_probability,
                "register": (self.register_low, self.register_high),
            },
            "articulation": {
                "type": self.articulation,
                "sustain_ratio": self.sustain_ratio,
            },
            "humanization": {
                "timing": self.humanize_timing,
                "velocity": self.humanize_velocity,
            },
        }


# Pre-defined emotional presets
EMOTIONAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "grief": {
        "valence": -0.8,
        "arousal": 0.3,
        "tempo_range": (55, 75),
        "mode": Mode.MINOR,
        "timing_feel": TimingFeel.BEHIND,
        "harmonic_complexity": 0.6,
        "dissonance": 0.4,
        "velocity_range": (35, 70),
        "density": 0.3,
        "space": 0.5,
        "articulation": "legato",
        "humanize": 0.7,
    },
    "sadness": {
        "valence": -0.6,
        "arousal": 0.25,
        "tempo_range": (60, 85),
        "mode": Mode.AEOLIAN,
        "timing_feel": TimingFeel.BEHIND,
        "harmonic_complexity": 0.5,
        "dissonance": 0.3,
        "velocity_range": (40, 75),
        "density": 0.35,
        "space": 0.4,
        "articulation": "legato",
        "humanize": 0.6,
    },
    "melancholy": {
        "valence": -0.4,
        "arousal": 0.2,
        "tempo_range": (65, 90),
        "mode": Mode.DORIAN,
        "timing_feel": TimingFeel.SWING,
        "harmonic_complexity": 0.55,
        "dissonance": 0.25,
        "velocity_range": (45, 80),
        "density": 0.4,
        "space": 0.35,
        "articulation": "tenuto",
        "humanize": 0.5,
    },
    "anger": {
        "valence": -0.7,
        "arousal": 0.85,
        "tempo_range": (110, 150),
        "mode": Mode.PHRYGIAN,
        "timing_feel": TimingFeel.AHEAD,
        "harmonic_complexity": 0.7,
        "dissonance": 0.6,
        "velocity_range": (80, 127),
        "density": 0.8,
        "space": 0.1,
        "articulation": "staccato",
        "humanize": 0.4,
    },
    "rage": {
        "valence": -0.9,
        "arousal": 1.0,
        "tempo_range": (130, 180),
        "mode": Mode.LOCRIAN,
        "timing_feel": TimingFeel.MECHANICAL,
        "harmonic_complexity": 0.8,
        "dissonance": 0.8,
        "velocity_range": (100, 127),
        "density": 0.9,
        "space": 0.05,
        "articulation": "marcato",
        "humanize": 0.2,
    },
    "anxiety": {
        "valence": -0.5,
        "arousal": 0.7,
        "tempo_range": (95, 125),
        "mode": Mode.LOCRIAN,
        "timing_feel": TimingFeel.AHEAD,
        "harmonic_complexity": 0.65,
        "dissonance": 0.5,
        "velocity_range": (55, 95),
        "density": 0.6,
        "space": 0.2,
        "articulation": "staccato",
        "humanize": 0.8,
    },
    "fear": {
        "valence": -0.6,
        "arousal": 0.75,
        "tempo_range": (85, 115),
        "mode": Mode.PHRYGIAN,
        "timing_feel": TimingFeel.RUBATO,
        "harmonic_complexity": 0.6,
        "dissonance": 0.55,
        "velocity_range": (40, 90),
        "density": 0.5,
        "space": 0.3,
        "articulation": "tenuto",
        "humanize": 0.85,
    },
    "joy": {
        "valence": 0.8,
        "arousal": 0.7,
        "tempo_range": (110, 140),
        "mode": Mode.MAJOR,
        "timing_feel": TimingFeel.SWING,
        "harmonic_complexity": 0.45,
        "dissonance": 0.15,
        "velocity_range": (70, 110),
        "density": 0.65,
        "space": 0.2,
        "articulation": "staccato",
        "humanize": 0.5,
    },
    "euphoria": {
        "valence": 1.0,
        "arousal": 0.9,
        "tempo_range": (125, 160),
        "mode": Mode.LYDIAN,
        "timing_feel": TimingFeel.AHEAD,
        "harmonic_complexity": 0.5,
        "dissonance": 0.1,
        "velocity_range": (85, 127),
        "density": 0.75,
        "space": 0.1,
        "articulation": "marcato",
        "humanize": 0.3,
    },
    "serenity": {
        "valence": 0.6,
        "arousal": 0.15,
        "tempo_range": (55, 75),
        "mode": Mode.MAJOR,
        "timing_feel": TimingFeel.RUBATO,
        "harmonic_complexity": 0.4,
        "dissonance": 0.1,
        "velocity_range": (35, 65),
        "density": 0.25,
        "space": 0.5,
        "articulation": "legato",
        "humanize": 0.6,
    },
    "contentment": {
        "valence": 0.5,
        "arousal": 0.3,
        "tempo_range": (70, 95),
        "mode": Mode.MIXOLYDIAN,
        "timing_feel": TimingFeel.SWING,
        "harmonic_complexity": 0.35,
        "dissonance": 0.15,
        "velocity_range": (50, 80),
        "density": 0.4,
        "space": 0.35,
        "articulation": "tenuto",
        "humanize": 0.5,
    },
    "hope": {
        "valence": 0.4,
        "arousal": 0.5,
        "tempo_range": (85, 110),
        "mode": Mode.MAJOR,
        "timing_feel": TimingFeel.STRAIGHT,
        "harmonic_complexity": 0.5,
        "dissonance": 0.2,
        "velocity_range": (55, 90),
        "density": 0.5,
        "space": 0.3,
        "articulation": "legato",
        "humanize": 0.5,
    },
    "nostalgia": {
        "valence": 0.1,
        "arousal": 0.25,
        "tempo_range": (70, 90),
        "mode": Mode.MIXOLYDIAN,
        "timing_feel": TimingFeel.BEHIND,
        "harmonic_complexity": 0.55,
        "dissonance": 0.25,
        "velocity_range": (45, 75),
        "density": 0.4,
        "space": 0.4,
        "articulation": "legato",
        "humanize": 0.7,
    },
    "longing": {
        "valence": -0.2,
        "arousal": 0.35,
        "tempo_range": (65, 85),
        "mode": Mode.DORIAN,
        "timing_feel": TimingFeel.BEHIND,
        "harmonic_complexity": 0.6,
        "dissonance": 0.3,
        "velocity_range": (40, 70),
        "density": 0.35,
        "space": 0.45,
        "articulation": "legato",
        "humanize": 0.65,
    },
    "defiance": {
        "valence": -0.3,
        "arousal": 0.8,
        "tempo_range": (105, 135),
        "mode": Mode.DORIAN,
        "timing_feel": TimingFeel.AHEAD,
        "harmonic_complexity": 0.6,
        "dissonance": 0.4,
        "velocity_range": (75, 115),
        "density": 0.7,
        "space": 0.15,
        "articulation": "marcato",
        "humanize": 0.35,
    },
    "emptiness": {
        "valence": -0.5,
        "arousal": 0.1,
        "tempo_range": (45, 65),
        "mode": Mode.AEOLIAN,
        "timing_feel": TimingFeel.RUBATO,
        "harmonic_complexity": 0.3,
        "dissonance": 0.2,
        "velocity_range": (25, 55),
        "density": 0.15,
        "space": 0.7,
        "articulation": "legato",
        "humanize": 0.8,
    },
    "wonder": {
        "valence": 0.3,
        "arousal": 0.45,
        "tempo_range": (75, 100),
        "mode": Mode.LYDIAN,
        "timing_feel": TimingFeel.RUBATO,
        "harmonic_complexity": 0.65,
        "dissonance": 0.2,
        "velocity_range": (50, 85),
        "density": 0.45,
        "space": 0.35,
        "articulation": "legato",
        "humanize": 0.6,
    },
    "determination": {
        "valence": 0.2,
        "arousal": 0.65,
        "tempo_range": (95, 120),
        "mode": Mode.DORIAN,
        "timing_feel": TimingFeel.STRAIGHT,
        "harmonic_complexity": 0.5,
        "dissonance": 0.25,
        "velocity_range": (65, 100),
        "density": 0.6,
        "space": 0.2,
        "articulation": "tenuto",
        "humanize": 0.4,
    },
}


def get_parameters_for_state(state: EmotionalState) -> MusicalParameters:
    """Generate musical parameters from an emotional state."""
    # Try to find preset match
    preset = EMOTIONAL_PRESETS.get(state.primary_emotion.lower())
    
    if preset:
        tempo_range = preset["tempo_range"]
        tempo_suggested = int((tempo_range[0] + tempo_range[1]) / 2)
        
        return MusicalParameters(
            tempo_suggested=tempo_suggested,
            tempo_min=tempo_range[0],
            tempo_max=tempo_range[1],
            mode=preset["mode"],
            harmonic_complexity=preset["harmonic_complexity"],
            dissonance=preset["dissonance"],
            timing_feel=preset["timing_feel"],
            swing_amount=0.15 if preset["timing_feel"] == TimingFeel.SWING else 0,
            rhythmic_density=preset["density"],
            velocity_min=preset["velocity_range"][0],
            velocity_max=preset["velocity_range"][1],
            dynamic_variance=0.3,
            density_suggested=preset["density"],
            space_probability=preset["space"],
            register_low=36,
            register_high=84,
            articulation=preset["articulation"],
            sustain_ratio=0.85 if preset["articulation"] == "legato" else 0.6,
            humanize_timing=preset["humanize"],
            humanize_velocity=preset["humanize"] * 0.8,
        )
    
    # Generate from dimensions
    tempo_base = 60 + int(state.arousal * 80)
    
    # Mode from valence
    if state.valence > 0.3:
        mode = Mode.MAJOR if state.arousal < 0.6 else Mode.LYDIAN
    elif state.valence > -0.3:
        mode = Mode.DORIAN if state.arousal > 0.5 else Mode.MIXOLYDIAN
    else:
        mode = Mode.MINOR if state.arousal < 0.6 else Mode.PHRYGIAN
    
    # Timing feel from arousal
    if state.arousal > 0.7:
        timing = TimingFeel.AHEAD
    elif state.arousal < 0.3:
        timing = TimingFeel.BEHIND
    else:
        timing = TimingFeel.STRAIGHT
    
    return MusicalParameters(
        tempo_suggested=tempo_base,
        tempo_min=tempo_base - 15,
        tempo_max=tempo_base + 15,
        mode=mode,
        harmonic_complexity=0.3 + state.arousal * 0.4,
        dissonance=max(0, -state.valence * 0.5) + state.intensity * 0.2,
        timing_feel=timing,
        swing_amount=0.15 if state.valence > 0 else 0,
        rhythmic_density=0.3 + state.arousal * 0.5,
        velocity_min=30 + int(state.intensity * 40),
        velocity_max=70 + int(state.intensity * 50),
        dynamic_variance=state.intensity * 0.4,
        density_suggested=0.3 + state.arousal * 0.4,
        space_probability=0.5 - state.arousal * 0.3,
        register_low=36,
        register_high=84,
        articulation="legato" if state.arousal < 0.5 else "staccato",
        sustain_ratio=0.9 - state.arousal * 0.3,
        humanize_timing=0.5,
        humanize_velocity=0.4,
    )


def describe_parameters(params: MusicalParameters) -> str:
    """Generate human-readable description of parameters."""
    lines = [
        f"Tempo: {params.tempo_suggested} BPM ({params.tempo_min}-{params.tempo_max})",
        f"Mode: {params.mode.value}",
        f"Feel: {params.timing_feel.value}",
        f"Dynamics: {params.velocity_min}-{params.velocity_max}",
        f"Density: {params.density_suggested:.0%}",
        f"Dissonance: {params.dissonance:.0%}",
        f"Articulation: {params.articulation}",
    ]
    return "\n".join(lines)


def interpolate_states(
    state_a: EmotionalState,
    state_b: EmotionalState,
    t: float
) -> EmotionalState:
    """Interpolate between two emotional states (for transitions)."""
    return EmotionalState(
        valence=state_a.valence + (state_b.valence - state_a.valence) * t,
        arousal=state_a.arousal + (state_b.arousal - state_a.arousal) * t,
        primary_emotion=state_a.primary_emotion if t < 0.5 else state_b.primary_emotion,
        intensity=state_a.intensity + (state_b.intensity - state_a.intensity) * t,
    )
