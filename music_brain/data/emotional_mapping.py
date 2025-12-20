# music_brain/models/emotional_mapping.py
"""
Emotional-to-Musical Parameter Mapping for DAiW

Based on Russell's Circumplex Model (arousal x valence) and 
research on music-emotion correlations.

This module provides the vocabulary for emotional interrogation,
not automated output generation.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional


class Valence(Enum):
    """Emotional valence (positive/negative)"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


class Arousal(Enum):
    """Emotional arousal (energy level)"""
    VERY_LOW = -2
    LOW = -1
    NEUTRAL = 0
    HIGH = 1
    VERY_HIGH = 2


class TimingFeel(Enum):
    """Where notes sit relative to the beat"""
    AHEAD = "ahead"      # Anxious, urgent
    ON = "on"            # Present, assertive
    BEHIND = "behind"    # Relaxed, grief, resignation


class Mode(Enum):
    """Musical modes with emotional associations"""
    MAJOR = "major"              # Bright, resolved
    MINOR = "minor"              # Sad, introspective
    DORIAN = "dorian"            # Minor but hopeful
    PHRYGIAN = "phrygian"        # Dark, exotic, anxious
    LYDIAN = "lydian"            # Dreamy, floating
    MIXOLYDIAN = "mixolydian"    # Bluesy major, nostalgic
    LOCRIAN = "locrian"          # Unstable, diminished
    

@dataclass
class MusicalParameters:
    """
    Musical parameters derived from emotional state.
    
    These are suggestions for interrogation, not mandates.
    """
    # Tempo
    tempo_min: int = 60
    tempo_max: int = 120
    tempo_suggested: int = 90
    
    # Mode preferences (weights 0-1)
    mode_weights: dict[Mode, float] = field(default_factory=dict)
    
    # Register
    register_low: int = 36   # MIDI note (C2)
    register_high: int = 84  # MIDI note (C6)
    register_center: int = 60  # Where melody gravitates
    
    # Harmonic rhythm (chord changes per bar)
    harmonic_rhythm_min: float = 0.5
    harmonic_rhythm_max: float = 4.0
    harmonic_rhythm_suggested: float = 1.0
    
    # Dissonance level (0-1)
    dissonance: float = 0.3
    
    # Timing feel
    timing_feel: TimingFeel = TimingFeel.ON
    
    # Note density (notes per beat, roughly)
    density_min: float = 0.25
    density_max: float = 4.0
    density_suggested: float = 1.0
    
    # Dynamics (0-1, maps to velocity)
    dynamics_floor: float = 0.3
    dynamics_ceiling: float = 0.7
    dynamics_variance: float = 0.2  # How much it moves
    
    # Silence/space (probability of rest)
    space_probability: float = 0.2


@dataclass
class EmotionalState:
    """
    Core emotional state from interrogation.
    
    This is extracted through conversation, not assumed.
    """
    # Primary dimensions
    valence: Valence = Valence.NEUTRAL
    arousal: Arousal = Arousal.NEUTRAL
    
    # Named emotion (optional, for presets)
    primary_emotion: Optional[str] = None
    
    # Compound/layered emotions
    secondary_emotions: list[str] = field(default_factory=list)
    
    # Intrusion events (for trauma/PTSD patterns)
    has_intrusions: bool = False
    intrusion_probability: float = 0.0
    intrusion_types: list[str] = field(default_factory=list)
    
    # Context
    context_notes: str = ""


# =============================================================================
# EMOTIONAL PRESETS
# =============================================================================
# These are starting points for interrogation, not final answers.
# The interrogation engine should ask: "Does this feel right? What's missing?"

EMOTIONAL_PRESETS: dict[str, MusicalParameters] = {
    
    "grief": MusicalParameters(
        tempo_min=60,
        tempo_max=82,
        tempo_suggested=72,
        mode_weights={
            Mode.MINOR: 0.4,
            Mode.DORIAN: 0.4,
            Mode.MAJOR: 0.2,  # For borrowed chords
        },
        register_low=48,   # C3
        register_high=72,  # C5
        register_center=58,  # Bb3 - not too high, not too low
        harmonic_rhythm_suggested=1.0,  # One chord per bar
        dissonance=0.3,
        timing_feel=TimingFeel.BEHIND,
        density_suggested=0.5,  # Sparse
        dynamics_floor=0.2,
        dynamics_ceiling=0.6,
        dynamics_variance=0.3,  # Swells
        space_probability=0.3,  # Lots of space
    ),
    
    "anxiety": MusicalParameters(
        tempo_min=100,
        tempo_max=140,
        tempo_suggested=120,
        mode_weights={
            Mode.MINOR: 0.3,
            Mode.PHRYGIAN: 0.3,
            Mode.LOCRIAN: 0.2,
            Mode.DORIAN: 0.2,
        },
        register_low=60,   # C4 - compressed higher
        register_high=84,  # C6
        register_center=72,  # C5
        harmonic_rhythm_suggested=2.0,  # Faster changes
        dissonance=0.6,
        timing_feel=TimingFeel.AHEAD,
        density_suggested=2.0,  # Busier
        dynamics_floor=0.4,
        dynamics_ceiling=0.8,
        dynamics_variance=0.4,  # Restless
        space_probability=0.1,  # Less space
    ),
    
    "nostalgia": MusicalParameters(
        tempo_min=70,
        tempo_max=90,
        tempo_suggested=78,
        mode_weights={
            Mode.MAJOR: 0.4,
            Mode.MIXOLYDIAN: 0.3,
            Mode.DORIAN: 0.2,
            Mode.MINOR: 0.1,  # For borrowed iv
        },
        register_low=48,
        register_high=76,
        register_center=62,  # D4 - warm middle
        harmonic_rhythm_suggested=1.5,
        dissonance=0.25,
        timing_feel=TimingFeel.BEHIND,
        density_suggested=1.0,
        dynamics_floor=0.3,
        dynamics_ceiling=0.6,
        dynamics_variance=0.2,
        space_probability=0.2,
    ),
    
    "anger": MusicalParameters(
        tempo_min=120,
        tempo_max=160,
        tempo_suggested=138,
        mode_weights={
            Mode.PHRYGIAN: 0.3,
            Mode.MINOR: 0.4,
            Mode.LOCRIAN: 0.2,
            Mode.DORIAN: 0.1,
        },
        register_low=36,   # C2 - heavy low
        register_high=72,  # C5
        register_center=48,  # C3
        harmonic_rhythm_suggested=2.0,
        dissonance=0.5,
        timing_feel=TimingFeel.AHEAD,
        density_suggested=2.5,
        dynamics_floor=0.6,
        dynamics_ceiling=1.0,
        dynamics_variance=0.2,  # Consistently loud
        space_probability=0.05,
    ),
    
    "calm": MusicalParameters(
        tempo_min=60,
        tempo_max=80,
        tempo_suggested=68,
        mode_weights={
            Mode.MAJOR: 0.5,
            Mode.LYDIAN: 0.3,
            Mode.MIXOLYDIAN: 0.2,
        },
        register_low=48,
        register_high=76,
        register_center=60,
        harmonic_rhythm_suggested=0.5,  # Very slow changes
        dissonance=0.1,
        timing_feel=TimingFeel.BEHIND,
        density_suggested=0.5,
        dynamics_floor=0.2,
        dynamics_ceiling=0.5,
        dynamics_variance=0.1,
        space_probability=0.35,
    ),
    
    "tension_building": MusicalParameters(
        tempo_min=80,
        tempo_max=110,
        tempo_suggested=96,
        mode_weights={
            Mode.MINOR: 0.3,
            Mode.PHRYGIAN: 0.3,
            Mode.DORIAN: 0.2,
            Mode.LOCRIAN: 0.2,
        },
        register_low=40,
        register_high=80,
        register_center=60,
        harmonic_rhythm_suggested=1.5,
        dissonance=0.5,  # Increasing
        timing_feel=TimingFeel.ON,
        density_suggested=1.5,
        dynamics_floor=0.3,
        dynamics_ceiling=0.9,
        dynamics_variance=0.5,  # Building
        space_probability=0.15,
    ),
}


# =============================================================================
# COMPOUND EMOTION MODIFIERS
# =============================================================================
# Apply these on top of base presets for layered emotions

@dataclass
class EmotionModifier:
    """Modifier to layer on base emotional parameters"""
    name: str
    
    # Additive adjustments
    tempo_adjust: int = 0
    dissonance_adjust: float = 0.0
    density_adjust: float = 0.0
    register_shift: int = 0  # Semitones
    
    # Override
    timing_override: Optional[TimingFeel] = None
    
    # Intrusion pattern
    intrusion_probability: float = 0.0
    intrusion_types: list[str] = field(default_factory=list)


EMOTION_MODIFIERS: dict[str, EmotionModifier] = {
    
    "ptsd_intrusion": EmotionModifier(
        name="PTSD Intrusive Memories",
        intrusion_probability=0.15,
        intrusion_types=[
            "register_spike",      # Sudden high notes
            "harmonic_rush",       # Chords speed up briefly
            "unresolved_dissonance",  # Dissonance that doesn't resolve
            "dynamic_spike",       # Sudden loud
            "rhythmic_stumble",    # Timing disruption
        ],
    ),
    
    "dissociation": EmotionModifier(
        name="Dissociative Quality",
        dissonance_adjust=-0.1,  # Oddly smooth
        density_adjust=-0.3,     # Sparse
        register_shift=5,        # Higher, detached
        timing_override=TimingFeel.BEHIND,  # Disconnected from beat
    ),
    
    "misdirection": EmotionModifier(
        name="Emotional Misdirection",
        # Surface reads as one emotion, undertow is another
        # Implementation: use "positive" parameters with "negative" harmonic choices
        dissonance_adjust=0.1,  # Subtle wrongness
        # Key: avoid perfect cadences, use inversions
    ),
    
    "suppressed": EmotionModifier(
        name="Suppressed Emotion",
        density_adjust=-0.5,
        dissonance_adjust=0.15,  # Tension underneath
        timing_override=TimingFeel.BEHIND,
        # Dynamics stay controlled despite content
    ),
    
    "cathartic_release": EmotionModifier(
        name="Cathartic Release",
        tempo_adjust=10,
        dissonance_adjust=-0.2,  # Resolving
        density_adjust=0.5,
        # Dynamics expand
    ),
}


# =============================================================================
# INTERVAL EMOTIONAL WEIGHTS
# =============================================================================
# Used for melody generation and harmonic voicing decisions

INTERVAL_EMOTIONS: dict[int, dict] = {
    # Interval in semitones: associations
    1: {  # Minor 2nd
        "tension": 0.9,
        "associations": ["discomfort", "dread", "anxiety"],
        "use_for": ["tension_peak", "wrong_feeling"],
    },
    2: {  # Major 2nd
        "tension": 0.4,
        "associations": ["suspension", "yearning", "anticipation"],
        "use_for": ["unresolved_longing", "sus2_chords"],
    },
    3: {  # Minor 3rd
        "tension": 0.3,
        "associations": ["sadness", "introspection", "tenderness"],
        "use_for": ["grief", "melancholy", "minor_quality"],
    },
    4: {  # Major 3rd
        "tension": 0.2,
        "associations": ["brightness", "clarity", "resolution"],
        "use_for": ["joy", "major_quality", "resolution"],
    },
    5: {  # Perfect 4th
        "tension": 0.3,
        "associations": ["openness", "questioning", "folk"],
        "use_for": ["uncertainty", "sus4_chords", "ambiguity"],
    },
    6: {  # Tritone
        "tension": 1.0,
        "associations": ["instability", "danger", "wrongness"],
        "use_for": ["tension_peak", "dominant_function", "unease"],
    },
    7: {  # Perfect 5th
        "tension": 0.1,
        "associations": ["stability", "power", "grounding"],
        "use_for": ["resolution", "strength", "foundation"],
    },
    8: {  # Minor 6th
        "tension": 0.6,
        "associations": ["anguish", "dramatic", "operatic"],
        "use_for": ["grief_climax", "dramatic_loss"],
    },
    9: {  # Major 6th
        "tension": 0.25,
        "associations": ["warmth", "sweetness", "hope"],
        "use_for": ["nostalgia", "tenderness", "gentle_joy"],
    },
    10: {  # Minor 7th
        "tension": 0.5,
        "associations": ["bluesy", "unresolved", "longing"],
        "use_for": ["incompleteness", "dominant_seventh", "yearning"],
    },
    11: {  # Major 7th
        "tension": 0.55,
        "associations": ["sophisticated", "bittersweet", "complex"],
        "use_for": ["jazz_voicing", "complex_emotion", "maj7_chords"],
    },
}


# =============================================================================
# MAPPING FUNCTIONS
# =============================================================================

def get_parameters_for_state(state: EmotionalState) -> MusicalParameters:
    """
    Get musical parameters for an emotional state.
    
    This is a starting point, not the final answer.
    The interrogation engine should validate with the user.
    """
    # Start with base from primary emotion if available
    if state.primary_emotion and state.primary_emotion in EMOTIONAL_PRESETS:
        params = EMOTIONAL_PRESETS[state.primary_emotion]
    else:
        # Derive from arousal/valence
        params = _derive_from_dimensions(state.valence, state.arousal)
    
    # Apply modifiers for secondary/compound emotions
    for modifier_name in state.secondary_emotions:
        if modifier_name in EMOTION_MODIFIERS:
            params = _apply_modifier(params, EMOTION_MODIFIERS[modifier_name])
    
    # Handle intrusions
    if state.has_intrusions:
        modifier = EMOTION_MODIFIERS.get("ptsd_intrusion")
        if modifier:
            params = _apply_modifier(params, modifier)
    
    return params


def _derive_from_dimensions(valence: Valence, arousal: Arousal) -> MusicalParameters:
    """Derive parameters from arousal/valence when no named emotion."""
    # Map to closest preset based on quadrant
    if arousal.value <= 0:
        if valence.value <= 0:
            return EMOTIONAL_PRESETS["grief"]
        else:
            return EMOTIONAL_PRESETS["calm"]
    else:
        if valence.value <= 0:
            return EMOTIONAL_PRESETS["anxiety"]
        else:
            # No "joy" preset yet - use calm with adjustments
            # Use replace() to avoid mutating the global preset
            params = replace(EMOTIONAL_PRESETS["calm"], 
                           tempo_suggested=EMOTIONAL_PRESETS["calm"].tempo_suggested + 20)
            return params


def _apply_modifier(
    params: MusicalParameters, 
    modifier: EmotionModifier
) -> MusicalParameters:
    """Apply a modifier to musical parameters."""
    # Create new instance to avoid mutating original
    new_params = replace(params)
    
    new_params.tempo_suggested += modifier.tempo_adjust
    new_params.dissonance = min(1.0, max(0.0, 
        new_params.dissonance + modifier.dissonance_adjust))
    new_params.density_suggested += modifier.density_adjust
    new_params.register_center += modifier.register_shift
    
    if modifier.timing_override:
        new_params.timing_feel = modifier.timing_override
    
    return new_params


# =============================================================================
# INTERROGATION HELPERS
# =============================================================================

def get_interrogation_prompts(params: MusicalParameters) -> list[str]:
    """
    Generate questions to validate parameters with the user.
    
    This is the "Interrogate Before Generate" philosophy in action.
    """
    prompts = []
    
    # Tempo
    prompts.append(
        f"The emotional mapping suggests {params.tempo_suggested} BPM "
        f"(range {params.tempo_min}-{params.tempo_max}). "
        f"Does that feel right, or does this need to breathe more/less?"
    )
    
    # Timing feel
    feel_descriptions = {
        TimingFeel.AHEAD: "pushing forward, urgent",
        TimingFeel.ON: "steady, present",
        TimingFeel.BEHIND: "laid back, heavy, resigned",
    }
    prompts.append(
        f"The timing feel is '{params.timing_feel.value}' - "
        f"{feel_descriptions[params.timing_feel]}. "
        f"Does that match the emotional weight?"
    )
    
    # Dissonance
    if params.dissonance > 0.5:
        prompts.append(
            "This maps to high dissonance - lots of tension that doesn't resolve. "
            "Is that the texture you want, or should some of it resolve?"
        )
    elif params.dissonance < 0.2:
        prompts.append(
            "This maps to low dissonance - mostly consonant, resolved. "
            "Should there be more grit or tension underneath?"
        )
    
    # Space
    if params.space_probability > 0.25:
        prompts.append(
            "There's a lot of space/silence mapped here. "
            "Is this song about absence and what's not said?"
        )
    
    return prompts


def describe_parameters(params: MusicalParameters) -> str:
    """Human-readable description of parameters."""
    mode_str = ", ".join(
        f"{m.value} ({w*100:.0f}%)" 
        for m, w in sorted(params.mode_weights.items(), key=lambda x: -x[1])
        if w > 0.1
    )
    
    return f"""
Tempo: {params.tempo_suggested} BPM (range {params.tempo_min}-{params.tempo_max})
Modes: {mode_str}
Register: MIDI {params.register_low}-{params.register_high}, centered on {params.register_center}
Harmonic rhythm: {params.harmonic_rhythm_suggested} changes per bar
Dissonance level: {params.dissonance:.0%}
Timing feel: {params.timing_feel.value}
Density: {params.density_suggested} notes per beat
Dynamics: {params.dynamics_floor:.0%}-{params.dynamics_ceiling:.0%} with {params.dynamics_variance:.0%} variance
Space probability: {params.space_probability:.0%}
""".strip()
