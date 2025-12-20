"""
Emotional expression mapping for music theory rules.

Maps rule violations/applications to emotional effects in music.
"""

from enum import Enum, auto
from typing import List, Dict, Set
from dataclasses import dataclass


class Emotion(Enum):
    """
    Emotional states expressible through music theory choices.
    
    Based on Meyer's "Emotion and Meaning in Music" and Huron's "Sweet Anticipation".
    """
    # Core emotions (Plutchik's wheel)
    JOY = auto()
    SADNESS = auto()
    ANGER = auto()
    FEAR = auto()
    SURPRISE = auto()
    ANTICIPATION = auto()
    
    # Musical emotions (Zentner/Eerola taxonomy)
    TENSION = auto()
    RESOLUTION = auto()
    NOSTALGIA = auto()
    TRANSCENDENCE = auto()
    TENDERNESS = auto()
    POWER = auto()
    PEACEFULNESS = auto()
    
    # Complex states
    GRIEF = auto()
    TRIUMPH = auto()
    YEARNING = auto()
    MYSTERY = auto()
    
    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class EmotionalMapping:
    """
    Maps a music theory technique to its emotional effect.
    
    Attributes:
        rule_name: The technique/rule being applied or broken
        emotion: Primary emotional effect
        intensity: 1-10 scale of emotional impact
        context_dependencies: Musical contexts where this mapping is strongest
        explanation: Why this technique evokes this emotion
    """
    rule_name: str
    emotion: Emotion
    intensity: int  # 1-10
    context_dependencies: Set[str]
    explanation: str


# Emotional mappings database
EMOTION_TO_TECHNIQUES: Dict[Emotion, List[str]] = {
    Emotion.GRIEF: [
        "non_resolution",  # Unresolved suspensions
        "modal_interchange",  # Borrowed minor chords
        "tempo_fluctuation",  # Rubato, slowing
        "descending_chromatic_bass",  # Lament bass
        "avoid_perfect_cadences",  # Lack of closure
    ],
    
    Emotion.POWER: [
        "parallel_fifths",  # Medieval organum, rock power chords
        "parallel_octaves",  # Unison doubling
        "root_position_triads",  # Stable, grounded harmony
        "strong_downbeats",  # Metric emphasis
        "wide_spacing",  # Orchestral doubling
    ],
    
    Emotion.TENSION: [
        "unprepared_dissonance",  # Sudden harmonic clash
        "augmented_intervals",  # Tritones, aug 2nds
        "unresolved_leading_tone",  # Incomplete harmonic motion
        "metric_displacement",  # Syncopation against pulse
        "dense_voicing",  # Cluster chords
    ],
    
    Emotion.RESOLUTION: [
        "stepwise_contrary_motion",  # V→I voice leading
        "prepared_suspensions",  # 4-3, 7-6 resolutions
        "authentic_cadence",  # V→I closure
        "tonic_pedal",  # Harmonic stability
        "consonant_intervals",  # 3rds, 6ths
    ],
    
    Emotion.YEARNING: [
        "dominant_prolongation",  # Extended V chords
        "secondary_dominants",  # Tonicization of non-tonic
        "appoggiatura",  # Leaning notes
        "suspension_chains",  # 7-6, 6-5, 5-4 sequences
        "raised_scale_degrees",  # Leading tones, alterations
    ],
    
    Emotion.MYSTERY: [
        "whole_tone_scale",  # Ambiguous tonality
        "diminished_seventh",  # Symmetrical, directionless
        "unresolved_augmented_sixth",  # Hanging dominant prep
        "parallel_quartal_harmony",  # Modal ambiguity
        "avoid_cadences",  # No tonal center establishment
    ],
    
    Emotion.PEACEFULNESS: [
        "consonant_intervals",  # Pure 3rds, 6ths
        "stepwise_motion",  # Smooth melodic contour
        "diatonic_harmony",  # No chromaticism
        "regular_phrase_lengths",  # Metric predictability
        "slow_harmonic_rhythm",  # Infrequent chord changes
    ],
    
    Emotion.TRIUMPH: [
        "parallel_octaves",  # Orchestral unison
        "perfect_authentic_cadence",  # V→I with scale degrees 7→1
        "ascending_sequences",  # Rising melodic/harmonic patterns
        "major_mode",  # Bright tonality
        "forte_dynamics",  # Loudness (performance aspect)
    ],
}


# Reverse mapping: technique → emotions
TECHNIQUE_TO_EMOTIONS: Dict[str, List[EmotionalMapping]] = {}

# Build reverse mapping
for emotion, techniques in EMOTION_TO_TECHNIQUES.items():
    for technique in techniques:
        if technique not in TECHNIQUE_TO_EMOTIONS:
            TECHNIQUE_TO_EMOTIONS[technique] = []
        
        # Create emotional mapping with context
        mapping = EmotionalMapping(
            rule_name=technique,
            emotion=emotion,
            intensity=7,  # Default intensity
            context_dependencies={"classical", "romantic", "film"},
            explanation=f"{technique} commonly evokes {emotion.name.lower()} in tonal music"
        )
        TECHNIQUE_TO_EMOTIONS[technique].append(mapping)


def get_techniques_for_emotion(emotion: Emotion) -> List[str]:
    """
    Get music theory techniques that evoke a specific emotion.
    
    Args:
        emotion: Target emotional state
    
    Returns:
        List of rule/technique names
    
    Example:
        >>> get_techniques_for_emotion(Emotion.GRIEF)
        ['non_resolution', 'modal_interchange', 'tempo_fluctuation', ...]
    """
    return EMOTION_TO_TECHNIQUES.get(emotion, [])


def get_emotions_for_technique(technique: str) -> List[EmotionalMapping]:
    """
    Get emotional effects of a music theory technique.
    
    Args:
        technique: Rule or technique name
    
    Returns:
        List of emotional mappings with intensity and context
    
    Example:
        >>> get_emotions_for_technique("parallel_fifths")
        [EmotionalMapping(emotion=POWER, intensity=8, ...)]
    """
    return TECHNIQUE_TO_EMOTIONS.get(technique, [])
