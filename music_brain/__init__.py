"""
Music Brain - Intelligent Music Analysis Toolkit

A Python package for music production analysis:
- Groove extraction and application
- Chord progression analysis
- Section detection
- Feel/timing analysis
- DAW integration
- Therapy-to-music pipeline (Comprehensive Engine)
- Lyrical fragment generation
- Reference track DNA analysis
- AI-powered instrument education (Learning Module)
"""

__version__ = "1.0.0"
__author__ = "Sean Burdges"

from music_brain.groove import extract_groove, apply_groove, GrooveTemplate
from music_brain.structure import analyze_chords, detect_sections, ChordProgression
from music_brain.audio import analyze_feel, AudioFeatures

# Harmony generation
from music_brain.harmony import HarmonyGenerator, HarmonyResult, generate_midi_from_harmony

# New comprehensive engine exports
from music_brain.structure.comprehensive_engine import (
    AffectAnalyzer,
    TherapySession,
    HarmonyPlan,
    render_plan_to_midi,
)
from music_brain.groove_engine import apply_groove as apply_groove_events
from music_brain.text.lyrical_mirror import generate_lyrical_fragments

# Emotion API (Emotion -> Music -> Mixer)
from music_brain.emotion_api import (
    MusicBrain,
    GeneratedMusic,
    FluentChain,
    quick_generate,
    quick_export,
    INTENT_EXAMPLES,
)

# Emotional mapping
from music_brain.data.emotional_mapping import (
    EmotionalState,
    MusicalParameters,
    Valence,
    Arousal,
    TimingFeel,
    Mode,
    EMOTIONAL_PRESETS,
    get_parameters_for_state,
)

# Mixer parameters
from music_brain.daw.mixer_params import (
    MixerParameters,
    EmotionMapper,
    export_to_logic_automation,
    MIXER_PRESETS,
)

# Learning Module - AI-powered instrument education
from music_brain.learning import (
    DifficultyLevel,
    InstrumentFamily,
    Instrument,
    INSTRUMENTS,
    get_instrument,
    get_instruments_by_family,
    get_beginner_instruments,
    CurriculumBuilder,
    LearningPath,
    PedagogyEngine,
    AdaptiveTeacher,
    generate_learning_plan,
)

__all__ = [
    # Groove (file-based)
    "extract_groove",
    "apply_groove",
    "GrooveTemplate",
    # Groove (event-based)
    "apply_groove_events",
    # Structure
    "analyze_chords",
    "detect_sections",
    "ChordProgression",
    # Audio
    "analyze_feel",
    "AudioFeatures",
    # Harmony
    "HarmonyGenerator",
    "HarmonyResult",
    "generate_midi_from_harmony",
    # Comprehensive Engine
    "AffectAnalyzer",
    "TherapySession",
    "HarmonyPlan",
    "render_plan_to_midi",
    # Text/Lyrical
    "generate_lyrical_fragments",
    # Emotion API
    "MusicBrain",
    "GeneratedMusic",
    "FluentChain",
    "quick_generate",
    "quick_export",
    "INTENT_EXAMPLES",
    # Emotional mapping
    "EmotionalState",
    "MusicalParameters",
    "Valence",
    "Arousal",
    "TimingFeel",
    "Mode",
    "EMOTIONAL_PRESETS",
    "get_parameters_for_state",
    # Mixer parameters
    "MixerParameters",
    "EmotionMapper",
    "export_to_logic_automation",
    "MIXER_PRESETS",
    # Learning Module
    "DifficultyLevel",
    "InstrumentFamily",
    "Instrument",
    "INSTRUMENTS",
    "get_instrument",
    "get_instruments_by_family",
    "get_beginner_instruments",
    "CurriculumBuilder",
    "LearningPath",
    "PedagogyEngine",
    "AdaptiveTeacher",
    "generate_learning_plan",
]
