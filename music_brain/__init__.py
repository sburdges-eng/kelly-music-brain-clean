# flake8: noqa
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

# Groove
from music_brain.groove import (
    apply_groove,
    extract_groove,
    GrooveTemplate,
    list_genre_templates,
)
from music_brain.groove.drum_analysis import (
    DrumAnalyzer,
    DrumTechniqueProfile,
    HiHatAlternation,
    SnareBounceSignature,
    analyze_drum_technique,
)
from music_brain.groove.drum_humanizer import DrumHumanizer
from music_brain.groove_engine import apply_groove as apply_groove_events

# Structure / audio / harmony
from music_brain.structure import (
    ChordProgression,
    analyze_chords,
    detect_sections,
)
from music_brain.audio import AudioFeatures, analyze_feel

# Comprehensive engine
from music_brain.structure.comprehensive_engine import (
    AffectAnalyzer,
    TherapySession,
    HarmonyPlan,
    render_plan_to_midi,
)

# Text / lyrical
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

# Emotion-to-production bridge
from music_brain.emotion import EmotionProductionMapper, ProductionPreset
from music_brain.production import DynamicsEngine, SectionDynamics

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
    EmotionMapper,
    MixerParameters,
    MIXER_PRESETS,
    export_to_logic_automation,
)

# Samples
from music_brain.samples import EmotionScaleSampler, FreesoundFetcher

# Emotion-driven sampling (Freesound integration)
from music_brain.emotion_sampler import (
    EmotionSampler,
    BASE_EMOTIONS,
    INSTRUMENTS as SAMPLER_INSTRUMENTS,
)

# ML OSC Learning - pattern learning for OSC control
from music_brain.ml_osc import (
    OSCRecorder,
    OSCSequence,
    OSCDataset,
    OSCMessageEncoder,
    OSCPatternLearner,
    OSCPredictor,
    OSCSequenceGenerator,
)

# Learning Module - AI-powered instrument education
from music_brain.learning import (
    AdaptiveTeacher,
    CurriculumBuilder,
    DifficultyLevel,
    INSTRUMENTS,
    Instrument,
    InstrumentFamily,
    LearningPath,
    PedagogyEngine,
    get_beginner_instruments,
    get_instrument,
    get_instruments_by_family,
    generate_learning_plan,
)

__all__ = [
    # Groove (file-based)
    "extract_groove",
    "apply_groove",
    "GrooveTemplate",
    "list_genre_templates",
    # Groove (event-based)
    "apply_groove_events",
    # Groove (analysis/humanization)
    "DrumAnalyzer",
    "DrumTechniqueProfile",
    "HiHatAlternation",
    "SnareBounceSignature",
    "analyze_drum_technique",
    "DrumHumanizer",
    # Structure
    "analyze_chords",
    "detect_sections",
    "ChordProgression",
    # Audio
    "analyze_feel",
    "AudioFeatures",
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
    # Emotion-to-production bridge
    "EmotionProductionMapper",
    "ProductionPreset",
    "DynamicsEngine",
    "SectionDynamics",
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
    # Samples
    "EmotionScaleSampler",
    "FreesoundFetcher",
    # Emotion-driven sampling
    "EmotionSampler",
    "BASE_EMOTIONS",
    "SAMPLER_INSTRUMENTS",
    # ML OSC Learning
    "OSCRecorder",
    "OSCSequence",
    "OSCDataset",
    "OSCMessageEncoder",
    "OSCPatternLearner",
    "OSCPredictor",
    "OSCSequenceGenerator",
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
