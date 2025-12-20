"""Kelly - Therapeutic iDAW translating emotions to music.

Philosophy: "Interrogate Before Generate" - The tool shouldn't finish art 
for people; it should make them braver.

Three-Phase Intent System:
1. Wound → Identify the emotional trigger
2. Emotion → Map to the 216-node emotion thesaurus
3. Rule-breaks → Express through intentional musical violations
"""

__version__ = "0.1.0"
__author__ = "Kelly Development Team"

from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionNode, EmotionCategory
from kelly.core.intent_processor import IntentProcessor, Wound, RuleBreak
from kelly.core.midi_generator import MidiGenerator
from kelly.core.intent_schema import (
    CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
    HarmonyRuleBreak, RhythmRuleBreak, ArrangementRuleBreak,
    ProductionRuleBreak, MelodyRuleBreak, TextureRuleBreak,
)
from kelly.core.emotional_mapping import (
    EmotionalState, MusicalParameters, Valence, Arousal, TimingFeel, Mode,
    get_parameters_for_state, EMOTIONAL_PRESETS,
)

from kelly.engines.groove_engine import GrooveEngine, GrooveSettings
from kelly.engines.bass_engine import BassEngine, BassConfig, BassPattern
from kelly.engines.melody_engine import MelodyEngine, MelodyConfig, MelodyContour
from kelly.engines.rhythm_engine import RhythmEngine, RhythmConfig
from kelly.engines.pad_engine import PadEngine, PadConfig, PadTexture
from kelly.engines.string_engine import StringEngine, StringConfig
from kelly.engines.counter_melody_engine import CounterMelodyEngine
from kelly.engines.fill_engine import FillEngine, FillConfig
from kelly.engines.dynamics_engine import DynamicsEngine, DynamicShape
from kelly.engines.transition_engine import TransitionEngine, TransitionType
from kelly.engines.arrangement_engine import ArrangementEngine, ArrangementPlan
from kelly.engines.variation_engine import VariationEngine, VariationType
from kelly.engines.tension_engine import TensionEngine, TensionCurve
from kelly.engines.voice_leading import VoiceLeadingEngine

__all__ = [
    # Version
    "__version__",
    # Core
    "EmotionThesaurus", "EmotionNode", "EmotionCategory",
    "IntentProcessor", "Wound", "RuleBreak",
    "MidiGenerator",
    "CompleteSongIntent", "SongRoot", "SongIntent", "TechnicalConstraints",
    "HarmonyRuleBreak", "RhythmRuleBreak", "ArrangementRuleBreak",
    "ProductionRuleBreak", "MelodyRuleBreak", "TextureRuleBreak",
    "EmotionalState", "MusicalParameters", "Valence", "Arousal", "TimingFeel", "Mode",
    "get_parameters_for_state", "EMOTIONAL_PRESETS",
    # Engines
    "GrooveEngine", "GrooveSettings",
    "BassEngine", "BassConfig", "BassPattern",
    "MelodyEngine", "MelodyConfig", "MelodyContour",
    "RhythmEngine", "RhythmConfig",
    "PadEngine", "PadConfig", "PadTexture",
    "StringEngine", "StringConfig",
    "CounterMelodyEngine",
    "FillEngine", "FillConfig",
    "DynamicsEngine", "DynamicShape",
    "TransitionEngine", "TransitionType",
    "ArrangementEngine", "ArrangementPlan",
    "VariationEngine", "VariationType",
    "TensionEngine", "TensionCurve",
    "VoiceLeadingEngine",
]
