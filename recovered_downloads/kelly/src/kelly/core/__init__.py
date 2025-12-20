"""Kelly Core - Emotion processing and MIDI generation."""

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

__all__ = [
    "EmotionThesaurus", "EmotionNode", "EmotionCategory",
    "IntentProcessor", "Wound", "RuleBreak",
    "MidiGenerator",
    "CompleteSongIntent", "SongRoot", "SongIntent", "TechnicalConstraints",
    "HarmonyRuleBreak", "RhythmRuleBreak", "ArrangementRuleBreak",
    "ProductionRuleBreak", "MelodyRuleBreak", "TextureRuleBreak",
    "EmotionalState", "MusicalParameters", "Valence", "Arousal", "TimingFeel", "Mode",
    "get_parameters_for_state", "EMOTIONAL_PRESETS",
]
