"""Kelly - Therapeutic iDAW translating emotions to music."""

__version__ = "0.1.0"
__author__ = "Kelly Development Team"

from kelly.core.emotion_thesaurus import EmotionThesaurus
from kelly.core.intent_processor import IntentProcessor
from kelly.core.midi_generator import MidiGenerator

__all__ = ["EmotionThesaurus", "IntentProcessor", "MidiGenerator"]
