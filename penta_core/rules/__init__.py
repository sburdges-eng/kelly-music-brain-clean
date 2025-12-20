"""
Penta Core Rules Package
========================

Comprehensive music theory rules with context-dependent severity.
"""

from .severity import RuleSeverity
from .species import Species
from .context import MusicalContext, CONTEXT_GROUPS
from .base import Rule, RuleViolation, RuleBreakSuggestion
from .emotion import Emotion, get_techniques_for_emotion, get_emotions_for_technique
from .timing import TimingPocket, SwingType, get_genre_pocket, apply_pocket_to_midi
from .voice_leading import VoiceLeadingRules
from .harmony_rules import HarmonyRules
from .counterpoint_rules import CounterpointRules
from .rhythm_rules import RhythmRules

__all__ = [
    # Enums
    "RuleSeverity",
    "Species",
    "MusicalContext",
    "CONTEXT_GROUPS",
    "Emotion",
    "SwingType",
    # Base classes
    "Rule",
    "RuleViolation",
    "RuleBreakSuggestion",
    "TimingPocket",
    # Rule collections
    "VoiceLeadingRules",
    "HarmonyRules",
    "CounterpointRules",
    "RhythmRules",
    # Functions
    "get_techniques_for_emotion",
    "get_emotions_for_technique",
    "get_genre_pocket",
    "apply_pocket_to_midi",
]
