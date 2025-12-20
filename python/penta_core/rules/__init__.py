"""
Re-export rule modules so tests can import from python.penta_core.rules.
"""

from penta_core.rules.severity import RuleSeverity
from penta_core.rules.context import MusicalContext, CONTEXT_GROUPS, get_context_group
from penta_core.rules.base import Rule, RuleViolation, RuleBreakSuggestion
from penta_core.rules.emotion import (
    Emotion,
    EmotionalMapping,
    EMOTION_TO_TECHNIQUES,
    TECHNIQUE_TO_EMOTIONS,
    get_techniques_for_emotion,
    get_emotions_for_technique,
)
from penta_core.rules.timing import (
    SwingType,
    TimingPocket,
    GENRE_POCKETS,
    get_genre_pocket,
    apply_pocket_to_midi,
)

__all__ = [
    "RuleSeverity",
    "MusicalContext",
    "CONTEXT_GROUPS",
    "get_context_group",
    "Rule",
    "RuleViolation",
    "RuleBreakSuggestion",
    "Emotion",
    "EmotionalMapping",
    "EMOTION_TO_TECHNIQUES",
    "TECHNIQUE_TO_EMOTIONS",
    "get_techniques_for_emotion",
    "get_emotions_for_technique",
    "SwingType",
    "TimingPocket",
    "GENRE_POCKETS",
    "get_genre_pocket",
    "apply_pocket_to_midi",
]
