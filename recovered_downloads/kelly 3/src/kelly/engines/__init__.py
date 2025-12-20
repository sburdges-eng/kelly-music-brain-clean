"""Kelly Engines - Specialized music generation engines."""

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
