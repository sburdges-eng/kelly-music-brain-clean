"""Song Intent Schema - Structured deep interrogation for songwriting.

Comprehensive rule-breaking enums and data structures for intentional
creative choices in music generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class HarmonyRuleBreak(Enum):
    """Harmony rules to intentionally break."""
    AVOID_TONIC_RESOLUTION = "HARMONY_AvoidTonicResolution"
    PARALLEL_MOTION = "HARMONY_ParallelMotion"
    MODAL_INTERCHANGE = "HARMONY_ModalInterchange"
    TRITONE_SUBSTITUTION = "HARMONY_TritoneSubstitution"
    POLYTONALITY = "HARMONY_Polytonality"
    UNRESOLVED_DISSONANCE = "HARMONY_UnresolvedDissonance"
    CHROMATIC_MEDIANTS = "HARMONY_ChromaticMediants"
    DECEPTIVE_CADENCE = "HARMONY_DeceptiveCadence"


class RhythmRuleBreak(Enum):
    """Rhythm rules to intentionally break."""
    CONSTANT_DISPLACEMENT = "RHYTHM_ConstantDisplacement"
    TEMPO_FLUCTUATION = "RHYTHM_TempoFluctuation"
    METRIC_MODULATION = "RHYTHM_MetricModulation"
    POLYRHYTHMIC_LAYERS = "RHYTHM_PolyrhythmicLayers"
    DROPPED_BEATS = "RHYTHM_DroppedBeats"
    ASYMMETRIC_GROUPINGS = "RHYTHM_AsymmetricGroupings"


class ArrangementRuleBreak(Enum):
    """Arrangement rules to intentionally break."""
    UNBALANCED_DYNAMICS = "ARRANGEMENT_UnbalancedDynamics"
    STRUCTURAL_MISMATCH = "ARRANGEMENT_StructuralMismatch"
    BURIED_VOCALS = "ARRANGEMENT_BuriedVocals"
    EXTREME_DYNAMIC_RANGE = "ARRANGEMENT_ExtremeDynamicRange"
    PREMATURE_CLIMAX = "ARRANGEMENT_PrematureClimax"
    ANTI_DROP = "ARRANGEMENT_AntiDrop"


class ProductionRuleBreak(Enum):
    """Production rules to intentionally break."""
    EXCESSIVE_MUD = "PRODUCTION_ExcessiveMud"
    PITCH_IMPERFECTION = "PRODUCTION_PitchImperfection"
    ROOM_NOISE = "PRODUCTION_RoomNoise"
    DISTORTION = "PRODUCTION_Distortion"
    MONO_COLLAPSE = "PRODUCTION_MonoCollapse"
    LO_FI_DEGRADATION = "PRODUCTION_LoFiDegradation"
    SILENCE_AS_INSTRUMENT = "PRODUCTION_SilenceAsInstrument"
    CLIPPING_PEAKS = "PRODUCTION_ClippingPeaks"
    EXCESSIVE_REVERB = "PRODUCTION_ExcessiveReverb"


class MelodyRuleBreak(Enum):
    """Melody rules to intentionally break."""
    AVOID_RESOLUTION = "MELODY_AvoidResolution"
    EXCESSIVE_REPETITION = "MELODY_ExcessiveRepetition"
    ANGULAR_INTERVALS = "MELODY_AngularIntervals"
    ANTI_CLIMAX = "MELODY_AntiClimax"
    MONOTONE_DRONE = "MELODY_MonotoneDrone"
    FRAGMENTED_PHRASES = "MELODY_FragmentedPhrases"


class TextureRuleBreak(Enum):
    """Texture rules to intentionally break."""
    EXTREME_DENSITY = "TEXTURE_ExtremeDensity"
    SPARSE = "TEXTURE_Sparse"
    FREQUENCY_MASKING = "TEXTURE_FrequencyMasking"
    TIMBRAL_CLASH = "TEXTURE_TimbralClash"


class NarrativeArc(Enum):
    """Emotional journey structures."""
    LINEAR_BUILD = "linear_build"
    LINEAR_DESCENT = "linear_descent"
    WAVE = "wave"
    EXPLOSION = "explosion"
    IMPLOSION = "implosion"
    TENSION_RELEASE = "tension_release"
    CIRCULAR = "circular"
    FRAGMENTED = "fragmented"


class Vulnerability(Enum):
    """Vulnerability levels in expression."""
    GUARDED = "guarded"
    CAUTIOUS = "cautious"
    OPEN = "open"
    RAW = "raw"
    EXPOSED = "exposed"


@dataclass
class SongRoot:
    """Phase 0: Core wound/desire - the deepest layer."""
    core_wound: str
    desired_state: str
    blocking_belief: Optional[str] = None
    hidden_need: Optional[str] = None
    transformation_goal: Optional[str] = None


@dataclass
class SongIntent:
    """Phase 1: Emotional and expressive intent."""
    primary_emotion: str
    secondary_emotions: List[str] = field(default_factory=list)
    emotional_intensity: float = 0.7
    vulnerability_level: Vulnerability = Vulnerability.OPEN
    narrative_arc: NarrativeArc = NarrativeArc.WAVE
    imagery: List[str] = field(default_factory=list)
    misdirection: Optional[str] = None  # Surface emotion vs true emotion


@dataclass
class TechnicalConstraints:
    """Phase 2: Technical implementation constraints."""
    key: str = "C"
    mode: str = "minor"
    tempo_bpm: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    groove_feel: str = "straight"
    target_duration_bars: int = 32
    
    # Rule breaks to apply
    harmony_breaks: List[HarmonyRuleBreak] = field(default_factory=list)
    rhythm_breaks: List[RhythmRuleBreak] = field(default_factory=list)
    arrangement_breaks: List[ArrangementRuleBreak] = field(default_factory=list)
    production_breaks: List[ProductionRuleBreak] = field(default_factory=list)
    melody_breaks: List[MelodyRuleBreak] = field(default_factory=list)
    texture_breaks: List[TextureRuleBreak] = field(default_factory=list)


@dataclass
class SystemDirective:
    """Generation directives for the system."""
    generate_midi: bool = True
    generate_audio: bool = False
    include_chord_progression: bool = True
    include_melody: bool = True
    include_bass: bool = True
    include_drums: bool = True
    include_pads: bool = False
    include_strings: bool = False
    export_stems: bool = False


@dataclass
class CompleteSongIntent:
    """Complete three-phase song intent structure."""
    root: SongRoot
    intent: SongIntent
    constraints: TechnicalConstraints
    directive: SystemDirective = field(default_factory=SystemDirective)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": {
                "core_wound": self.root.core_wound,
                "desired_state": self.root.desired_state,
                "blocking_belief": self.root.blocking_belief,
                "hidden_need": self.root.hidden_need,
                "transformation_goal": self.root.transformation_goal,
            },
            "intent": {
                "primary_emotion": self.intent.primary_emotion,
                "secondary_emotions": self.intent.secondary_emotions,
                "emotional_intensity": self.intent.emotional_intensity,
                "vulnerability_level": self.intent.vulnerability_level.value,
                "narrative_arc": self.intent.narrative_arc.value,
                "imagery": self.intent.imagery,
                "misdirection": self.intent.misdirection,
            },
            "constraints": {
                "key": self.constraints.key,
                "mode": self.constraints.mode,
                "tempo_bpm": self.constraints.tempo_bpm,
                "time_signature": self.constraints.time_signature,
                "groove_feel": self.constraints.groove_feel,
                "target_duration_bars": self.constraints.target_duration_bars,
                "harmony_breaks": [h.value for h in self.constraints.harmony_breaks],
                "rhythm_breaks": [r.value for r in self.constraints.rhythm_breaks],
                "arrangement_breaks": [a.value for a in self.constraints.arrangement_breaks],
                "production_breaks": [p.value for p in self.constraints.production_breaks],
                "melody_breaks": [m.value for m in self.constraints.melody_breaks],
                "texture_breaks": [t.value for t in self.constraints.texture_breaks],
            },
            "metadata": self.metadata,
        }


# Emotion to rule break suggestions
AFFECT_MODE_MAP = {
    "grief": {
        "mode": "minor",
        "tempo_range": (60, 80),
        "harmony_breaks": [HarmonyRuleBreak.UNRESOLVED_DISSONANCE, HarmonyRuleBreak.AVOID_TONIC_RESOLUTION],
        "rhythm_breaks": [RhythmRuleBreak.DROPPED_BEATS],
        "production_breaks": [ProductionRuleBreak.SILENCE_AS_INSTRUMENT],
    },
    "anger": {
        "mode": "phrygian",
        "tempo_range": (100, 140),
        "harmony_breaks": [HarmonyRuleBreak.PARALLEL_MOTION],
        "rhythm_breaks": [RhythmRuleBreak.POLYRHYTHMIC_LAYERS],
        "production_breaks": [ProductionRuleBreak.DISTORTION],
    },
    "anxiety": {
        "mode": "locrian",
        "tempo_range": (90, 120),
        "harmony_breaks": [HarmonyRuleBreak.AVOID_TONIC_RESOLUTION],
        "rhythm_breaks": [RhythmRuleBreak.TEMPO_FLUCTUATION, RhythmRuleBreak.CONSTANT_DISPLACEMENT],
        "production_breaks": [ProductionRuleBreak.ROOM_NOISE],
    },
    "hope": {
        "mode": "major",
        "tempo_range": (80, 110),
        "harmony_breaks": [HarmonyRuleBreak.MODAL_INTERCHANGE],
        "arrangement_breaks": [ArrangementRuleBreak.EXTREME_DYNAMIC_RANGE],
    },
    "nostalgia": {
        "mode": "mixolydian",
        "tempo_range": (70, 95),
        "harmony_breaks": [HarmonyRuleBreak.MODAL_INTERCHANGE, HarmonyRuleBreak.CHROMATIC_MEDIANTS],
        "production_breaks": [ProductionRuleBreak.LO_FI_DEGRADATION],
    },
    "defiance": {
        "mode": "dorian",
        "tempo_range": (95, 130),
        "harmony_breaks": [HarmonyRuleBreak.PARALLEL_MOTION],
        "rhythm_breaks": [RhythmRuleBreak.CONSTANT_DISPLACEMENT],
        "arrangement_breaks": [ArrangementRuleBreak.EXTREME_DYNAMIC_RANGE],
    },
    "emptiness": {
        "mode": "aeolian",
        "tempo_range": (50, 70),
        "texture_breaks": [TextureRuleBreak.SPARSE],
        "production_breaks": [ProductionRuleBreak.SILENCE_AS_INSTRUMENT, ProductionRuleBreak.EXCESSIVE_REVERB],
        "melody_breaks": [MelodyRuleBreak.MONOTONE_DRONE],
    },
}


def suggest_rule_break(emotion: str, intensity: float) -> List[str]:
    """Suggest rule breaks based on emotion and intensity."""
    mapping = AFFECT_MODE_MAP.get(emotion.lower(), {})
    breaks = []
    
    for key in ["harmony_breaks", "rhythm_breaks", "arrangement_breaks", 
                "production_breaks", "melody_breaks", "texture_breaks"]:
        if key in mapping:
            for rule_break in mapping[key]:
                if intensity > 0.5 or rule_break.value not in [rb for rb in breaks]:
                    breaks.append(rule_break.value)
    
    return breaks


def get_affect_mapping(emotion: str) -> Dict[str, Any]:
    """Get musical parameters for an emotion."""
    return AFFECT_MODE_MAP.get(emotion.lower(), {
        "mode": "minor",
        "tempo_range": (80, 100),
    })


def validate_intent(intent: CompleteSongIntent) -> Tuple[bool, List[str]]:
    """Validate a complete song intent structure."""
    errors = []
    
    if not intent.root.core_wound:
        errors.append("Core wound must be specified")
    
    if not intent.intent.primary_emotion:
        errors.append("Primary emotion must be specified")
    
    if not 0 <= intent.intent.emotional_intensity <= 1:
        errors.append("Emotional intensity must be between 0 and 1")
    
    if intent.constraints.tempo_bpm < 30 or intent.constraints.tempo_bpm > 300:
        errors.append("Tempo must be between 30 and 300 BPM")
    
    return len(errors) == 0, errors


def create_intent(
    wound: str,
    emotion: str,
    intensity: float = 0.7,
    key: str = "C",
    tempo: int = 100,
    **kwargs
) -> CompleteSongIntent:
    """Quick helper to create a complete intent."""
    mapping = get_affect_mapping(emotion)
    
    return CompleteSongIntent(
        root=SongRoot(
            core_wound=wound,
            desired_state=kwargs.get("desired_state", "processing"),
        ),
        intent=SongIntent(
            primary_emotion=emotion,
            emotional_intensity=intensity,
            vulnerability_level=Vulnerability.OPEN,
            narrative_arc=NarrativeArc.WAVE,
        ),
        constraints=TechnicalConstraints(
            key=key,
            mode=mapping.get("mode", "minor"),
            tempo_bpm=tempo,
            harmony_breaks=mapping.get("harmony_breaks", []),
            rhythm_breaks=mapping.get("rhythm_breaks", []),
            production_breaks=mapping.get("production_breaks", []),
        ),
    )
