"""
Advanced Harmony - Sophisticated harmonic analysis and generation.

Provides:
- Jazz voicing generation
- Neo-Riemannian transformations
- Counterpoint generation
- Tension/release analysis
- Microtonal support (24-TET, just intonation)
"""

from python.penta_core.harmony.jazz_voicings import (
    JazzVoicing,
    VoicingStyle,
    generate_jazz_voicing,
    voice_lead_progression,
    get_drop_voicing,
    get_rootless_voicing,
)

from python.penta_core.harmony.neo_riemannian import (
    NeoRiemannianTransform,
    apply_transform,
    get_transform_path,
    parallel_transform,
    relative_transform,
    leading_tone_transform,
)

from python.penta_core.harmony.counterpoint import (
    Species,
    CounterpointVoice,
    generate_counterpoint,
    check_counterpoint_rules,
    get_species_rules,
)

from python.penta_core.harmony.tension import (
    TensionLevel,
    TensionAnalysis,
    analyze_tension,
    plan_tension_curve,
    suggest_tension_chords,
)

from python.penta_core.harmony.microtonal import (
    TuningSystem,
    MicrotonalPitch,
    cents_to_ratio,
    ratio_to_cents,
    just_intonation,
    equal_temperament,
)

__all__ = [
    # Jazz Voicings
    "JazzVoicing",
    "VoicingStyle",
    "generate_jazz_voicing",
    "voice_lead_progression",
    "get_drop_voicing",
    "get_rootless_voicing",
    # Neo-Riemannian
    "NeoRiemannianTransform",
    "apply_transform",
    "get_transform_path",
    "parallel_transform",
    "relative_transform",
    "leading_tone_transform",
    # Counterpoint
    "Species",
    "CounterpointVoice",
    "generate_counterpoint",
    "check_counterpoint_rules",
    "get_species_rules",
    # Tension
    "TensionLevel",
    "TensionAnalysis",
    "analyze_tension",
    "plan_tension_curve",
    "suggest_tension_chords",
    # Microtonal
    "TuningSystem",
    "MicrotonalPitch",
    "cents_to_ratio",
    "ratio_to_cents",
    "just_intonation",
    "equal_temperament",
]
