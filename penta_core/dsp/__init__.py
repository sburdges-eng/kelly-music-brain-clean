"""
DSP Module - Digital Signal Processing utilities for iDAW.

Provides:
- Envelope follower and pattern automation (Trace DSP)
- Sample playback engine and pitch shifting (Parrot DSP)
- Common DSP building blocks
"""

from python.penta_core.dsp.trace_dsp import (
    EnvelopeFollower,
    EnvelopeMode,
    PatternAutomation,
    AutomationCurve,
    create_envelope_follower,
    follow_envelope,
    apply_pattern_automation,
    generate_lfo_pattern,
)

from python.penta_core.dsp.parrot_dsp import (
    SamplePlayback,
    PitchShifter,
    GrainCloud,
    PlaybackMode,
    create_pitch_shifter,
    shift_pitch,
    time_stretch,
    create_grain_cloud,
)

__all__ = [
    # Trace DSP
    "EnvelopeFollower",
    "EnvelopeMode",
    "PatternAutomation",
    "AutomationCurve",
    "create_envelope_follower",
    "follow_envelope",
    "apply_pattern_automation",
    "generate_lfo_pattern",
    # Parrot DSP
    "SamplePlayback",
    "PitchShifter",
    "GrainCloud",
    "PlaybackMode",
    "create_pitch_shifter",
    "shift_pitch",
    "time_stretch",
    "create_grain_cloud",
]
