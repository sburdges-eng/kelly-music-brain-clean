"""
Voice Processing - Auto-tune, modulation, and voice synthesis.

This module provides voice processing capabilities including:
- AutoTuneProcessor: Pitch correction for vocals
- VoiceModulator: Voice character modification
- VoiceSynthesizer: Text-to-speech and guide vocal generation
"""

from music_brain.voice.auto_tune import (
    AutoTuneProcessor,
    AutoTuneSettings,
    get_auto_tune_preset,
)
from music_brain.voice.modulator import (
    ModulationSettings,
    VoiceModulator,
    get_modulation_preset,
)
from music_brain.voice.synthesizer import (
    SynthConfig,
    VoiceSynthesizer,
    get_voice_profile,
)

__all__ = [
    # Auto-tune
    "AutoTuneProcessor",
    "AutoTuneSettings",
    "ModulationSettings",
    "SynthConfig",
    # Modulation
    "VoiceModulator",
    # Synthesis
    "VoiceSynthesizer",
    "get_auto_tune_preset",
    "get_modulation_preset",
    "get_voice_profile",
]
