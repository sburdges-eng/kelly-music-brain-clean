"""
DAW Mixer Parameter Mapping - Emotion to Audio Production Parameters

Translates emotional states into concrete mixer automation values.
This is where the magic happens - emotion becomes sound.

Philosophy: "Interrogate Before Generate" - Each mixer setting is justified
by emotional intent, not arbitrary technical choices.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum
import json
from pathlib import Path

# Import from existing modules
from music_brain.data.emotional_mapping import (
    EmotionalState,
    MusicalParameters,
    Valence,
    Arousal,
    TimingFeel,
    EMOTIONAL_PRESETS,
    get_parameters_for_state,
)


class SaturationType(Enum):
    """Saturation/distortion character types."""
    TAPE = "tape"           # Warm, soft compression
    TUBE = "tube"           # Harmonic richness, even harmonics
    TRANSISTOR = "transistor"  # Aggressive, odd harmonics
    DIGITAL = "digital"     # Hard, harsh clipping


class FilterType(Enum):
    """Filter types for tone shaping."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"


class ReverbType(Enum):
    """Reverb character types."""
    ROOM = "room"           # Small, intimate
    HALL = "hall"           # Large, classical
    PLATE = "plate"         # Bright, vintage
    CHAMBER = "chamber"     # Warm, diffuse
    SPRING = "spring"       # Lo-fi, vintage
    SHIMMER = "shimmer"     # Ethereal, modulated


@dataclass
class MixerParameters:
    """
    DAW mixer automation parameters derived from emotional state.
    All values are normalized and ready for Logic Pro automation.
    """

    # EQ (per frequency band) - dB adjustments (-12 to +12)
    eq_sub_bass: float = 0.0      # 20-60 Hz
    eq_bass: float = 0.0          # 60-250 Hz
    eq_low_mid: float = 0.0       # 250-500 Hz
    eq_mid: float = 0.0           # 500-2000 Hz
    eq_high_mid: float = 0.0      # 2-6 kHz
    eq_presence: float = 0.0      # 6-12 kHz
    eq_air: float = 0.0           # 12-20 kHz

    # Dynamics (Compressor)
    compression_ratio: float = 1.0        # 1:1 to 20:1
    compression_threshold: float = -20.0  # dB
    compression_attack: float = 10.0      # ms
    compression_release: float = 100.0    # ms
    compression_knee: float = 0.0         # dB (0=hard, 10=soft)
    compression_makeup: float = 0.0       # dB auto-makeup gain

    # Space (Reverb)
    reverb_type: str = "hall"     # room, hall, plate, chamber, spring, shimmer
    reverb_mix: float = 0.0       # 0.0-1.0 (dry to wet)
    reverb_decay: float = 2.0     # seconds
    reverb_predelay: float = 20.0  # ms
    reverb_size: float = 0.5      # 0.0-1.0 (small to large)
    reverb_damping: float = 0.5   # 0.0-1.0 (bright to dark)

    # Delay
    delay_mix: float = 0.0        # 0.0-1.0
    delay_time: float = 500.0     # ms
    delay_feedback: float = 0.3   # 0.0-1.0
    delay_sync: bool = False      # Sync to tempo
    delay_filter: float = 0.5     # 0.0-1.0 (lo-pass filter on delay)

    # Stereo
    stereo_width: float = 1.0     # 0.0 (mono) to 2.0 (super wide)
    pan_position: float = 0.0     # -1.0 (left) to +1.0 (right)
    mid_side_ratio: float = 1.0   # 0.0 (all mid) to 2.0 (all side)

    # Saturation/Distortion
    saturation: float = 0.0       # 0.0-1.0
    saturation_type: str = "tape"  # tape, tube, transistor, digital

    # Filter
    filter_type: str = "lowpass"     # lowpass, highpass, bandpass, notch
    filter_cutoff: float = 20000.0  # Hz
    filter_resonance: float = 0.0   # 0.0-1.0

    # Master
    master_gain: float = 0.0      # dB adjustment
    limiter_threshold: float = -1.0  # dB

    # Metadata
    description: str = ""         # Why these settings?
    tags: List[str] = field(default_factory=list)
    emotional_justification: str = ""  # Emotional reasoning

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "eq": {
                "sub_bass": self.eq_sub_bass,
                "bass": self.eq_bass,
                "low_mid": self.eq_low_mid,
                "mid": self.eq_mid,
                "high_mid": self.eq_high_mid,
                "presence": self.eq_presence,
                "air": self.eq_air,
            },
            "compression": {
                "ratio": self.compression_ratio,
                "threshold": self.compression_threshold,
                "attack": self.compression_attack,
                "release": self.compression_release,
                "knee": self.compression_knee,
                "makeup": self.compression_makeup,
            },
            "reverb": {
                "type": self.reverb_type,
                "mix": self.reverb_mix,
                "decay": self.reverb_decay,
                "predelay": self.reverb_predelay,
                "size": self.reverb_size,
                "damping": self.reverb_damping,
            },
            "delay": {
                "mix": self.delay_mix,
                "time": self.delay_time,
                "feedback": self.delay_feedback,
                "sync": self.delay_sync,
                "filter": self.delay_filter,
            },
            "stereo": {
                "width": self.stereo_width,
                "pan": self.pan_position,
                "mid_side_ratio": self.mid_side_ratio,
            },
            "saturation": {
                "amount": self.saturation,
                "type": self.saturation_type,
            },
            "filter": {
                "type": self.filter_type,
                "cutoff": self.filter_cutoff,
                "resonance": self.filter_resonance,
            },
            "master": {
                "gain": self.master_gain,
                "limiter_threshold": self.limiter_threshold,
            },
            "metadata": {
                "description": self.description,
                "tags": self.tags,
                "emotional_justification": self.emotional_justification,
            },
        }


# =============================================================================
# EMOTION PRESETS
# =============================================================================

MIXER_PRESETS: Dict[str, MixerParameters] = {

    "grief": MixerParameters(
        # Dark, spacious, intimate - lo-fi bedroom aesthetic
        eq_high_mid=-4.0,
        eq_presence=-6.0,
        eq_air=-8.0,
        eq_low_mid=+2.0,
        eq_bass=+1.0,
        reverb_type="hall",
        reverb_mix=0.45,
        reverb_decay=3.5,
        reverb_predelay=35.0,
        reverb_size=0.7,
        reverb_damping=0.6,
        compression_ratio=2.5,
        compression_threshold=-18.0,
        compression_attack=30.0,
        compression_release=250.0,
        compression_knee=6.0,
        saturation=0.2,
        saturation_type="tape",
        filter_cutoff=12000.0,
        stereo_width=0.8,
        delay_mix=0.15,
        delay_time=375.0,
        delay_feedback=0.35,
        delay_filter=0.3,
        description="Deep, spacious, lo-fi grief processing",
        tags=["grief", "lo-fi", "intimate", "dark", "spacious"],
        emotional_justification="Rolled-off highs create distance from reality, "
                               "long reverb provides space for the weight of loss, "
                               "tape saturation adds warmth and imperfection"
    ),

    "anxiety": MixerParameters(
        # Tight, compressed, hyper-present
        eq_presence=+3.0,
        eq_high_mid=+2.0,
        eq_mid=+1.0,
        eq_sub_bass=-2.0,
        reverb_type="room",
        reverb_mix=0.1,
        reverb_decay=0.8,
        reverb_size=0.3,
        compression_ratio=8.0,
        compression_threshold=-30.0,
        compression_attack=1.0,
        compression_release=50.0,
        compression_knee=0.0,
        delay_mix=0.15,
        delay_time=125.0,
        delay_feedback=0.2,
        delay_sync=True,
        stereo_width=0.6,
        saturation=0.1,
        saturation_type="transistor",
        description="Tight, anxious, hyper-present",
        tags=["anxiety", "tight", "compressed", "nervous", "present"],
        emotional_justification="Fast compression attack mimics racing heart, "
                               "narrow stereo field creates claustrophobia, "
                               "enhanced presence frequencies heighten alertness"
    ),

    "anger": MixerParameters(
        # Saturated, forward, aggressive
        eq_bass=+4.0,
        eq_sub_bass=+2.0,
        eq_high_mid=+5.0,
        eq_presence=+6.0,
        eq_air=+2.0,
        reverb_type="room",
        reverb_mix=0.05,
        reverb_decay=0.5,
        compression_ratio=6.0,
        compression_threshold=-25.0,
        compression_attack=0.5,
        compression_release=30.0,
        saturation=0.7,
        saturation_type="transistor",
        stereo_width=1.5,
        master_gain=+2.0,
        filter_cutoff=18000.0,
        description="Aggressive, saturated, in-your-face",
        tags=["anger", "aggressive", "saturated", "loud", "forward"],
        emotional_justification="Heavy saturation represents burning intensity, "
                               "boosted bass provides physical weight, "
                               "minimal reverb keeps the attack immediate and present"
    ),

    "nostalgia": MixerParameters(
        # Warm, slightly degraded, dreamy
        eq_air=-6.0,
        eq_presence=-2.0,
        eq_low_mid=+3.0,
        eq_bass=+2.0,
        reverb_type="plate",
        reverb_mix=0.35,
        reverb_decay=2.8,
        reverb_predelay=25.0,
        reverb_damping=0.7,
        delay_mix=0.25,
        delay_time=375.0,
        delay_feedback=0.4,
        delay_filter=0.2,
        saturation=0.4,
        saturation_type="tape",
        filter_cutoff=14000.0,
        filter_resonance=0.2,
        stereo_width=1.2,
        description="Warm, dreamy, slightly degraded nostalgia",
        tags=["nostalgia", "warm", "dreamy", "vintage", "memory"],
        emotional_justification="Lo-fi degradation represents the imperfection of memory, "
                               "plate reverb evokes vintage recordings, "
                               "modulated delay creates dreamlike quality"
    ),

    "hope": MixerParameters(
        # Bright, open, lifting
        eq_presence=+2.0,
        eq_air=+4.0,
        eq_high_mid=+1.0,
        eq_bass=-1.0,
        reverb_type="hall",
        reverb_mix=0.25,
        reverb_decay=2.2,
        reverb_predelay=15.0,
        reverb_size=0.6,
        reverb_damping=0.3,
        compression_ratio=2.0,
        compression_threshold=-15.0,
        compression_attack=20.0,
        compression_release=150.0,
        stereo_width=1.3,
        saturation=0.1,
        saturation_type="tube",
        description="Bright, open, uplifting",
        tags=["hope", "bright", "open", "lifting", "optimistic"],
        emotional_justification="Enhanced air frequencies create sense of space and possibility, "
                               "wide stereo field suggests openness, "
                               "gentle tube saturation adds warmth without harshness"
    ),

    "calm": MixerParameters(
        # Smooth, warm, gentle
        eq_presence=-2.0,
        eq_high_mid=-1.0,
        eq_low_mid=+1.0,
        eq_bass=+1.0,
        reverb_type="chamber",
        reverb_mix=0.30,
        reverb_decay=2.5,
        reverb_predelay=20.0,
        reverb_damping=0.6,
        compression_ratio=1.5,
        compression_threshold=-12.0,
        compression_attack=50.0,
        compression_release=300.0,
        compression_knee=10.0,
        stereo_width=1.0,
        saturation=0.05,
        saturation_type="tape",
        filter_cutoff=16000.0,
        description="Smooth, warm, peaceful",
        tags=["calm", "peaceful", "warm", "gentle", "serene"],
        emotional_justification="Soft compression knee creates gentle dynamics, "
                               "rolled-off highs reduce harshness, "
                               "chamber reverb provides enveloping warmth"
    ),

    "tension": MixerParameters(
        # Building, unsettled, edge
        eq_high_mid=+3.0,
        eq_presence=+4.0,
        eq_bass=-2.0,
        eq_sub_bass=+3.0,
        reverb_type="room",
        reverb_mix=0.15,
        reverb_decay=1.2,
        reverb_size=0.4,
        compression_ratio=4.0,
        compression_threshold=-22.0,
        compression_attack=5.0,
        compression_release=80.0,
        delay_mix=0.1,
        delay_time=187.5,
        delay_feedback=0.15,
        stereo_width=0.7,
        saturation=0.3,
        saturation_type="transistor",
        filter_resonance=0.3,
        description="Tense, building, on edge",
        tags=["tension", "building", "unsettled", "suspense"],
        emotional_justification="Boosted sub bass creates physical unease, "
                               "enhanced presence frequencies create alertness, "
                               "narrow stereo field suggests confinement"
    ),

    "catharsis": MixerParameters(
        # Full, releasing, overwhelming
        eq_bass=+3.0,
        eq_sub_bass=+2.0,
        eq_mid=+1.0,
        eq_high_mid=+2.0,
        eq_presence=+3.0,
        eq_air=+2.0,
        reverb_type="hall",
        reverb_mix=0.40,
        reverb_decay=4.0,
        reverb_size=0.9,
        reverb_damping=0.4,
        compression_ratio=3.0,
        compression_threshold=-20.0,
        compression_attack=10.0,
        compression_release=200.0,
        stereo_width=1.8,
        saturation=0.4,
        saturation_type="tube",
        master_gain=+1.0,
        description="Full, releasing, overwhelming catharsis",
        tags=["catharsis", "release", "full", "overwhelming", "climax"],
        emotional_justification="Maximum stereo width suggests breaking free, "
                               "long reverb represents the release washing over, "
                               "full frequency boost creates sense of totality"
    ),

    "dissociation": MixerParameters(
        # Distant, hazy, disconnected
        eq_mid=-3.0,
        eq_presence=-4.0,
        eq_air=-2.0,
        eq_low_mid=+2.0,
        reverb_type="shimmer",
        reverb_mix=0.50,
        reverb_decay=5.0,
        reverb_predelay=50.0,
        reverb_size=0.8,
        reverb_damping=0.5,
        delay_mix=0.30,
        delay_time=500.0,
        delay_feedback=0.5,
        delay_filter=0.2,
        compression_ratio=1.5,
        compression_threshold=-15.0,
        stereo_width=1.6,
        saturation=0.15,
        saturation_type="tape",
        filter_cutoff=10000.0,
        description="Distant, hazy, disconnected",
        tags=["dissociation", "distant", "hazy", "detached", "floating"],
        emotional_justification="Heavy reverb and delay create distance from reality, "
                               "scooped mids represent hollowness, "
                               "shimmer reverb adds otherworldly quality"
    ),

    "intimacy": MixerParameters(
        # Close, dry, personal
        eq_presence=+2.0,
        eq_air=-3.0,
        eq_low_mid=+1.0,
        reverb_type="room",
        reverb_mix=0.08,
        reverb_decay=0.6,
        reverb_size=0.2,
        compression_ratio=3.0,
        compression_threshold=-18.0,
        compression_attack=15.0,
        compression_release=100.0,
        stereo_width=0.7,
        saturation=0.1,
        saturation_type="tape",
        filter_cutoff=15000.0,
        description="Close, dry, intimate and personal",
        tags=["intimacy", "close", "dry", "personal", "vulnerable"],
        emotional_justification="Minimal reverb creates sense of physical closeness, "
                               "narrow stereo width suggests whispered confession, "
                               "gentle tape saturation adds warmth without distance"
    ),
}


# =============================================================================
# EMOTION MAPPER CLASS
# =============================================================================

class EmotionMapper:
    """
    Maps emotional states to mixer parameters.
    This is where the magic happens - emotion becomes sound.
    """

    def __init__(self):
        self.presets = MIXER_PRESETS

    def map_emotion_to_mixer(
        self,
        emotional_state: EmotionalState,
        musical_params: Optional[MusicalParameters] = None
    ) -> MixerParameters:
        """
        Main mapping function: emotional state -> mixer parameters.

        Args:
            emotional_state: The emotional state to map
            musical_params: Optional musical parameters for context

        Returns:
            MixerParameters ready for DAW automation
        """
        # Get primary emotion key
        emotion_key = self._get_emotion_key(emotional_state)

        # Start with preset if available
        if emotion_key in self.presets:
            mixer = self._copy_preset(self.presets[emotion_key])
        else:
            mixer = MixerParameters()

        # Apply valence-based adjustments
        mixer = self._apply_valence(mixer, emotional_state.valence)

        # Apply arousal-based adjustments
        mixer = self._apply_arousal(mixer, emotional_state.arousal)

        # Apply musical context if provided
        if musical_params:
            mixer = self._apply_musical_context(mixer, musical_params)

        return mixer

    def _get_emotion_key(self, emotional_state: EmotionalState) -> str:
        """Extract emotion key from emotional state."""
        if emotional_state.primary_emotion:
            return emotional_state.primary_emotion.lower()

        # Derive from valence/arousal
        if isinstance(emotional_state.valence, Valence):
            v = emotional_state.valence.value
        else:
            v = float(emotional_state.valence)

        if isinstance(emotional_state.arousal, Arousal):
            a = emotional_state.arousal.value
        else:
            a = float(emotional_state.arousal)

        # Map quadrants to emotions
        if v < 0 and a < 0:
            return "grief"
        elif v < 0 and a >= 0:
            return "anxiety"
        elif v >= 0 and a < 0:
            return "calm"
        else:
            return "hope"

    def _copy_preset(self, preset: MixerParameters) -> MixerParameters:
        """Create a copy of a preset for modification."""
        return MixerParameters(
            eq_sub_bass=preset.eq_sub_bass,
            eq_bass=preset.eq_bass,
            eq_low_mid=preset.eq_low_mid,
            eq_mid=preset.eq_mid,
            eq_high_mid=preset.eq_high_mid,
            eq_presence=preset.eq_presence,
            eq_air=preset.eq_air,
            compression_ratio=preset.compression_ratio,
            compression_threshold=preset.compression_threshold,
            compression_attack=preset.compression_attack,
            compression_release=preset.compression_release,
            compression_knee=preset.compression_knee,
            compression_makeup=preset.compression_makeup,
            reverb_type=preset.reverb_type,
            reverb_mix=preset.reverb_mix,
            reverb_decay=preset.reverb_decay,
            reverb_predelay=preset.reverb_predelay,
            reverb_size=preset.reverb_size,
            reverb_damping=preset.reverb_damping,
            delay_mix=preset.delay_mix,
            delay_time=preset.delay_time,
            delay_feedback=preset.delay_feedback,
            delay_sync=preset.delay_sync,
            delay_filter=preset.delay_filter,
            stereo_width=preset.stereo_width,
            pan_position=preset.pan_position,
            mid_side_ratio=preset.mid_side_ratio,
            saturation=preset.saturation,
            saturation_type=preset.saturation_type,
            filter_type=preset.filter_type,
            filter_cutoff=preset.filter_cutoff,
            filter_resonance=preset.filter_resonance,
            master_gain=preset.master_gain,
            limiter_threshold=preset.limiter_threshold,
            description=preset.description,
            tags=list(preset.tags),
            emotional_justification=preset.emotional_justification,
        )

    def _apply_valence(
        self,
        mixer: MixerParameters,
        valence: float
    ) -> MixerParameters:
        """Adjust mixer based on valence (negative = dark, positive = bright)."""
        # Convert Valence enum if needed
        if isinstance(valence, Valence):
            v = valence.value / 2  # Normalize to -1 to 1
        else:
            v = float(valence)

        if v < 0:  # Negative = darker, roll off highs
            mixer.eq_presence += -4 * abs(v)
            mixer.eq_air += -6 * abs(v)
            mixer.filter_cutoff *= (1 - abs(v) * 0.3)
            mixer.reverb_damping = min(1.0, mixer.reverb_damping + abs(v) * 0.2)
        else:  # Positive = brighter
            mixer.eq_presence += 3 * v
            mixer.eq_air += 2 * v
            mixer.reverb_damping = max(0.0, mixer.reverb_damping - v * 0.2)

        return mixer

    def _apply_arousal(
        self,
        mixer: MixerParameters,
        arousal: float
    ) -> MixerParameters:
        """Adjust mixer based on arousal (low = spacious, high = tight)."""
        # Convert Arousal enum if needed
        if isinstance(arousal, Arousal):
            a = (arousal.value + 2) / 4  # Normalize to 0 to 1
        else:
            a = float(arousal)

        if a > 0.6:  # High energy = more compression, less space
            mixer.compression_ratio = max(mixer.compression_ratio, 4.0 + a * 4)
            mixer.compression_threshold = -30 + (a * 15)
            mixer.compression_attack = 1.0 + (1 - a) * 10
            mixer.reverb_mix = max(0.05, mixer.reverb_mix * (1 - a * 0.5))
            mixer.stereo_width = max(0.5, mixer.stereo_width - a * 0.3)
        elif a < 0.4:  # Low energy = gentle, spacious
            mixer.compression_ratio = 1.5 + a * 2
            mixer.compression_attack = 20.0 + (1 - a) * 30
            mixer.reverb_mix = min(0.6, mixer.reverb_mix + (1 - a) * 0.2)
            mixer.reverb_decay = mixer.reverb_decay + (1 - a) * 2
            mixer.stereo_width = min(2.0, mixer.stereo_width + (1 - a) * 0.3)

        return mixer

    def _apply_musical_context(
        self,
        mixer: MixerParameters,
        musical_params: MusicalParameters
    ) -> MixerParameters:
        """Adjust mixer based on musical parameters."""
        # More dissonance = more saturation, edge
        if musical_params.dissonance > 0.4:
            mixer.saturation = max(mixer.saturation, musical_params.dissonance * 0.6)
            mixer.eq_high_mid += musical_params.dissonance * 3

        # Slower tempo = more space
        if musical_params.tempo_suggested < 90:
            tempo_factor = (90 - musical_params.tempo_suggested) / 30
            mixer.reverb_decay += tempo_factor
            mixer.delay_time += tempo_factor * 100

        # Timing feel affects reverb predelay
        if musical_params.timing_feel == TimingFeel.BEHIND:
            mixer.reverb_predelay += 15
            mixer.compression_attack += 10
        elif musical_params.timing_feel == TimingFeel.AHEAD:
            mixer.reverb_predelay = max(5, mixer.reverb_predelay - 10)
            mixer.compression_attack = max(0.5, mixer.compression_attack - 5)

        return mixer

    def get_preset(self, emotion: str) -> Optional[MixerParameters]:
        """Get a preset by emotion name."""
        return self.presets.get(emotion.lower())

    def list_presets(self) -> List[str]:
        """List all available preset names."""
        return list(self.presets.keys())


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_logic_automation(
    mixer_params: MixerParameters,
    output_path: str
) -> str:
    """
    Export mixer parameters as Logic Pro automation data.

    Creates a JSON file that can be imported into Logic Pro
    using custom scripting or the LogicProject export system.

    Args:
        mixer_params: The mixer parameters to export
        output_path: Path to save the automation file

    Returns:
        Path to the created file
    """
    automation_data = {
        "daw": "Logic Pro",
        "format_version": "1.0",
        "parameters": {
            "channel_eq": {
                "sub_bass_gain": mixer_params.eq_sub_bass,
                "bass_gain": mixer_params.eq_bass,
                "low_mid_gain": mixer_params.eq_low_mid,
                "mid_gain": mixer_params.eq_mid,
                "high_mid_gain": mixer_params.eq_high_mid,
                "presence_gain": mixer_params.eq_presence,
                "air_gain": mixer_params.eq_air,
            },
            "compressor": {
                "ratio": mixer_params.compression_ratio,
                "threshold": mixer_params.compression_threshold,
                "attack": mixer_params.compression_attack,
                "release": mixer_params.compression_release,
                "knee": mixer_params.compression_knee,
                "makeup": mixer_params.compression_makeup,
            },
            "reverb": {
                "type": mixer_params.reverb_type,
                "mix": mixer_params.reverb_mix * 100,  # Convert to percentage
                "decay": mixer_params.reverb_decay,
                "predelay": mixer_params.reverb_predelay,
                "size": mixer_params.reverb_size * 100,
                "damping": mixer_params.reverb_damping * 100,
            },
            "delay": {
                "mix": mixer_params.delay_mix * 100,
                "time": mixer_params.delay_time,
                "feedback": mixer_params.delay_feedback * 100,
                "sync": mixer_params.delay_sync,
                "filter": mixer_params.delay_filter * 100,
            },
            "stereo": {
                "width": mixer_params.stereo_width * 100,
                "pan": mixer_params.pan_position * 100,
                "mid_side": mixer_params.mid_side_ratio * 100,
            },
            "saturation": {
                "amount": mixer_params.saturation * 100,
                "type": mixer_params.saturation_type,
            },
            "filter": {
                "type": mixer_params.filter_type,
                "cutoff": mixer_params.filter_cutoff,
                "resonance": mixer_params.filter_resonance * 100,
            },
            "master": {
                "gain": mixer_params.master_gain,
                "limiter_threshold": mixer_params.limiter_threshold,
            },
        },
        "metadata": {
            "description": mixer_params.description,
            "tags": mixer_params.tags,
            "emotional_justification": mixer_params.emotional_justification,
        },
    }

    with open(output_path, 'w') as f:
        json.dump(automation_data, f, indent=2)

    return output_path


def export_mixer_settings(
    mixer_params: MixerParameters,
    output_path: str,
    format: str = "json"
) -> str:
    """
    Export mixer parameters in various formats.

    Args:
        mixer_params: The mixer parameters to export
        output_path: Path to save the file
        format: Export format (json, logic, text)

    Returns:
        Path to the created file
    """
    if format == "logic":
        return export_to_logic_automation(mixer_params, output_path)
    elif format == "json":
        with open(output_path, 'w') as f:
            json.dump(mixer_params.to_dict(), f, indent=2)
        return output_path
    elif format == "text":
        with open(output_path, 'w') as f:
            f.write(describe_mixer_params(mixer_params))
        return output_path
    else:
        raise ValueError(f"Unknown format: {format}")


def describe_mixer_params(mixer: MixerParameters) -> str:
    """Generate human-readable description of mixer parameters."""
    lines = [
        "=" * 60,
        "MIXER PARAMETERS",
        "=" * 60,
        "",
        f"Description: {mixer.description}",
        f"Tags: {', '.join(mixer.tags)}",
        "",
        "EQ:",
        f"  Sub Bass (20-60Hz): {mixer.eq_sub_bass:+.1f}dB",
        f"  Bass (60-250Hz): {mixer.eq_bass:+.1f}dB",
        f"  Low Mid (250-500Hz): {mixer.eq_low_mid:+.1f}dB",
        f"  Mid (500-2kHz): {mixer.eq_mid:+.1f}dB",
        f"  High Mid (2-6kHz): {mixer.eq_high_mid:+.1f}dB",
        f"  Presence (6-12kHz): {mixer.eq_presence:+.1f}dB",
        f"  Air (12-20kHz): {mixer.eq_air:+.1f}dB",
        "",
        "Compression:",
        f"  Ratio: {mixer.compression_ratio:.1f}:1",
        f"  Threshold: {mixer.compression_threshold:.1f}dB",
        f"  Attack: {mixer.compression_attack:.1f}ms",
        f"  Release: {mixer.compression_release:.1f}ms",
        f"  Knee: {mixer.compression_knee:.1f}dB",
        "",
        "Reverb:",
        f"  Type: {mixer.reverb_type}",
        f"  Mix: {mixer.reverb_mix:.0%}",
        f"  Decay: {mixer.reverb_decay:.1f}s",
        f"  Predelay: {mixer.reverb_predelay:.0f}ms",
        f"  Size: {mixer.reverb_size:.0%}",
        "",
        "Delay:",
        f"  Mix: {mixer.delay_mix:.0%}",
        f"  Time: {mixer.delay_time:.0f}ms",
        f"  Feedback: {mixer.delay_feedback:.0%}",
        "",
        "Stereo:",
        f"  Width: {mixer.stereo_width:.0%}",
        f"  Pan: {mixer.pan_position:+.0%}",
        "",
        "Saturation:",
        f"  Amount: {mixer.saturation:.0%}",
        f"  Type: {mixer.saturation_type}",
        "",
        "Filter:",
        f"  Type: {mixer.filter_type}",
        f"  Cutoff: {mixer.filter_cutoff:.0f}Hz",
        f"  Resonance: {mixer.filter_resonance:.0%}",
        "",
        "Master:",
        f"  Gain: {mixer.master_gain:+.1f}dB",
        "",
        "Emotional Justification:",
        f"  {mixer.emotional_justification}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EMOTION -> MIXER PARAMETER MAPPING")
    print("=" * 70)

    mapper = EmotionMapper()

    # Test grief mapping
    grief_state = EmotionalState(
        valence=Valence.VERY_NEGATIVE,
        arousal=Arousal.LOW,
        primary_emotion="grief"
    )

    mixer = mapper.map_emotion_to_mixer(grief_state)

    print(f"\nGRIEF Mixer Settings:")
    print(f"  EQ: Presence {mixer.eq_presence:+.1f}dB, Air {mixer.eq_air:+.1f}dB")
    print(f"  Compression: {mixer.compression_ratio:.1f}:1 @ {mixer.compression_threshold:.1f}dB")
    print(f"  Reverb: {mixer.reverb_mix:.1%} mix, {mixer.reverb_decay:.1f}s decay")
    print(f"  Saturation: {mixer.saturation:.1%} ({mixer.saturation_type})")
    print(f"  Filter: {mixer.filter_cutoff:.0f}Hz cutoff")
    print(f"\n  Description: {mixer.description}")

    # Test anxiety mapping
    anxiety_state = EmotionalState(
        valence=Valence.NEGATIVE,
        arousal=Arousal.HIGH,
        primary_emotion="anxiety"
    )

    mixer_anxiety = mapper.map_emotion_to_mixer(anxiety_state)

    print(f"\nANXIETY Mixer Settings:")
    print(f"  EQ: Presence {mixer_anxiety.eq_presence:+.1f}dB")
    print(f"  Compression: {mixer_anxiety.compression_ratio:.1f}:1 @ {mixer_anxiety.compression_threshold:.1f}dB")
    print(f"  Reverb: {mixer_anxiety.reverb_mix:.1%} mix")
    print(f"  Stereo Width: {mixer_anxiety.stereo_width:.0%}")
    print(f"\n  Description: {mixer_anxiety.description}")

    # Export example
    print("\nExporting grief mixer settings...")
    export_path = export_to_logic_automation(mixer, "grief_automation.json")
    print(f"  Saved to: {export_path}")

    print("\n" + "=" * 70)
    print("Available presets:", ", ".join(mapper.list_presets()))
    print("=" * 70)
