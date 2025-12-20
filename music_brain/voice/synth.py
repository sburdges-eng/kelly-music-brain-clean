"""
Voice Synthesizer - Local TTS-based voice synthesis.

Provides text-to-speech synthesis for generating guide vocals,
spoken prompts, and creative vocal content.

Supports multiple platforms via pyttsx3 backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import tempfile
import platform
import os

# Optional TTS backends
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False


class VoiceGender(Enum):
    """Voice gender options."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class EmotionalTone(Enum):
    """Emotional tone presets for voice synthesis."""
    VULNERABLE = "vulnerable"
    CONFIDENT = "confident"
    TENDER = "tender"
    DEFIANT = "defiant"
    INTIMATE = "intimate"
    DETACHED = "detached"
    HOPEFUL = "hopeful"
    MELANCHOLIC = "melancholic"


@dataclass
class VoiceProfile:
    """Voice profile configuration for synthesis."""
    name: str
    gender: VoiceGender = VoiceGender.NEUTRAL
    tone: EmotionalTone = EmotionalTone.CONFIDENT
    rate: int = 150  # Words per minute (100-200 typical)
    pitch: float = 1.0  # Pitch multiplier (0.5-2.0)
    volume: float = 0.9  # Volume (0.0-1.0)

    # Platform-specific voice name overrides
    voice_id_macos: Optional[str] = None
    voice_id_windows: Optional[str] = None
    voice_id_linux: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gender": self.gender.value,
            "tone": self.tone.value,
            "rate": self.rate,
            "pitch": self.pitch,
            "volume": self.volume,
        }


# Built-in voice profiles
VOICE_PROFILES: Dict[str, VoiceProfile] = {
    "guide_vulnerable": VoiceProfile(
        name="Guide Vulnerable",
        gender=VoiceGender.FEMALE,
        tone=EmotionalTone.VULNERABLE,
        rate=130,
        pitch=1.0,
        volume=0.85,
    ),
    "guide_confident": VoiceProfile(
        name="Guide Confident",
        gender=VoiceGender.MALE,
        tone=EmotionalTone.CONFIDENT,
        rate=160,
        pitch=0.95,
        volume=0.95,
    ),
    "guide_tender": VoiceProfile(
        name="Guide Tender",
        gender=VoiceGender.FEMALE,
        tone=EmotionalTone.TENDER,
        rate=120,
        pitch=1.05,
        volume=0.8,
    ),
    "guide_defiant": VoiceProfile(
        name="Guide Defiant",
        gender=VoiceGender.MALE,
        tone=EmotionalTone.DEFIANT,
        rate=170,
        pitch=0.9,
        volume=1.0,
    ),
    "narrator_neutral": VoiceProfile(
        name="Narrator",
        gender=VoiceGender.NEUTRAL,
        tone=EmotionalTone.CONFIDENT,
        rate=145,
        pitch=1.0,
        volume=0.9,
    ),
    "whisper_intimate": VoiceProfile(
        name="Whisper Intimate",
        gender=VoiceGender.FEMALE,
        tone=EmotionalTone.INTIMATE,
        rate=110,
        pitch=1.1,
        volume=0.65,
    ),
}


def get_voice_profile(profile_name: str) -> VoiceProfile:
    """
    Get a voice profile by name.

    Args:
        profile_name: Name of the preset profile

    Returns:
        VoiceProfile configuration

    Raises:
        KeyError: If profile not found
    """
    if profile_name not in VOICE_PROFILES:
        available = ", ".join(VOICE_PROFILES.keys())
        raise KeyError(
            f"Voice profile '{profile_name}' not found. "
            f"Available profiles: {available}"
        )
    return VOICE_PROFILES[profile_name]


def list_voice_profiles() -> List[str]:
    """List available voice profile names."""
    return list(VOICE_PROFILES.keys())


@dataclass
class SynthConfig:
    """Configuration for voice synthesis."""
    output_format: str = "wav"  # wav, mp3, aiff
    sample_rate: int = 44100
    channels: int = 1  # mono
    bit_depth: int = 16
    normalize: bool = True
    add_reverb: bool = False
    reverb_amount: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_format": self.output_format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "normalize": self.normalize,
            "add_reverb": self.add_reverb,
            "reverb_amount": self.reverb_amount,
        }


class LocalVoiceSynth:
    """
    Local TTS engine wrapper.

    Uses pyttsx3 for cross-platform text-to-speech.
    Falls back gracefully if TTS is unavailable.
    """

    def __init__(self, profile: Optional[VoiceProfile] = None):
        """
        Initialize local voice synth.

        Args:
            profile: Optional voice profile to use
        """
        self.profile = profile or VOICE_PROFILES["narrator_neutral"]
        self._engine = None
        self._available = PYTTSX3_AVAILABLE
        self._platform = platform.system().lower()

        if self._available:
            self._init_engine()

    def _init_engine(self) -> None:
        """Initialize the TTS engine."""
        try:
            self._engine = pyttsx3.init()
            self._apply_profile()
        except Exception as e:
            self._available = False
            self._engine = None

    def _apply_profile(self) -> None:
        """Apply voice profile settings to engine."""
        if not self._engine:
            return

        try:
            self._engine.setProperty('rate', self.profile.rate)
            self._engine.setProperty('volume', self.profile.volume)

            # Try to set voice based on platform and gender
            voices = self._engine.getProperty('voices')
            if voices:
                # Try to find a matching voice
                target_voice = None

                # Platform-specific voice ID
                if self._platform == "darwin" and self.profile.voice_id_macos:
                    for v in voices:
                        if self.profile.voice_id_macos in v.id:
                            target_voice = v
                            break
                elif self._platform == "windows" and self.profile.voice_id_windows:
                    for v in voices:
                        if self.profile.voice_id_windows in v.id:
                            target_voice = v
                            break
                elif self._platform == "linux" and self.profile.voice_id_linux:
                    for v in voices:
                        if self.profile.voice_id_linux in v.id:
                            target_voice = v
                            break

                # Fallback: match by gender
                if not target_voice:
                    gender_map = {
                        VoiceGender.FEMALE: ["female", "woman", "fiona", "samantha", "victoria"],
                        VoiceGender.MALE: ["male", "man", "alex", "daniel", "david"],
                    }
                    keywords = gender_map.get(self.profile.gender, [])

                    for v in voices:
                        voice_id_lower = v.id.lower()
                        if any(kw in voice_id_lower for kw in keywords):
                            target_voice = v
                            break

                if target_voice:
                    self._engine.setProperty('voice', target_voice.id)
        except Exception:
            pass  # Keep default voice on error

    def set_profile(self, profile: VoiceProfile) -> None:
        """Change the voice profile."""
        self.profile = profile
        if self._engine:
            self._apply_profile()

    @property
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self._available and self._engine is not None

    def list_available_voices(self) -> List[Dict[str, str]]:
        """List available system voices."""
        if not self._engine:
            return []

        voices = self._engine.getProperty('voices')
        return [
            {"id": v.id, "name": v.name, "languages": getattr(v, 'languages', [])}
            for v in voices
        ]

    def speak(self, text: str) -> None:
        """
        Speak text immediately (blocking).

        Args:
            text: Text to speak
        """
        if not self._engine:
            return

        self._engine.say(text)
        self._engine.runAndWait()

    def save_to_file(
        self,
        text: str,
        output_path: str,
    ) -> Optional[str]:
        """
        Save spoken text to audio file.

        Args:
            text: Text to synthesize
            output_path: Output file path

        Returns:
            Output path if successful, None otherwise
        """
        if not self._engine:
            return None

        output_path = str(Path(output_path).absolute())

        try:
            self._engine.save_to_file(text, output_path)
            self._engine.runAndWait()

            if Path(output_path).exists():
                return output_path
        except Exception:
            pass

        return None


class VoiceSynthesizer:
    """
    High-level voice synthesizer for DAiW.

    Combines TTS with music production features for creating
    guide vocals and spoken content.
    """

    def __init__(self, config: Optional[SynthConfig] = None):
        """
        Initialize voice synthesizer.

        Args:
            config: Synthesis configuration
        """
        self.config = config or SynthConfig()
        self._local_synth: Optional[LocalVoiceSynth] = None

    def _get_synth(self, profile: VoiceProfile) -> LocalVoiceSynth:
        """Get or create local synth with profile."""
        if self._local_synth is None:
            self._local_synth = LocalVoiceSynth(profile)
        else:
            self._local_synth.set_profile(profile)
        return self._local_synth

    @property
    def is_available(self) -> bool:
        """Check if voice synthesis is available."""
        if self._local_synth is None:
            self._local_synth = LocalVoiceSynth()
        return self._local_synth.is_available

    def speak_text(
        self,
        text: str,
        output_path: str = "spoken_prompt.wav",
        profile: str = "narrator_neutral",
    ) -> Optional[str]:
        """
        Generate spoken audio from text.

        Args:
            text: Text to speak
            output_path: Output audio file path
            profile: Voice profile name

        Returns:
            Output path if successful, None otherwise
        """
        voice_profile = get_voice_profile(profile)
        synth = self._get_synth(voice_profile)

        if not synth.is_available:
            return None

        return synth.save_to_file(text, output_path)

    def synthesize_guide(
        self,
        lyrics: str,
        melody_midi: Optional[List[int]] = None,
        tempo_bpm: int = 82,
        output_path: str = "guide_vocal.wav",
        profile: str = "guide_vulnerable",
    ) -> Optional[str]:
        """
        Synthesize guide vocal from lyrics.

        Note: This is a simplified implementation that speaks the lyrics.
        For melodic synthesis, integrate with a melody-following TTS
        system or external service.

        Args:
            lyrics: Lyrics text
            melody_midi: Optional MIDI note sequence (for future melodic synthesis)
            tempo_bpm: Tempo in BPM
            output_path: Output audio file path
            profile: Voice profile name

        Returns:
            Output path if successful, None otherwise
        """
        voice_profile = get_voice_profile(profile)

        # Adjust rate based on tempo
        adjusted_profile = VoiceProfile(
            name=voice_profile.name,
            gender=voice_profile.gender,
            tone=voice_profile.tone,
            rate=int(voice_profile.rate * (tempo_bpm / 120.0)),  # Scale rate with tempo
            pitch=voice_profile.pitch,
            volume=voice_profile.volume,
        )

        synth = self._get_synth(adjusted_profile)

        if not synth.is_available:
            return None

        return synth.save_to_file(lyrics, output_path)

    def list_profiles(self) -> List[str]:
        """List available voice profiles."""
        return list_voice_profiles()

    def get_profile_info(self, profile_name: str) -> Dict[str, Any]:
        """Get information about a voice profile."""
        profile = get_voice_profile(profile_name)
        return profile.to_dict()

    def list_system_voices(self) -> List[Dict[str, str]]:
        """List available system TTS voices."""
        if self._local_synth is None:
            self._local_synth = LocalVoiceSynth()
        return self._local_synth.list_available_voices()
