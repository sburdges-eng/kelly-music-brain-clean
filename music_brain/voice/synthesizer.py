"""
VoiceSynthesizer - Text-to-speech and guide vocal generation.

Provides voice synthesis capabilities for generating guide vocals,
spoken prompts, and text-to-speech with emotional character.
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import numpy as np


@dataclass
class SynthConfig:
    """Configuration for voice synthesis."""

    # Voice character
    voice_type: str = "neutral"  # neutral, warm, bright, dark, breathy

    # Pitch settings
    base_pitch_hz: float = 220.0  # Base fundamental frequency
    pitch_variation: float = 0.1  # Amount of natural pitch variation

    # Timing settings
    syllable_duration_ms: float = 200.0  # Base syllable duration
    pause_between_words_ms: float = 100.0  # Pause between words

    # Expression settings
    expressiveness: float = 0.5  # Amount of dynamic variation
    breathiness: float = 0.2  # Breath in voice
    vibrato_rate: float = 5.0  # Vibrato frequency in Hz
    vibrato_depth: float = 0.02  # Vibrato depth (semitones)

    # Quality settings
    formant_emphasis: float = 0.5  # Formant strength
    attack_sharpness: float = 0.5  # Note attack character

    def to_dict(self) -> dict:
        return {
            "voice_type": self.voice_type,
            "base_pitch_hz": self.base_pitch_hz,
            "pitch_variation": self.pitch_variation,
            "syllable_duration_ms": self.syllable_duration_ms,
            "pause_between_words_ms": self.pause_between_words_ms,
            "expressiveness": self.expressiveness,
            "breathiness": self.breathiness,
            "vibrato_rate": self.vibrato_rate,
            "vibrato_depth": self.vibrato_depth,
            "formant_emphasis": self.formant_emphasis,
            "attack_sharpness": self.attack_sharpness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SynthConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Voice profile presets
VOICE_PROFILES = {
    "guide_vulnerable": SynthConfig(
        voice_type="warm",
        base_pitch_hz=200.0,
        pitch_variation=0.15,
        syllable_duration_ms=250.0,
        expressiveness=0.7,
        breathiness=0.4,
        vibrato_rate=4.5,
        vibrato_depth=0.03,
        formant_emphasis=0.4,
        attack_sharpness=0.3,
    ),
    "guide_confident": SynthConfig(
        voice_type="bright",
        base_pitch_hz=180.0,
        pitch_variation=0.1,
        syllable_duration_ms=180.0,
        expressiveness=0.5,
        breathiness=0.1,
        vibrato_rate=5.5,
        vibrato_depth=0.02,
        formant_emphasis=0.6,
        attack_sharpness=0.6,
    ),
    "guide_ethereal": SynthConfig(
        voice_type="breathy",
        base_pitch_hz=300.0,
        pitch_variation=0.2,
        syllable_duration_ms=300.0,
        expressiveness=0.6,
        breathiness=0.6,
        vibrato_rate=4.0,
        vibrato_depth=0.04,
        formant_emphasis=0.3,
        attack_sharpness=0.2,
    ),
    "guide_powerful": SynthConfig(
        voice_type="dark",
        base_pitch_hz=150.0,
        pitch_variation=0.08,
        syllable_duration_ms=160.0,
        expressiveness=0.4,
        breathiness=0.1,
        vibrato_rate=5.0,
        vibrato_depth=0.015,
        formant_emphasis=0.7,
        attack_sharpness=0.7,
    ),
    "narrator": SynthConfig(
        voice_type="neutral",
        base_pitch_hz=170.0,
        pitch_variation=0.12,
        syllable_duration_ms=200.0,
        expressiveness=0.3,
        breathiness=0.15,
        vibrato_rate=4.0,
        vibrato_depth=0.01,
        formant_emphasis=0.5,
        attack_sharpness=0.4,
    ),
}


def get_voice_profile(name: str) -> SynthConfig:
    """Get a voice profile by name."""
    if name not in VOICE_PROFILES:
        raise ValueError(f"Unknown profile: {name}. Available: {list(VOICE_PROFILES.keys())}")
    return VOICE_PROFILES[name]


class VoiceSynthesizer:
    """
    Voice synthesizer for guide vocals and text-to-speech.

    Example:
        >>> synth = VoiceSynthesizer(get_voice_profile("guide_vulnerable"))
        >>> synth.synthesize_guide(
        ...     lyrics="Hello world",
        ...     melody_midi=[60, 62, 64],
        ...     tempo_bpm=120,
        ...     output_path="guide.wav"
        ... )
    """

    # Phoneme durations (relative)
    PHONEME_DURATIONS = {
        "a": 1.0, "e": 0.9, "i": 0.8, "o": 1.0, "u": 0.9,
        "b": 0.3, "c": 0.3, "d": 0.3, "f": 0.4, "g": 0.3,
        "h": 0.3, "j": 0.4, "k": 0.3, "l": 0.4, "m": 0.4,
        "n": 0.4, "p": 0.3, "q": 0.3, "r": 0.4, "s": 0.5,
        "t": 0.3, "v": 0.4, "w": 0.4, "x": 0.4, "y": 0.4, "z": 0.4,
        " ": 0.5, ".": 1.0, ",": 0.6, "!": 1.0, "?": 1.0,
    }

    # Formant frequencies for vowels (F1, F2, F3 in Hz)
    VOWEL_FORMANTS = {
        "a": (800, 1200, 2500),
        "e": (400, 2200, 2800),
        "i": (300, 2300, 3000),
        "o": (500, 900, 2500),
        "u": (350, 700, 2500),
    }

    def __init__(self, config: Optional[SynthConfig] = None, sample_rate: int = 44100):
        """Initialize synthesizer with configuration."""
        self.config = config or SynthConfig()
        self.sample_rate = sample_rate

    def synthesize_guide(
        self,
        lyrics: str,
        melody_midi: List[int],
        tempo_bpm: int = 82,
        output_path: str = "guide_vocal.wav",
    ) -> str:
        """
        Synthesize a guide vocal track.

        Args:
            lyrics: Lyrics text to sing
            melody_midi: List of MIDI note numbers for melody
            tempo_bpm: Tempo in BPM
            output_path: Path for output WAV file

        Returns:
            Path to generated audio file
        """
        # Calculate timing
        beat_duration = 60.0 / tempo_bpm

        # Split lyrics into syllables (simplified)
        syllables = self._split_syllables(lyrics)

        # Match syllables to notes
        notes_per_syllable = max(1, len(melody_midi) // max(1, len(syllables)))

        # Generate audio
        samples = []
        note_idx = 0

        for syllable in syllables:
            if note_idx >= len(melody_midi):
                note_idx = len(melody_midi) - 1

            midi_note = melody_midi[note_idx]
            frequency = self._midi_to_hz(midi_note)

            # Generate syllable audio
            duration = beat_duration * (self.config.syllable_duration_ms / 250.0)
            syllable_audio = self._synthesize_syllable(syllable, frequency, duration)
            samples.extend(syllable_audio)

            note_idx = min(note_idx + notes_per_syllable, len(melody_midi) - 1)

        # Convert to numpy array
        audio = np.array(samples)

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9

        # Save
        self._save_audio(audio, output_path)

        return output_path

    def speak_text(
        self,
        text: str,
        output_path: str = "spoken.wav",
        tempo_bpm: int = 80,
    ) -> str:
        """
        Synthesize spoken text (text-to-speech).

        Args:
            text: Text to speak
            output_path: Path for output WAV file
            tempo_bpm: Speaking rate (affects pacing)

        Returns:
            Path to generated audio file
        """
        # Split into words
        words = text.split()

        # Base pitch with variation
        samples = []

        for i, word in enumerate(words):
            # Natural pitch contour
            pitch_mult = 1.0 + np.sin(i / len(words) * np.pi) * 0.1
            frequency = self.config.base_pitch_hz * pitch_mult

            # Synthesize word
            syllables = self._split_syllables(word)
            for syllable in syllables:
                duration = self.config.syllable_duration_ms / 1000.0
                syllable_audio = self._synthesize_syllable(syllable, frequency, duration)
                samples.extend(syllable_audio)

            # Pause between words
            pause_samples = int(self.sample_rate * self.config.pause_between_words_ms / 1000)
            samples.extend([0.0] * pause_samples)

        # Convert and normalize
        audio = np.array(samples)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9

        # Save
        self._save_audio(audio, output_path)

        return output_path

    def _split_syllables(self, text: str) -> List[str]:
        """Split text into syllables (simplified)."""
        vowels = "aeiouAEIOU"
        syllables = []
        current = ""

        for char in text.lower():
            current += char
            if char in vowels and len(current) > 0:
                syllables.append(current)
                current = ""

        if current:
            if syllables:
                syllables[-1] += current
            else:
                syllables.append(current)

        return syllables if syllables else [text]

    def _synthesize_syllable(
        self, syllable: str, frequency: float, duration: float
    ) -> List[float]:
        """Synthesize a single syllable."""
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples) / self.sample_rate

        # Find main vowel for formants
        vowel = None
        for char in syllable.lower():
            if char in self.VOWEL_FORMANTS:
                vowel = char
                break
        vowel = vowel or "a"

        # Generate base waveform (glottal pulse approximation)
        waveform = self._generate_glottal_pulse(frequency, t)

        # Apply formants
        waveform = self._apply_formants(waveform, vowel)

        # Add vibrato
        if self.config.vibrato_depth > 0:
            vibrato = np.sin(2 * np.pi * self.config.vibrato_rate * t)
            pitch_mod = 2 ** (vibrato * self.config.vibrato_depth / 12)
            # Simplified vibrato application
            waveform = waveform * (1 + vibrato * self.config.vibrato_depth * 0.1)

        # Add breathiness
        if self.config.breathiness > 0:
            noise = np.random.randn(num_samples) * self.config.breathiness * 0.2
            waveform = waveform + noise

        # Apply envelope
        envelope = self._generate_envelope(num_samples)
        waveform = waveform * envelope

        return waveform.tolist()

    def _generate_glottal_pulse(self, frequency: float, t: np.ndarray) -> np.ndarray:
        """Generate a glottal pulse waveform."""
        # LF model approximation
        period = 1.0 / frequency
        phase = (t % period) / period

        # Glottal pulse shape
        pulse = np.sin(np.pi * phase) ** 2
        pulse = pulse * (1 - phase)  # Asymmetric falloff

        # Add harmonics
        for h in range(2, 6):
            harmonic = np.sin(2 * np.pi * frequency * h * t) / h
            pulse = pulse + harmonic * 0.3 / h

        return pulse

    def _apply_formants(self, waveform: np.ndarray, vowel: str) -> np.ndarray:
        """Apply formant filtering to shape vowel sound."""
        if vowel not in self.VOWEL_FORMANTS:
            return waveform

        f1, f2, f3 = self.VOWEL_FORMANTS[vowel]

        # Apply resonant filters at formant frequencies
        # Simplified: just boost energy at formant frequencies
        spectrum = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(len(waveform), 1 / self.sample_rate)

        # Create formant envelope
        envelope = np.ones_like(freqs)
        bandwidth = 100  # Hz

        for formant_freq in [f1, f2, f3]:
            # Gaussian bump at each formant
            envelope = envelope + np.exp(-((freqs - formant_freq) ** 2) / (2 * bandwidth ** 2))

        # Apply envelope
        spectrum = spectrum * envelope * self.config.formant_emphasis

        return np.fft.irfft(spectrum, n=len(waveform))

    def _generate_envelope(self, num_samples: int) -> np.ndarray:
        """Generate ADSR-like envelope."""
        attack = int(num_samples * 0.1 * self.config.attack_sharpness)
        decay = int(num_samples * 0.1)
        release = int(num_samples * 0.2)
        sustain_level = 0.7 + 0.3 * (1 - self.config.expressiveness)

        envelope = np.ones(num_samples)

        # Attack
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)

        # Decay
        if decay > 0 and attack + decay < num_samples:
            envelope[attack : attack + decay] = np.linspace(1, sustain_level, decay)

        # Sustain (already at sustain_level)
        sustain_end = num_samples - release
        envelope[attack + decay : sustain_end] = sustain_level

        # Release
        if release > 0 and sustain_end < num_samples:
            envelope[sustain_end:] = np.linspace(sustain_level, 0, release)

        return envelope

    def _midi_to_hz(self, midi_note: int) -> float:
        """Convert MIDI note number to frequency in Hz."""
        return 440.0 * (2 ** ((midi_note - 69) / 12))

    def _save_audio(self, audio: np.ndarray, output_path: str) -> None:
        """Save audio to file."""
        try:
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)
        except ImportError:
            try:
                from scipy.io import wavfile
                wavfile.write(output_path, self.sample_rate, (audio * 32767).astype(np.int16))
            except ImportError:
                # Manual WAV writing
                self._write_wav_manual(audio, output_path)

    def _write_wav_manual(self, audio: np.ndarray, output_path: str) -> None:
        """Write WAV file without external dependencies."""
        import struct

        # Convert to 16-bit PCM
        audio_int = (audio * 32767).astype(np.int16)

        with open(output_path, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(audio_int) * 2))
            f.write(b"WAVE")

            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))  # Chunk size
            f.write(struct.pack("<H", 1))   # Audio format (PCM)
            f.write(struct.pack("<H", 1))   # Channels
            f.write(struct.pack("<I", self.sample_rate))  # Sample rate
            f.write(struct.pack("<I", self.sample_rate * 2))  # Byte rate
            f.write(struct.pack("<H", 2))   # Block align
            f.write(struct.pack("<H", 16))  # Bits per sample

            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", len(audio_int) * 2))
            f.write(audio_int.tobytes())
