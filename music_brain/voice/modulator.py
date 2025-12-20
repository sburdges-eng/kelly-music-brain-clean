"""
VoiceModulator - Voice character modification.

Provides voice transformation effects including pitch shifting,
formant adjustment, and character presets for emotional expression.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np


@dataclass
class ModulationSettings:
    """Settings for voice modulation."""

    # Pitch shift in semitones (-12 to +12)
    pitch_shift: float = 0.0

    # Formant shift in semitones (changes voice character without pitch)
    formant_shift: float = 0.0

    # Breathiness amount (0.0 = none, 1.0 = maximum)
    breathiness: float = 0.0

    # Whisper mix (0.0 = normal voice, 1.0 = full whisper)
    whisper_mix: float = 0.0

    # Robotic/vocoder effect (0.0 = off, 1.0 = full)
    robotic: float = 0.0

    # Warmth (low frequency boost) (0.0 = neutral, 1.0 = warm)
    warmth: float = 0.5

    # Presence (high-mid boost for clarity)
    presence: float = 0.5

    # Room reverb amount
    reverb: float = 0.1

    # Compression amount (0.0 = none, 1.0 = heavy)
    compression: float = 0.3

    def to_dict(self) -> dict:
        return {
            "pitch_shift": self.pitch_shift,
            "formant_shift": self.formant_shift,
            "breathiness": self.breathiness,
            "whisper_mix": self.whisper_mix,
            "robotic": self.robotic,
            "warmth": self.warmth,
            "presence": self.presence,
            "reverb": self.reverb,
            "compression": self.compression,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModulationSettings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Preset configurations for different emotional characters
MODULATION_PRESETS = {
    "intimate_whisper": ModulationSettings(
        pitch_shift=0.0,
        formant_shift=0.0,
        breathiness=0.4,
        whisper_mix=0.3,
        robotic=0.0,
        warmth=0.7,
        presence=0.4,
        reverb=0.15,
        compression=0.4,
    ),
    "vulnerable": ModulationSettings(
        pitch_shift=0.0,
        formant_shift=0.5,
        breathiness=0.3,
        whisper_mix=0.1,
        robotic=0.0,
        warmth=0.6,
        presence=0.5,
        reverb=0.2,
        compression=0.3,
    ),
    "powerful": ModulationSettings(
        pitch_shift=0.0,
        formant_shift=-0.5,
        breathiness=0.1,
        whisper_mix=0.0,
        robotic=0.0,
        warmth=0.5,
        presence=0.7,
        reverb=0.1,
        compression=0.5,
    ),
    "ethereal": ModulationSettings(
        pitch_shift=0.0,
        formant_shift=1.0,
        breathiness=0.5,
        whisper_mix=0.2,
        robotic=0.0,
        warmth=0.4,
        presence=0.3,
        reverb=0.4,
        compression=0.2,
    ),
    "dark": ModulationSettings(
        pitch_shift=-2.0,
        formant_shift=-2.0,
        breathiness=0.2,
        whisper_mix=0.1,
        robotic=0.0,
        warmth=0.8,
        presence=0.3,
        reverb=0.25,
        compression=0.4,
    ),
    "bright": ModulationSettings(
        pitch_shift=0.0,
        formant_shift=1.5,
        breathiness=0.1,
        whisper_mix=0.0,
        robotic=0.0,
        warmth=0.3,
        presence=0.8,
        reverb=0.1,
        compression=0.3,
    ),
    "robot": ModulationSettings(
        pitch_shift=0.0,
        formant_shift=0.0,
        breathiness=0.0,
        whisper_mix=0.0,
        robotic=0.8,
        warmth=0.3,
        presence=0.6,
        reverb=0.15,
        compression=0.6,
    ),
    "neutral": ModulationSettings(),
}


def get_modulation_preset(name: str) -> ModulationSettings:
    """Get a preset configuration by name."""
    if name not in MODULATION_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(MODULATION_PRESETS.keys())}")
    return MODULATION_PRESETS[name]


class VoiceModulator:
    """
    Voice character modification processor.

    Example:
        >>> modulator = VoiceModulator(get_modulation_preset("intimate_whisper"))
        >>> output_path = modulator.process_file("vocals.wav")
    """

    def __init__(self, settings: Optional[ModulationSettings] = None):
        """Initialize with optional settings."""
        self.settings = settings or ModulationSettings()

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Process an audio file with voice modulation.

        Args:
            input_path: Path to input audio file
            output_path: Path for output (default: input_modulated.wav)

        Returns:
            Path to processed audio file
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = str(input_path.with_stem(input_path.stem + "_modulated"))

        # Load audio
        try:
            import soundfile as sf
            samples, sample_rate = sf.read(str(input_path))
        except ImportError:
            try:
                import librosa
                samples, sample_rate = librosa.load(str(input_path), sr=None, mono=True)
            except ImportError:
                raise ImportError("Requires 'soundfile' or 'librosa': pip install soundfile")

        # Convert to mono if stereo
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        # Process
        processed = self.process_samples(samples, sample_rate)

        # Save
        try:
            import soundfile as sf
            sf.write(output_path, processed, sample_rate)
        except ImportError:
            from scipy.io import wavfile
            wavfile.write(output_path, sample_rate, (processed * 32767).astype(np.int16))

        return output_path

    def process_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Process audio samples with voice modulation.

        Args:
            samples: Audio samples (mono)
            sample_rate: Sample rate in Hz

        Returns:
            Processed audio samples
        """
        output = samples.copy()

        # Apply pitch shift
        if abs(self.settings.pitch_shift) > 0.01:
            output = self._apply_pitch_shift(output, sample_rate, self.settings.pitch_shift)

        # Apply formant shift (simplified - real implementation would use vocoder)
        if abs(self.settings.formant_shift) > 0.01:
            output = self._apply_formant_shift(output, sample_rate, self.settings.formant_shift)

        # Add breathiness
        if self.settings.breathiness > 0:
            output = self._add_breathiness(output, self.settings.breathiness)

        # Apply whisper mix
        if self.settings.whisper_mix > 0:
            output = self._apply_whisper(output, sample_rate, self.settings.whisper_mix)

        # Apply robotic effect
        if self.settings.robotic > 0:
            output = self._apply_robotic(output, sample_rate, self.settings.robotic)

        # Apply warmth (low boost)
        output = self._apply_eq(output, sample_rate, "warmth", self.settings.warmth)

        # Apply presence (high-mid boost)
        output = self._apply_eq(output, sample_rate, "presence", self.settings.presence)

        # Apply compression
        if self.settings.compression > 0:
            output = self._apply_compression(output, self.settings.compression)

        # Apply reverb
        if self.settings.reverb > 0:
            output = self._apply_reverb(output, sample_rate, self.settings.reverb)

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95

        return output

    def _apply_pitch_shift(
        self, samples: np.ndarray, sample_rate: int, semitones: float
    ) -> np.ndarray:
        """Apply pitch shift in semitones."""
        ratio = 2 ** (semitones / 12)

        # Simple resampling-based shift
        indices = np.arange(0, len(samples) * ratio, ratio)
        indices = indices[indices < len(samples) - 1].astype(int)

        if len(indices) == 0:
            return samples

        shifted = samples[indices]

        # Resample back to original length
        x_old = np.linspace(0, 1, len(shifted))
        x_new = np.linspace(0, 1, len(samples))
        return np.interp(x_new, x_old, shifted)

    def _apply_formant_shift(
        self, samples: np.ndarray, sample_rate: int, semitones: float
    ) -> np.ndarray:
        """Apply formant shift (simplified spectral envelope modification)."""
        # Simplified: pitch shift then shift back (real vocoder would be more sophisticated)
        ratio = 2 ** (semitones / 12)

        # This is a very simplified approximation
        # Real formant shifting requires LPC or vocoder analysis
        frame_size = int(sample_rate * 0.03)  # 30ms frames
        hop_size = frame_size // 2
        output = np.zeros_like(samples)

        for i in range(0, len(samples) - frame_size, hop_size):
            frame = samples[i : i + frame_size]

            # Apply spectral envelope shift via frequency warping
            spectrum = np.fft.rfft(frame)
            freqs = np.fft.rfftfreq(frame_size, 1 / sample_rate)

            # Warp frequencies
            warped_freqs = freqs * ratio
            warped_freqs = np.clip(warped_freqs, 0, sample_rate / 2)

            # Interpolate spectrum at warped frequencies
            warped_spectrum = np.interp(freqs, warped_freqs, np.abs(spectrum))
            warped_spectrum = warped_spectrum * np.exp(1j * np.angle(spectrum))

            # Back to time domain
            warped_frame = np.fft.irfft(warped_spectrum, n=frame_size)

            # Overlap-add
            output[i : i + frame_size] += warped_frame * np.hanning(frame_size)

        return output

    def _add_breathiness(self, samples: np.ndarray, amount: float) -> np.ndarray:
        """Add breathiness (filtered noise following envelope)."""
        # Generate noise
        noise = np.random.randn(len(samples))

        # Follow amplitude envelope
        envelope = self._get_envelope(samples, 100)
        shaped_noise = noise * envelope * amount * 0.3

        return samples + shaped_noise

    def _apply_whisper(
        self, samples: np.ndarray, sample_rate: int, amount: float
    ) -> np.ndarray:
        """Apply whisper effect (remove voiced component, add noise)."""
        # Generate whisper from noise shaped by spectral envelope
        noise = np.random.randn(len(samples))

        # Apply original spectral envelope to noise
        frame_size = 2048
        hop_size = 512
        whisper = np.zeros_like(samples)

        for i in range(0, len(samples) - frame_size, hop_size):
            orig_frame = samples[i : i + frame_size]
            noise_frame = noise[i : i + frame_size]

            # Get spectral envelope from original
            orig_spectrum = np.abs(np.fft.rfft(orig_frame))

            # Apply to noise
            noise_spectrum = np.fft.rfft(noise_frame)
            shaped_spectrum = orig_spectrum * np.exp(1j * np.angle(noise_spectrum))

            # Back to time domain
            shaped_frame = np.fft.irfft(shaped_spectrum, n=frame_size)
            whisper[i : i + frame_size] += shaped_frame * np.hanning(frame_size)

        # Mix original and whisper
        return samples * (1 - amount) + whisper * amount

    def _apply_robotic(
        self, samples: np.ndarray, sample_rate: int, amount: float
    ) -> np.ndarray:
        """Apply robotic/vocoder effect."""
        # Simple vocoder-like effect using ring modulation
        carrier_freq = 100  # Hz
        t = np.arange(len(samples)) / sample_rate
        carrier = np.sin(2 * np.pi * carrier_freq * t)

        # Ring modulate
        robotic = samples * carrier

        # Mix
        return samples * (1 - amount) + robotic * amount

    def _apply_eq(
        self, samples: np.ndarray, sample_rate: int, band: str, amount: float
    ) -> np.ndarray:
        """Apply simple EQ adjustment."""
        # Normalize amount to -0.5 to +0.5 range centered at neutral
        gain = (amount - 0.5) * 12  # ±6 dB

        if abs(gain) < 0.5:
            return samples

        # Simple filter based on band
        if band == "warmth":
            # Low shelf around 200 Hz
            cutoff = 200 / (sample_rate / 2)
        elif band == "presence":
            # High-mid boost around 3 kHz
            cutoff = 3000 / (sample_rate / 2)
        else:
            return samples

        cutoff = min(0.99, max(0.01, cutoff))

        # Simple first-order filter
        alpha = cutoff / (cutoff + 1)

        if band == "warmth":
            # Low pass for warmth
            output = np.zeros_like(samples)
            output[0] = samples[0]
            for i in range(1, len(samples)):
                output[i] = alpha * samples[i] + (1 - alpha) * output[i - 1]
            # Blend based on gain
            mix = 0.5 + gain / 12 * 0.5  # 0 to 1 based on gain
            return samples * (1 - mix * 0.3) + output * mix * 0.3 + samples
        else:
            # High pass for presence
            output = np.zeros_like(samples)
            output[0] = samples[0]
            for i in range(1, len(samples)):
                output[i] = (1 - alpha) * (output[i - 1] + samples[i] - samples[i - 1])
            mix = 0.5 + gain / 12 * 0.5
            return samples + output * mix * 0.3

    def _apply_compression(self, samples: np.ndarray, amount: float) -> np.ndarray:
        """Apply dynamic range compression."""
        threshold = 1.0 - amount * 0.5  # Lower threshold with more compression
        ratio = 1 + amount * 3  # Higher ratio with more compression

        output = samples.copy()
        for i, sample in enumerate(samples):
            if abs(sample) > threshold:
                sign = np.sign(sample)
                excess = abs(sample) - threshold
                output[i] = sign * (threshold + excess / ratio)

        return output

    def _apply_reverb(
        self, samples: np.ndarray, sample_rate: int, amount: float
    ) -> np.ndarray:
        """Apply simple reverb effect."""
        # Simple comb filter reverb
        delay_ms = 30
        delay_samples = int(sample_rate * delay_ms / 1000)
        decay = 0.5 * amount

        output = samples.copy()
        for delay_mult in [1, 1.5, 2.3, 3.1]:
            delay = int(delay_samples * delay_mult)
            for i in range(delay, len(samples)):
                output[i] += samples[i - delay] * decay / delay_mult

        # Mix dry and wet
        return samples * (1 - amount * 0.5) + output * amount * 0.5

    def _get_envelope(self, samples: np.ndarray, window_size: int) -> np.ndarray:
        """Get amplitude envelope of signal."""
        envelope = np.zeros_like(samples)
        for i, _ in enumerate(samples):
            start = max(0, i - window_size // 2)
            end = min(len(samples), i + window_size // 2)
            envelope[i] = np.max(np.abs(samples[start:end]))
        return envelope
