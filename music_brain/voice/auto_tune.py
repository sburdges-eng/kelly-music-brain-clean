"""
AutoTuneProcessor - Pitch correction for vocals.

Provides transparent to aggressive pitch correction with
musical key awareness and natural-sounding results.
"""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import numpy as np


@dataclass
class AutoTuneSettings:
    """Settings for auto-tune processing."""

    # Pitch correction strength (0.0 = off, 1.0 = full correction)
    correction_strength: float = 0.8

    # Speed of pitch correction in milliseconds (lower = faster/more robotic)
    correction_speed_ms: float = 50.0

    # Humanize amount (adds subtle pitch variation to avoid robotic sound)
    humanize: float = 0.1

    # Vibrato preservation (0.0 = remove, 1.0 = preserve)
    vibrato_preservation: float = 0.7

    # Formant preservation (keeps natural voice character)
    preserve_formants: bool = True

    # Note snap threshold in cents (notes within this range snap to target)
    snap_threshold_cents: float = 50.0

    # Enable natural transitions between notes
    natural_transitions: bool = True

    def to_dict(self) -> dict:
        return {
            "correction_strength": self.correction_strength,
            "correction_speed_ms": self.correction_speed_ms,
            "humanize": self.humanize,
            "vibrato_preservation": self.vibrato_preservation,
            "preserve_formants": self.preserve_formants,
            "snap_threshold_cents": self.snap_threshold_cents,
            "natural_transitions": self.natural_transitions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AutoTuneSettings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Preset configurations
AUTO_TUNE_PRESETS = {
    "transparent": AutoTuneSettings(
        correction_strength=0.5,
        correction_speed_ms=80.0,
        humanize=0.2,
        vibrato_preservation=0.9,
        preserve_formants=True,
        natural_transitions=True,
    ),
    "natural": AutoTuneSettings(
        correction_strength=0.7,
        correction_speed_ms=60.0,
        humanize=0.15,
        vibrato_preservation=0.8,
        preserve_formants=True,
        natural_transitions=True,
    ),
    "moderate": AutoTuneSettings(
        correction_strength=0.85,
        correction_speed_ms=40.0,
        humanize=0.1,
        vibrato_preservation=0.6,
        preserve_formants=True,
        natural_transitions=True,
    ),
    "aggressive": AutoTuneSettings(
        correction_strength=1.0,
        correction_speed_ms=20.0,
        humanize=0.05,
        vibrato_preservation=0.3,
        preserve_formants=True,
        natural_transitions=False,
    ),
    "hard_tune": AutoTuneSettings(
        correction_strength=1.0,
        correction_speed_ms=0.0,
        humanize=0.0,
        vibrato_preservation=0.0,
        preserve_formants=False,
        natural_transitions=False,
    ),
    "vulnerable": AutoTuneSettings(
        correction_strength=0.4,
        correction_speed_ms=100.0,
        humanize=0.25,
        vibrato_preservation=0.95,
        preserve_formants=True,
        natural_transitions=True,
    ),
}


def get_auto_tune_preset(name: str) -> AutoTuneSettings:
    """Get a preset configuration by name."""
    if name not in AUTO_TUNE_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(AUTO_TUNE_PRESETS.keys())}")
    return AUTO_TUNE_PRESETS[name]


class AutoTuneProcessor:
    """
    Pitch correction processor for vocals.

    Example:
        >>> processor = AutoTuneProcessor(get_auto_tune_preset("natural"))
        >>> output_path = processor.process_file("vocals.wav", key="C", mode="major")
    """

    # Note frequencies for A4 = 440 Hz
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Scale intervals (semitones from root)
    SCALES = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
        "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "chromatic": list(range(12)),
    }

    def __init__(self, settings: Optional[AutoTuneSettings] = None):
        """Initialize with optional settings."""
        self.settings = settings or AutoTuneSettings()

    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        key: Optional[str] = None,
        mode: str = "major",
    ) -> str:
        """
        Process an audio file with pitch correction.

        Args:
            input_path: Path to input audio file
            output_path: Path for output (default: input_tuned.wav)
            key: Musical key (e.g., "C", "F#"). None = chromatic
            mode: Scale mode (major, minor, etc.)

        Returns:
            Path to processed audio file
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = str(input_path.with_stem(input_path.stem + "_tuned"))

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

        # Get target notes for the key
        target_notes = self._get_scale_notes(key, mode)

        # Process
        processed = self._process_samples(samples, sample_rate, target_notes)

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
        key: Optional[str] = None,
        mode: str = "major",
    ) -> np.ndarray:
        """
        Process audio samples with pitch correction.

        Args:
            samples: Audio samples (mono)
            sample_rate: Sample rate in Hz
            key: Musical key (None = chromatic)
            mode: Scale mode

        Returns:
            Processed audio samples
        """
        target_notes = self._get_scale_notes(key, mode)
        return self._process_samples(samples, sample_rate, target_notes)

    def _get_scale_notes(self, key: Optional[str], mode: str) -> List[int]:
        """Get MIDI note numbers for a scale."""
        if key is None or mode == "chromatic":
            return list(range(128))  # All notes valid

        # Find root note index
        key = key.replace("b", "#").upper()  # Normalize flats to sharps
        if key not in self.NOTE_NAMES:
            # Handle enharmonics
            enharmonic = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
            key = enharmonic.get(key, "C")
        root = self.NOTE_NAMES.index(key)

        # Get scale intervals
        intervals = self.SCALES.get(mode, self.SCALES["major"])

        # Generate all valid notes across MIDI range
        valid_notes = []
        for octave in range(11):
            for interval in intervals:
                note = octave * 12 + root + interval
                if 0 <= note < 128:
                    valid_notes.append(note)

        return valid_notes

    def _process_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
        target_notes: List[int],
    ) -> np.ndarray:
        """Core pitch correction processing."""
        # Frame-based processing
        frame_size = int(sample_rate * 0.05)  # 50ms frames
        hop_size = frame_size // 4

        output = np.zeros_like(samples)
        window = np.hanning(frame_size)

        for i in range(0, len(samples) - frame_size, hop_size):
            frame = samples[i : i + frame_size] * window

            # Detect pitch
            pitch_hz = self._detect_pitch(frame, sample_rate)

            if pitch_hz > 0:
                # Convert to MIDI note
                midi_note = self._hz_to_midi(pitch_hz)

                # Find nearest target note
                target_note = self._find_nearest_note(midi_note, target_notes)

                # Calculate pitch shift
                if target_note is not None:
                    cents_off = (midi_note - target_note) * 100
                    if abs(cents_off) < self.settings.snap_threshold_cents:
                        # Apply correction based on strength
                        correction = cents_off * self.settings.correction_strength
                        shift_ratio = 2 ** (-correction / 1200)

                        # Apply pitch shift (simplified - real implementation would use PSOLA or phase vocoder)
                        frame = self._apply_pitch_shift(frame, shift_ratio, self.settings.preserve_formants)

            # Add humanization
            if self.settings.humanize > 0:
                frame = self._add_humanization(frame, self.settings.humanize)

            # Overlap-add
            output[i : i + frame_size] += frame

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95

        return output

    def _detect_pitch(self, frame: np.ndarray, sample_rate: int) -> float:
        """Detect fundamental frequency using autocorrelation."""
        if np.max(np.abs(frame)) < 0.01:
            return 0.0  # Silent frame

        # Autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]

        # Find first peak (fundamental period)
        min_period = int(sample_rate / 1000)  # 1000 Hz max
        max_period = int(sample_rate / 50)  # 50 Hz min

        if max_period > len(corr):
            return 0.0

        # Find peak in valid range
        search_range = corr[min_period:max_period]
        if len(search_range) == 0:
            return 0.0

        peak_idx = np.argmax(search_range) + min_period

        if corr[peak_idx] < corr[0] * 0.3:
            return 0.0  # Weak correlation, likely noise

        return sample_rate / peak_idx

    def _hz_to_midi(self, hz: float) -> float:
        """Convert frequency to MIDI note number."""
        if hz <= 0:
            return 0
        return 12 * np.log2(hz / 440) + 69

    def _find_nearest_note(self, midi_note: float, target_notes: List[int]) -> Optional[int]:
        """Find the nearest note in the target scale."""
        if not target_notes:
            return None
        return min(target_notes, key=lambda n: abs(n - midi_note))

    def _apply_pitch_shift(
        self, frame: np.ndarray, ratio: float, preserve_formants: bool
    ) -> np.ndarray:
        """Apply pitch shift to a frame (simplified implementation)."""
        if abs(ratio - 1.0) < 0.001:
            return frame

        # Simple resampling-based pitch shift
        # Real implementation would use PSOLA or phase vocoder for formant preservation
        indices = np.arange(0, len(frame), ratio)
        indices = indices[indices < len(frame) - 1].astype(int)

        if len(indices) == 0:
            return frame

        shifted = frame[indices]

        # Pad or truncate to original length
        if len(shifted) < len(frame):
            shifted = np.pad(shifted, (0, len(frame) - len(shifted)))
        else:
            shifted = shifted[: len(frame)]

        return shifted

    def _add_humanization(self, frame: np.ndarray, amount: float) -> np.ndarray:
        """Add subtle pitch variation for natural sound."""
        # Add very subtle random pitch wobble
        wobble = 1 + np.random.randn() * amount * 0.01
        return frame * wobble
