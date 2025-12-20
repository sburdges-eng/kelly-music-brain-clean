"""
AudioAnalyzer - Unified audio analysis interface.

Combines tempo detection, key detection, spectral analysis, and chord detection
into a single comprehensive analyzer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from music_brain.utils.path_utils import safe_path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False

from music_brain.audio.feel import analyze_feel, AudioFeatures
from music_brain.audio.chord_detection import ChordDetector, ChordProgressionDetection
from music_brain.audio.frequency_analysis import analyze_frequency_bands, FrequencyProfile


@dataclass
class AudioAnalysis:
    """
    Comprehensive audio analysis result.

    Combines tempo, key, spectral characteristics, and chord detection
    into a single analysis object.
    """
    # File info
    filename: str = ""
    duration_seconds: float = 0.0
    sample_rate: int = 44100

    # Tempo/rhythm
    tempo_bpm: float = 120.0
    tempo_confidence: float = 0.0
    beat_positions: List[float] = field(default_factory=list)

    # Key detection
    detected_key: Optional[str] = None
    key_mode: str = "major"  # major or minor

    # Spectral characteristics
    spectral_centroid: float = 0.0
    brightness: float = 0.0
    warmth: float = 0.0

    # Dynamics
    dynamic_range_db: float = 0.0
    rms_mean: float = 0.0

    # Groove
    swing_estimate: float = 0.0
    groove_regularity: float = 0.0

    # Chords (optional)
    chord_sequence: List[str] = field(default_factory=list)
    chord_progression: Optional[ChordProgressionDetection] = None

    # Frequency profile (optional)
    frequency_profile: Optional[FrequencyProfile] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "file_info": {
                "filename": self.filename,
                "duration_seconds": self.duration_seconds,
                "sample_rate": self.sample_rate,
            },
            "tempo": {
                "bpm": self.tempo_bpm,
                "confidence": self.tempo_confidence,
                "beat_count": len(self.beat_positions),
            },
            "key": {
                "detected": self.detected_key,
                "mode": self.key_mode,
            },
            "spectral": {
                "centroid": self.spectral_centroid,
                "brightness": self.brightness,
                "warmth": self.warmth,
            },
            "dynamics": {
                "range_db": self.dynamic_range_db,
                "rms_mean": self.rms_mean,
            },
            "groove": {
                "swing_estimate": self.swing_estimate,
                "regularity": self.groove_regularity,
            },
            "chords": self.chord_sequence,
        }

        if self.frequency_profile:
            result["frequency_bands"] = self.frequency_profile.to_dict()

        return result


class AudioAnalyzer:
    """
    Unified audio analysis interface.

    Combines multiple analysis methods:
    - Tempo and beat detection
    - Key detection
    - Spectral analysis
    - Chord detection
    - Frequency band analysis
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
    ):
        """
        Initialize audio analyzer.

        Args:
            sample_rate: Sample rate for analysis
            hop_length: Hop length for spectral analysis
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa required for AudioAnalyzer. "
                "Install with: pip install librosa numpy"
            )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self._chord_detector = ChordDetector(hop_length=hop_length)

    def analyze_file(
        self,
        audio_path: str,
        include_chords: bool = True,
        include_frequency_bands: bool = True,
    ) -> AudioAnalysis:
        """
        Perform comprehensive analysis on an audio file.

        Args:
            audio_path: Path to audio file
            include_chords: Whether to detect chords
            include_frequency_bands: Whether to analyze frequency bands

        Returns:
            AudioAnalysis with all detected features
        """
        audio_safe = safe_path(audio_path)
        if not audio_safe.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_safe}")

        # Get basic feel analysis
        feel = analyze_feel(str(audio_safe), hop_length=self.hop_length)

        # Initialize analysis with feel data
        analysis = AudioAnalysis(
            filename=str(audio_safe),
            duration_seconds=feel.duration_seconds,
            sample_rate=feel.sample_rate,
            tempo_bpm=feel.tempo_bpm,
            tempo_confidence=feel.tempo_confidence,
            beat_positions=feel.beat_positions,
            spectral_centroid=feel.spectral_centroid_mean,
            dynamic_range_db=feel.dynamic_range_db,
            rms_mean=feel.rms_mean,
            swing_estimate=feel.swing_estimate,
            groove_regularity=feel.groove_regularity,
        )

        # Detect chords and key
        if include_chords:
            chord_prog = self._chord_detector.detect_progression(str(audio_safe))
            analysis.chord_progression = chord_prog
            analysis.chord_sequence = chord_prog.chord_sequence

            if chord_prog.estimated_key:
                key_parts = chord_prog.estimated_key.split()
                analysis.detected_key = key_parts[0] if key_parts else None
                analysis.key_mode = key_parts[1] if len(key_parts) > 1 else "major"

        # Frequency band analysis
        if include_frequency_bands:
            freq_profile = analyze_frequency_bands(str(audio_safe))
            analysis.frequency_profile = freq_profile
            analysis.brightness = freq_profile.brightness
            analysis.warmth = freq_profile.warmth

        return analysis

    def analyze_waveform(
        self,
        samples: "np.ndarray",
        sample_rate: int,
    ) -> AudioAnalysis:
        """
        Analyze audio from raw waveform data.

        Args:
            samples: Audio samples (mono)
            sample_rate: Sample rate in Hz

        Returns:
            AudioAnalysis with detected features
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for waveform analysis")

        # Ensure mono
        if len(samples.shape) > 1:
            samples = np.mean(samples, axis=1)

        duration = len(samples) / sample_rate

        # Tempo and beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            y=samples, sr=sample_rate, hop_length=self.hop_length
        )
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sample_rate, hop_length=self.hop_length
        )

        # RMS energy
        rms = librosa.feature.rms(y=samples, hop_length=self.hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms)
        dynamic_range = float(np.max(rms_db) - np.min(rms_db[rms_db > -60]))

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=samples, sr=sample_rate, hop_length=self.hop_length
        )[0]

        # Chromagram for key detection
        chroma = librosa.feature.chroma_cqt(
            y=samples, sr=sample_rate, hop_length=self.hop_length
        )
        key, mode = self._detect_key_from_chroma(chroma)

        return AudioAnalysis(
            filename="<waveform>",
            duration_seconds=duration,
            sample_rate=sample_rate,
            tempo_bpm=float(tempo),
            beat_positions=beat_times.tolist(),
            detected_key=key,
            key_mode=mode,
            spectral_centroid=float(np.mean(spectral_centroid)),
            dynamic_range_db=dynamic_range,
            rms_mean=float(np.mean(rms)),
        )

    def detect_bpm(
        self,
        samples: "np.ndarray",
        sample_rate: int,
    ) -> float:
        """
        Detect tempo (BPM) from audio samples.

        Args:
            samples: Audio samples (mono)
            sample_rate: Sample rate in Hz

        Returns:
            Detected BPM
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for BPM detection")

        # Ensure mono
        if len(samples.shape) > 1:
            samples = np.mean(samples, axis=1)

        tempo, _ = librosa.beat.beat_track(
            y=samples, sr=sample_rate, hop_length=self.hop_length
        )
        return float(tempo)

    def detect_key(
        self,
        samples: "np.ndarray",
        sample_rate: int,
    ) -> Tuple[str, str]:
        """
        Detect musical key from audio samples.

        Args:
            samples: Audio samples (mono)
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (key_name, mode) e.g., ("C", "major")
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for key detection")

        # Ensure mono
        if len(samples.shape) > 1:
            samples = np.mean(samples, axis=1)

        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(
            y=samples, sr=sample_rate, hop_length=self.hop_length
        )

        return self._detect_key_from_chroma(chroma)

    def _detect_key_from_chroma(
        self,
        chroma: "np.ndarray",
    ) -> Tuple[str, str]:
        """
        Detect key from chromagram.

        Uses Krumhansl-Schmuckler key-finding algorithm.
        """
        NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Major and minor key profiles (Krumhansl)
        MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Average chroma
        chroma_mean = np.mean(chroma, axis=1)

        best_corr = -1
        best_key = 0
        best_mode = "major"

        for shift in range(12):
            # Rotate chroma to align with each possible key
            rotated = np.roll(chroma_mean, -shift)

            # Correlate with major profile
            major_corr = np.corrcoef(rotated, MAJOR_PROFILE)[0, 1]
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = shift
                best_mode = "major"

            # Correlate with minor profile
            minor_corr = np.corrcoef(rotated, MINOR_PROFILE)[0, 1]
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = shift
                best_mode = "minor"

        return NOTE_NAMES[best_key], best_mode


# Convenience function
def analyze_audio(
    audio_path: str,
    include_chords: bool = True,
    include_frequency_bands: bool = True,
) -> AudioAnalysis:
    """
    Convenience function to analyze an audio file.

    Args:
        audio_path: Path to audio file
        include_chords: Whether to detect chords
        include_frequency_bands: Whether to analyze frequency bands

    Returns:
        AudioAnalysis result
    """
    analyzer = AudioAnalyzer()
    return analyzer.analyze_file(
        audio_path,
        include_chords=include_chords,
        include_frequency_bands=include_frequency_bands,
    )
