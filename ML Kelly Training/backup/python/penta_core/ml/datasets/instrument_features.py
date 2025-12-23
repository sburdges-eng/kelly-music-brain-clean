"""
Instrument Feature Extraction for Kelly ML Training.

Extracts BOTH technical and emotional features from instrument audio:

Technical Features:
- Spectral envelope (timbre signature)
- Attack/decay characteristics
- Harmonic structure
- Playing technique indicators

Emotional Features:
- Expression dynamics
- Energy contour
- Sentiment indicators
- Human feel vs mechanical
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Instrument Features Dataclass
# =============================================================================


@dataclass
class InstrumentFeatures:
    """Extracted features from an instrument audio sample."""
    
    # File metadata
    file_path: str = ""
    duration_sec: float = 0.0
    sample_rate: int = 22050
    
    # ==========================================================================
    # TECHNICAL FEATURES (What instrument + How it's played)
    # ==========================================================================
    
    # Spectral envelope (timbre signature) - 40 dims
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_bandwidth_std: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_rolloff_std: float = 0.0
    spectral_flatness_mean: float = 0.0
    spectral_flatness_std: float = 0.0
    spectral_contrast: List[float] = field(default_factory=list)  # 7 bands
    mfcc_mean: List[float] = field(default_factory=list)  # 20 coefficients
    mfcc_std: List[float] = field(default_factory=list)   # 20 coefficients
    
    # Harmonic structure - 16 dims
    harmonic_ratio: float = 0.0           # Harmonic vs noise
    fundamental_freq: float = 0.0         # Detected pitch
    pitch_confidence: float = 0.0         # How clear is the pitch
    inharmonicity: float = 0.0            # Deviation from pure harmonics
    odd_even_ratio: float = 0.0           # Odd vs even harmonics
    spectral_peaks: List[float] = field(default_factory=list)  # Top 10 peaks
    
    # Attack/decay characteristics - 12 dims
    attack_time_ms: float = 0.0           # Time to reach peak
    decay_time_ms: float = 0.0            # Time to decay
    attack_slope: float = 0.0             # Steepness of attack
    release_slope: float = 0.0            # Steepness of release
    onset_strength_mean: float = 0.0
    onset_strength_std: float = 0.0
    onset_count: int = 0                  # Number of note onsets
    onset_regularity: float = 0.0         # How regular are onsets
    transient_sharpness: float = 0.0      # How sharp are transients
    sustain_level: float = 0.0            # Average sustain level
    adsr_envelope: List[float] = field(default_factory=list)  # 4-point ADSR
    
    # Technique indicators - 12 dims
    vibrato_rate: float = 0.0             # Vibrato frequency (Hz)
    vibrato_depth: float = 0.0            # Vibrato amount (cents)
    tremolo_rate: float = 0.0             # Amplitude modulation rate
    tremolo_depth: float = 0.0            # Amplitude modulation depth
    pitch_bend_range: float = 0.0         # Range of pitch bending
    legato_score: float = 0.0             # How connected are notes
    staccato_score: float = 0.0           # How detached are notes
    articulation_vector: List[float] = field(default_factory=list)  # 4 dims
    
    # Register/range - 4 dims
    pitch_mean: float = 0.0               # Average pitch (MIDI note)
    pitch_std: float = 0.0                # Pitch variation
    pitch_range: float = 0.0              # Max - min pitch
    register_class: int = 0               # 0=bass, 1=mid, 2=treble
    
    # ==========================================================================
    # EMOTIONAL FEATURES (How it feels)
    # ==========================================================================
    
    # Expression dynamics - 16 dims
    velocity_mean: float = 0.0            # Average loudness
    velocity_std: float = 0.0             # Dynamic variation
    velocity_range: float = 0.0           # Dynamic range
    crescendo_count: int = 0              # Number of crescendos
    decrescendo_count: int = 0            # Number of decrescendos
    dynamic_curve: List[float] = field(default_factory=list)  # 8 points
    accent_frequency: float = 0.0         # How often accents occur
    accent_strength: float = 0.0          # How strong are accents
    
    # Energy contour - 12 dims
    rms_mean: float = 0.0
    rms_std: float = 0.0
    energy_curve: List[float] = field(default_factory=list)  # 8 points
    energy_buildup: float = 0.0           # Increasing energy trend
    energy_release: float = 0.0           # Decreasing energy trend
    
    # Sentiment indicators - 12 dims
    brightness_score: float = 0.0         # Spectral brightness → mood
    warmth_score: float = 0.0             # Low-frequency presence
    tension_score: float = 0.0            # Dissonance/instability
    resolution_score: float = 0.0         # Consonance/stability
    major_minor_score: float = 0.0        # Major (+) vs minor (-) feel
    
    # Human feel - 8 dims
    timing_humanization: float = 0.0      # Deviation from grid
    velocity_humanization: float = 0.0    # Velocity variation pattern
    micro_timing_variance: float = 0.0    # Sub-beat timing variation
    groove_consistency: float = 0.0       # Pattern regularity
    expressiveness_score: float = 0.0     # Overall human expression
    
    # Composite scores (derived)
    technical_confidence: float = 0.0     # How confident in technical classification
    emotional_confidence: float = 0.0     # How confident in emotional classification
    
    # Full feature vectors for ML
    technical_vector: List[float] = field(default_factory=list)  # 80 dims
    emotional_vector: List[float] = field(default_factory=list)  # 80 dims
    combined_vector: List[float] = field(default_factory=list)   # 160 dims
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return asdict(self)
    
    def save(self, path: Path):
        """Save features to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_npz(self, path: Path):
        """Save features to npz for training."""
        np.savez_compressed(
            path,
            technical=np.array(self.technical_vector, dtype=np.float32),
            emotional=np.array(self.emotional_vector, dtype=np.float32),
            combined=np.array(self.combined_vector, dtype=np.float32),
        )
    
    @classmethod
    def load(cls, path: Path) -> "InstrumentFeatures":
        """Load features from JSON."""
        with open(path) as f:
            return cls(**json.load(f))


# =============================================================================
# Instrument Feature Extractor
# =============================================================================


class InstrumentFeatureExtractor:
    """
    Extracts technical and emotional features from instrument audio.
    
    Usage:
        extractor = InstrumentFeatureExtractor()
        features = extractor.extract("guitar_sample.wav")
        
        # Access technical features
        print(f"Attack time: {features.attack_time_ms}ms")
        print(f"Vibrato rate: {features.vibrato_rate}Hz")
        
        # Access emotional features
        print(f"Expression score: {features.expressiveness_score}")
        print(f"Energy: {features.rms_mean}")
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_mfcc: int = 20,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract(self, audio_path: Path) -> InstrumentFeatures:
        """Extract all features from an audio file."""
        try:
            import librosa
        except ImportError:
            logger.error("librosa required for audio feature extraction")
            return InstrumentFeatures(file_path=str(audio_path))
        
        audio_path = Path(audio_path)
        features = InstrumentFeatures(file_path=str(audio_path))
        
        # Load audio
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        except Exception as e:
            logger.error(f"Failed to load {audio_path}: {e}")
            return features
        
        features.sample_rate = sr
        features.duration_sec = len(y) / sr
        
        # Extract technical features
        self._extract_spectral_features(y, sr, features)
        self._extract_harmonic_features(y, sr, features)
        self._extract_attack_decay_features(y, sr, features)
        self._extract_technique_features(y, sr, features)
        self._extract_register_features(y, sr, features)
        
        # Extract emotional features
        self._extract_expression_features(y, sr, features)
        self._extract_energy_features(y, sr, features)
        self._extract_sentiment_features(y, sr, features)
        self._extract_human_feel_features(y, sr, features)
        
        # Build feature vectors
        self._build_feature_vectors(features)
        
        return features
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int, features: InstrumentFeatures):
        """Extract spectral envelope features (timbre signature)."""
        import librosa
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        features.spectral_centroid_mean = float(np.mean(centroid))
        features.spectral_centroid_std = float(np.std(centroid))
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        features.spectral_bandwidth_mean = float(np.mean(bandwidth))
        features.spectral_bandwidth_std = float(np.std(bandwidth))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        features.spectral_rolloff_mean = float(np.mean(rolloff))
        features.spectral_rolloff_std = float(np.std(rolloff))
        
        # Spectral flatness (noise vs tone)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        features.spectral_flatness_mean = float(np.mean(flatness))
        features.spectral_flatness_std = float(np.std(flatness))
        
        # Spectral contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
        features.spectral_contrast = np.mean(contrast, axis=1).tolist()
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
        features.mfcc_mean = np.mean(mfcc, axis=1).tolist()
        features.mfcc_std = np.std(mfcc, axis=1).tolist()
    
    def _extract_harmonic_features(self, y: np.ndarray, sr: int, features: InstrumentFeatures):
        """Extract harmonic structure features."""
        import librosa
        
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        total_energy = np.sum(y ** 2)
        features.harmonic_ratio = float(harmonic_energy / (total_energy + 1e-8))
        
        # Pitch detection
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=self.hop_length)
        
        # Get most confident pitch per frame
        pitch_idx = np.argmax(magnitudes, axis=0)
        pitch_values = pitches[pitch_idx, np.arange(pitches.shape[1])]
        
        # Filter out zero pitches
        valid_pitches = pitch_values[pitch_values > 0]
        if len(valid_pitches) > 0:
            features.fundamental_freq = float(np.median(valid_pitches))
            features.pitch_confidence = float(len(valid_pitches) / len(pitch_values))
        
        # Spectral peaks (top 10 frequencies)
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        avg_spectrum = np.mean(S, axis=1)
        peak_indices = np.argsort(avg_spectrum)[-10:][::-1]
        peak_freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)[peak_indices]
        features.spectral_peaks = peak_freqs.tolist()
        
        # Inharmonicity (deviation from perfect harmonics)
        if features.fundamental_freq > 0:
            harmonics = np.arange(1, 11) * features.fundamental_freq
            actual_peaks = np.array(features.spectral_peaks[:10])
            if len(actual_peaks) > 0:
                # Find closest actual peak to each expected harmonic
                deviations = []
                for h in harmonics:
                    closest = actual_peaks[np.argmin(np.abs(actual_peaks - h))]
                    deviations.append(abs(closest - h) / h)
                features.inharmonicity = float(np.mean(deviations))
    
    def _extract_attack_decay_features(self, y: np.ndarray, sr: int, features: InstrumentFeatures):
        """Extract attack/decay envelope characteristics."""
        import librosa
        
        # RMS envelope
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Find attack (time to reach peak)
        if len(rms) > 0:
            peak_idx = np.argmax(rms)
            attack_samples = peak_idx * self.hop_length
            features.attack_time_ms = float(attack_samples / sr * 1000)
            
            # Attack slope (how steep)
            if peak_idx > 0:
                features.attack_slope = float((rms[peak_idx] - rms[0]) / peak_idx)
            
            # Decay time (time to drop to 60% of peak)
            peak_val = rms[peak_idx]
            decay_threshold = peak_val * 0.6
            decay_frames = np.where(rms[peak_idx:] < decay_threshold)[0]
            if len(decay_frames) > 0:
                decay_samples = decay_frames[0] * self.hop_length
                features.decay_time_ms = float(decay_samples / sr * 1000)
            
            # Sustain level (average after decay)
            sustain_start = peak_idx + len(decay_frames) if len(decay_frames) > 0 else peak_idx
            if sustain_start < len(rms):
                features.sustain_level = float(np.mean(rms[sustain_start:]))
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        features.onset_strength_mean = float(np.mean(onset_env))
        features.onset_strength_std = float(np.std(onset_env))
        
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=self.hop_length, units='time'
        )
        features.onset_count = len(onsets)
        
        # Onset regularity
        if len(onsets) > 1:
            intervals = np.diff(onsets)
            features.onset_regularity = float(1.0 - np.std(intervals) / (np.mean(intervals) + 1e-8))
        
        # ADSR envelope (simplified 4-point)
        if len(rms) >= 4:
            indices = np.linspace(0, len(rms) - 1, 4).astype(int)
            features.adsr_envelope = rms[indices].tolist()
    
    def _extract_technique_features(self, y: np.ndarray, sr: int, features: InstrumentFeatures):
        """Extract playing technique indicators."""
        import librosa
        
        # Vibrato detection (pitch modulation analysis)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=self.hop_length)
        pitch_idx = np.argmax(magnitudes, axis=0)
        pitch_track = pitches[pitch_idx, np.arange(pitches.shape[1])]
        
        valid_pitch = pitch_track[pitch_track > 0]
        if len(valid_pitch) > 10:
            # Vibrato rate: look for periodic pitch variation
            pitch_diff = np.diff(valid_pitch)
            
            # Simple vibrato rate estimation via zero crossings
            zero_crossings = np.where(np.diff(np.sign(pitch_diff)))[0]
            if len(zero_crossings) > 2:
                avg_period = np.mean(np.diff(zero_crossings))
                if avg_period > 0:
                    features.vibrato_rate = float(sr / (avg_period * self.hop_length))
            
            # Vibrato depth (pitch variation in cents)
            pitch_cents = 1200 * np.log2(valid_pitch / np.median(valid_pitch) + 1e-8)
            features.vibrato_depth = float(np.std(pitch_cents))
        
        # Tremolo detection (amplitude modulation)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        if len(rms) > 10:
            rms_detrended = rms - np.mean(rms)
            
            # Tremolo rate via autocorrelation
            autocorr = np.correlate(rms_detrended, rms_detrended, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first peak after zero lag
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
                    break
            
            if peaks:
                tremolo_period_frames = peaks[0]
                features.tremolo_rate = float(sr / (tremolo_period_frames * self.hop_length))
            
            features.tremolo_depth = float(np.std(rms) / (np.mean(rms) + 1e-8))
        
        # Legato vs staccato scoring
        # Based on note overlap and sustain characteristics
        if features.decay_time_ms > 0 and features.attack_time_ms > 0:
            sustain_ratio = features.sustain_level / (features.onset_strength_mean + 1e-8)
            features.legato_score = min(1.0, float(sustain_ratio))
            features.staccato_score = 1.0 - features.legato_score
        
        # Articulation vector (normalized)
        features.articulation_vector = [
            features.legato_score,
            features.staccato_score,
            min(1.0, features.attack_slope * 10) if features.attack_slope else 0.0,
            features.onset_regularity,
        ]
    
    def _extract_register_features(self, y: np.ndarray, sr: int, features: InstrumentFeatures):
        """Extract pitch register features."""
        import librosa
        
        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=self.hop_length)
        pitch_idx = np.argmax(magnitudes, axis=0)
        pitch_hz = pitches[pitch_idx, np.arange(pitches.shape[1])]
        
        valid_pitch = pitch_hz[pitch_hz > 0]
        if len(valid_pitch) > 0:
            # Convert to MIDI note numbers
            midi_notes = librosa.hz_to_midi(valid_pitch)
            features.pitch_mean = float(np.mean(midi_notes))
            features.pitch_std = float(np.std(midi_notes))
            features.pitch_range = float(np.max(midi_notes) - np.min(midi_notes))
            
            # Register classification
            # Bass: < 48 (C3), Mid: 48-72 (C3-C5), Treble: > 72 (C5)
            if features.pitch_mean < 48:
                features.register_class = 0  # Bass
            elif features.pitch_mean < 72:
                features.register_class = 1  # Mid
            else:
                features.register_class = 2  # Treble
    
    def _extract_expression_features(self, y: np.ndarray, sr: int, features: InstrumentFeatures):
        """Extract expression/dynamics features."""
        import librosa
        
        # RMS for dynamics
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        features.velocity_mean = float(np.mean(rms))
        features.velocity_std = float(np.std(rms))
        features.velocity_range = float(np.max(rms) - np.min(rms))
        
        # Detect crescendos/decrescendos
        smoothed_rms = np.convolve(rms, np.ones(5)/5, mode='valid')
        if len(smoothed_rms) > 10:
            diff = np.diff(smoothed_rms)
            
            # Crescendo: sustained increase
            crescendo_regions = np.where(diff > np.std(diff))[0]
            features.crescendo_count = len(np.split(crescendo_regions, 
                np.where(np.diff(crescendo_regions) > 3)[0] + 1))
            
            # Decrescendo: sustained decrease
            decrescendo_regions = np.where(diff < -np.std(diff))[0]
            features.decrescendo_count = len(np.split(decrescendo_regions,
                np.where(np.diff(decrescendo_regions) > 3)[0] + 1))
        
        # Dynamic curve (8 points)
        if len(rms) >= 8:
            indices = np.linspace(0, len(rms) - 1, 8).astype(int)
            features.dynamic_curve = rms[indices].tolist()
        
        # Accent detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        peaks = onset_env[onset_env > np.mean(onset_env) + np.std(onset_env)]
        features.accent_frequency = float(len(peaks) / len(onset_env))
        features.accent_strength = float(np.mean(peaks)) if len(peaks) > 0 else 0.0
    
    def _extract_energy_features(self, y: np.ndarray, sr: int, features: InstrumentFeatures):
        """Extract energy contour features."""
        import librosa
        
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        features.rms_mean = float(np.mean(rms))
        features.rms_std = float(np.std(rms))
        
        # Energy curve (8 points)
        if len(rms) >= 8:
            indices = np.linspace(0, len(rms) - 1, 8).astype(int)
            features.energy_curve = rms[indices].tolist()
        
        # Energy trends
        if len(rms) > 1:
            half = len(rms) // 2
            first_half_energy = np.mean(rms[:half])
            second_half_energy = np.mean(rms[half:])
            
            if first_half_energy < second_half_energy:
                features.energy_buildup = float((second_half_energy - first_half_energy) / 
                                                (first_half_energy + 1e-8))
            else:
                features.energy_release = float((first_half_energy - second_half_energy) /
                                                (first_half_energy + 1e-8))
    
    def _extract_sentiment_features(self, y: np.ndarray, sr: int, features: InstrumentFeatures):
        """Extract sentiment/mood indicators."""
        import librosa
        
        # Brightness (spectral centroid relative to max)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.brightness_score = float(np.mean(centroid) / (sr / 2))  # Normalized
        
        # Warmth (low frequency content)
        S = np.abs(librosa.stft(y, n_fft=self.n_fft))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        low_freq_mask = freqs < 500
        low_energy = np.mean(S[low_freq_mask, :])
        total_energy = np.mean(S)
        features.warmth_score = float(low_energy / (total_energy + 1e-8))
        
        # Tension/resolution (simplified via dissonance estimation)
        # Using spectral flatness as proxy (higher = more noise/tension)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features.tension_score = float(np.mean(flatness))
        features.resolution_score = 1.0 - features.tension_score
        
        # Major/minor feel (very simplified - based on spectral characteristics)
        # Brighter, more regular harmonics → major feel
        # Darker, less regular → minor feel
        features.major_minor_score = float(
            features.brightness_score * 0.5 + 
            features.harmonic_ratio * 0.3 +
            (1.0 - features.tension_score) * 0.2
        ) * 2 - 1  # Scale to -1 (minor) to +1 (major)
    
    def _extract_human_feel_features(self, y: np.ndarray, sr: int, features: InstrumentFeatures):
        """Extract human feel vs mechanical indicators."""
        import librosa
        
        # Timing humanization (micro-timing variations)
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', hop_length=self.hop_length)
        
        if len(onset_times) > 2:
            intervals = np.diff(onset_times)
            
            # Detect probable tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            beat_duration = 60.0 / float(tempo) if np.isscalar(tempo) else 60.0 / float(tempo[0])
            
            # Quantize intervals to nearest beat subdivision
            subdivisions = intervals / (beat_duration / 4)  # 16th notes
            quantized = np.round(subdivisions)
            deviations = (subdivisions - quantized) * (beat_duration / 4) * 1000  # in ms
            
            features.timing_humanization = float(np.std(deviations))
            features.micro_timing_variance = float(np.mean(np.abs(deviations)))
        
        # Velocity humanization
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames', hop_length=self.hop_length)
        
        if len(onset_frames) > 1:
            onset_velocities = rms[np.clip(onset_frames, 0, len(rms) - 1)]
            features.velocity_humanization = float(np.std(onset_velocities) / (np.mean(onset_velocities) + 1e-8))
        
        # Groove consistency
        if len(onset_times) > 4:
            # Look for repeating patterns
            intervals = np.diff(onset_times)
            autocorr = np.correlate(intervals, intervals, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if len(autocorr) > 1:
                features.groove_consistency = float(np.max(autocorr[1:]) / (autocorr[0] + 1e-8))
        
        # Overall expressiveness score
        features.expressiveness_score = float(
            features.timing_humanization * 0.2 +
            features.velocity_humanization * 0.2 +
            features.vibrato_depth * 0.002 +  # Scale down vibrato
            features.velocity_range * 5 +      # Scale up dynamic range
            features.tremolo_depth * 0.3
        )
    
    def _build_feature_vectors(self, features: InstrumentFeatures):
        """Build normalized feature vectors for ML."""
        # Technical vector (80 dims)
        technical = []
        
        # Spectral (40 dims)
        technical.append(features.spectral_centroid_mean / 5000)
        technical.append(features.spectral_centroid_std / 2000)
        technical.append(features.spectral_bandwidth_mean / 3000)
        technical.append(features.spectral_bandwidth_std / 1000)
        technical.append(features.spectral_rolloff_mean / 10000)
        technical.append(features.spectral_rolloff_std / 3000)
        technical.append(features.spectral_flatness_mean)
        technical.append(features.spectral_flatness_std)
        technical.extend(features.spectral_contrast[:7] if features.spectral_contrast else [0]*7)
        technical.extend([m / 100 for m in (features.mfcc_mean[:13] if features.mfcc_mean else [0]*13)])
        technical.extend([s / 50 for s in (features.mfcc_std[:7] if features.mfcc_std else [0]*7)])
        
        # Harmonic (12 dims)
        technical.append(features.harmonic_ratio)
        technical.append(features.fundamental_freq / 1000)
        technical.append(features.pitch_confidence)
        technical.append(features.inharmonicity)
        technical.append(features.odd_even_ratio)
        technical.extend([p / 5000 for p in (features.spectral_peaks[:7] if features.spectral_peaks else [0]*7)])
        
        # Attack/decay (12 dims)
        technical.append(features.attack_time_ms / 500)
        technical.append(features.decay_time_ms / 1000)
        technical.append(min(1.0, features.attack_slope * 10))
        technical.append(min(1.0, features.release_slope * 10) if features.release_slope else 0)
        technical.append(features.onset_strength_mean)
        technical.append(features.onset_strength_std)
        technical.append(features.onset_count / 20)
        technical.append(features.onset_regularity)
        technical.extend(features.adsr_envelope[:4] if features.adsr_envelope else [0]*4)
        
        # Technique (8 dims)
        technical.append(features.vibrato_rate / 10)
        technical.append(features.vibrato_depth / 100)
        technical.append(features.tremolo_rate / 10)
        technical.append(features.tremolo_depth)
        technical.append(features.legato_score)
        technical.append(features.staccato_score)
        technical.extend(features.articulation_vector[:2] if features.articulation_vector else [0]*2)
        
        # Register (4 dims)
        technical.append(features.pitch_mean / 127)
        technical.append(features.pitch_std / 24)
        technical.append(features.pitch_range / 48)
        technical.append(features.register_class / 2)
        
        # Pad to 80
        while len(technical) < 80:
            technical.append(0.0)
        features.technical_vector = technical[:80]
        
        # Emotional vector (80 dims)
        emotional = []
        
        # Expression (16 dims)
        emotional.append(features.velocity_mean * 10)
        emotional.append(features.velocity_std * 10)
        emotional.append(features.velocity_range * 5)
        emotional.append(features.crescendo_count / 5)
        emotional.append(features.decrescendo_count / 5)
        emotional.extend(features.dynamic_curve[:8] if features.dynamic_curve else [0]*8)
        emotional.append(features.accent_frequency)
        emotional.append(features.accent_strength)
        
        # Energy (12 dims)
        emotional.append(features.rms_mean * 10)
        emotional.append(features.rms_std * 10)
        emotional.extend(features.energy_curve[:8] if features.energy_curve else [0]*8)
        emotional.append(features.energy_buildup)
        emotional.append(features.energy_release)
        
        # Sentiment (12 dims)
        emotional.append(features.brightness_score)
        emotional.append(features.warmth_score)
        emotional.append(features.tension_score)
        emotional.append(features.resolution_score)
        emotional.append((features.major_minor_score + 1) / 2)  # Normalize to 0-1
        emotional.extend([0.0] * 7)  # Placeholder for more sentiment features
        
        # Human feel (8 dims)
        emotional.append(features.timing_humanization / 50)
        emotional.append(features.velocity_humanization)
        emotional.append(features.micro_timing_variance / 20)
        emotional.append(features.groove_consistency)
        emotional.append(features.expressiveness_score / 2)
        emotional.extend([0.0] * 3)  # Placeholder
        
        # Composite (4 dims)
        technical_conf = 0.5 + 0.5 * features.pitch_confidence
        emotional_conf = 0.5 + 0.5 * features.expressiveness_score
        emotional.append(technical_conf)
        emotional.append(emotional_conf)
        emotional.extend([0.0] * 2)
        
        # Padding
        emotional.extend([0.0] * (32 - len(emotional) % 32 if len(emotional) % 32 != 0 else 0))
        
        while len(emotional) < 80:
            emotional.append(0.0)
        features.emotional_vector = emotional[:80]
        
        # Combined vector
        features.combined_vector = features.technical_vector + features.emotional_vector


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_instrument_features(audio_path: Path, **kwargs) -> InstrumentFeatures:
    """Extract features from an instrument audio file."""
    extractor = InstrumentFeatureExtractor(**kwargs)
    return extractor.extract(audio_path)


def extract_technical_features(audio_path: Path) -> Dict[str, Any]:
    """Extract only technical features."""
    features = extract_instrument_features(audio_path)
    return {
        'spectral_centroid': features.spectral_centroid_mean,
        'harmonic_ratio': features.harmonic_ratio,
        'attack_time_ms': features.attack_time_ms,
        'vibrato_rate': features.vibrato_rate,
        'register': features.register_class,
        'technical_vector': features.technical_vector,
    }


def extract_emotional_features(audio_path: Path) -> Dict[str, Any]:
    """Extract only emotional features."""
    features = extract_instrument_features(audio_path)
    return {
        'expressiveness': features.expressiveness_score,
        'energy_mean': features.rms_mean,
        'brightness': features.brightness_score,
        'tension': features.tension_score,
        'humanization': features.timing_humanization,
        'emotional_vector': features.emotional_vector,
    }

