"""
Audio Feature Extraction for Kelly ML Training.

Extracts features from audio files for training:
- EmotionRecognizer: Spectral features → emotion embedding
- DynamicsEngine: Envelope features → expression parameters
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Audio Features Dataclass
# =============================================================================


@dataclass
class AudioFeatures:
    """Extracted features from an audio file."""
    
    # File metadata
    file_path: str = ""
    duration_sec: float = 0.0
    sample_rate: int = 44100
    channels: int = 1
    
    # Spectral features (for EmotionRecognizer)
    mel_spectrogram: List[List[float]] = field(default_factory=list)  # (n_mels, time)
    mfcc: List[List[float]] = field(default_factory=list)  # (n_mfcc, time)
    chroma: List[List[float]] = field(default_factory=list)  # (12, time)
    spectral_centroid: List[float] = field(default_factory=list)
    spectral_rolloff: List[float] = field(default_factory=list)
    spectral_bandwidth: List[float] = field(default_factory=list)
    zero_crossing_rate: List[float] = field(default_factory=list)
    
    # Temporal features
    rms_energy: List[float] = field(default_factory=list)
    onset_strength: List[float] = field(default_factory=list)
    tempo_bpm: float = 0.0
    beat_frames: List[int] = field(default_factory=list)
    
    # Statistical summaries
    mfcc_mean: List[float] = field(default_factory=list)
    mfcc_std: List[float] = field(default_factory=list)
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    rms_mean: float = 0.0
    rms_std: float = 0.0
    
    # Emotion-related features
    valence_estimate: float = 0.0  # From spectral brightness/tempo
    arousal_estimate: float = 0.0  # From energy/tempo
    
    # Feature vector for ML
    feature_vector: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (without large spectrograms)."""
        d = asdict(self)
        # Truncate large arrays for JSON serialization
        if len(d['mel_spectrogram']) > 0:
            d['mel_spectrogram_shape'] = [len(d['mel_spectrogram']), len(d['mel_spectrogram'][0]) if d['mel_spectrogram'] else 0]
            d['mel_spectrogram'] = []  # Too large for JSON
        if len(d['mfcc']) > 0:
            d['mfcc_shape'] = [len(d['mfcc']), len(d['mfcc'][0]) if d['mfcc'] else 0]
            d['mfcc'] = []
        if len(d['chroma']) > 0:
            d['chroma_shape'] = [len(d['chroma']), len(d['chroma'][0]) if d['chroma'] else 0]
            d['chroma'] = []
        return d
    
    def save(self, path: Path, include_spectrograms: bool = False):
        """Save features to file."""
        if include_spectrograms:
            # Save as npz for large arrays
            np.savez(path, **asdict(self))
        else:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
    
    def save_npz(self, path: Path):
        """Save all features including spectrograms to npz."""
        data = {k: np.array(v) if isinstance(v, list) else v for k, v in asdict(self).items()}
        np.savez_compressed(path, **data)
    
    @classmethod
    def load_npz(cls, path: Path) -> "AudioFeatures":
        """Load features from npz file."""
        data = np.load(path, allow_pickle=True)
        kwargs = {}
        for key in cls.__dataclass_fields__:
            if key in data:
                val = data[key]
                if isinstance(val, np.ndarray):
                    val = val.tolist() if val.ndim > 0 else val.item()
                kwargs[key] = val
        return cls(**kwargs)


# =============================================================================
# Audio Feature Extractor
# =============================================================================


class AudioFeatureExtractor:
    """
    Extracts training features from audio files.
    
    Uses librosa for audio analysis. Optimized for emotion recognition.
    
    Usage:
        extractor = AudioFeatureExtractor()
        features = extractor.extract("song.wav")
        features.save_npz("song_features.npz")
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 64,
        n_mfcc: int = 20,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_duration: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration = max_duration
    
    def extract(self, audio_path: Path) -> AudioFeatures:
        """Extract all features from an audio file."""
        try:
            import librosa
        except ImportError:
            logger.error("librosa required for audio feature extraction")
            return AudioFeatures(file_path=str(audio_path))
        
        audio_path = Path(audio_path)
        features = AudioFeatures(file_path=str(audio_path))
        
        # Load audio
        try:
            y, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                duration=self.max_duration,
                mono=True,
            )
        except Exception as e:
            logger.error(f"Failed to load {audio_path}: {e}")
            return features
        
        features.sample_rate = sr
        features.duration_sec = len(y) / sr
        features.channels = 1
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.mel_spectrogram = mel_spec_db.tolist()
        
        # MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        features.mfcc = mfcc.tolist()
        features.mfcc_mean = np.mean(mfcc, axis=1).tolist()
        features.mfcc_std = np.std(mfcc, axis=1).tolist()
        
        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        features.chroma = chroma.tolist()
        
        # Spectral features
        features.spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.hop_length
        )[0].tolist()
        features.spectral_centroid_mean = float(np.mean(features.spectral_centroid))
        features.spectral_centroid_std = float(np.std(features.spectral_centroid))
        
        features.spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=self.hop_length
        )[0].tolist()
        
        features.spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=self.hop_length
        )[0].tolist()
        
        features.zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y, hop_length=self.hop_length
        )[0].tolist()
        
        # Temporal features
        features.rms_energy = librosa.feature.rms(
            y=y, hop_length=self.hop_length
        )[0].tolist()
        features.rms_mean = float(np.mean(features.rms_energy))
        features.rms_std = float(np.std(features.rms_energy))
        
        # Onset strength
        features.onset_strength = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        ).tolist()
        
        # Tempo and beats
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=self.hop_length
        )
        features.tempo_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        features.beat_frames = beat_frames.tolist()
        
        # Estimate emotional features
        self._estimate_emotion_features(features)
        
        # Build feature vector
        features.feature_vector = self._build_feature_vector(features)
        
        return features
    
    def _estimate_emotion_features(self, features: AudioFeatures):
        """Estimate valence/arousal from acoustic features."""
        # Simple heuristics (should be replaced by trained model)
        
        # Valence: correlated with spectral brightness (centroid) and major/minor
        # Higher centroid → brighter → more positive valence
        centroid_norm = features.spectral_centroid_mean / 5000  # Normalize
        features.valence_estimate = np.clip(centroid_norm * 2 - 1, -1, 1)
        
        # Arousal: correlated with RMS energy and tempo
        energy_norm = features.rms_mean * 10  # Normalize
        tempo_norm = features.tempo_bpm / 180  # Normalize
        features.arousal_estimate = np.clip((energy_norm + tempo_norm) / 2, -1, 1)
    
    def _build_feature_vector(self, features: AudioFeatures) -> List[float]:
        """Build a combined feature vector for ML input (128 dims)."""
        vec = []
        
        # MFCC statistics (20 * 2 = 40)
        vec.extend(features.mfcc_mean[:20] if features.mfcc_mean else [0] * 20)
        vec.extend(features.mfcc_std[:20] if features.mfcc_std else [0] * 20)
        
        # Chroma mean (12)
        if features.chroma:
            chroma_mean = np.mean(features.chroma, axis=1).tolist()
            vec.extend(chroma_mean[:12])
        else:
            vec.extend([0] * 12)
        
        # Spectral features (8)
        vec.append(features.spectral_centroid_mean / 5000)
        vec.append(features.spectral_centroid_std / 2000)
        vec.append(np.mean(features.spectral_rolloff) / 10000 if features.spectral_rolloff else 0)
        vec.append(np.mean(features.spectral_bandwidth) / 3000 if features.spectral_bandwidth else 0)
        vec.append(np.mean(features.zero_crossing_rate) * 10 if features.zero_crossing_rate else 0)
        vec.append(features.rms_mean * 10)
        vec.append(features.rms_std * 10)
        vec.append(features.tempo_bpm / 200)
        
        # Onset strength statistics (4)
        if features.onset_strength:
            vec.append(np.mean(features.onset_strength))
            vec.append(np.std(features.onset_strength))
            vec.append(np.max(features.onset_strength))
            vec.append(len(features.beat_frames) / features.duration_sec if features.duration_sec > 0 else 0)
        else:
            vec.extend([0] * 4)
        
        # RMS envelope (32 points, normalized)
        if features.rms_energy:
            # Resample to 32 points
            rms = np.array(features.rms_energy)
            indices = np.linspace(0, len(rms) - 1, 32).astype(int)
            rms_resampled = rms[indices]
            rms_norm = rms_resampled / (np.max(rms_resampled) + 1e-6)
            vec.extend(rms_norm.tolist())
        else:
            vec.extend([0] * 32)
        
        # Mel spectrogram average over time (32, from n_mels)
        if features.mel_spectrogram:
            mel = np.array(features.mel_spectrogram)
            mel_mean = np.mean(mel, axis=1)
            # Resample to 32
            indices = np.linspace(0, len(mel_mean) - 1, 32).astype(int)
            mel_resampled = mel_mean[indices]
            # Normalize
            mel_norm = (mel_resampled - np.min(mel_resampled)) / (np.max(mel_resampled) - np.min(mel_resampled) + 1e-6)
            vec.extend(mel_norm.tolist())
        else:
            vec.extend([0] * 32)
        
        # Pad to 128 if needed
        while len(vec) < 128:
            vec.append(0)
        
        return vec[:128]
    
    def extract_batch(self, audio_paths: List[Path], show_progress: bool = True) -> List[AudioFeatures]:
        """Extract features from multiple audio files."""
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                paths = tqdm(audio_paths, desc="Extracting audio features")
            except ImportError:
                paths = audio_paths
        else:
            paths = audio_paths
        
        for path in paths:
            try:
                features = self.extract(path)
                results.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features from {path}: {e}")
                results.append(AudioFeatures(file_path=str(path)))
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_audio_features(audio_path: Path, **kwargs) -> AudioFeatures:
    """Extract features from an audio file."""
    extractor = AudioFeatureExtractor(**kwargs)
    return extractor.extract(audio_path)


def extract_emotion_features(audio_path: Path) -> Dict[str, Any]:
    """Extract emotion-specific features for EmotionRecognizer."""
    features = extract_audio_features(audio_path)
    return {
        'mfcc_mean': features.mfcc_mean,
        'mfcc_std': features.mfcc_std,
        'spectral_centroid_mean': features.spectral_centroid_mean,
        'rms_mean': features.rms_mean,
        'tempo_bpm': features.tempo_bpm,
        'valence_estimate': features.valence_estimate,
        'arousal_estimate': features.arousal_estimate,
        'feature_vector': features.feature_vector,
    }


def extract_spectral_features(audio_path: Path) -> Dict[str, Any]:
    """Extract spectral features."""
    features = extract_audio_features(audio_path)
    return {
        'spectral_centroid': features.spectral_centroid,
        'spectral_rolloff': features.spectral_rolloff,
        'spectral_bandwidth': features.spectral_bandwidth,
        'zero_crossing_rate': features.zero_crossing_rate,
    }

