"""
Audio Dataset Module - Mac-optimized lazy-loading audio datasets.

Provides memory-efficient data loading for training on limited RAM systems.
Uses lazy loading with torchaudio/soundfile to avoid loading entire datasets.

Key Features:
- Lazy audio loading (only loads when accessed)
- Pre-computed mel spectrogram caching
- Metadata-driven dataset organization
- Memory-safe for systems with < 16GB RAM
- Supports 16kHz mono audio (optimized for training)

Usage:
    from python.penta_core.ml.audio_dataset import AudioDataset, create_dataloaders
    
    dataset = AudioDataset(
        data_dir="data/raw",
        metadata_file="data/metadata.csv",
        sample_rate=16000,
        max_duration=10.0,
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, batch_size=16, split_ratios=(0.8, 0.1, 0.1)
    )
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Represents a single audio sample with metadata."""
    
    path: Path
    label: Optional[str] = None
    label_id: Optional[int] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    
    # Optional metadata
    emotion: Optional[str] = None
    genre: Optional[str] = None
    tempo: Optional[float] = None
    key: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Cached features (populated on demand)
    _waveform: Optional[np.ndarray] = field(default=None, repr=False)
    _mel_spec: Optional[np.ndarray] = field(default=None, repr=False)


class AudioDataset:
    """
    Memory-efficient audio dataset with lazy loading.
    
    Designed for Mac systems with limited RAM. Audio is loaded on-demand
    and features are computed lazily to minimize memory usage.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000,
        mono: bool = True,
        max_duration: Optional[float] = None,
        min_duration: float = 0.5,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        precompute_features: bool = False,
        feature_cache_dir: Optional[Union[str, Path]] = None,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        """
        Initialize AudioDataset.
        
        Args:
            data_dir: Directory containing audio files
            metadata_file: Optional CSV/JSON file with labels and metadata
            sample_rate: Target sample rate (default 16kHz for efficiency)
            mono: Convert to mono (recommended for training)
            max_duration: Maximum audio duration in seconds (truncate longer)
            min_duration: Minimum audio duration in seconds (skip shorter)
            transform: Optional transform applied to waveform
            target_transform: Optional transform applied to labels
            precompute_features: If True, compute mel spectrograms on init
            feature_cache_dir: Directory to cache computed features
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.mono = mono
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.transform = transform
        self.target_transform = target_transform
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Feature caching
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        if self.feature_cache_dir:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load samples
        self.samples: List[AudioSample] = []
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        
        self._load_samples(metadata_file)
        
        if precompute_features:
            self._precompute_all_features()
        
        logger.info(
            f"AudioDataset initialized: {len(self.samples)} samples, "
            f"{len(self.label_to_id)} classes"
        )
    
    def _load_samples(self, metadata_file: Optional[Union[str, Path]]) -> None:
        """Load samples from directory and optional metadata file."""
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}
        
        if metadata_file:
            self._load_from_metadata(Path(metadata_file))
        else:
            # Auto-discover audio files
            for audio_path in self.data_dir.rglob("*"):
                if audio_path.suffix.lower() in audio_extensions:
                    # Infer label from parent directory
                    label = audio_path.parent.name if audio_path.parent != self.data_dir else None
                    
                    sample = AudioSample(
                        path=audio_path,
                        label=label,
                    )
                    self.samples.append(sample)
        
        # Build label mappings
        labels = set(s.label for s in self.samples if s.label)
        for i, label in enumerate(sorted(labels)):
            self.label_to_id[label] = i
            self.id_to_label[i] = label
        
        # Assign label IDs
        for sample in self.samples:
            if sample.label:
                sample.label_id = self.label_to_id[sample.label]
    
    def _load_from_metadata(self, metadata_file: Path) -> None:
        """Load samples from metadata file (CSV or JSON)."""
        if metadata_file.suffix == ".json":
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            # Handle both root array and root object with "samples" key
            entries = metadata if isinstance(metadata, list) else metadata.get("samples", [])
            
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                    
                path = self.data_dir / entry.get("file", entry.get("path", ""))
                if path.exists():
                    duration = entry.get("duration")
                    
                    # Filter by minimum duration if provided in metadata
                    if duration is not None and duration < self.min_duration:
                        continue
                        
                    sample = AudioSample(
                        path=path,
                        label=entry.get("label"),
                        duration=duration,
                        sample_rate=entry.get("sample_rate"),
                        emotion=entry.get("emotion"),
                        genre=entry.get("genre"),
                        tempo=entry.get("tempo"),
                        key=entry.get("key"),
                        tags=entry.get("tags", []),
                    )
                    self.samples.append(sample)
        
        elif metadata_file.suffix == ".csv":
            import csv
            
            with open(metadata_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    path = self.data_dir / row.get("file", row.get("path", ""))
                    if path.exists():
                        duration = float(row["duration"]) if row.get("duration") else None
                        
                        # Filter by minimum duration if provided in metadata
                        if duration is not None and duration < self.min_duration:
                            continue
                            
                        sample = AudioSample(
                            path=path,
                            label=row.get("label"),
                            duration=duration,
                            emotion=row.get("emotion"),
                            genre=row.get("genre"),
                            tempo=float(row["tempo"]) if row.get("tempo") else None,
                            key=row.get("key"),
                        )
                        self.samples.append(sample)
    
    def _load_audio(self, sample: AudioSample) -> np.ndarray:
        """
        Lazy-load audio from disk.
        
        Uses soundfile for efficiency (no librosa overhead on load).
        Falls back to librosa if soundfile is unavailable.
        """
        sf = None
        try:
            import soundfile as _sf  # type: ignore
            sf = _sf
        except ImportError:
            sf = None

        # Fallback: librosa load if soundfile not available
        if sf is None:
            import librosa

            waveform, sr = librosa.load(
                sample.path,
                sr=self.sample_rate,
                mono=self.mono,
                duration=self.max_duration,
            )
            # librosa.load already returns the desired sample rate
        else:
            # Load with soundfile (faster)
            waveform, sr = sf.read(sample.path, dtype="float32")
        
        # Convert to mono if needed
        if self.mono and len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        
        # Resample if needed
        if sr != self.sample_rate:
            try:
                import resampy

                waveform = resampy.resample(waveform, sr, self.sample_rate)
            except ImportError:
                import librosa

                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
        
        # Truncate if needed
        if self.max_duration:
            max_samples = int(self.max_duration * self.sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
        
        return waveform
    
    def _compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from waveform."""
        try:
            import librosa
            
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
            
        except ImportError:
            # Fallback: simple FFT-based approximation
            logger.warning("librosa not available, using simple spectrogram")
            from scipy import signal
            
            _, _, Sxx = signal.spectrogram(
                waveform,
                fs=self.sample_rate,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
            )
            
            return 10 * np.log10(Sxx + 1e-10)
    
    def _get_cache_filename(self, sample: AudioSample) -> str:
        """
        Generate a unique cache filename for a sample.
        
        Uses hash of full path to prevent collisions when files have
        identical names in different subdirectories (e.g., happy/audio_001.wav
        and sad/audio_001.wav).
        """
        import hashlib
        # Use full path to ensure uniqueness
        path_hash = hashlib.md5(str(sample.path).encode()).hexdigest()[:12]
        return f"{sample.path.stem}_{path_hash}.npy"
    
    def _get_cached_features(self, sample: AudioSample) -> Optional[np.ndarray]:
        """Load cached features if available."""
        if not self.feature_cache_dir:
            return None
        
        cache_path = self.feature_cache_dir / self._get_cache_filename(sample)
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def _cache_features(self, sample: AudioSample, features: np.ndarray) -> None:
        """Cache computed features to disk."""
        if not self.feature_cache_dir:
            return
        
        cache_path = self.feature_cache_dir / self._get_cache_filename(sample)
        np.save(cache_path, features)
    
    def _precompute_all_features(self) -> None:
        """Precompute mel spectrograms for all samples."""
        from tqdm import tqdm
        
        logger.info("Precomputing mel spectrograms...")
        
        for sample in tqdm(self.samples, desc="Computing features"):
            if self._get_cached_features(sample) is None:
                waveform = self._load_audio(sample)
                mel_spec = self._compute_mel_spectrogram(waveform)
                self._cache_features(sample, mel_spec)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[int]]:
        """
        Get a sample by index.
        
        Returns:
            Tuple of (features, label_id)
        """
        sample = self.samples[idx]
        
        # Try to get cached features
        features = self._get_cached_features(sample)
        
        if features is None:
            # Load audio and compute features
            waveform = self._load_audio(sample)
            features = self._compute_mel_spectrogram(waveform)
            
            # Cache for future use
            self._cache_features(sample, features)
        
        # Apply transforms
        if self.transform:
            features = self.transform(features)
        
        label = sample.label_id
        if self.target_transform and label is not None:
            label = self.target_transform(label)
        
        return features, label
    
    def get_sample_info(self, idx: int) -> AudioSample:
        """Get full sample information."""
        return self.samples[idx]
    
    @property
    def num_classes(self) -> int:
        """Number of unique classes."""
        return len(self.label_to_id)
    
    @property
    def class_names(self) -> List[str]:
        """List of class names in order."""
        return [self.id_to_label[i] for i in range(len(self.id_to_label))]


class AudioDatasetTorch:
    """
    PyTorch-compatible wrapper for AudioDataset.
    
    Provides proper tensor conversion and batching support.
    """
    
    def __init__(self, audio_dataset: AudioDataset, return_waveform: bool = False):
        """
        Initialize PyTorch wrapper.
        
        Args:
            audio_dataset: Base AudioDataset
            return_waveform: If True, return raw waveform instead of mel spec
        """
        self.dataset = audio_dataset
        self.return_waveform = return_waveform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        """Get item as PyTorch tensors."""
        import torch
        
        if self.return_waveform:
            sample = self.dataset.get_sample_info(idx)
            waveform = self.dataset._load_audio(sample)
            features = waveform
            label = sample.label_id
        else:
            features, label = self.dataset[idx]
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float()
        
        # Add channel dimension if needed (for CNN input)
        if features_tensor.dim() == 2:
            features_tensor = features_tensor.unsqueeze(0)
        elif features_tensor.dim() == 1:
            # Waveform: (T,) -> (1, T)
            features_tensor = features_tensor.unsqueeze(0)
        
        if label is not None:
            label_tensor = torch.tensor(label, dtype=torch.long)
            return features_tensor, label_tensor
        
        return features_tensor, torch.tensor(-1, dtype=torch.long)


def create_dataloaders(
    dataset: AudioDataset,
    batch_size: int = 16,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_file: Optional[Union[str, Path]] = None,
    num_workers: int = 0,  # 0 for Mac compatibility
    seed: int = 42,
) -> Tuple[Any, Any, Any]:
    """
    Create train/val/test dataloaders from AudioDataset.
    
    Args:
        dataset: AudioDataset instance
        batch_size: Batch size (keep small for Mac: 8-16)
        split_ratios: (train, val, test) ratios
        split_file: Optional file with predefined splits
        num_workers: DataLoader workers (0 recommended for Mac)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    
    torch_dataset = AudioDatasetTorch(dataset)
    n_samples = len(torch_dataset)
    
    if split_file and Path(split_file).exists():
        # Load predefined splits
        with open(split_file) as f:
            splits = json.load(f)
        
        train_indices = splits.get("train", [])
        val_indices = splits.get("val", [])
        test_indices = splits.get("test", [])
    else:
        # Random split
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        
        train_size = int(n_samples * split_ratios[0])
        val_size = int(n_samples * split_ratios[1])
        
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:train_size + val_size].tolist()
        test_indices = indices[train_size + val_size:].tolist()
    
    train_dataset = Subset(torch_dataset, train_indices)
    val_dataset = Subset(torch_dataset, val_indices)
    test_dataset = Subset(torch_dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable for Mac compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    logger.info(
        f"Created dataloaders: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    
    return train_loader, val_loader, test_loader


def create_metadata_template(
    data_dir: Union[str, Path],
    output_file: Union[str, Path] = "metadata.csv",
) -> Path:
    """
    Create a metadata template CSV from audio files in a directory.
    
    Args:
        data_dir: Directory containing audio files
        output_file: Output CSV file path
    
    Returns:
        Path to created metadata file
    """
    import csv
    
    data_dir = Path(data_dir)
    output_path = data_dir / output_file
    
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "label", "duration", "emotion", "genre", "tempo", "key", "tags"
        ])
        
        for audio_path in sorted(data_dir.rglob("*")):
            if audio_path.suffix.lower() in audio_extensions:
                rel_path = audio_path.relative_to(data_dir)
                label = audio_path.parent.name if audio_path.parent != data_dir else ""
                
                writer.writerow([
                    str(rel_path), label, "", "", "", "", "", ""
                ])
    
    logger.info(f"Created metadata template: {output_path}")
    return output_path

