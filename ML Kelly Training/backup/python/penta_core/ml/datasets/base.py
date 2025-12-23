"""
Base classes and utilities for Kelly ML datasets.

Defines the core data structures:
- DatasetConfig: Dataset configuration
- DatasetManifest: Index of all samples with metadata
- Sample: Individual training sample
- SampleAnnotation: Labels and features for a sample
"""

from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class SampleType(Enum):
    """Type of training sample."""
    MIDI = "midi"
    AUDIO = "audio"
    FEATURES = "features"  # Pre-extracted features
    SYNTHETIC = "synthetic"


class EmotionCategory(Enum):
    """Standard emotion categories."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    PEACEFUL = "peaceful"
    TENSE = "tense"
    MELANCHOLIC = "melancholic"
    ENERGETIC = "energetic"
    NEUTRAL = "neutral"


class GrooveType(Enum):
    """Standard groove/feel types."""
    STRAIGHT = "straight"
    SWING = "swing"
    SHUFFLE = "shuffle"
    LAID_BACK = "laid_back"
    RUSHED = "rushed"
    TRIPLET = "triplet"
    DOTTED = "dotted"


class ArticulationType(Enum):
    """Standard articulation types."""
    LEGATO = "legato"
    STACCATO = "staccato"
    ACCENT = "accent"
    MARCATO = "marcato"
    TENUTO = "tenuto"
    NORMAL = "normal"


class DatasetSplit(Enum):
    """Dataset split type."""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    
    # Identity
    dataset_id: str = "emotion_dataset_v1"
    version: str = "1.0.0"
    description: str = ""
    
    # Target model
    target_model: str = "emotionrecognizer"  # Which model this trains
    
    # Structure
    sample_type: str = "midi"  # midi, audio, features
    sample_rate: int = 44100  # For audio
    max_duration_sec: float = 30.0
    min_duration_sec: float = 1.0
    
    # Splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Size targets
    target_samples_per_category: int = 1000
    max_total_size_gb: float = 100.0  # Part of 3TB limit
    
    # Feature extraction
    extract_mel_spectrogram: bool = True
    extract_mfcc: bool = True
    extract_chroma: bool = True
    n_mels: int = 64
    n_mfcc: int = 20
    hop_length: int = 512
    
    # Labels
    emotion_labels: List[str] = field(default_factory=lambda: [
        "happy", "sad", "angry", "peaceful", "tense", "melancholic", "energetic", "neutral"
    ])
    groove_labels: List[str] = field(default_factory=lambda: [
        "straight", "swing", "shuffle", "laid_back", "rushed"
    ])
    
    # Metadata
    license: str = ""
    source: str = ""
    author: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        """Create from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DatasetConfig":
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: Union[str, Path]):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Sample Annotation
# =============================================================================


@dataclass
class SampleAnnotation:
    """Annotation/labels for a single sample."""
    
    # Emotion (for EmotionRecognizer)
    emotion: str = ""
    valence: float = 0.0  # -1 (negative) to 1 (positive)
    arousal: float = 0.0  # -1 (calm) to 1 (excited)
    dominance: float = 0.0  # -1 (submissive) to 1 (dominant)
    
    # Music theory (for HarmonyPredictor)
    key: str = ""  # e.g., "C major", "A minor"
    mode: str = ""  # major, minor, dorian, etc.
    tempo_bpm: float = 0.0
    time_signature: str = "4/4"
    
    # Chords (for HarmonyPredictor)
    chord_progression: List[str] = field(default_factory=list)  # ["Cmaj", "Am", "F", "G"]
    
    # Groove (for GroovePredictor)
    groove_type: str = "straight"
    swing_ratio: float = 0.5  # 0.5 = straight, 0.67 = swing
    timing_offsets_ms: List[float] = field(default_factory=list)
    
    # Articulation (for DynamicsEngine)
    articulation: str = "normal"
    dynamic_range: str = "mf"  # pp, p, mp, mf, f, ff
    velocity_curve: List[int] = field(default_factory=list)
    
    # Melody (for MelodyTransformer)
    melodic_contour: str = ""  # ascending, descending, arched, etc.
    pitch_range: str = ""  # narrow, medium, wide
    
    # Quality flags
    is_verified: bool = False
    quality_score: float = 0.0  # 0-1, human-assessed quality
    rule_breaks: List[str] = field(default_factory=list)  # Intentional rule violations
    
    # Free-form
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return asdict(self)
    
    @classmethod  
    def from_dict(cls, data: Dict[str, Any]) -> "SampleAnnotation":
        """Create from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Sample
# =============================================================================


@dataclass
class Sample:
    """A single training sample with file reference and annotations."""
    
    # Identity
    sample_id: str = ""
    file_path: str = ""  # Relative to dataset root
    file_type: str = "midi"  # midi, wav, mp3, npz
    
    # Metadata
    sha256: str = ""
    file_size_bytes: int = 0
    duration_sec: float = 0.0
    created_at: str = ""
    modified_at: str = ""
    
    # Dataset split
    split: str = "train"  # train, val, test
    
    # Annotations
    annotations: Optional[SampleAnnotation] = None
    
    # Feature cache (path to pre-extracted features)
    features_path: str = ""
    
    # Source tracking
    source: str = ""  # Where it came from
    is_synthetic: bool = False
    parent_id: str = ""  # If augmented, ID of original
    augmentation_type: str = ""  # transpose, stretch, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        d = asdict(self)
        if self.annotations:
            d['annotations'] = self.annotations.to_dict()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        """Create from dict."""
        annotations = None
        if data.get('annotations'):
            annotations = SampleAnnotation.from_dict(data['annotations'])
        
        return cls(
            sample_id=data.get('sample_id', ''),
            file_path=data.get('file_path', ''),
            file_type=data.get('file_type', 'midi'),
            sha256=data.get('sha256', ''),
            file_size_bytes=data.get('file_size_bytes', 0),
            duration_sec=data.get('duration_sec', 0.0),
            created_at=data.get('created_at', ''),
            modified_at=data.get('modified_at', ''),
            split=data.get('split', 'train'),
            annotations=annotations,
            features_path=data.get('features_path', ''),
            source=data.get('source', ''),
            is_synthetic=data.get('is_synthetic', False),
            parent_id=data.get('parent_id', ''),
            augmentation_type=data.get('augmentation_type', ''),
        )
    
    def compute_hash(self, dataset_root: Path) -> str:
        """Compute SHA256 hash of file."""
        full_path = dataset_root / self.file_path
        if not full_path.exists():
            return ""
        
        sha256 = hashlib.sha256()
        with open(full_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


# =============================================================================
# Dataset Manifest
# =============================================================================


@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    total_duration_hours: float = 0.0
    total_size_gb: float = 0.0
    samples_per_category: Dict[str, int] = field(default_factory=dict)
    avg_duration_sec: float = 0.0
    min_duration_sec: float = 0.0
    max_duration_sec: float = 0.0


@dataclass
class DatasetManifest:
    """Manifest file for a complete dataset."""
    
    # Metadata
    config: DatasetConfig = field(default_factory=DatasetConfig)
    created_at: str = ""
    updated_at: str = ""
    
    # Samples
    samples: List[Sample] = field(default_factory=list)
    
    # Computed statistics
    stats: DatasetStats = field(default_factory=DatasetStats)
    
    def add_sample(self, sample: Sample):
        """Add a sample to the manifest."""
        self.samples.append(sample)
        self._update_stats()
    
    def remove_sample(self, sample_id: str) -> bool:
        """Remove a sample by ID."""
        for i, s in enumerate(self.samples):
            if s.sample_id == sample_id:
                del self.samples[i]
                self._update_stats()
                return True
        return False
    
    def get_sample(self, sample_id: str) -> Optional[Sample]:
        """Get sample by ID."""
        for s in self.samples:
            if s.sample_id == sample_id:
                return s
        return None
    
    def get_samples_by_split(self, split: str) -> List[Sample]:
        """Get all samples in a split."""
        return [s for s in self.samples if s.split == split]
    
    def get_samples_by_category(self, category: str, field: str = "emotion") -> List[Sample]:
        """Get all samples in a category."""
        result = []
        for s in self.samples:
            if s.annotations:
                if getattr(s.annotations, field, "") == category:
                    result.append(s)
        return result
    
    def _update_stats(self):
        """Recompute statistics."""
        self.stats = DatasetStats()
        self.stats.total_samples = len(self.samples)
        
        total_duration = 0.0
        total_size = 0
        durations = []
        categories: Dict[str, int] = {}
        
        for s in self.samples:
            # Split counts
            if s.split == "train":
                self.stats.train_samples += 1
            elif s.split == "val":
                self.stats.val_samples += 1
            elif s.split == "test":
                self.stats.test_samples += 1
            
            # Duration
            total_duration += s.duration_sec
            durations.append(s.duration_sec)
            
            # Size
            total_size += s.file_size_bytes
            
            # Categories
            if s.annotations and s.annotations.emotion:
                cat = s.annotations.emotion
                categories[cat] = categories.get(cat, 0) + 1
        
        self.stats.total_duration_hours = total_duration / 3600
        self.stats.total_size_gb = total_size / (1024**3)
        self.stats.samples_per_category = categories
        
        if durations:
            self.stats.avg_duration_sec = sum(durations) / len(durations)
            self.stats.min_duration_sec = min(durations)
            self.stats.max_duration_sec = max(durations)
        
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "config": self.config.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "samples": [s.to_dict() for s in self.samples],
            "stats": asdict(self.stats),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        """Create from dict."""
        manifest = cls()
        manifest.config = DatasetConfig.from_dict(data.get('config', {}))
        manifest.created_at = data.get('created_at', '')
        manifest.updated_at = data.get('updated_at', '')
        manifest.samples = [Sample.from_dict(s) for s in data.get('samples', [])]
        
        if data.get('stats'):
            manifest.stats = DatasetStats(**data['stats'])
        else:
            manifest._update_stats()
        
        return manifest


# =============================================================================
# Utility Functions
# =============================================================================


def create_dataset_structure(
    root_path: Union[str, Path],
    config: DatasetConfig,
) -> DatasetManifest:
    """
    Create the standard dataset directory structure.
    
    Structure:
        datasets/<dataset_id>/
        ├── manifest.json
        ├── config.json
        ├── raw/
        │   ├── midi/
        │   └── audio/
        ├── processed/
        │   ├── features/
        │   └── augmented/
        ├── annotations/
        │   └── *.json (per-file annotations)
        └── splits/
            ├── train.txt
            ├── val.txt
            └── test.txt
    """
    root = Path(root_path) / config.dataset_id
    
    # Create directories
    dirs = [
        root,
        root / "raw" / "midi",
        root / "raw" / "audio",
        root / "processed" / "features",
        root / "processed" / "augmented",
        root / "annotations",
        root / "splits",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {d}")
    
    # Create manifest
    manifest = DatasetManifest(
        config=config,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    
    # Save config
    config.save(root / "config.json")
    
    # Save manifest
    save_manifest(manifest, root / "manifest.json")
    
    # Create empty split files
    for split in ["train", "val", "test"]:
        (root / "splits" / f"{split}.txt").touch()
    
    logger.info(f"Dataset structure created: {root}")
    return manifest


def load_manifest(path: Union[str, Path]) -> DatasetManifest:
    """Load manifest from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return DatasetManifest.from_dict(data)


def save_manifest(manifest: DatasetManifest, path: Union[str, Path]):
    """Save manifest to JSON file."""
    with open(path, 'w') as f:
        json.dump(manifest.to_dict(), f, indent=2)


def create_sample_from_file(
    file_path: Path,
    dataset_root: Path,
    split: str = "train",
    annotations: Optional[SampleAnnotation] = None,
) -> Sample:
    """Create a Sample object from a file."""
    import uuid
    
    rel_path = file_path.relative_to(dataset_root)
    
    # Determine file type
    suffix = file_path.suffix.lower()
    if suffix in ['.mid', '.midi']:
        file_type = 'midi'
    elif suffix in ['.wav', '.mp3', '.flac', '.ogg']:
        file_type = 'audio'
    elif suffix in ['.npz', '.npy']:
        file_type = 'features'
    else:
        file_type = 'unknown'
    
    # Get file info
    stat = file_path.stat()
    
    sample = Sample(
        sample_id=str(uuid.uuid4())[:8],
        file_path=str(rel_path),
        file_type=file_type,
        file_size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
        modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        split=split,
        annotations=annotations or SampleAnnotation(),
    )
    
    # Compute hash
    sample.sha256 = sample.compute_hash(dataset_root)
    
    return sample


def generate_sample_id(prefix: str = "") -> str:
    """Generate a unique sample ID."""
    import uuid
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:6]
    if prefix:
        return f"{prefix}_{ts}_{uid}"
    return f"{ts}_{uid}"

