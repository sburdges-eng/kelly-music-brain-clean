"""
Kelly ML Datasets - Audio data management and downloading.

All audio data is stored on external SSD at:
    /Volumes/Extreme SSD/kelly-audio-data/

Directory structure:
    /Volumes/Extreme SSD/kelly-audio-data/
    ├── raw/           # Original audio files
    ├── processed/     # Pre-processed features (mel specs, etc.)
    ├── downloads/     # Downloaded datasets
    └── cache/         # Temporary cache

Usage:
    from python.penta_core.ml.datasets import get_audio_data_root, AudioDownloader
    
    # Get the audio data root
    root = get_audio_data_root()
    
    # Download datasets
    downloader = AudioDownloader()
    downloader.download_freesound_pack("emotion_sounds", output_dir=root / "raw" / "emotions")
"""

from pathlib import Path
from typing import Optional
import os

# Primary audio data location - external SSD
AUDIO_DATA_ROOT = Path("/Volumes/Extreme SSD/kelly-audio-data")

# Fallback location if SSD not mounted
FALLBACK_AUDIO_ROOT = Path(__file__).parent.parent.parent.parent.parent / "data" / "audio"


def get_audio_data_root() -> Path:
    """
    Get the root directory for audio data.
    
    Returns /Volumes/Extreme SSD/kelly-audio-data if available,
    otherwise falls back to project data/audio directory.
    """
    if AUDIO_DATA_ROOT.exists():
        return AUDIO_DATA_ROOT
    
    # Create fallback if needed
    FALLBACK_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    return FALLBACK_AUDIO_ROOT


def get_raw_audio_dir() -> Path:
    """Get directory for raw audio files."""
    path = get_audio_data_root() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_processed_dir() -> Path:
    """Get directory for processed features."""
    path = get_audio_data_root() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_downloads_dir() -> Path:
    """Get directory for downloaded datasets."""
    path = get_audio_data_root() / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir() -> Path:
    """Get directory for temporary cache."""
    path = get_audio_data_root() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_audio_directories() -> dict:
    """
    Ensure all audio data directories exist.
    
    Returns dict with all directory paths.
    """
    return {
        "root": get_audio_data_root(),
        "raw": get_raw_audio_dir(),
        "processed": get_processed_dir(),
        "downloads": get_downloads_dir(),
        "cache": get_cache_dir(),
    }


# Import submodules
try:
    from .audio_downloader import (
        AudioDownloader,
        DownloadResult,
        download_audio,
    )
    _HAS_DOWNLOADER = True
except ImportError:
    _HAS_DOWNLOADER = False

try:
    from .audio_features import (
        AudioFeatures,
        AudioFeatureExtractor,
        extract_audio_features,
        extract_emotion_features,
    )
    _HAS_FEATURES = True
except ImportError:
    _HAS_FEATURES = False

try:
    from .thesaurus_loader import (
        ThesaurusLoader,
        EmotionNode,
        ThesaurusLabels,
        load_thesaurus,
        get_node_label_tensor,
        validate_thesaurus_completeness,
    )
    _HAS_THESAURUS = True
except ImportError:
    _HAS_THESAURUS = False


__all__ = [
    "AUDIO_DATA_ROOT",
    "get_audio_data_root",
    "get_raw_audio_dir",
    "get_processed_dir",
    "get_downloads_dir",
    "get_cache_dir",
    "ensure_audio_directories",
]

if _HAS_DOWNLOADER:
    __all__.extend([
        "AudioDownloader",
        "DownloadResult",
        "download_audio",
    ])

if _HAS_FEATURES:
    __all__.extend([
        "AudioFeatures",
        "AudioFeatureExtractor",
        "extract_audio_features",
        "extract_emotion_features",
    ])

if _HAS_THESAURUS:
    __all__.extend([
        "ThesaurusLoader",
        "EmotionNode",
        "ThesaurusLabels",
        "load_thesaurus",
        "get_node_label_tensor",
        "validate_thesaurus_completeness",
    ])
