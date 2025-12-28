"""
Kelly ML Datasets - Audio data management and downloading.

Audio data storage is configured via environment variables:
    KELLY_AUDIO_DATA_ROOT: Full path to audio data directory
    KELLY_SSD_PATH: Path to external SSD mount point

Default directory structure:
    <audio_data_root>/
    ├── raw/           # Original audio files
    ├── processed/     # Pre-processed features (mel specs, etc.)
    ├── downloads/     # Downloaded datasets
    ├── cache/         # Temporary cache
    └── manifests/     # Data manifest files

Usage:
    from python.penta_core.ml.datasets import get_audio_data_root, AudioDownloader

    # Get the audio data root
    root = get_audio_data_root()

    # Download datasets
    downloader = AudioDownloader()
    downloader.download_freesound_pack("emotion_sounds", output_dir=root / "raw" / "emotions")

See configs/storage.py for full configuration options.
"""

from pathlib import Path
from typing import Optional
import os
import sys

# Try to import centralized storage config
try:
    # Add configs to path if needed
    _configs_path = Path(__file__).parent.parent.parent.parent.parent / "configs"
    if _configs_path.exists() and str(_configs_path) not in sys.path:
        sys.path.insert(0, str(_configs_path.parent))

    from configs.storage import (
        StorageConfig,
        get_storage_config,
        reset_storage_config,
        get_audio_data_root as _get_audio_root,
        get_raw_audio_dir as _get_raw,
        get_processed_dir as _get_processed,
        get_downloads_dir as _get_downloads,
        get_cache_dir as _get_cache,
        get_manifests_dir as _get_manifests,
        ensure_storage_directories as _ensure_dirs,
    )
    _HAS_STORAGE_CONFIG = True
except ImportError:
    _HAS_STORAGE_CONFIG = False

# Fallback if storage config not available
if not _HAS_STORAGE_CONFIG:
    import platform

    # Default paths based on platform
    _DEFAULT_SSD_PATHS = {
        "Darwin": "/Volumes/Extreme SSD",
        "Linux": "/mnt/ssd",
        "Windows": "D:\\",
    }

    def _find_audio_root() -> Path:
        """Find audio data root from environment or defaults."""
        # Check environment variable first
        env_root = os.environ.get("KELLY_AUDIO_DATA_ROOT")
        if env_root:
            return Path(env_root)

        # Check SSD path environment variable
        env_ssd = os.environ.get("KELLY_SSD_PATH")
        if env_ssd:
            return Path(env_ssd) / "kelly-audio-data"

        # Try platform default
        system = platform.system()
        default_ssd = _DEFAULT_SSD_PATHS.get(system)
        if default_ssd and Path(default_ssd).exists():
            return Path(default_ssd) / "kelly-audio-data"

        # Fallback to project local
        return Path(__file__).parent.parent.parent.parent.parent / "data" / "audio"

# Legacy constant for backward compatibility
AUDIO_DATA_ROOT = _get_audio_root() if _HAS_STORAGE_CONFIG else _find_audio_root()


def get_audio_data_root() -> Path:
    """
    Get the root directory for audio data.

    Uses centralized storage config if available, otherwise
    falls back to environment variables or platform defaults.

    Configure via:
        - KELLY_AUDIO_DATA_ROOT environment variable
        - KELLY_SSD_PATH environment variable
        - configs/storage.py settings
    """
    if _HAS_STORAGE_CONFIG:
        return _get_audio_root()
    return _find_audio_root()


def get_raw_audio_dir() -> Path:
    """Get directory for raw audio files."""
    if _HAS_STORAGE_CONFIG:
        return _get_raw()
    path = get_audio_data_root() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_processed_dir() -> Path:
    """Get directory for processed features."""
    if _HAS_STORAGE_CONFIG:
        return _get_processed()
    path = get_audio_data_root() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_downloads_dir() -> Path:
    """Get directory for downloaded datasets."""
    if _HAS_STORAGE_CONFIG:
        return _get_downloads()
    path = get_audio_data_root() / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir() -> Path:
    """Get directory for temporary cache."""
    if _HAS_STORAGE_CONFIG:
        return _get_cache()
    path = get_audio_data_root() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_manifests_dir() -> Path:
    """Get directory for data manifest files."""
    if _HAS_STORAGE_CONFIG:
        return _get_manifests()
    path = get_audio_data_root() / "manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_audio_directories() -> dict:
    """
    Ensure all audio data directories exist.

    Returns dict with all directory paths.
    """
    if _HAS_STORAGE_CONFIG:
        return _ensure_dirs()
    return {
        "root": get_audio_data_root(),
        "raw": get_raw_audio_dir(),
        "processed": get_processed_dir(),
        "downloads": get_downloads_dir(),
        "cache": get_cache_dir(),
        "manifests": get_manifests_dir(),
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
    "get_manifests_dir",
    "ensure_audio_directories",
    "DatasetCompressor",
    "DatasetArchiveManifest",
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

from .compressor import (  # noqa: E402
    DatasetCompressor,
    DatasetArchiveManifest,
)

# Unified dataset generator
try:
    from .unified_generator import (
        UnifiedDatasetGenerator,
        UnifiedDataset,
        DataSample,
        DatasetType,
        StaticDataLoader,
        DownloadedSamplesLoader,
        OSCPatternLoader,
    )
    _HAS_UNIFIED = True
except ImportError:
    _HAS_UNIFIED = False

# MIDI generator
try:
    from .midi_generator import (
        MIDIGenerator,
        MIDISequence,
        MIDINote,
        MIDIDatasetGenerator,
        ChordParser,
        EmotionProfile,
        EmotionModulator,
        GenerationConfig,
    )
    _HAS_MIDI_GEN = True
except ImportError:
    _HAS_MIDI_GEN = False

if _HAS_UNIFIED:
    __all__.extend([
        "UnifiedDatasetGenerator",
        "UnifiedDataset",
        "DataSample",
        "DatasetType",
        "StaticDataLoader",
        "DownloadedSamplesLoader",
        "OSCPatternLoader",
    ])

if _HAS_MIDI_GEN:
    __all__.extend([
        "MIDIGenerator",
        "MIDISequence",
        "MIDINote",
        "MIDIDatasetGenerator",
        "ChordParser",
        "EmotionProfile",
        "EmotionModulator",
        "GenerationConfig",
    ])
