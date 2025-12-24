"""
Kelly Storage Configuration

Centralized configuration for external storage paths.
Supports environment variables for flexible deployment.

Environment Variables:
    KELLY_AUDIO_DATA_ROOT: Primary audio data directory
    KELLY_SSD_PATH: External SSD mount point (default: /Volumes/Extreme SSD)

Usage:
    from configs.storage import StorageConfig

    config = StorageConfig()
    audio_root = config.audio_data_root
    raw_dir = config.raw_audio_dir
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Default paths based on platform
DEFAULT_SSD_PATHS = {
    "Darwin": [
        "/Volumes/Extreme SSD",
        "/Volumes/External SSD",
        "/Volumes/Audio Data",
    ],
    "Linux": [
        "/mnt/ssd",
        "/media/ssd",
        "/mnt/external",
        os.path.expanduser("~/audio-data"),
    ],
    "Windows": [
        "D:\\",
        "E:\\",
        "F:\\",
    ],
}


@dataclass
class StorageConfig:
    """
    Storage configuration with environment variable support.

    Priority:
    1. KELLY_AUDIO_DATA_ROOT environment variable
    2. KELLY_SSD_PATH environment variable + /kelly-audio-data
    3. Default SSD paths (platform-specific)
    4. Project-local fallback: data/audio

    Attributes:
        ssd_path: Path to the external SSD mount point
        audio_data_root: Root directory for all audio data
        subdirs: List of subdirectories to create
    """

    ssd_path: Optional[Path] = field(default=None)
    audio_data_root: Optional[Path] = field(default=None)
    subdirs: List[str] = field(default_factory=lambda: [
        "raw",
        "raw/emotions",
        "raw/melodies",
        "raw/chord_progressions",
        "raw/grooves",
        "raw/expression",
        "raw/instruments",
        "raw/emotion_thesaurus",
        "processed",
        "processed/mel_spectrograms",
        "processed/embeddings",
        "downloads",
        "cache",
        "manifests",
    ])

    def __post_init__(self):
        """Initialize paths from environment or defaults."""
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """Resolve storage paths from environment or defaults."""
        # Check for explicit audio data root
        env_audio_root = os.environ.get("KELLY_AUDIO_DATA_ROOT")
        if env_audio_root:
            self.audio_data_root = Path(env_audio_root)
            self.ssd_path = self.audio_data_root.parent
            return

        # Check for SSD path environment variable
        env_ssd_path = os.environ.get("KELLY_SSD_PATH")
        if env_ssd_path:
            self.ssd_path = Path(env_ssd_path)
            self.audio_data_root = self.ssd_path / "kelly-audio-data"
            return

        # Try default SSD paths for current platform
        system = platform.system()
        default_paths = DEFAULT_SSD_PATHS.get(system, [])

        for ssd_path in default_paths:
            path = Path(ssd_path)
            if path.exists() and path.is_dir():
                self.ssd_path = path
                self.audio_data_root = path / "kelly-audio-data"
                return

        # Fallback to project-local directory
        project_root = Path(__file__).parent.parent
        self.ssd_path = None
        self.audio_data_root = project_root / "data" / "audio"

    @property
    def raw_audio_dir(self) -> Path:
        """Get directory for raw audio files."""
        return self.audio_data_root / "raw"

    @property
    def processed_dir(self) -> Path:
        """Get directory for processed features."""
        return self.audio_data_root / "processed"

    @property
    def downloads_dir(self) -> Path:
        """Get directory for downloaded datasets."""
        return self.audio_data_root / "downloads"

    @property
    def cache_dir(self) -> Path:
        """Get directory for temporary cache."""
        return self.audio_data_root / "cache"

    @property
    def manifests_dir(self) -> Path:
        """Get directory for data manifests."""
        return self.audio_data_root / "manifests"

    @property
    def is_external_ssd(self) -> bool:
        """Check if using external SSD storage."""
        return self.ssd_path is not None and self.ssd_path.exists()

    @property
    def storage_type(self) -> str:
        """Get human-readable storage type description."""
        if self.is_external_ssd:
            return f"External SSD ({self.ssd_path})"
        return "Local (project directory)"

    def ensure_directories(self) -> dict:
        """
        Create all required directories.

        Returns:
            Dict mapping directory names to paths
        """
        paths = {}

        # Create root
        self.audio_data_root.mkdir(parents=True, exist_ok=True)
        paths["root"] = self.audio_data_root

        # Create subdirectories
        for subdir in self.subdirs:
            path = self.audio_data_root / subdir
            path.mkdir(parents=True, exist_ok=True)
            paths[subdir.replace("/", "_")] = path

        return paths

    def get_data_path(self, dataset_name: str, subdirectory: str = "raw") -> Path:
        """
        Get path for a specific dataset.

        Args:
            dataset_name: Name of the dataset (e.g., "emotions", "melodies")
            subdirectory: Subdirectory type ("raw", "processed", etc.)

        Returns:
            Full path to the dataset directory
        """
        return self.audio_data_root / subdirectory / dataset_name

    def get_manifest_path(self, manifest_name: str) -> Path:
        """
        Get path for a data manifest file.

        Args:
            manifest_name: Name of the manifest (without extension)

        Returns:
            Full path to the manifest file
        """
        if not manifest_name.endswith(".jsonl"):
            manifest_name = f"{manifest_name}.jsonl"
        return self.manifests_dir / manifest_name

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "ssd_path": str(self.ssd_path) if self.ssd_path else None,
            "audio_data_root": str(self.audio_data_root),
            "storage_type": self.storage_type,
            "is_external_ssd": self.is_external_ssd,
            "subdirs": self.subdirs,
        }

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate storage configuration.

        Returns:
            Tuple of (is_valid, list of warning/error messages)
        """
        messages = []
        is_valid = True

        if not self.audio_data_root:
            messages.append("ERROR: No audio data root configured")
            is_valid = False
            return is_valid, messages

        if not self.audio_data_root.exists():
            messages.append(f"WARNING: Audio data root does not exist: {self.audio_data_root}")
            messages.append("  Run config.ensure_directories() to create it")

        if not self.is_external_ssd:
            messages.append("WARNING: Using local storage, not external SSD")
            messages.append("  Set KELLY_AUDIO_DATA_ROOT or mount external SSD")

        # Check disk space (basic check)
        try:
            import shutil
            if self.audio_data_root.exists():
                total, used, free = shutil.disk_usage(self.audio_data_root)
                free_gb = free / (1024 ** 3)
                if free_gb < 10:
                    messages.append(f"WARNING: Low disk space: {free_gb:.1f} GB free")
        except Exception:
            pass

        return is_valid, messages

    def __str__(self) -> str:
        return f"StorageConfig(root={self.audio_data_root}, type={self.storage_type})"

    def __repr__(self) -> str:
        return self.__str__()


# Singleton instance for easy access
_storage_config: Optional[StorageConfig] = None


def get_storage_config() -> StorageConfig:
    """
    Get the global storage configuration instance.

    Creates a new instance on first call, then returns cached instance.
    """
    global _storage_config
    if _storage_config is None:
        _storage_config = StorageConfig()
    return _storage_config


def get_audio_data_root() -> Path:
    """Get the audio data root directory."""
    return get_storage_config().audio_data_root


def get_raw_audio_dir() -> Path:
    """Get directory for raw audio files."""
    return get_storage_config().raw_audio_dir


def get_processed_dir() -> Path:
    """Get directory for processed features."""
    return get_storage_config().processed_dir


def get_downloads_dir() -> Path:
    """Get directory for downloaded datasets."""
    return get_storage_config().downloads_dir


def get_cache_dir() -> Path:
    """Get directory for temporary cache."""
    return get_storage_config().cache_dir


def get_manifests_dir() -> Path:
    """Get directory for data manifests."""
    return get_storage_config().manifests_dir


def ensure_storage_directories() -> dict:
    """Create all storage directories and return paths dict."""
    return get_storage_config().ensure_directories()


# Legacy compatibility
AUDIO_DATA_ROOT = get_audio_data_root()


if __name__ == "__main__":
    # Test the configuration
    config = StorageConfig()
    print(f"Storage Configuration:")
    print(f"  SSD Path: {config.ssd_path}")
    print(f"  Audio Root: {config.audio_data_root}")
    print(f"  Storage Type: {config.storage_type}")
    print(f"  Is External SSD: {config.is_external_ssd}")
    print()

    is_valid, messages = config.validate()
    print(f"Validation: {'VALID' if is_valid else 'INVALID'}")
    for msg in messages:
        print(f"  {msg}")
