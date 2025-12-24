"""
Kelly Storage Configuration

Single source of truth for audio data storage paths.
Configuration is done exclusively via environment variables.

Environment Variables:
    KELLY_AUDIO_DATA_ROOT: Full path to audio data directory (required for production)
    KELLY_SSD_PATH: External SSD mount point (kelly-audio-data created inside)

Priority:
    1. KELLY_AUDIO_DATA_ROOT (explicit, takes precedence)
    2. KELLY_SSD_PATH/kelly-audio-data
    3. Platform default SSD paths (auto-detected)
    4. Safe fallback: ~/.kelly/audio-data (always writable)

Usage:
    from configs.storage import get_storage_config, reset_storage_config

    config = get_storage_config()
    audio_root = config.audio_data_root

    # After changing env vars, reset to pick up changes
    reset_storage_config()
"""

from __future__ import annotations

import os
import platform
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Platform-specific default SSD mount points to auto-detect
_PLATFORM_SSD_PATHS = {
    "Darwin": [
        "/Volumes/Extreme SSD",
        "/Volumes/External SSD",
        "/Volumes/Audio Data",
    ],
    "Linux": [
        "/mnt/ssd",
        "/media/ssd",
        "/mnt/external",
    ],
    "Windows": [
        "D:\\",
        "E:\\",
        "F:\\",
    ],
}

# Standard subdirectories to create
_SUBDIRS = [
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
]


def _is_path_writable(path: Path) -> bool:
    """Check if a path is writable (or can be created)."""
    try:
        if path.exists():
            # Check if we can write to it
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        else:
            # Check if parent exists and is writable
            parent = path.parent
            if parent.exists():
                test_file = parent / ".write_test"
                test_file.touch()
                test_file.unlink()
                return True
            # Try to create the path
            path.mkdir(parents=True, exist_ok=True)
            path.rmdir()
            return True
    except (PermissionError, OSError):
        return False


def _get_safe_fallback() -> Path:
    """Get a safe, always-writable fallback location."""
    # Try user home directory first
    home_path = Path.home() / ".kelly" / "audio-data"
    if _is_path_writable(home_path.parent):
        return home_path

    # Last resort: temp directory
    return Path(tempfile.gettempdir()) / "kelly-audio-data"


def _ensure_dir_exists(path: Path) -> Path:
    """Ensure a directory exists (with parents) and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class StorageConfig:
    """
    Storage configuration resolved from environment variables.

    Attributes:
        audio_data_root: Root directory for all audio data
        source: How the path was determined ('env', 'ssd_env', 'auto_ssd', 'fallback')
        configured_ssd_path: The SSD path if explicitly configured or auto-detected
    """

    audio_data_root: Path = field(default_factory=Path)
    source: str = field(default="")
    configured_ssd_path: Optional[Path] = field(default=None)
    subdirs: List[str] = field(default_factory=lambda: _SUBDIRS.copy())

    def __post_init__(self):
        """Resolve paths from environment."""
        if not self.audio_data_root or not self.source:
            self._resolve_paths()

    def _resolve_paths(self) -> None:
        """Resolve storage paths from environment variables."""
        # 1. Check KELLY_AUDIO_DATA_ROOT (explicit full path)
        env_root = os.environ.get("KELLY_AUDIO_DATA_ROOT")
        if env_root:
            path = Path(env_root)
            if _is_path_writable(path):
                self.audio_data_root = path
                self.source = "env"
                self.configured_ssd_path = path.parent
                return
            # Env var set but not writable - this is an error condition
            # Fall through to fallback but mark it

        # 2. Check KELLY_SSD_PATH (SSD mount point)
        env_ssd = os.environ.get("KELLY_SSD_PATH")
        if env_ssd:
            ssd_path = Path(env_ssd)
            if ssd_path.exists() and ssd_path.is_dir():
                audio_root = ssd_path / "kelly-audio-data"
                if _is_path_writable(audio_root):
                    self.audio_data_root = audio_root
                    self.source = "ssd_env"
                    self.configured_ssd_path = ssd_path
                    return

        # 3. Auto-detect platform SSD paths
        system = platform.system()
        for ssd_path_str in _PLATFORM_SSD_PATHS.get(system, []):
            ssd_path = Path(ssd_path_str)
            if ssd_path.exists() and ssd_path.is_dir():
                audio_root = ssd_path / "kelly-audio-data"
                if _is_path_writable(audio_root):
                    self.audio_data_root = audio_root
                    self.source = "auto_ssd"
                    self.configured_ssd_path = ssd_path
                    return

        # 4. Safe fallback (always writable)
        self.audio_data_root = _get_safe_fallback()
        self.source = "fallback"
        self.configured_ssd_path = None

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
    def is_configured(self) -> bool:
        """True if path was explicitly configured via environment variable."""
        return self.source in ("env", "ssd_env")

    @property
    def is_auto_detected(self) -> bool:
        """True if path was auto-detected from platform defaults."""
        return self.source == "auto_ssd"

    @property
    def is_fallback(self) -> bool:
        """True if using safe fallback location (no SSD found/configured)."""
        return self.source == "fallback"

    @property
    def storage_type(self) -> str:
        """Human-readable description of storage configuration."""
        if self.source == "env":
            return f"Configured ({self.audio_data_root})"
        elif self.source == "ssd_env":
            return f"SSD via env ({self.configured_ssd_path})"
        elif self.source == "auto_ssd":
            return f"Auto-detected SSD ({self.configured_ssd_path})"
        else:
            return f"Fallback ({self.audio_data_root})"

    def ensure_directories(self) -> dict:
        """
        Create all required directories.

        Returns:
            Dict mapping directory names to paths
        """
        paths = {"root": self.audio_data_root}
        self.audio_data_root.mkdir(parents=True, exist_ok=True)

        for subdir in self.subdirs:
            path = self.audio_data_root / subdir
            path.mkdir(parents=True, exist_ok=True)
            paths[subdir.replace("/", "_")] = path

        return paths

    def get_data_path(self, dataset_name: str, subdirectory: str = "raw") -> Path:
        """Get path for a specific dataset."""
        return self.audio_data_root / subdirectory / dataset_name

    def get_manifest_path(self, manifest_name: str) -> Path:
        """Get path for a data manifest file."""
        if not manifest_name.endswith(".jsonl"):
            manifest_name = f"{manifest_name}.jsonl"
        return self.manifests_dir / manifest_name

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "audio_data_root": str(self.audio_data_root),
            "source": self.source,
            "configured_ssd_path": str(self.configured_ssd_path) if self.configured_ssd_path else None,
            "storage_type": self.storage_type,
            "is_configured": self.is_configured,
            "is_fallback": self.is_fallback,
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
            messages.append("ERROR: No audio data root resolved")
            return False, messages

        if self.is_fallback:
            messages.append("WARNING: Using fallback storage location")
            messages.append(f"  Path: {self.audio_data_root}")
            messages.append("  Set KELLY_AUDIO_DATA_ROOT for production use")

        if not self.audio_data_root.exists():
            messages.append(f"INFO: Audio data root does not exist yet: {self.audio_data_root}")
            messages.append("  Run ensure_directories() to create it")

        # Check disk space
        try:
            import shutil
            check_path = self.audio_data_root if self.audio_data_root.exists() else self.audio_data_root.parent
            if check_path.exists():
                total, used, free = shutil.disk_usage(check_path)
                free_gb = free / (1024 ** 3)
                if free_gb < 10:
                    messages.append(f"WARNING: Low disk space: {free_gb:.1f} GB free")
        except Exception:
            pass

        return is_valid, messages

    def __str__(self) -> str:
        return f"StorageConfig({self.storage_type})"

    def __repr__(self) -> str:
        return f"StorageConfig(audio_data_root={self.audio_data_root!r}, source={self.source!r})"


# Singleton instance
_storage_config: Optional[StorageConfig] = None


def get_storage_config() -> StorageConfig:
    """
    Get the global storage configuration instance.

    Creates a new instance on first call, then returns cached instance.
    Use reset_storage_config() to force re-resolution after env changes.
    """
    global _storage_config
    if _storage_config is None:
        _storage_config = StorageConfig()
    return _storage_config


def reset_storage_config() -> StorageConfig:
    """
    Reset and re-resolve storage configuration.

    Call this after changing environment variables to pick up new values.

    Returns:
        The new StorageConfig instance
    """
    global _storage_config
    _storage_config = StorageConfig()
    return _storage_config


def get_audio_data_root() -> Path:
    """Get the audio data root directory."""
    return get_storage_config().audio_data_root


def get_raw_audio_dir() -> Path:
    """Get directory for raw audio files and ensure it exists."""
    return _ensure_dir_exists(get_storage_config().raw_audio_dir)


def get_processed_dir() -> Path:
    """Get directory for processed features and ensure it exists."""
    return _ensure_dir_exists(get_storage_config().processed_dir)


def get_downloads_dir() -> Path:
    """Get directory for downloaded datasets and ensure it exists."""
    return _ensure_dir_exists(get_storage_config().downloads_dir)


def get_cache_dir() -> Path:
    """Get directory for temporary cache and ensure it exists."""
    return _ensure_dir_exists(get_storage_config().cache_dir)


def get_manifests_dir() -> Path:
    """Get directory for data manifests and ensure it exists."""
    return _ensure_dir_exists(get_storage_config().manifests_dir)


def ensure_storage_directories() -> dict:
    """Create all storage directories and return paths dict."""
    return get_storage_config().ensure_directories()


# Note: AUDIO_DATA_ROOT is evaluated at import time.
# Use get_audio_data_root() for dynamic access after env changes.
AUDIO_DATA_ROOT = get_audio_data_root()


if __name__ == "__main__":
    import sys

    config = get_storage_config()
    print("Storage Configuration")
    print("=" * 50)
    print(f"  Audio Root:  {config.audio_data_root}")
    print(f"  Source:      {config.source}")
    print(f"  SSD Path:    {config.configured_ssd_path}")
    print(f"  Type:        {config.storage_type}")
    print()
    print("Status:")
    print(f"  Configured:  {config.is_configured}")
    print(f"  Auto-detect: {config.is_auto_detected}")
    print(f"  Fallback:    {config.is_fallback}")
    print()

    is_valid, messages = config.validate()
    if messages:
        print("Validation:")
        for msg in messages:
            print(f"  {msg}")
    else:
        print("Validation: OK")

    sys.exit(0 if is_valid else 1)
