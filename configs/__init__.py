"""
Kelly Configuration Module

Centralized configuration for the Kelly project.

Storage Configuration:
    Set KELLY_AUDIO_DATA_ROOT or KELLY_SSD_PATH environment variables
    to configure external SSD storage. See .env.example for details.

Usage:
    from configs import get_audio_data_root, load_config

    # Get storage paths
    audio_root = get_audio_data_root()

    # Load training config with env var expansion
    config = load_config("configs/emotion_recognizer.yaml")
"""

from .storage import (
    StorageConfig,
    get_storage_config,
    reset_storage_config,
    get_audio_data_root,
    get_raw_audio_dir,
    get_processed_dir,
    get_downloads_dir,
    get_cache_dir,
    get_manifests_dir,
    ensure_storage_directories,
    AUDIO_DATA_ROOT,
)

from .config_loader import (
    load_config,
    load_yaml_with_expansion,
    expand_env_vars,
    get_data_path,
    resolve_manifest_path,
)

__all__ = [
    # Storage
    "StorageConfig",
    "get_storage_config",
    "reset_storage_config",
    "get_audio_data_root",
    "get_raw_audio_dir",
    "get_processed_dir",
    "get_downloads_dir",
    "get_cache_dir",
    "get_manifests_dir",
    "ensure_storage_directories",
    "AUDIO_DATA_ROOT",
    # Config loading
    "load_config",
    "load_yaml_with_expansion",
    "expand_env_vars",
    "get_data_path",
    "resolve_manifest_path",
]

