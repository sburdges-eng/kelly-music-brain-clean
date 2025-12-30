"""Centralized path configuration management for RR CLI"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class FileMapping:
    """Represents a file mapping between repositories"""

    kelly_path: str
    midikompanion_path: str
    shared_identifier: str
    priority: int = 1
    description: str = ""

    def kelly_full_path(self, kelly_root: str) -> str:
        """Get full path for kelly-project file"""
        return os.path.join(kelly_root, self.kelly_path)

    def midikompanion_full_path(self, midi_root: str) -> str:
        """Get full path for MidiKompanion file"""
        return os.path.join(midi_root, self.midikompanion_path)

    def __lt__(self, other: "FileMapping") -> bool:
        """Support sorting by priority (higher first)"""
        return self.priority > other.priority


class PathConfig:
    """Manages centralized path configuration"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize path configuration from YAML file"""
        self.config_path = config_path or self._find_config()
        self.config: Dict[str, Any] = {}
        self.mappings: Dict[str, FileMapping] = {}
        self.kelly_root = ""
        self.midikompanion_root = ""

        if self.config_path:
            self.load_config(self.config_path)

    def _find_config(self) -> Optional[str]:
        """Find paths_config.yaml in the package directory"""
        package_dir = Path(__file__).parent
        config_file = package_dir / "paths_config.yaml"
        if config_file.exists():
            return str(config_file)
        return None

    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        if not yaml:
            raise ImportError("PyYAML is required for path config loading")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f) or {}

        # Load and resolve root paths
        self._load_roots()

        # Load and validate mappings
        self._load_mappings()

    def _load_roots(self) -> None:
        """Load and resolve root paths with environment variable support"""
        roots = self.config.get("roots", {})

        # Resolve kelly root
        kelly_env = roots.get("kelly", "")
        if kelly_env.startswith("${") and kelly_env.endswith("}"):
            # Parse environment variable syntax: ${VAR:default}
            env_part = kelly_env[2:-1]  # Remove ${ and }
            if ":" in env_part:
                var_name, default = env_part.split(":", 1)
                self.kelly_root = os.getenv(var_name, default or ".")
            else:
                self.kelly_root = os.getenv(env_part, ".")
        else:
            self.kelly_root = kelly_env or "."

        # Resolve midikompanion root
        midi_env = roots.get("midikompanion", "")
        if midi_env.startswith("${") and midi_env.endswith("}"):
            env_part = midi_env[2:-1]
            if ":" in env_part:
                var_name, default = env_part.split(":", 1)
                self.midikompanion_root = os.getenv(var_name, default or ".")
            else:
                self.midikompanion_root = os.getenv(env_part, ".")
        else:
            self.midikompanion_root = midi_env or "."

    def _load_mappings(self) -> None:
        """Load and convert mappings to FileMapping objects"""
        mappings_config = self.config.get("mappings", {})

        for key, mapping_data in mappings_config.items():
            if isinstance(mapping_data, dict):
                try:
                    mapping = FileMapping(
                        kelly_path=mapping_data.get("kelly_path", ""),
                        midikompanion_path=mapping_data.get(
                            "midikompanion_path", ""
                        ),
                        shared_identifier=mapping_data.get(
                            "shared_identifier", ""
                        ),
                        priority=mapping_data.get("priority", 1),
                        description=mapping_data.get("description", ""),
                    )
                    self.mappings[key] = mapping
                except (KeyError, TypeError) as e:
                    raise ValueError(
                        f"Invalid mapping configuration for '{key}': {e}"
                    )

    def get_mapping(self, key: str) -> Optional[FileMapping]:
        """Get a specific mapping by key"""
        return self.mappings.get(key)

    def get_all_mappings(self) -> List[FileMapping]:
        """Get all mappings sorted by priority"""
        return sorted(self.mappings.values())

    def validate_paths(self) -> tuple[bool, List[str]]:
        """Validate that all mapped paths exist"""
        errors = []

        # Check root directories
        if not os.path.isdir(self.kelly_root):
            errors.append(f"Kelly root not found: {self.kelly_root}")
        if not os.path.isdir(self.midikompanion_root):
            errors.append(
                f"MidiKompanion root not found: {self.midikompanion_root}"
            )

        # Check file mappings
        for key, mapping in self.mappings.items():
            kelly_path = mapping.kelly_full_path(self.kelly_root)
            midi_path = mapping.midikompanion_full_path(self.midikompanion_root)

            if not os.path.exists(kelly_path):
                errors.append(f"Kelly file not found: {kelly_path}")
            if not os.path.exists(midi_path):
                errors.append(f"MidiKompanion file not found: {midi_path}")

        return len(errors) == 0, errors

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information for display"""
        return {
            "config_file": self.config_path,
            "kelly_root": self.kelly_root,
            "midikompanion_root": self.midikompanion_root,
            "mappings_count": len(self.mappings),
            "mappings": {
                key: {
                    "kelly": mapping.kelly_path,
                    "midi": mapping.midikompanion_path,
                    "priority": mapping.priority,
                }
                for key, mapping in self.mappings.items()
            },
        }

    def to_sync_mappings(self) -> List[Dict[str, str]]:
        """Convert to format compatible with DualRepoSync"""
        result = []
        for mapping in self.get_all_mappings():
            result.append(
                {
                    "kelly_path": mapping.kelly_path,
                    "midikompanion_path": mapping.midikompanion_path,
                    "shared_identifier": mapping.shared_identifier,
                    "priority": mapping.priority,
                }
            )
        return result
