#!/usr/bin/env python3
"""
Model Versioning System
=======================
Tracks model versions, compatibility, and provides migration utilities.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime


class ModelVersion:
    """Model version information."""

    def __init__(self, major: int = 1, minor: int = 0, patch: int = 0):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModelVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other) -> bool:
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def is_compatible_with(self, other) -> bool:
        """Check if versions are compatible (same major version)."""
        if not isinstance(other, ModelVersion):
            return False
        return self.major == other.major

    @classmethod
    def from_string(cls, version_str: str) -> 'ModelVersion':
        """Parse version string like '2.1.3'."""
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))


def get_model_version(model_path: Path) -> Optional[ModelVersion]:
    """Extract version from model JSON file."""
    try:
        with open(model_path, 'r') as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        export_version = metadata.get("export_version", "1.0")

        # Handle both string and float versions
        if isinstance(export_version, str):
            return ModelVersion.from_string(export_version)
        elif isinstance(export_version, (int, float)):
            # Convert 1.0 -> "1.0.0", 2.0 -> "2.0.0"
            version_str = f"{int(export_version)}.0.0"
            return ModelVersion.from_string(version_str)
        else:
            return None
    except Exception as e:
        print(f"Error reading model version: {e}")
        return None


def add_version_metadata(model_path: Path, version: Optional[ModelVersion] = None) -> bool:
    """Add or update version metadata in model JSON file."""
    if version is None:
        version = ModelVersion(2, 0, 0)  # Current export version

    try:
        with open(model_path, 'r') as f:
            data = json.load(f)

        if "metadata" not in data:
            data["metadata"] = {}

        metadata = data["metadata"]
        metadata["export_version"] = str(version)
        metadata["export_date"] = datetime.now().isoformat()
        metadata["format_version"] = "2.0"  # RTNeural JSON format version

        with open(model_path, 'w') as f:
            json.dump(data, f, indent=2)

        return True
    except Exception as e:
        print(f"Error updating version metadata: {e}")
        return False


def check_model_compatibility(model_path: Path, required_version: Optional[ModelVersion] = None) -> Tuple[bool, str]:
    """
    Check if model is compatible with current system.
    Returns (is_compatible, message).
    """
    model_version = get_model_version(model_path)

    if model_version is None:
        return False, "Could not determine model version"

    if required_version is None:
        required_version = ModelVersion(2, 0, 0)  # Current required version

    if model_version.is_compatible_with(required_version):
        return True, f"Model version {model_version} is compatible"
    else:
        return False, f"Model version {model_version} is incompatible (requires {required_version.major}.x.x)"


def migrate_model_v1_to_v2(model_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Migrate model from version 1.0 format to 2.0 format.
    Version 1.0 had different JSON structure.
    """
    try:
        with open(model_path, 'r') as f:
            data = json.load(f)

        # Check if already v2
        metadata = data.get("metadata", {})
        export_version = metadata.get("export_version", "1.0")

        if isinstance(export_version, str) and export_version.startswith("2."):
            print("Model is already version 2.0")
            return True

        # V1 format had: model_name, model_type, input_size, output_size at top level
        # V2 format has: layers, metadata
        if "layers" not in data and "model_type" in data:
            # This is v1 format, needs migration
            print("Migrating model from v1.0 to v2.0 format...")

            # Extract layers (v1 might have had different structure)
            # For now, just update metadata
            if "metadata" not in data:
                data["metadata"] = {}

            data["metadata"]["export_version"] = "2.0.0"
            data["metadata"]["export_date"] = datetime.now().isoformat()
            data["metadata"]["format_version"] = "2.0"

            # Preserve old metadata
            if "model_name" in data:
                data["metadata"]["model_name"] = data["model_name"]
            if "input_size" in data:
                data["metadata"]["input_size"] = data["input_size"]
            if "output_size" in data:
                data["metadata"]["output_size"] = data["output_size"]

        # Save migrated model
        if output_path is None:
            output_path = model_path

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Migration complete: {output_path}")
        return True

    except Exception as e:
        print(f"Error migrating model: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model versioning utilities")
    parser.add_argument("command", choices=["check", "version", "migrate", "update"],
                        help="Command to execute")
    parser.add_argument("model_file", type=str, help="Path to model JSON file")
    parser.add_argument("--output", "-o", type=str, help="Output file (for migrate)")

    args = parser.parse_args()

    model_path = Path(args.model_file)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        exit(1)

    if args.command == "check":
        compatible, message = check_model_compatibility(model_path)
        print(message)
        exit(0 if compatible else 1)

    elif args.command == "version":
        version = get_model_version(model_path)
        if version:
            print(f"Model version: {version}")
        else:
            print("Could not determine model version")
            exit(1)

    elif args.command == "migrate":
        output_path = Path(args.output) if args.output else None
        success = migrate_model_v1_to_v2(model_path, output_path)
        exit(0 if success else 1)

    elif args.command == "update":
        success = add_version_metadata(model_path)
        exit(0 if success else 1)
