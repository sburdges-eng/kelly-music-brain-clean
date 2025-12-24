"""
Configuration Loader with Environment Variable Expansion

Loads YAML configuration files with support for environment variable substitution.

Syntax:
    ${VAR_NAME}             - Expands to env var VAR_NAME (error if not set)
    ${VAR_NAME:-default}    - Expands to VAR_NAME or 'default' if not set
    ${VAR_NAME:=value}      - Expands and sets VAR_NAME to 'value' if not set

Usage:
    from configs.config_loader import load_config

    config = load_config("configs/emotion_recognizer.yaml")
    print(config["data_path"])  # Expanded path
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


# Pattern to match ${VAR:-default} or ${VAR} or ${VAR:=value}
ENV_VAR_PATTERN = re.compile(
    r'\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<op>[-=])(?P<default>[^}]*))?\}'
)


def expand_env_vars(value: str) -> str:
    """
    Expand environment variables in a string.

    Supports:
        ${VAR}          - Required variable
        ${VAR:-default} - Default value if not set
        ${VAR:=value}   - Set and use default if not set

    Args:
        value: String potentially containing ${VAR} patterns

    Returns:
        String with all variables expanded

    Raises:
        ValueError: If a required variable is not set
    """
    def replace_match(match: re.Match) -> str:
        var_name = match.group("name")
        operator = match.group("op")
        default = match.group("default")

        env_value = os.environ.get(var_name)

        if env_value is not None:
            return env_value

        if operator == "-":
            # Use default but don't set
            return default if default is not None else ""
        elif operator == "=":
            # Set the environment variable and use default
            if default is not None:
                os.environ[var_name] = default
                return default
            return ""
        else:
            # No default specified, variable required
            if env_value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' is not set and no default provided"
                )
            return env_value

    return ENV_VAR_PATTERN.sub(replace_match, value)


def expand_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables in dict values.

    Args:
        data: Dictionary with potentially unexpanded values

    Returns:
        New dictionary with all string values expanded
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = expand_env_vars(value)
        elif isinstance(value, dict):
            result[key] = expand_dict_values(value)
        elif isinstance(value, list):
            result[key] = [
                expand_env_vars(item) if isinstance(item, str)
                else expand_dict_values(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def load_yaml_with_expansion(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file and expand all environment variables.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with expanded values
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    return expand_dict_values(data)


def load_config(
    path: Union[str, Path],
    defaults: Optional[Dict[str, Any]] = None,
    required_keys: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Load a configuration file with environment variable expansion.

    Args:
        path: Path to YAML configuration file
        defaults: Default values to merge (config overrides)
        required_keys: List of keys that must be present

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required keys are missing or env vars not set
    """
    config = load_yaml_with_expansion(path)

    # Merge with defaults (config values take precedence)
    if defaults:
        merged = {**defaults, **config}
        config = merged

    # Validate required keys
    if required_keys:
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    return config


def get_data_path(config: Dict[str, Any], key: str = "data_path") -> Path:
    """
    Get a data path from config, ensuring it's a Path object.

    Args:
        config: Configuration dictionary
        key: Key containing the path

    Returns:
        Path object for the data path
    """
    path_str = config.get(key, "")
    if not path_str:
        from configs.storage import get_audio_data_root
        return get_audio_data_root()
    return Path(path_str)


def resolve_manifest_path(manifest_path: str) -> Path:
    """
    Resolve a manifest path, expanding env vars and ensuring it exists.

    Args:
        manifest_path: Path to manifest file (may contain ${VAR})

    Returns:
        Resolved Path object
    """
    expanded = expand_env_vars(manifest_path)
    return Path(expanded)


if __name__ == "__main__":
    # Test the loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python config_loader.py <config.yaml>")
        print("\nExample:")
        print("  KELLY_AUDIO_DATA_ROOT=/data python config_loader.py configs/emotion_recognizer.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        config = load_config(config_path)
        print(f"Loaded config from: {config_path}")
        print()

        # Pretty print config
        import json
        print(json.dumps(config, indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
