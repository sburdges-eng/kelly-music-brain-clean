"""
Validate models/registry.json against its schema and optional golden vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import jsonschema
except ImportError:  # pragma: no cover - handled by test skip
    jsonschema = None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
def test_registry_schema_validation() -> None:
    root = Path(__file__).resolve().parents[2]
    registry_path = root / "models" / "registry.json"
    schema_path = root / "models" / "registry.schema.json"

    assert registry_path.exists(), "models/registry.json missing"
    assert schema_path.exists(), "models/registry.schema.json missing"

    registry = _load_json(registry_path)
    schema = _load_json(schema_path)

    jsonschema.validate(instance=registry, schema=schema)

    # Basic sanity: unique IDs
    ids = [m["id"] for m in registry.get("models", [])]
    assert len(ids) == len(set(ids)), "Duplicate model ids found"


def test_optional_golden_vectors() -> None:
    """
    If a golden vector file exists for a model, ensure it has required fields.

    Does not enforce existence for all models to keep stubs passing.
    """
    root = Path(__file__).resolve().parents[2]
    registry_path = root / "models" / "registry.json"
    registry = _load_json(registry_path)

    for model in registry.get("models", []):
        golden_path = root / "models" / "golden" / f"{model['id']}.json"
        if not golden_path.exists():
            continue

        golden = _load_json(golden_path)
        required = {"input_checksum", "output_checksum", "feature_version"}
        missing = required - set(golden.keys())
        assert not missing, f"Golden vector missing fields for {model['id']}: {missing}"

        # Optional: allow storing sample_rate / input_size to guard preprocessing drift
        if "input_size" in golden:
            assert golden["input_size"] > 0
        if "sample_rate" in golden:
            assert golden["sample_rate"] > 0

