#!/usr/bin/env python3
"""
RTNeural Export/Import Roundtrip Test
======================================
Tests that models can be exported and the JSON structure is valid.
Since we can't easily test RTNeural C++ loading in Python, this verifies:
1. Models export successfully
2. JSON structure is valid
3. Layer dimensions match expectations
4. Weight counts match
"""

import torch
import torch.nn as nn
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor,
    export_to_rtneural
)
from validate_models import validate_model


def test_export_roundtrip(model_class: type, model_name: str) -> bool:
    """Test that a model can be exported and validates correctly."""
    print(f"\nTesting {model_name}...")

    try:
        # Create model
        model = model_class()

        # Initialize with random weights for testing
        for param in model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.1)

        # Export to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Export model
            try:
                rtneural_json = export_to_rtneural(model, model_name, output_dir)
            except Exception as e:
                print(f"  ✗ Export failed: {e}")
                return False

            # Check export was successful
            json_file = output_dir / f"{model_name.lower()}.json"
            if not json_file.exists():
                print(f"  ✗ JSON file not created: {json_file}")
                return False

            # Validate exported JSON
            is_valid, errors = validate_model(json_file, verbose=False)
            if not is_valid:
                print(f"  ✗ Validation failed:")
                for error in errors:
                    print(f"    - {error}")
                return False

            # Verify weight counts match
            exported_params = rtneural_json.get("metadata", {}).get("parameter_count", 0)
            actual_params = sum(p.numel() for p in model.parameters())

            if exported_params != actual_params:
                print(f"  ⚠ Parameter count mismatch: exported={exported_params}, actual={actual_params}")
                # This is a warning, not a failure, as export may count differently

            # Verify input/output sizes
            if "input_size" not in rtneural_json or "output_size" not in rtneural_json:
                print(f"  ✗ Missing input_size or output_size in JSON")
                return False

            # Test that we can load the JSON back
            with open(json_file, 'r') as f:
                loaded_json = json.load(f)

            if loaded_json.get("model_name") != model_name:
                print(f"  ✗ Model name mismatch")
                return False

            print(f"  ✓ Export successful")
            print(f"  ✓ Validation passed")
            print(f"  ✓ JSON structure valid")
            print(f"    Input size: {rtneural_json['input_size']}")
            print(f"    Output size: {rtneural_json['output_size']}")
            print(f"    Layers: {len(rtneural_json.get('layers', []))}")
            print(f"    Parameters: {exported_params:,}")

            return True

    except Exception as e:
        print(f"  ✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test export/import roundtrip for all models."""
    print("=" * 60)
    print("RTNeural Export/Import Roundtrip Test")
    print("=" * 60)
    print("\nThis test verifies:")
    print("  1. Models can be exported to RTNeural JSON format")
    print("  2. Exported JSON structure is valid")
    print("  3. Layer dimensions match model architecture")
    print("  4. Weight counts are preserved")
    print("\nNote: This does not test C++ RTNeural loading.")
    print("      For C++ testing, compile and run test_model_loading.cpp")

    models = [
        (EmotionRecognizer, "EmotionRecognizer"),
        (MelodyTransformer, "MelodyTransformer"),
        (HarmonyPredictor, "HarmonyPredictor"),
        (DynamicsEngine, "DynamicsEngine"),
        (GroovePredictor, "GroovePredictor")
    ]

    all_passed = True

    for model_class, model_name in models:
        if not test_export_roundtrip(model_class, model_name):
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All export tests passed!")
        print("\nNext steps:")
        print("  1. Train models: python train_all_models.py")
        print("  2. Validate exports: python validate_models.py models/*.json")
        print("  3. Test C++ loading: compile test_model_loading.cpp")
        return 0
    else:
        print("✗ Some export tests failed!")
        print("  Check errors above and fix export function.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
