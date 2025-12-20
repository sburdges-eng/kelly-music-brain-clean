#!/usr/bin/env python3
"""
Integration Test for ML Training Pipeline
==========================================
Tests the full pipeline: training → export → validation → C++ compatibility
"""

import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor,
    export_to_rtneural
)
from validate_models import validate_rtneural_json, MODEL_SPECS


def test_training_export_validation():
    """Test: Train models → Export → Validate."""
    print("=" * 60)
    print("Integration Test: Training → Export → Validation")
    print("=" * 60)
    print()

    models = {
        'EmotionRecognizer': EmotionRecognizer(),
        'MelodyTransformer': MelodyTransformer(),
        'HarmonyPredictor': HarmonyPredictor(),
        'DynamicsEngine': DynamicsEngine(),
        'GroovePredictor': GroovePredictor()
    }

    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    all_passed = True

    for model_name, model in models.items():
        print(f"Testing {model_name}...")

        # 1. Export model
        try:
            export_to_rtneural(model, model_name, output_dir)
            json_path = output_dir / f"{model_name.lower()}.json"

            if not json_path.exists():
                print(f"  ✗ Export failed: {json_path} not found")
                all_passed = False
                continue

            print(f"  ✓ Exported to {json_path}")

            # 2. Validate JSON structure
            is_valid, errors = validate_rtneural_json(json_path)
            if not is_valid:
                print(f"  ✗ Validation failed:")
                for error in errors:
                    print(f"    - {error}")
                all_passed = False
                continue

            print(f"  ✓ JSON structure valid")

            # 3. Check against expected specs
            model_file = json_path.name.lower()
            expected_specs = MODEL_SPECS.get(model_file)
            if expected_specs:
                from validate_models import validate_model_specs
                spec_valid, spec_errors = validate_model_specs(json_path, expected_specs)
                if not spec_valid:
                    print(f"  ✗ Specification mismatch:")
                    for error in spec_errors:
                        print(f"    - {error}")
                    all_passed = False
                    continue
                print(f"  ✓ Specifications match")

            # 4. Test model can be loaded and used
            with open(json_path, 'r') as f:
                data = json.load(f)

            layers = data.get("layers", [])
            metadata = data.get("metadata", {})

            if len(layers) == 0:
                print(f"  ✗ No layers found in exported model")
                all_passed = False
                continue

            print(f"  ✓ Model has {len(layers)} layers")
            print(f"    Input: {metadata.get('input_size')}, Output: {metadata.get('output_size')}")
            print(f"    Parameters: {metadata.get('parameter_count', 0):,}")
            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            print()

    print("=" * 60)
    if all_passed:
        print("✓ All integration tests passed")
        return 0
    else:
        print("✗ Some integration tests failed")
        return 1


def test_model_inference():
    """Test: Model inference produces correct output shapes."""
    print("=" * 60)
    print("Integration Test: Model Inference")
    print("=" * 60)
    print()

    models = {
        'EmotionRecognizer': (EmotionRecognizer(), torch.randn(1, 128)),
        'MelodyTransformer': (MelodyTransformer(), torch.randn(1, 64)),
        'HarmonyPredictor': (HarmonyPredictor(), torch.randn(1, 128)),
        'DynamicsEngine': (DynamicsEngine(), torch.randn(1, 32)),
        'GroovePredictor': (GroovePredictor(), torch.randn(1, 64))
    }

    expected_outputs = {
        'EmotionRecognizer': 64,
        'MelodyTransformer': 128,
        'HarmonyPredictor': 64,
        'DynamicsEngine': 16,
        'GroovePredictor': 32
    }

    all_passed = True

    for model_name, (model, test_input) in models.items():
        model.eval()
        with torch.no_grad():
            output = model(test_input)

        expected_size = expected_outputs[model_name]
        actual_size = output.shape[-1]

        if actual_size == expected_size:
            print(f"✓ {model_name}: Output shape correct ({actual_size})")
        else:
            print(f"✗ {model_name}: Output shape mismatch (expected {expected_size}, got {actual_size})")
            all_passed = False

    print()
    return 0 if all_passed else 1


def main():
    """Run all integration tests."""
    results = []

    # Test 1: Model inference
    results.append(test_model_inference())
    print()

    # Test 2: Training → Export → Validation
    results.append(test_training_export_validation())
    print()

    # Summary
    print("=" * 60)
    print("Integration Test Summary")
    print("=" * 60)

    if all(r == 0 for r in results):
        print("✓ All integration tests passed")
        return 0
    else:
        print("✗ Some integration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
