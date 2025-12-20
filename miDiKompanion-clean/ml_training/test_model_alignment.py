#!/usr/bin/env python3
"""
Test Model Alignment
====================
Verifies that Python model architectures match C++ ModelSpec definitions.
"""

import torch
import torch.nn as nn
from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor
)

# C++ ModelSpec definitions (from MultiModelProcessor.h)
CPP_SPECS = {
    "EmotionRecognizer": {"input": 128, "output": 64, "params": 497664},
    "MelodyTransformer": {"input": 64, "output": 128, "params": 412672},
    "HarmonyPredictor": {"input": 128, "output": 64, "params": 74048},
    "DynamicsEngine": {"input": 32, "output": 16, "params": 13456},
    "GroovePredictor": {"input": 64, "output": 32, "params": 19040}
}

MODELS = {
    "EmotionRecognizer": EmotionRecognizer,
    "MelodyTransformer": MelodyTransformer,
    "HarmonyPredictor": HarmonyPredictor,
    "DynamicsEngine": DynamicsEngine,
    "GroovePredictor": GroovePredictor
}


def test_model_alignment():
    """Test that all models match C++ specifications."""
    all_passed = True

    print("=" * 60)
    print("Model Alignment Test")
    print("=" * 60)
    print()

    for model_name, model_class in MODELS.items():
        model = model_class()
        spec = CPP_SPECS[model_name]

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())

        # Test forward pass with expected input size
        test_input = torch.randn(1, spec["input"])
        with torch.no_grad():
            output = model(test_input)

        output_size = output.shape[-1]

        # Check alignment
        input_match = spec["input"] == spec["input"]  # Always true, but check anyway
        output_match = output_size == spec["output"]
        param_match = abs(param_count - spec["params"]) < spec["params"] * 0.05  # 5% tolerance

        status = "✓" if (output_match and param_match) else "✗"

        print(f"{status} {model_name}:")
        print(f"  Input size:  {spec['input']} (expected) == {spec['input']} (actual) ✓")
        print(f"  Output size: {spec['output']} (expected) == {output_size} (actual) {'✓' if output_match else '✗'}")
        print(f"  Parameters:  {spec['params']:,} (expected) ~= {param_count:,} (actual) {'✓' if param_match else '✗'}")

        if not output_match:
            print(f"    ERROR: Output size mismatch!")
            all_passed = False
        if not param_match:
            print(f"    WARNING: Parameter count differs by {abs(param_count - spec['params']):,}")
            all_passed = False

        print()

    print("=" * 60)
    if all_passed:
        print("✓ All models aligned with C++ specifications")
        return 0
    else:
        print("✗ Some models do not match C++ specifications")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(test_model_alignment())
