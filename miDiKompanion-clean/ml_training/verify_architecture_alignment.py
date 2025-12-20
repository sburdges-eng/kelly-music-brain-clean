#!/usr/bin/env python3
"""
Architecture Alignment Verification
====================================
Verifies that Python model architectures match C++ ModelSpec definitions.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Import models from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor
)

# C++ ModelSpec definitions (from MultiModelProcessor.h)
CPP_SPECS = {
    "EmotionRecognizer": {
        "inputSize": 128,
        "outputSize": 64,
        "estimatedParams": 497664
    },
    "MelodyTransformer": {
        "inputSize": 64,
        "outputSize": 128,
        "estimatedParams": 412672
    },
    "HarmonyPredictor": {
        "inputSize": 128,
        "outputSize": 64,
        "estimatedParams": 74048
    },
    "DynamicsEngine": {
        "inputSize": 32,
        "outputSize": 16,
        "estimatedParams": 13456
    },
    "GroovePredictor": {
        "inputSize": 64,
        "outputSize": 32,
        "estimatedParams": 19040
    }
}


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def test_model_forward(model: nn.Module, input_size: int, output_size: int, model_name: str) -> bool:
    """Test that model forward pass produces correct output shape."""
    try:
        model.eval()
        # Create dummy input
        dummy_input = torch.randn(1, input_size)

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        # Check output shape
        if output.shape[1] != output_size:
            print(f"  ✗ Output size mismatch: expected {output_size}, got {output.shape[1]}")
            return False

        # Check that output is not all zeros or NaNs
        if torch.isnan(output).any():
            print(f"  ✗ Output contains NaN values")
            return False

        if (output == 0).all():
            print(f"  ✗ Output is all zeros (model may not be initialized)")
            return False

        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False


def verify_model(model_class: type, model_name: str, cpp_spec: dict) -> bool:
    """Verify a single model matches C++ specification."""
    print(f"\nVerifying {model_name}...")

    # Create model instance
    model = model_class()

    # Check input size (by testing forward pass)
    input_size = cpp_spec["inputSize"]
    output_size = cpp_spec["outputSize"]

    # Verify forward pass
    forward_ok = test_model_forward(model, input_size, output_size, model_name)
    if not forward_ok:
        return False

    # Count parameters
    param_count = count_parameters(model)
    expected_params = cpp_spec["estimatedParams"]

    # Allow 5% tolerance for parameter count (due to LSTM implementation differences)
    tolerance = expected_params * 0.05
    param_diff = abs(param_count - expected_params)

    print(f"  Input size: {input_size} ✓")
    print(f"  Output size: {output_size} ✓")
    print(f"  Parameters: {param_count:,} (expected: {expected_params:,})")

    if param_diff > tolerance:
        print(f"  ✗ Parameter count mismatch: difference of {param_diff:,} exceeds tolerance")
        return False
    else:
        if param_diff > 0:
            print(f"  ⚠ Parameter count difference: {param_diff:,} (within tolerance)")
        else:
            print(f"  Parameters: ✓")

    return True


def main():
    """Verify all models match C++ specifications."""
    print("=" * 60)
    print("Model Architecture Alignment Verification")
    print("=" * 60)

    models = [
        (EmotionRecognizer, "EmotionRecognizer"),
        (MelodyTransformer, "MelodyTransformer"),
        (HarmonyPredictor, "HarmonyPredictor"),
        (DynamicsEngine, "DynamicsEngine"),
        (GroovePredictor, "GroovePredictor")
    ]

    all_verified = True

    for model_class, model_name in models:
        if model_name not in CPP_SPECS:
            print(f"\n✗ {model_name}: No C++ specification found")
            all_verified = False
            continue

        cpp_spec = CPP_SPECS[model_name]
        if not verify_model(model_class, model_name, cpp_spec):
            all_verified = False

    print("\n" + "=" * 60)
    if all_verified:
        print("✓ All models verified successfully!")
        print("  Python architectures match C++ ModelSpec definitions.")
        return 0
    else:
        print("✗ Verification failed!")
        print("  Some models do not match C++ specifications.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
