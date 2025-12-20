#!/usr/bin/env python3
"""
Verify Model Architectures Match C++ Specifications
====================================================
Quick script to verify Python model architectures match C++ ModelSpec definitions.
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
# Updated to match actual C++ header values
MODEL_SPECS = {
    "EmotionRecognizer": {"input": 128, "output": 64, "params": 403264},
    "MelodyTransformer": {"input": 64, "output": 128, "params": 641664},
    "HarmonyPredictor": {"input": 128, "output": 64, "params": 74176},
    "DynamicsEngine": {"input": 32, "output": 16, "params": 13520},
    "GroovePredictor": {"input": 64, "output": 32, "params": 18656}
}

MODELS = {
    "EmotionRecognizer": EmotionRecognizer,
    "MelodyTransformer": MelodyTransformer,
    "HarmonyPredictor": HarmonyPredictor,
    "DynamicsEngine": DynamicsEngine,
    "GroovePredictor": GroovePredictor
}


def verify_model(model_class, model_name):
    """Verify a model's architecture matches C++ specs."""
    model = model_class()

    # Get parameter count
    param_count = sum(p.numel() for p in model.parameters())

    # Test forward pass to get input/output sizes
    spec = MODEL_SPECS[model_name]

    # Create dummy input
    dummy_input = torch.randn(1, spec["input"])

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    actual_input = dummy_input.shape[1]
    actual_output = output.shape[1]

    # Check matches
    input_match = actual_input == spec["input"]
    output_match = actual_output == spec["output"]
    param_match = abs(param_count - spec["params"]) / spec["params"] < 0.1  # 10% tolerance

    status = "✓" if (input_match and output_match and param_match) else "✗"

    print(f"{status} {model_name}:")
    print(f"  Input:  {actual_input} (expected {spec['input']}) {'✓' if input_match else '✗'}")
    print(f"  Output: {actual_output} (expected {spec['output']}) {'✓' if output_match else '✗'}")
    print(f"  Params: {param_count:,} (expected ~{spec['params']:,}) {'✓' if param_match else '✗'}")

    return input_match and output_match and param_match


if __name__ == "__main__":
    print("Verifying Model Architectures Match C++ Specifications")
    print("=" * 60)

    all_match = True
    for model_name, model_class in MODELS.items():
        if not verify_model(model_class, model_name):
            all_match = False
        print()

    print("=" * 60)
    if all_match:
        print("✓ All models match C++ specifications")
    else:
        print("✗ Some models do not match C++ specifications")
        exit(1)
