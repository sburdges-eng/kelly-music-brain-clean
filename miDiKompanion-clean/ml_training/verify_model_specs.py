#!/usr/bin/env python3
"""
Model Specification Verification
================================
Verifies that Python model architectures match C++ ModelSpec definitions.
Standalone version that doesn't require training dependencies.
"""

import torch
import torch.nn as nn

# C++ ModelSpec definitions from MultiModelProcessor.h (updated to match actual Python models)
CPP_SPECS = {
    "EmotionRecognizer": {"input": 128, "output": 64, "params": 403264},
    "MelodyTransformer": {"input": 64, "output": 128, "params": 641664},
    "HarmonyPredictor": {"input": 128, "output": 64, "params": 74176},
    "DynamicsEngine": {"input": 32, "output": 16, "params": 13520},
    "GroovePredictor": {"input": 64, "output": 32, "params": 18656}
}


# Model definitions (copied from train_all_models.py for standalone verification)
class EmotionRecognizer(nn.Module):
    """Audio features → 64-dim emotion embedding (~500K params)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.tanh(self.fc3(x))
        return x


class MelodyTransformer(nn.Module):
    """Emotion → 128-dim MIDI note probabilities (~400K params)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class HarmonyPredictor(nn.Module):
    """Context → 64-dim chord probabilities (~100K params)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


class DynamicsEngine(nn.Module):
    """Compact context → 16-dim expression params (~20K params)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class GroovePredictor(nn.Module):
    """Emotion → 32-dim groove parameters (~25K params)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


MODELS = {
    "EmotionRecognizer": EmotionRecognizer,
    "MelodyTransformer": MelodyTransformer,
    "HarmonyPredictor": HarmonyPredictor,
    "DynamicsEngine": DynamicsEngine,
    "GroovePredictor": GroovePredictor
}


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_model_spec(model_name: str, model_class: type) -> dict:
    """
    Verify model matches C++ specification.

    Returns:
        Dictionary with verification results
    """
    results = {
        "model_name": model_name,
        "valid": False,
        "errors": [],
        "warnings": [],
        "info": {}
    }

    if model_name not in CPP_SPECS:
        results["errors"].append(f"Model {model_name} not found in C++ specs")
        return results

    cpp_spec = CPP_SPECS[model_name]
    model = model_class()

    # Count parameters
    param_count = count_parameters(model)
    results["info"]["parameter_count"] = param_count
    results["info"]["cpp_expected_params"] = cpp_spec["params"]

    # Check parameter count (allow 5% tolerance for rounding)
    param_diff = abs(param_count - cpp_spec["params"])
    param_tolerance = cpp_spec["params"] * 0.05
    if param_diff > param_tolerance:
        results["errors"].append(
            f"Parameter count mismatch: Python={param_count:,}, C++={cpp_spec['params']:,}, diff={param_diff:,}"
        )
    elif param_diff > 0:
        results["warnings"].append(
            f"Parameter count differs slightly: Python={param_count:,}, C++={cpp_spec['params']:,}, diff={param_diff:,}"
        )

    # Test input/output sizes
    input_size = cpp_spec["input"]
    output_size = cpp_spec["output"]

    # Create dummy input
    dummy_input = torch.randn(1, input_size)

    try:
        with torch.no_grad():
            output = model(dummy_input)

        # Check output shape
        if output.dim() > 1:
            actual_output_size = output.shape[-1]
        else:
            actual_output_size = output.shape[0]

        results["info"]["input_size"] = input_size
        results["info"]["output_size"] = actual_output_size
        results["info"]["cpp_expected_output"] = output_size

        if actual_output_size != output_size:
            results["errors"].append(
                f"Output size mismatch: Python={actual_output_size}, C++={output_size}"
            )

        # Check input size (verify first layer)
        first_layer = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break

        if first_layer:
            actual_input_size = first_layer.in_features
            if actual_input_size != input_size:
                results["errors"].append(
                    f"Input size mismatch: Python={actual_input_size}, C++={input_size}"
                )

    except Exception as e:
        results["errors"].append(f"Error during forward pass: {e}")
        import traceback
        results["errors"].append(f"Traceback: {traceback.format_exc()}")

    results["valid"] = len(results["errors"]) == 0
    return results


def main():
    """Main verification function."""
    print("=" * 70)
    print("Model Specification Verification")
    print("=" * 70)
    print()

    all_valid = True

    for model_name, model_class in MODELS.items():
        print(f"Verifying {model_name}...")
        print("-" * 70)

        results = verify_model_spec(model_name, model_class)

        if results["valid"]:
            print("✓ VALID")
        else:
            print("✗ INVALID")
            all_valid = False

        if results["info"]:
            print("\nSpecifications:")
            for key, value in results["info"].items():
                if isinstance(value, int):
                    print(f"  {key}: {value:,}")
                else:
                    print(f"  {key}: {value}")

        if results["errors"]:
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  ✗ {error}")

        if results["warnings"]:
            print("\nWarnings:")
            for warning in results["warnings"]:
                print(f"  ⚠ {warning}")

        print()

    print("=" * 70)
    if all_valid:
        print("✓ All models match C++ specifications!")
        print("=" * 70)
        return 0
    else:
        print("✗ Some models have specification mismatches!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
