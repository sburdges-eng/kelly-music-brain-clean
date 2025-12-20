#!/usr/bin/env python3
"""
RTNeural Export Validation Script
==================================
Validates that exported JSON models are compatible with RTNeural format.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def validate_json_structure(json_data: Dict) -> tuple[bool, List[str]]:
    """Validate JSON structure matches RTNeural format."""
    errors = []

    # Check for required top-level keys
    if "layers" not in json_data:
        errors.append("Missing 'layers' key")
        return False, errors

    if "metadata" not in json_data:
        errors.append("Missing 'metadata' key")
        return False, errors

    layers = json_data["layers"]
    if not isinstance(layers, list):
        errors.append("'layers' must be a list")
        return False, errors

    if len(layers) == 0:
        errors.append("'layers' list is empty")
        return False, errors

    # Validate each layer
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            errors.append(f"Layer {i} is not a dictionary")
            continue

        layer_type = layer.get("type")
        if not layer_type:
            errors.append(f"Layer {i} missing 'type' field")
            continue

        if layer_type == "dense":
            errors.extend(validate_dense_layer(layer, i))
        elif layer_type == "lstm":
            errors.extend(validate_lstm_layer(layer, i))
        else:
            errors.append(f"Layer {i} has unknown type: {layer_type}")

    return len(errors) == 0, errors


def validate_dense_layer(layer: Dict, index: int) -> List[str]:
    """Validate a dense layer structure."""
    errors = []

    required_fields = ["in_size", "out_size", "activation", "weights", "bias"]
    for field in required_fields:
        if field not in layer:
            errors.append(f"Layer {index} (dense) missing '{field}' field")

    if "in_size" in layer and "out_size" in layer and "weights" in layer:
        in_size = layer["in_size"]
        out_size = layer["out_size"]
        weights = layer["weights"]

        if not isinstance(weights, list):
            errors.append(f"Layer {index} weights must be a list")
        elif len(weights) != out_size:
            errors.append(f"Layer {index} weights length ({len(weights)}) != out_size ({out_size})")
        elif len(weights) > 0 and len(weights[0]) != in_size:
            errors.append(f"Layer {index} weight row length ({len(weights[0])}) != in_size ({in_size})")

    if "bias" in layer:
        bias = layer["bias"]
        if not isinstance(bias, list):
            errors.append(f"Layer {index} bias must be a list")
        elif "out_size" in layer and len(bias) != layer["out_size"]:
            errors.append(f"Layer {index} bias length ({len(bias)}) != out_size ({layer['out_size']})")

    valid_activations = ["tanh", "relu", "sigmoid", "softmax", "linear"]
    if "activation" in layer and layer["activation"] not in valid_activations:
        errors.append(f"Layer {index} invalid activation: {layer['activation']}")

    return errors


def validate_lstm_layer(layer: Dict, index: int) -> List[str]:
    """Validate an LSTM layer structure."""
    errors = []

    required_fields = ["in_size", "out_size", "weights_ih", "weights_hh", "bias_ih", "bias_hh"]
    for field in required_fields:
        if field not in layer:
            errors.append(f"Layer {index} (lstm) missing '{field}' field")

    if "out_size" in layer and "weights_ih" in layer:
        out_size = layer["out_size"]
        weights_ih = layer["weights_ih"]

        if not isinstance(weights_ih, list):
            errors.append(f"Layer {index} weights_ih must be a list")
        elif len(weights_ih) != 4:
            errors.append(f"Layer {index} weights_ih must have 4 gates, got {len(weights_ih)}")
        elif len(weights_ih) > 0:
            if not isinstance(weights_ih[0], list):
                errors.append(f"Layer {index} weights_ih[0] must be a list")
            elif len(weights_ih[0]) != out_size:
                errors.append(f"Layer {index} weights_ih[0] length ({len(weights_ih[0])}) != out_size ({out_size})")

    if "in_size" in layer and "weights_ih" in layer and len(layer["weights_ih"]) > 0:
        in_size = layer["in_size"]
        if isinstance(layer["weights_ih"][0], list) and len(layer["weights_ih"][0]) > 0:
            if isinstance(layer["weights_ih"][0][0], list):
                if len(layer["weights_ih"][0][0]) != in_size:
                    errors.append(f"Layer {index} weights_ih inner dimension != in_size ({in_size})")

    return errors


def validate_model_dimensions(json_data: Dict) -> tuple[bool, List[str]]:
    """Validate that layer dimensions are consistent."""
    errors = []
    layers = json_data.get("layers", [])

    if len(layers) == 0:
        return True, errors

    # Check first layer input size matches metadata
    first_layer = layers[0]
    if "in_size" in first_layer:
        metadata_input = json_data.get("metadata", {}).get("input_size")
        if metadata_input is not None and metadata_input != first_layer["in_size"]:
            errors.append(f"Metadata input_size ({metadata_input}) != first layer in_size ({first_layer['in_size']})")

    # Check last layer output size matches metadata
    last_layer = layers[-1]
    if "out_size" in last_layer:
        metadata_output = json_data.get("metadata", {}).get("output_size")
        if metadata_output is not None and metadata_output != last_layer["out_size"]:
            errors.append(f"Metadata output_size ({metadata_output}) != last layer out_size ({last_layer['out_size']})")

    # Check layer chaining (output of layer N should match input of layer N+1)
    for i in range(len(layers) - 1):
        current = layers[i]
        next_layer = layers[i + 1]

        if "out_size" in current and "in_size" in next_layer:
            if current["out_size"] != next_layer["in_size"]:
                errors.append(f"Layer {i} out_size ({current['out_size']}) != Layer {i+1} in_size ({next_layer['in_size']})")

    return len(errors) == 0, errors


def validate_model_file(model_path: Path) -> tuple[bool, List[str]]:
    """Validate a single model JSON file."""
    errors = []

    if not model_path.exists():
        return False, [f"Model file not found: {model_path}"]

    try:
        with open(model_path, 'r') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except Exception as e:
        return False, [f"Error reading file: {e}"]

    # Validate structure
    valid, struct_errors = validate_json_structure(json_data)
    errors.extend(struct_errors)

    if valid:
        # Validate dimensions
        dim_valid, dim_errors = validate_model_dimensions(json_data)
        errors.extend(dim_errors)

    return len(errors) == 0, errors


def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate RTNeural JSON model exports"
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to model JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed validation information"
    )

    args = parser.parse_args()

    model_path = Path(args.model_file)

    print(f"Validating RTNeural model: {model_path}")
    print("=" * 60)

    valid, errors = validate_model_file(model_path)

    if valid:
        print("✓ Model validation PASSED")

        if args.verbose:
            with open(model_path, 'r') as f:
                json_data = json.load(f)

            metadata = json_data.get("metadata", {})
            layers = json_data.get("layers", [])

            print(f"\nModel: {metadata.get('model_name', 'Unknown')}")
            print(f"Input size: {metadata.get('input_size', 'Unknown')}")
            print(f"Output size: {metadata.get('output_size', 'Unknown')}")
            print(f"Parameters: {metadata.get('parameter_count', 'Unknown'):,}")
            print(f"Layers: {len(layers)}")

            for i, layer in enumerate(layers):
                layer_type = layer.get("type", "unknown")
                in_size = layer.get("in_size", "?")
                out_size = layer.get("out_size", "?")
                activation = layer.get("activation", "N/A")
                print(f"  Layer {i}: {layer_type} ({in_size} → {out_size}) {activation}")

        return 0
    else:
        print("✗ Model validation FAILED")
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
