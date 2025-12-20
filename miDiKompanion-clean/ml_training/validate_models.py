#!/usr/bin/env python3
"""
Model Validation Script for Kelly ML Models
============================================
Validates exported RTNeural JSON models match C++ specifications.

Usage:
    python ml_training/validate_models.py <model_dir> [--model <model_name>] [--json <output.json>]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


# Model specifications from C++ MultiModelProcessor.h
MODEL_SPECS = {
    "emotionrecognizer": {
        "name": "EmotionRecognizer",
        "input_size": 128,
        "output_size": 64,
        "estimated_params": 403264,
        "tolerance": 0.1,  # 10% tolerance for parameter count
        "layer_count": 4,  # FC(128→512), FC(512→256), LSTM(256→128), FC(128→64)
        "has_lstm": True
    },
    "melodytransformer": {
        "name": "MelodyTransformer",
        "input_size": 64,
        "output_size": 128,
        "estimated_params": 641664,
        "tolerance": 0.1,
        "layer_count": 4,  # FC(64→256), LSTM(256→256), FC(256→256), FC(256→128)
        "has_lstm": True
    },
    "harmonypredictor": {
        "name": "HarmonyPredictor",
        "input_size": 128,
        "output_size": 64,
        "estimated_params": 74176,
        "tolerance": 0.1,
        "layer_count": 3,  # FC(128→256), FC(256→128), FC(128→64)
        "has_lstm": False
    },
    "dynamicsengine": {
        "name": "DynamicsEngine",
        "input_size": 32,
        "output_size": 16,
        "estimated_params": 13520,
        "tolerance": 0.1,
        "layer_count": 3,  # FC(32→128), FC(128→64), FC(64→16)
        "has_lstm": False
    },
    "groovepredictor": {
        "name": "GroovePredictor",
        "input_size": 64,
        "output_size": 32,
        "estimated_params": 18656,
        "tolerance": 0.1,
        "layer_count": 3,  # FC(64→128), FC(128→64), FC(64→32)
        "has_lstm": False
    }
}

# Valid RTNeural layer types
VALID_LAYER_TYPES = {"dense", "lstm"}

# Valid activation functions
VALID_ACTIVATIONS = {"linear", "tanh", "relu", "sigmoid", "softmax"}


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_json_structure(data: Dict, model_name: str) -> List[str]:
    """Validate JSON structure has required fields."""
    errors = []

    if "layers" not in data:
        errors.append(f"{model_name}: Missing 'layers' array")
    elif not isinstance(data["layers"], list):
        errors.append(f"{model_name}: 'layers' must be an array")
    elif len(data["layers"]) == 0:
        errors.append(f"{model_name}: 'layers' array is empty")

    if "metadata" not in data:
        errors.append(f"{model_name}: Missing 'metadata' object")
    elif not isinstance(data["metadata"], dict):
        errors.append(f"{model_name}: 'metadata' must be an object")
    else:
        metadata = data["metadata"]
        required_fields = ["model_name", "input_size", "output_size", "parameter_count"]
        for field in required_fields:
            if field not in metadata:
                errors.append(f"{model_name}: Missing '{field}' in metadata")

    return errors


def validate_layer(layer: Dict, layer_idx: int, model_name: str) -> List[str]:
    """Validate a single layer structure."""
    errors = []

    if "type" not in layer:
        errors.append(f"{model_name} Layer {layer_idx}: Missing 'type'")
        return errors

    layer_type = layer["type"]

    if layer_type not in VALID_LAYER_TYPES:
        errors.append(f"{model_name} Layer {layer_idx}: Invalid layer type '{layer_type}'")
        return errors

    # Common fields for all layers
    if "in_size" not in layer:
        errors.append(f"{model_name} Layer {layer_idx}: Missing 'in_size'")
    elif not isinstance(layer["in_size"], int) or layer["in_size"] <= 0:
        errors.append(f"{model_name} Layer {layer_idx}: Invalid 'in_size'")

    if "out_size" not in layer:
        errors.append(f"{model_name} Layer {layer_idx}: Missing 'out_size'")
    elif not isinstance(layer["out_size"], int) or layer["out_size"] <= 0:
        errors.append(f"{model_name} Layer {layer_idx}: Invalid 'out_size'")

    # Dense layer validation
    if layer_type == "dense":
        if "activation" not in layer:
            errors.append(f"{model_name} Layer {layer_idx}: Missing 'activation'")
        elif layer["activation"] not in VALID_ACTIVATIONS:
            errors.append(f"{model_name} Layer {layer_idx}: Invalid activation '{layer['activation']}'")

        if "weights" not in layer:
            errors.append(f"{model_name} Layer {layer_idx}: Missing 'weights'")
        elif not isinstance(layer["weights"], list):
            errors.append(f"{model_name} Layer {layer_idx}: 'weights' must be an array")
        else:
            # Validate weight dimensions
            if len(layer["weights"]) != layer["out_size"]:
                errors.append(f"{model_name} Layer {layer_idx}: Weight rows ({len(layer['weights'])}) != out_size ({layer['out_size']})")
            elif len(layer["weights"]) > 0:
                if len(layer["weights"][0]) != layer["in_size"]:
                    errors.append(f"{model_name} Layer {layer_idx}: Weight cols ({len(layer['weights'][0])}) != in_size ({layer['in_size']})")

        if "bias" not in layer:
            errors.append(f"{model_name} Layer {layer_idx}: Missing 'bias'")
        elif not isinstance(layer["bias"], list):
            errors.append(f"{model_name} Layer {layer_idx}: 'bias' must be an array")
        elif len(layer["bias"]) != layer["out_size"]:
            errors.append(f"{model_name} Layer {layer_idx}: Bias size ({len(layer['bias'])}) != out_size ({layer['out_size']})")

    # LSTM layer validation
    elif layer_type == "lstm":
        required_fields = ["weights_ih", "weights_hh", "bias_ih", "bias_hh"]
        for field in required_fields:
            if field not in layer:
                errors.append(f"{model_name} Layer {layer_idx}: Missing '{field}'")
            elif not isinstance(layer[field], list):
                errors.append(f"{model_name} Layer {layer_idx}: '{field}' must be an array")

        # Validate LSTM has 4 gates
        if "weights_ih" in layer and isinstance(layer["weights_ih"], list):
            if len(layer["weights_ih"]) != 4:
                errors.append(f"{model_name} Layer {layer_idx}: LSTM weights_ih must have 4 gates, got {len(layer['weights_ih'])}")
            else:
                # Validate each gate has correct dimensions
                hidden_size = layer["out_size"]
                for gate_idx, gate_weights in enumerate(layer["weights_ih"]):
                    if not isinstance(gate_weights, list):
                        errors.append(f"{model_name} Layer {layer_idx}: LSTM weights_ih[{gate_idx}] must be an array")
                    elif len(gate_weights) != hidden_size:
                        errors.append(f"{model_name} Layer {layer_idx}: LSTM weights_ih[{gate_idx}] size ({len(gate_weights)}) != hidden_size ({hidden_size})")
                    elif len(gate_weights) > 0 and len(gate_weights[0]) != layer["in_size"]:
                        errors.append(f"{model_name} Layer {layer_idx}: LSTM weights_ih[{gate_idx}] cols ({len(gate_weights[0])}) != in_size ({layer['in_size']})")

        if "weights_hh" in layer and isinstance(layer["weights_hh"], list):
            if len(layer["weights_hh"]) != 4:
                errors.append(f"{model_name} Layer {layer_idx}: LSTM weights_hh must have 4 gates, got {len(layer['weights_hh'])}")

        if "bias_ih" in layer and isinstance(layer["bias_ih"], list):
            if len(layer["bias_ih"]) != 4:
                errors.append(f"{model_name} Layer {layer_idx}: LSTM bias_ih must have 4 gates, got {len(layer['bias_ih'])}")

        if "bias_hh" in layer and isinstance(layer["bias_hh"], list):
            if len(layer["bias_hh"]) != 4:
                errors.append(f"{model_name} Layer {layer_idx}: LSTM bias_hh must have 4 gates, got {len(layer['bias_hh'])}")

    return errors


def calculate_parameter_count(layers: List[Dict]) -> int:
    """Calculate total parameter count from layers."""
    total = 0

    for layer in layers:
        layer_type = layer.get("type")

        if layer_type == "dense":
            in_size = layer.get("in_size", 0)
            out_size = layer.get("out_size", 0)
            # Weights: in_size * out_size
            total += in_size * out_size
            # Bias: out_size
            if "bias" in layer and layer["bias"]:
                total += out_size

        elif layer_type == "lstm":
            in_size = layer.get("in_size", 0)
            out_size = layer.get("out_size", 0)
            # weights_ih: 4 gates * (in_size * hidden_size)
            # weights_hh: 4 gates * (hidden_size * hidden_size)
            # bias_ih: 4 gates * hidden_size
            # bias_hh: 4 gates * hidden_size
            total += 4 * (in_size * out_size)  # weights_ih
            total += 4 * (out_size * out_size)  # weights_hh
            total += 4 * out_size  # bias_ih
            total += 4 * out_size  # bias_hh

    return total


def validate_model_spec(data: Dict, model_name: str, spec: Dict) -> List[str]:
    """Validate model matches C++ specification."""
    errors = []
    metadata = data.get("metadata", {})
    layers = data.get("layers", [])

    # Check input/output sizes
    input_size = metadata.get("input_size")
    if input_size != spec["input_size"]:
        errors.append(f"{model_name}: Input size mismatch: expected {spec['input_size']}, got {input_size}")

    output_size = metadata.get("output_size")
    if output_size != spec["output_size"]:
        errors.append(f"{model_name}: Output size mismatch: expected {spec['output_size']}, got {output_size}")

    # Check first layer input size
    if layers and "in_size" in layers[0]:
        if layers[0]["in_size"] != spec["input_size"]:
            errors.append(f"{model_name}: First layer input size mismatch: expected {spec['input_size']}, got {layers[0]['in_size']}")

    # Check last layer output size
    if layers and "out_size" in layers[-1]:
        if layers[-1]["out_size"] != spec["output_size"]:
            errors.append(f"{model_name}: Last layer output size mismatch: expected {spec['output_size']}, got {layers[-1]['out_size']}")

    # Check parameter count
    param_count = metadata.get("parameter_count", 0)
    expected_params = spec["estimated_params"]
    tolerance = spec["tolerance"]
    min_params = int(expected_params * (1 - tolerance))
    max_params = int(expected_params * (1 + tolerance))

    if param_count < min_params or param_count > max_params:
        errors.append(f"{model_name}: Parameter count mismatch: expected ~{expected_params} (±{tolerance*100}%), got {param_count}")

    # Recalculate parameter count from layers
    calculated_params = calculate_parameter_count(layers)
    if abs(calculated_params - param_count) > 100:  # Allow small rounding differences
        errors.append(f"{model_name}: Parameter count inconsistency: metadata says {param_count}, calculated {calculated_params}")

    # Check layer count
    if len(layers) != spec["layer_count"]:
        errors.append(f"{model_name}: Layer count mismatch: expected {spec['layer_count']}, got {len(layers)}")

    # Check LSTM presence
    has_lstm = any(layer.get("type") == "lstm" for layer in layers)
    if has_lstm != spec["has_lstm"]:
        errors.append(f"{model_name}: LSTM presence mismatch: expected {spec['has_lstm']}, got {has_lstm}")

    return errors


def validate_model_file(model_path: Path) -> Tuple[bool, List[str], Dict]:
    """
    Validate a single model file.

    Returns:
        (is_valid, errors, report_dict)
    """
    errors = []
    warnings = []
    report = {
        "file": str(model_path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "metadata": {},
        "layer_count": 0,
        "parameter_count": 0
    }

    # Determine model name from filename
    model_name_lower = model_path.stem.lower()
    spec = MODEL_SPECS.get(model_name_lower)

    if not spec:
        errors.append(f"Unknown model: {model_name_lower}")
        report["errors"] = errors
        return False, errors, report

    # Load and parse JSON
    try:
        with open(model_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        report["errors"] = errors
        return False, errors, report
    except Exception as e:
        errors.append(f"Error reading file: {e}")
        report["errors"] = errors
        return False, errors, report

    # Validate JSON structure
    structure_errors = validate_json_structure(data, spec["name"])
    errors.extend(structure_errors)

    if structure_errors:
        report["errors"] = errors
        return False, errors, report

    # Validate layers
    layers = data.get("layers", [])
    for idx, layer in enumerate(layers):
        layer_errors = validate_layer(layer, idx, spec["name"])
        errors.extend(layer_errors)

    # Validate model specification
    spec_errors = validate_model_spec(data, spec["name"], spec)
    errors.extend(spec_errors)

    # Build report
    report["valid"] = len(errors) == 0
    report["errors"] = errors
    report["warnings"] = warnings
    report["metadata"] = data.get("metadata", {})
    report["layer_count"] = len(layers)
    report["parameter_count"] = data.get("metadata", {}).get("parameter_count", 0)

    return len(errors) == 0, errors, report


def validate_all_models(model_dir: Path, specific_model: Optional[str] = None) -> Dict:
    """Validate all models in directory."""
    model_dir = Path(model_dir)

    if not model_dir.exists():
        print(f"Error: Model directory does not exist: {model_dir}")
        sys.exit(1)

    # Find model files
    model_files = []
    if specific_model:
        model_file = model_dir / f"{specific_model.lower()}.json"
        if model_file.exists():
            model_files.append(model_file)
        else:
            print(f"Error: Model file not found: {model_file}")
            sys.exit(1)
    else:
        # Find all model files
        for model_name in MODEL_SPECS.keys():
            model_file = model_dir / f"{model_name}.json"
            if model_file.exists():
                model_files.append(model_file)

    if not model_files:
        print(f"Error: No model files found in {model_dir}")
        sys.exit(1)

    # Validate each model
    results = {
        "valid": True,
        "models": {},
        "summary": {
            "total": len(model_files),
            "valid": 0,
            "invalid": 0
        }
    }

    all_errors = []

    for model_file in sorted(model_files):
        is_valid, errors, report = validate_model_file(model_file)
        results["models"][model_file.stem] = report

        if is_valid:
            results["summary"]["valid"] += 1
            print(f"✓ {report['metadata'].get('model_name', model_file.stem)}: Valid")
        else:
            results["summary"]["invalid"] += 1
            results["valid"] = False
            all_errors.extend(errors)
            print(f"✗ {report['metadata'].get('model_name', model_file.stem)}: Invalid")
            for error in errors:
                print(f"  - {error}")

    # Print summary
    print(f"\nSummary: {results['summary']['valid']}/{results['summary']['total']} models valid")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate RTNeural JSON models match C++ specifications"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory containing model JSON files"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_SPECS.keys()),
        help="Validate specific model only"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Save validation report to JSON file"
    )

    args = parser.parse_args()

    # Validate models
    results = validate_all_models(Path(args.model_dir), args.model)

    # Save report if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nValidation report saved to {args.json}")

    # Exit with error code if validation failed
    sys.exit(0 if results["valid"] else 1)


if __name__ == "__main__":
    main()
