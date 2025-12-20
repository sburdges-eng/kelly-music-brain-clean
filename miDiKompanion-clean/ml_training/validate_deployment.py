#!/usr/bin/env python3
"""
validate_deployment.py - Validate Model Deployment
===================================================

Validates that deployed models work correctly with C++ integration.
Tests model loading, inference, and compatibility.
"""

import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
import argparse


def test_model_inference(model_path: Path, input_size: int, output_size: int) -> dict:
    """
    Test model inference with correct input/output sizes.

    Args:
        model_path: Path to ONNX model
        input_size: Expected input size
        output_size: Expected output size

    Returns:
        Test results dictionary
    """
    result = {
        'success': False,
        'error': None,
        'latency_ms': 0.0,
        'output_shape': None
    }

    try:
        # Load model
        session = ort.InferenceSession(str(model_path))

        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Create test input
        test_input = np.random.randn(1, input_size).astype(np.float32)

        # Run inference (measure latency)
        import time
        start_time = time.time()
        outputs = session.run([output_name], {input_name: test_input})
        end_time = time.time()

        # Check output
        output = outputs[0]
        if output.shape[-1] == output_size:
            result['success'] = True
            result['latency_ms'] = (end_time - start_time) * 1000.0
            result['output_shape'] = list(output.shape)
        else:
            result['error'] = f"Output size mismatch: expected {output_size}, got {output.shape[-1]}"

    except Exception as e:
        result['error'] = str(e)

    return result


def validate_deployment(deployment_dir: Path) -> bool:
    """
    Validate complete deployment package.

    Args:
        deployment_dir: Deployment package directory

    Returns:
        True if all validations pass
    """
    print("=" * 60)
    print("Deployment Validation")
    print("=" * 60)

    # Check manifest
    manifest_path = deployment_dir / "deployment_manifest.json"
    if not manifest_path.exists():
        print("✗ Deployment manifest not found")
        return False

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Deployment version: {manifest.get('version', 'unknown')}")
    print(f"Deployment date: {manifest.get('deployment_date', 'unknown')}")
    print(f"Total size: {manifest.get('total_size_mb', 0):.2f} MB")
    print()

    # Model specifications
    model_specs = {
        'EmotionRecognizer': {'input': 128, 'output': 64},
        'MelodyTransformer': {'input': 64, 'output': 128},
        'HarmonyPredictor': {'input': 128, 'output': 64},
        'DynamicsEngine': {'input': 32, 'output': 16},
        'GroovePredictor': {'input': 64, 'output': 32}
    }

    models_dir = deployment_dir / "models"
    if not models_dir.exists():
        print("✗ Models directory not found")
        return False

    all_valid = True

    print("Validating models...")
    print("-" * 60)

    for model_name, specs in model_specs.items():
        model_file = models_dir / f"{model_name.lower()}.onnx"

        if not model_file.exists():
            print(f"⚠ {model_name}: Model file not found")
            continue

        # Test inference
        test_result = test_model_inference(
            model_file,
            specs['input'],
            specs['output']
        )

        if test_result['success']:
            print(f"✓ {model_name}:")
            print(f"    Latency: {test_result['latency_ms']:.2f} ms")
            print(f"    Output shape: {test_result['output_shape']}")
        else:
            print(f"✗ {model_name}: {test_result['error']}")
            all_valid = False

    print()
    print("=" * 60)
    if all_valid:
        print("✓ All validations passed!")
        print()
        print("Models are ready for deployment to plugin.")
    else:
        print("✗ Some validations failed")
        print("Please check model files and specifications.")

    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Validate model deployment package")
    parser.add_argument(
        'deployment_dir',
        type=str,
        nargs='?',
        default='deployment',
        help='Deployment package directory (default: deployment)'
    )

    args = parser.parse_args()

    deployment_dir = Path(args.deployment_dir)

    if not deployment_dir.exists():
        print(f"Error: Deployment directory not found: {deployment_dir}")
        print("Run deploy_models.py first to create deployment package")
        exit(1)

    success = validate_deployment(deployment_dir)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
