#!/usr/bin/env python3
"""
export_to_onnx.py - Export Trained Models to ONNX Format
========================================================

Agent 2: ML Training Specialist (Week 3-6)
Purpose: Export trained PyTorch models to ONNX format for C++ inference.

This script:
1. Loads trained PyTorch models
2. Exports to ONNX format with optimization
3. Validates ONNX model structure
4. Packages models for distribution

ONNX Model Specifications (shared with Agent 3):
- EmotionRecognizer: Input [batch, 128] → Output [batch, 64]
- MelodyTransformer: Input [batch, 64] → Output [batch, 128]
- HarmonyPredictor: Input [batch, 128] → Output [batch, 64]
- DynamicsEngine: Input [batch, 32] → Output [batch, 16]
- GroovePredictor: Input [batch, 64] → Output [batch, 32]
"""

import torch
import torch.onnx
import onnx
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    print("Note: onnxruntime not available, skipping runtime validation")
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import json

# Import model definitions
try:
    from train_all_models import (
        EmotionRecognizer,
        MelodyTransformer,
        HarmonyPredictor,
        DynamicsEngine,
        GroovePredictor
    )
except ImportError:
    print("Error: Could not import model definitions from train_all_models.py")
    print("Make sure train_all_models.py is in the same directory")
    exit(1)


# Model input/output specifications
MODEL_SPECS = {
    'EmotionRecognizer': {
        'input_shape': (1, 128),  # (batch, features)
        'output_shape': (1, 64),
        'input_names': ['audio_features'],
        'output_names': ['emotion_embedding']
    },
    'MelodyTransformer': {
        'input_shape': (1, 64),
        'output_shape': (1, 128),
        'input_names': ['emotion_embedding'],
        'output_names': ['midi_notes']
    },
    'HarmonyPredictor': {
        'input_shape': (1, 128),
        'output_shape': (1, 64),
        'input_names': ['context'],
        'output_names': ['chord_probs']
    },
    'DynamicsEngine': {
        'input_shape': (1, 32),
        'output_shape': (1, 16),
        'input_names': ['context'],
        'output_names': ['expression_params']
    },
    'GroovePredictor': {
        'input_shape': (1, 64),
        'output_shape': (1, 32),
        'input_names': ['emotion_embedding'],
        'output_names': ['groove_params']
    }
}


def export_model_to_onnx(
    model: torch.nn.Module,
    model_name: str,
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 14
) -> bool:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model instance
        checkpoint_path: Path to model checkpoint (.pth file)
        output_path: Path to save ONNX model (.onnx file)
        opset_version: ONNX opset version (default: 14)

    Returns:
        True if export successful, False otherwise
    """
    print(f"\nExporting {model_name} to ONNX...")

    # Load checkpoint
    if not checkpoint_path.exists():
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"  ✓ Loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"  ✗ Failed to load checkpoint: {e}")
        return False

    # Get model specs
    if model_name not in MODEL_SPECS:
        print(f"  ✗ Unknown model: {model_name}")
        return False

    specs = MODEL_SPECS[model_name]
    input_shape = specs['input_shape']
    input_names = specs['input_names']
    output_names = specs['output_names']

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,  # Optimize constants
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                input_names[0]: {0: 'batch_size'},  # Allow variable batch size
                output_names[0]: {0: 'batch_size'}
            },
            verbose=False
        )
        print(f"  ✓ Exported to: {output_path}")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate ONNX model
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model validation passed")
    except Exception as e:
        print(f"  ✗ ONNX validation failed: {e}")
        return False

    # Test inference with ONNX Runtime (optional)
    if ONNXRUNTIME_AVAILABLE:
        try:
            ort_session = ort.InferenceSession(str(output_path))

            # Get input/output names
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name

            # Run inference
            test_input = np.random.randn(*input_shape).astype(np.float32)
            outputs = ort_session.run([output_name], {input_name: test_input})

            expected_output_shape = specs['output_shape']
            if outputs[0].shape == expected_output_shape:
                print(f"  ✓ ONNX Runtime inference test passed")
                print(f"    Input shape: {test_input.shape}")
                print(f"    Output shape: {outputs[0].shape}")
            else:
                print(f"  ✗ Output shape mismatch: expected {expected_output_shape}, got {outputs[0].shape}")
                return False
        except Exception as e:
            print(f"  ⚠ ONNX Runtime test failed (non-critical): {e}")
            # Don't fail export if runtime test fails
    else:
        print(f"  ⚠ ONNX Runtime not available, skipping inference test")

    return True


def export_all_models(
    models_dir: Path,
    output_dir: Path,
    opset_version: int = 14
) -> Dict[str, bool]:
    """
    Export all trained models to ONNX format.

    Args:
        models_dir: Directory containing model checkpoints
        output_dir: Directory to save ONNX models
        opset_version: ONNX opset version

    Returns:
        Dictionary mapping model names to export success status
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model class mapping
    model_classes = {
        'EmotionRecognizer': EmotionRecognizer,
        'MelodyTransformer': MelodyTransformer,
        'HarmonyPredictor': HarmonyPredictor,
        'DynamicsEngine': DynamicsEngine,
        'GroovePredictor': GroovePredictor
    }

    results = {}

    for model_name, model_class in model_classes.items():
        # Look for checkpoint file (try multiple naming conventions)
        checkpoint_path = None

        # Try various naming patterns
        patterns = [
            f"{model_name.lower()}.pth",
            f"{model_name.lower()}.pt",
            f"{model_name}_best.pth",
            f"{model_name}_best.pt",
            f"{model_name}_latest.pth",
            f"{model_name}_latest.pt",
        ]

        for pattern in patterns:
            test_path = models_dir / pattern
            if test_path.exists():
                checkpoint_path = test_path
                break

        # Also check in checkpoints subdirectory
        if checkpoint_path is None:
            for pattern in patterns:
                test_path = models_dir / "checkpoints" / pattern
                if test_path.exists():
                    checkpoint_path = test_path
                    break

        if checkpoint_path is None:
            print(f"\n⚠ {model_name}: No checkpoint found, skipping")
            print(f"   Looked for: {', '.join(patterns)}")
            results[model_name] = False
            continue

        print(f"\n✓ {model_name}: Found checkpoint: {checkpoint_path.name}")

        # Create model instance
        model = model_class()

        # Export to ONNX
        onnx_path = output_dir / f"{model_name.lower()}.onnx"
        success = export_model_to_onnx(
            model,
            model_name,
            checkpoint_path,
            onnx_path,
            opset_version
        )

        results[model_name] = success

    return results


def create_model_metadata(output_dir: Path) -> None:
    """
    Create metadata file for exported ONNX models.

    Args:
        output_dir: Directory containing ONNX models
    """
    metadata = {
        'format': 'ONNX',
        'opset_version': 14,
        'models': {}
    }

    for model_name, specs in MODEL_SPECS.items():
        onnx_path = output_dir / f"{model_name.lower()}.onnx"
        if onnx_path.exists():
            # Get model file size
            file_size = onnx_path.stat().st_size

            metadata['models'][model_name] = {
                'input_shape': specs['input_shape'],
                'output_shape': specs['output_shape'],
                'input_names': specs['input_names'],
                'output_names': specs['output_names'],
                'file': onnx_path.name,
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2)
            }

    metadata_path = output_dir / "onnx_models_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Created metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained models to ONNX format")
    parser.add_argument(
        '--models-dir',
        type=str,
        default='trained_models/checkpoints',
        help='Directory containing model checkpoints (default: trained_models/checkpoints)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='trained_models/onnx',
        help='Directory to save ONNX models (default: trained_models/onnx)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['EmotionRecognizer', 'MelodyTransformer', 'HarmonyPredictor', 'DynamicsEngine', 'GroovePredictor'],
        help='Export specific model only (default: export all)'
    )
    parser.add_argument(
        '--opset-version',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        print("Please train models first using train_all_models.py")
        exit(1)

    print("=" * 60)
    print("ONNX Model Export")
    print("=" * 60)
    print(f"Models directory: {models_dir}")
    print(f"Output directory: {output_dir}")
    print(f"ONNX opset version: {args.opset_version}")

    if args.model:
        # Export single model
        model_classes = {
            'EmotionRecognizer': EmotionRecognizer,
            'MelodyTransformer': MelodyTransformer,
            'HarmonyPredictor': HarmonyPredictor,
            'DynamicsEngine': DynamicsEngine,
            'GroovePredictor': GroovePredictor
        }

        model_class = model_classes[args.model]
        model = model_class()

        checkpoint_path = models_dir / f"{args.model.lower()}.pth"
        if not checkpoint_path.exists():
            checkpoint_path = models_dir / f"{args.model}_best.pth"

        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found for {args.model}")
            exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = output_dir / f"{args.model.lower()}.onnx"

        success = export_model_to_onnx(
            model,
            args.model,
            checkpoint_path,
            onnx_path,
            args.opset_version
        )

        if not success:
            exit(1)
    else:
        # Export all models
        results = export_all_models(models_dir, output_dir, args.opset_version)

        # Summary
        print("\n" + "=" * 60)
        print("Export Summary")
        print("=" * 60)

        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]

        if successful:
            print(f"\n✓ Successfully exported {len(successful)} model(s):")
            for name in successful:
                print(f"  - {name}")

        if failed:
            print(f"\n✗ Failed to export {len(failed)} model(s):")
            for name in failed:
                print(f"  - {name}")

        # Create metadata
        if successful:
            create_model_metadata(output_dir)

        if failed:
            exit(1)


if __name__ == "__main__":
    main()
