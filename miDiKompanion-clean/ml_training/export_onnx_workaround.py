#!/usr/bin/env python3
"""
ONNX Export Workaround - For Python 3.14 compatibility issues
=============================================================
Attempts to export models using alternative methods when onnxscript is unavailable.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Try to import ONNX
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available")

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
    print("Error: Could not import model definitions")
    sys.exit(1)


def export_with_torchscript(model, checkpoint_path, output_path):
    """Export model as TorchScript (alternative to ONNX)."""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Get input shape from model
        if hasattr(model, 'fc1'):
            input_size = model.fc1.in_features
        else:
            input_size = 128  # Default

        # Create dummy input
        dummy_input = torch.randn(1, input_size)

        # Export as TorchScript
        traced = torch.jit.trace(model, dummy_input)
        traced.save(str(output_path).replace('.onnx', '.pt'))

        print(f"  ✓ Exported as TorchScript: {output_path.replace('.onnx', '.pt')}")
        return True
    except Exception as e:
        print(f"  ✗ TorchScript export failed: {e}")
        return False


def main():
    """Export models using available methods."""
    import argparse

    parser = argparse.ArgumentParser(description="Export models (ONNX or TorchScript)")
    parser.add_argument("--models-dir", type=str, default="trained_models/checkpoints")
    parser.add_argument("--output-dir", type=str, default="trained_models/onnx")

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_configs = {
        'EmotionRecognizer': (EmotionRecognizer, 'EmotionRecognizer_best.pt', 128),
        'MelodyTransformer': (MelodyTransformer, 'MelodyTransformer_best.pt', 64),
        'HarmonyPredictor': (HarmonyPredictor, 'HarmonyPredictor_best.pt', 128),
        'DynamicsEngine': (DynamicsEngine, 'DynamicsEngine_best.pt', 32),
        'GroovePredictor': (GroovePredictor, 'GroovePredictor_best.pt', 64),
    }

    print("=" * 60)
    print("Model Export (Workaround)")
    print("=" * 60)

    for model_name, (model_class, checkpoint_name, input_size) in model_configs.items():
        checkpoint_path = models_dir / checkpoint_name

        if not checkpoint_path.exists():
            print(f"\n⚠ {model_name}: Checkpoint not found")
            continue

        print(f"\n{model_name}:")

        # Try ONNX first
        onnx_path = output_dir / f"{model_name.lower()}.onnx"
        onnx_success = False

        if ONNX_AVAILABLE:
            try:
                # Try standard ONNX export
                model = model_class()
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                model.eval()
                dummy_input = torch.randn(1, input_size)

                # Try export (may fail if onnxscript missing)
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    verbose=False
                )

                # Validate
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)

                print(f"  ✓ ONNX export successful: {onnx_path}")
                onnx_success = True
            except Exception as e:
                print(f"  ⚠ ONNX export failed: {e}")
                print(f"  → Falling back to TorchScript")

        # Fallback to TorchScript
        if not onnx_success:
            model = model_class()
            ts_path = output_dir / f"{model_name.lower()}.pt"
            export_with_torchscript(model, checkpoint_path, str(ts_path))

    print("\n" + "=" * 60)
    print("Export Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
