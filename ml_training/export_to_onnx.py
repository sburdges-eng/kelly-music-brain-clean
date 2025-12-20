#!/usr/bin/env python3
"""
ONNX Export Utility for MidiKompanion Models

Exports trained PyTorch models to ONNX format with optimization.

Usage:
    python export_to_onnx.py --input ./models --output ./onnx_models
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Error: PyTorch required for export")
    exit(1)

try:
    import onnx
    from onnxruntime import InferenceSession
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: ONNX not available for validation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import model definitions
from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor
)


MODEL_REGISTRY = {
    "emotion_recognizer": (EmotionRecognizer, 128, 64),
    "melody_transformer": (MelodyTransformer, 64, 128),
    "harmony_predictor": (HarmonyPredictor, 128, 64),
    "dynamics_engine": (DynamicsEngine, 32, 16),
    "groove_predictor": (GroovePredictor, 64, 32)
}


def export_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: Path,
    model_name: str,
    optimize: bool = True
) -> bool:
    """Export a PyTorch model to ONNX format."""
    
    logger.info(f"Exporting {model_name}...")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    if not HAS_ONNX:
        logger.warning("ONNX not available for validation, skipping checks")
        return True
    
    # Validate export
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Test inference matches
        session = InferenceSession(str(output_path))
        
        test_input = dummy_input.numpy()
        with torch.no_grad():
            torch_output = model(dummy_input).numpy()
        
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: test_input})[0]
        
        # Check outputs match
        diff = np.abs(torch_output - onnx_output).max()
        if diff > 1e-5:
            logger.warning(f"  Output mismatch: max diff = {diff}")
        
        logger.info(f"  ✓ Exported successfully: {output_path}")
        logger.info(f"    Size: {output_path.stat().st_size / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument("--input", type=str, default="./models",
                        help="Input directory with .pt files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (defaults to input)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to export (default: all)")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_export = [args.model] if args.model else list(MODEL_REGISTRY.keys())
    
    success_count = 0
    
    for model_name in models_to_export:
        if model_name not in MODEL_REGISTRY:
            logger.error(f"Unknown model: {model_name}")
            continue
        
        model_class, input_dim, output_dim = MODEL_REGISTRY[model_name]
        
        # Load PyTorch model
        pt_path = input_dir / f"{model_name}.pt"
        if not pt_path.exists():
            logger.warning(f"PyTorch model not found: {pt_path}")
            continue
        
        model = model_class(input_dim, output_dim)
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        
        # Export
        onnx_path = output_dir / f"{model_name}.onnx"
        if export_model(model, (input_dim,), onnx_path, model_name):
            success_count += 1
    
    logger.info(f"\nExported {success_count}/{len(models_to_export)} models")
    
    return 0 if success_count == len(models_to_export) else 1


if __name__ == "__main__":
    exit(main())
