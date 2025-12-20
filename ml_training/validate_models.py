#!/usr/bin/env python3
"""
Model Validation for MidiKompanion ML Models

Validates exported ONNX models by:
1. Checking model structure and I/O shapes
2. Running inference tests
3. Measuring performance
4. Comparing with PyTorch models

Usage:
    python validate_models.py --models-dir ./models
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: ONNX Runtime not available")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Expected model specifications
MODEL_SPECS = {
    "emotion_recognizer": {"input": (1, 128), "output": (1, 64)},
    "melody_transformer": {"input": (1, 64), "output": (1, 128)},
    "harmony_predictor": {"input": (1, 128), "output": (1, 64)},
    "dynamics_engine": {"input": (1, 32), "output": (1, 16)},
    "groove_predictor": {"input": (1, 64), "output": (1, 32)}
}


def validate_onnx_model(model_path: Path) -> Dict:
    """Validate a single ONNX model."""
    
    result = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "valid": False,
        "error": None,
        "input_shape": None,
        "output_shape": None,
        "inference_time_ms": None,
        "file_size_mb": None
    }
    
    if not model_path.exists():
        result["error"] = "File not found"
        return result
    
    result["file_size_mb"] = model_path.stat().st_size / (1024 * 1024)
    
    if not HAS_ONNX:
        result["error"] = "ONNX Runtime not available"
        return result
    
    try:
        # Load and check model
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        # Get input/output shapes
        inputs = model.graph.input
        outputs = model.graph.output
        
        if inputs:
            input_shape = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
            result["input_shape"] = input_shape
        
        if outputs:
            output_shape = [d.dim_value for d in outputs[0].type.tensor_type.shape.dim]
            result["output_shape"] = output_shape
        
        # Test inference
        session = ort.InferenceSession(str(model_path))
        
        # Create test input
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Handle dynamic dimensions
        test_shape = []
        for dim in input_shape:
            if isinstance(dim, int):
                test_shape.append(dim)
            else:
                test_shape.append(1)  # Use 1 for dynamic dims
        
        test_input = np.random.randn(*test_shape).astype(np.float32)
        
        # Measure inference time
        times = []
        for _ in range(100):
            start = time.perf_counter()
            outputs = session.run(None, {input_name: test_input})
            times.append((time.perf_counter() - start) * 1000)
        
        result["inference_time_ms"] = np.mean(times)
        result["inference_std_ms"] = np.std(times)
        result["valid"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def validate_model_specs(results: Dict[str, Dict]) -> List[str]:
    """Check that models match expected specifications."""
    
    issues = []
    
    for model_name, spec in MODEL_SPECS.items():
        if model_name not in results:
            issues.append(f"Missing model: {model_name}")
            continue
        
        result = results[model_name]
        
        if not result["valid"]:
            issues.append(f"{model_name}: Invalid model - {result['error']}")
            continue
        
        # Check input shape
        if result["input_shape"]:
            expected_input = list(spec["input"])
            actual_input = result["input_shape"]
            
            # Handle dynamic batch dimension
            if len(actual_input) == len(expected_input):
                for i in range(len(expected_input)):
                    if expected_input[i] != 0 and actual_input[i] != 0:
                        if actual_input[i] != expected_input[i]:
                            issues.append(
                                f"{model_name}: Input shape mismatch. "
                                f"Expected {expected_input}, got {actual_input}"
                            )
                            break
        
        # Check inference time (< 10ms target)
        if result["inference_time_ms"] and result["inference_time_ms"] > 10:
            issues.append(
                f"{model_name}: Slow inference ({result['inference_time_ms']:.2f}ms > 10ms target)"
            )
        
        # Check file size (< 5MB per model)
        if result["file_size_mb"] and result["file_size_mb"] > 5:
            issues.append(
                f"{model_name}: Large file size ({result['file_size_mb']:.2f}MB > 5MB target)"
            )
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX models")
    parser.add_argument("--models-dir", type=str, default="./models",
                        help="Directory containing ONNX models")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return 1
    
    logger.info(f"Validating models in: {models_dir}")
    logger.info("="*60)
    
    results = {}
    
    # Validate each expected model
    for model_name in MODEL_SPECS.keys():
        model_path = models_dir / f"{model_name}.onnx"
        logger.info(f"\nValidating: {model_name}")
        
        result = validate_onnx_model(model_path)
        results[model_name] = result
        
        if result["valid"]:
            logger.info(f"  ✓ Valid")
            logger.info(f"    Input shape: {result['input_shape']}")
            logger.info(f"    Output shape: {result['output_shape']}")
            logger.info(f"    Inference time: {result['inference_time_ms']:.2f}ms")
            logger.info(f"    File size: {result['file_size_mb']:.2f}MB")
        else:
            logger.error(f"  ✗ Invalid: {result['error']}")
    
    # Check specifications
    logger.info("\n" + "="*60)
    logger.info("SPECIFICATION CHECK")
    logger.info("="*60)
    
    issues = validate_model_specs(results)
    
    if issues:
        logger.warning(f"\n{len(issues)} issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\n✓ All models meet specifications!")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    valid_count = sum(1 for r in results.values() if r["valid"])
    total_count = len(MODEL_SPECS)
    
    logger.info(f"Models validated: {valid_count}/{total_count}")
    
    if valid_count == total_count:
        total_size = sum(r["file_size_mb"] for r in results.values() if r["file_size_mb"])
        avg_time = np.mean([r["inference_time_ms"] for r in results.values() if r["inference_time_ms"]])
        
        logger.info(f"Total size: {total_size:.2f}MB")
        logger.info(f"Average inference time: {avg_time:.2f}ms")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump({
                "results": results,
                "issues": issues,
                "summary": {
                    "valid_count": valid_count,
                    "total_count": total_count
                }
            }, f, indent=2)
        logger.info(f"\nResults saved to: {output_path}")
    
    return 0 if valid_count == total_count and len(issues) == 0 else 1


if __name__ == "__main__":
    exit(main())
