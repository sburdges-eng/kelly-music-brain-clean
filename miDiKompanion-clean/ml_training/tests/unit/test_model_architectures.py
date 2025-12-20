#!/usr/bin/env python3
"""
Unit Tests: Model Architectures
===============================
Verify all 5 models match C++ ModelSpec definitions exactly.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor
)

# C++ ModelSpec definitions (from MultiModelProcessor.h)
# Updated to match actual C++ header values
CPP_MODEL_SPECS = {
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


class TestModelArchitectures(unittest.TestCase):
    """Test that all models match C++ specifications."""

    def test_emotion_recognizer_specs(self):
        """Test EmotionRecognizer matches C++ specs."""
        model = EmotionRecognizer()
        spec = CPP_MODEL_SPECS["EmotionRecognizer"]

        # Test input/output dimensions
        test_input = torch.randn(1, spec["input"])
        with torch.no_grad():
            output = model(test_input)

        self.assertEqual(output.shape[-1], spec["output"],
                        f"Output size mismatch: expected {spec['output']}, got {output.shape[-1]}")

        # Test parameter count (±5% tolerance)
        param_count = sum(p.numel() for p in model.parameters())
        tolerance = spec["params"] * 0.05
        self.assertAlmostEqual(param_count, spec["params"], delta=tolerance,
                              msg=f"Parameter count mismatch: expected ~{spec['params']}, got {param_count}")

    def test_melody_transformer_specs(self):
        """Test MelodyTransformer matches C++ specs."""
        model = MelodyTransformer()
        spec = CPP_MODEL_SPECS["MelodyTransformer"]

        test_input = torch.randn(1, spec["input"])
        with torch.no_grad():
            output = model(test_input)

        self.assertEqual(output.shape[-1], spec["output"])

        param_count = sum(p.numel() for p in model.parameters())
        tolerance = spec["params"] * 0.05
        self.assertAlmostEqual(param_count, spec["params"], delta=tolerance)

    def test_harmony_predictor_specs(self):
        """Test HarmonyPredictor matches C++ specs."""
        model = HarmonyPredictor()
        spec = CPP_MODEL_SPECS["HarmonyPredictor"]

        test_input = torch.randn(1, spec["input"])
        with torch.no_grad():
            output = model(test_input)

        self.assertEqual(output.shape[-1], spec["output"])

        param_count = sum(p.numel() for p in model.parameters())
        tolerance = spec["params"] * 0.05
        self.assertAlmostEqual(param_count, spec["params"], delta=tolerance)

    def test_dynamics_engine_specs(self):
        """Test DynamicsEngine matches C++ specs."""
        model = DynamicsEngine()
        spec = CPP_MODEL_SPECS["DynamicsEngine"]

        test_input = torch.randn(1, spec["input"])
        with torch.no_grad():
            output = model(test_input)

        self.assertEqual(output.shape[-1], spec["output"])

        param_count = sum(p.numel() for p in model.parameters())
        tolerance = spec["params"] * 0.05
        self.assertAlmostEqual(param_count, spec["params"], delta=tolerance)

    def test_groove_predictor_specs(self):
        """Test GroovePredictor matches C++ specs."""
        model = GroovePredictor()
        spec = CPP_MODEL_SPECS["GroovePredictor"]

        test_input = torch.randn(1, spec["input"])
        with torch.no_grad():
            output = model(test_input)

        self.assertEqual(output.shape[-1], spec["output"])

        param_count = sum(p.numel() for p in model.parameters())
        tolerance = spec["params"] * 0.05
        self.assertAlmostEqual(param_count, spec["params"], delta=tolerance)

    def test_all_models_output_ranges(self):
        """Test that all models produce outputs in expected ranges."""
        test_cases = [
            (EmotionRecognizer(), torch.randn(1, 128), (-1.0, 1.0)),  # tanh output
            (MelodyTransformer(), torch.randn(1, 64), (0.0, 1.0)),    # sigmoid output
            (HarmonyPredictor(), torch.randn(1, 128), (0.0, 1.0)),   # softmax output
            (DynamicsEngine(), torch.randn(1, 32), (0.0, 1.0)),      # sigmoid output
            (GroovePredictor(), torch.randn(1, 64), (-1.0, 1.0))     # tanh output
        ]

        for model, test_input, (min_val, max_val) in test_cases:
            model.eval()
            with torch.no_grad():
                output = model(test_input)

            self.assertTrue(torch.all(output >= min_val),
                          f"{model.__class__.__name__}: Output below minimum {min_val}")
            self.assertTrue(torch.all(output <= max_val),
                          f"{model.__class__.__name__}: Output above maximum {max_val}")

    def test_batch_processing(self):
        """Test all models handle batch processing correctly."""
        batch_sizes = [1, 4, 16, 32]

        for model_name, model_class in MODELS.items():
            spec = CPP_MODEL_SPECS[model_name]
            model = model_class()
            model.eval()

            for batch_size in batch_sizes:
                test_input = torch.randn(batch_size, spec["input"])
                with torch.no_grad():
                    output = model(test_input)

                self.assertEqual(output.shape[0], batch_size,
                               f"{model_name}: Batch size mismatch")
                self.assertEqual(output.shape[-1], spec["output"],
                               f"{model_name}: Output size mismatch")

    def test_no_nan_or_inf(self):
        """Test that models never produce NaN or Inf values."""
        for model_name, model_class in MODELS.items():
            spec = CPP_MODEL_SPECS[model_name]
            model = model_class()
            model.eval()

            test_input = torch.randn(1, spec["input"])
            with torch.no_grad():
                output = model(test_input)

            self.assertFalse(torch.any(torch.isnan(output)),
                           f"{model_name}: Output contains NaN")
            self.assertFalse(torch.any(torch.isinf(output)),
                           f"{model_name}: Output contains Inf")


if __name__ == "__main__":
    unittest.main()
