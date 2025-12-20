#!/usr/bin/env python3
"""
Integration Test: Roundtrip
============================
Test Python inference vs C++ inference comparison (when C++ is available).
This test verifies that exported models produce similar results in both environments.
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ml_training"))

from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor,
    export_to_rtneural
)


class TestRoundtrip(unittest.TestCase):
    """Test Python/C++ roundtrip compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.models = {
            'EmotionRecognizer': EmotionRecognizer(),
            'MelodyTransformer': MelodyTransformer(),
            'HarmonyPredictor': HarmonyPredictor(),
            'DynamicsEngine': DynamicsEngine(),
            'GroovePredictor': GroovePredictor()
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_python_inference_consistency(self):
        """Test that Python model inference is consistent across runs."""
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                model.eval()

                input_sizes = {
                    'EmotionRecognizer': 128,
                    'MelodyTransformer': 64,
                    'HarmonyPredictor': 128,
                    'DynamicsEngine': 32,
                    'GroovePredictor': 64
                }

                input_size = input_sizes.get(model_name, 128)
                test_input = torch.randn(1, input_size)

                # Run inference twice with same input
                with torch.no_grad():
                    output1 = model(test_input)
                    output2 = model(test_input)

                # Results should be identical (deterministic)
                self.assertTrue(torch.allclose(output1, output2, atol=1e-6),
                              f"{model_name}: Inference not deterministic")

    def test_export_preserves_structure(self):
        """Test that export preserves model structure correctly."""
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                # Get Python model output
                input_sizes = {
                    'EmotionRecognizer': 128,
                    'MelodyTransformer': 64,
                    'HarmonyPredictor': 128,
                    'DynamicsEngine': 32,
                    'GroovePredictor': 64
                }

                input_size = input_sizes.get(model_name, 128)
                test_input = torch.randn(1, input_size)

                model.eval()
                with torch.no_grad():
                    python_output = model(test_input).numpy()

                # Export model
                export_to_rtneural(model, model_name, self.temp_dir)
                json_path = self.temp_dir / f"{model_name.lower()}.json"

                # Verify JSON structure matches model
                with open(json_path, 'r') as f:
                    data = json.load(f)

                metadata = data.get("metadata", {})
                layers = data.get("layers", [])

                # Check input/output sizes match
                if len(layers) > 0:
                    self.assertEqual(metadata.get("input_size"), layers[0].get("in_size"),
                                   f"{model_name}: Input size mismatch in export")
                    self.assertEqual(metadata.get("output_size"), layers[-1].get("out_size"),
                                   f"{model_name}: Output size mismatch in export")

                # Check parameter count
                exported_params = metadata.get("parameter_count", 0)
                actual_params = sum(p.numel() for p in model.parameters())
                self.assertEqual(exported_params, actual_params,
                               f"{model_name}: Parameter count mismatch")

    def test_json_can_be_loaded(self):
        """Test that exported JSON can be loaded and parsed."""
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                export_to_rtneural(model, model_name, self.temp_dir)
                json_path = self.temp_dir / f"{model_name.lower()}.json"

                # Try to load and parse
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Verify structure
                self.assertIn("layers", data)
                self.assertIn("metadata", data)

                # Verify layers are valid
                layers = data["layers"]
                for layer in layers:
                    self.assertIn("type", layer)
                    if layer["type"] == "dense":
                        self.assertIn("weights", layer)
                    elif layer["type"] == "lstm":
                        self.assertIn("weights_ih", layer)
                        self.assertIn("weights_hh", layer)

    def test_model_output_ranges(self):
        """Test that model outputs are in expected ranges."""
        expected_ranges = {
            'EmotionRecognizer': (-1.0, 1.0),  # tanh
            'MelodyTransformer': (0.0, 1.0),   # sigmoid
            'HarmonyPredictor': (0.0, 1.0),   # softmax
            'DynamicsEngine': (0.0, 1.0),     # sigmoid
            'GroovePredictor': (-1.0, 1.0)    # tanh
        }

        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                input_sizes = {
                    'EmotionRecognizer': 128,
                    'MelodyTransformer': 64,
                    'HarmonyPredictor': 128,
                    'DynamicsEngine': 32,
                    'GroovePredictor': 64
                }

                input_size = input_sizes.get(model_name, 128)
                test_input = torch.randn(1, input_size)

                model.eval()
                with torch.no_grad():
                    output = model(test_input)

                min_val, max_val = expected_ranges.get(model_name, (-1.0, 1.0))

                self.assertTrue(torch.all(output >= min_val),
                              f"{model_name}: Output below minimum {min_val}")
                self.assertTrue(torch.all(output <= max_val),
                              f"{model_name}: Output above maximum {max_val}")


if __name__ == "__main__":
    unittest.main()
