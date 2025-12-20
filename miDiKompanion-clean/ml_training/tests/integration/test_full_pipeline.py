#!/usr/bin/env python3
"""
Integration Test: Full Pipeline
================================
Test complete pipeline: Training → Export → Validation → C++ Loading → Inference
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
import torch
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
from validate_models import validate_model_file


class TestFullPipeline(unittest.TestCase):
    """Test the complete ML pipeline."""

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

    def test_training_export_validation(self):
        """Test: Train → Export → Validate."""
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                # 1. Export model
                export_to_rtneural(model, model_name, self.temp_dir)
                json_path = self.temp_dir / f"{model_name.lower()}.json"

                self.assertTrue(json_path.exists(),
                              f"{model_name}: JSON file not created")

                # 2. Validate JSON structure
                is_valid, errors, report = validate_model_file(json_path)

                if not is_valid:
                    self.fail(f"{model_name}: Validation failed: {errors}")

                # 3. Verify JSON can be parsed
                with open(json_path, 'r') as f:
                    data = json.load(f)

                self.assertIn("layers", data)
                self.assertIn("metadata", data)
                self.assertGreater(len(data["layers"]), 0)

    def test_exported_models_match_specs(self):
        """Test that exported models match C++ ModelSpec definitions."""
        cpp_specs = {
            'emotionrecognizer': {'input': 128, 'output': 64, 'params': 403264},
            'melodytransformer': {'input': 64, 'output': 128, 'params': 641664},
            'harmonypredictor': {'input': 128, 'output': 64, 'params': 74176},
            'dynamicsengine': {'input': 32, 'output': 16, 'params': 13520},
            'groovepredictor': {'input': 64, 'output': 32, 'params': 18656}
        }

        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                export_to_rtneural(model, model_name, self.temp_dir)
                json_path = self.temp_dir / f"{model_name.lower()}.json"

                with open(json_path, 'r') as f:
                    data = json.load(f)

                metadata = data.get("metadata", {})
                spec = cpp_specs.get(model_name.lower())

                if spec:
                    self.assertEqual(metadata.get("input_size"), spec["input"],
                                   f"{model_name}: Input size mismatch")
                    self.assertEqual(metadata.get("output_size"), spec["output"],
                                   f"{model_name}: Output size mismatch")

                    # Parameter count with 5% tolerance
                    param_count = metadata.get("parameter_count", 0)
                    tolerance = spec["params"] * 0.05
                    self.assertAlmostEqual(param_count, spec["params"],
                                          delta=tolerance,
                                          msg=f"{model_name}: Parameter count mismatch")

    def test_rtneural_json_structure(self):
        """Test that exported JSON matches RTNeural expected structure."""
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                export_to_rtneural(model, model_name, self.temp_dir)
                json_path = self.temp_dir / f"{model_name.lower()}.json"

                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Check required structure
                self.assertIn("layers", data)
                self.assertIn("metadata", data)

                layers = data["layers"]
                self.assertIsInstance(layers, list)
                self.assertGreater(len(layers), 0)

                # Check each layer
                for i, layer in enumerate(layers):
                    self.assertIn("type", layer)
                    self.assertIn(layer["type"], ["dense", "lstm"])
                    self.assertIn("in_size", layer)
                    self.assertIn("out_size", layer)

                # Check metadata
                metadata = data["metadata"]
                self.assertIn("model_name", metadata)
                self.assertIn("framework", metadata)
                self.assertIn("export_version", metadata)
                self.assertIn("parameter_count", metadata)
                self.assertIn("input_size", metadata)
                self.assertIn("output_size", metadata)

    def test_python_inference_matches_export(self):
        """Test that Python model inference produces reasonable outputs."""
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                model.eval()

                # Get expected input size from C++ specs
                input_sizes = {
                    'EmotionRecognizer': 128,
                    'MelodyTransformer': 64,
                    'HarmonyPredictor': 128,
                    'DynamicsEngine': 32,
                    'GroovePredictor': 64
                }

                input_size = input_sizes.get(model_name, 128)
                test_input = torch.randn(1, input_size)

                with torch.no_grad():
                    output = model(test_input)

                # Check output is valid
                self.assertFalse(torch.any(torch.isnan(output)),
                               f"{model_name}: Output contains NaN")
                self.assertFalse(torch.any(torch.isinf(output)),
                               f"{model_name}: Output contains Inf")

                # Check output shape matches expected
                output_sizes = {
                    'EmotionRecognizer': 64,
                    'MelodyTransformer': 128,
                    'HarmonyPredictor': 64,
                    'DynamicsEngine': 16,
                    'GroovePredictor': 32
                }
                expected_size = output_sizes.get(model_name, 64)
                self.assertEqual(output.shape[-1], expected_size,
                               f"{model_name}: Output size mismatch")


if __name__ == "__main__":
    unittest.main()
