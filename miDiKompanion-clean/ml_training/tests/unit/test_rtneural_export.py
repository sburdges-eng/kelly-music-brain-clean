#!/usr/bin/env python3
"""
Unit Tests: RTNeural Export
============================
Test RTNeural export format correctness, LSTM weight splitting, and JSON structure.
"""

import unittest
import json
import torch
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor,
    export_to_rtneural
)


class TestRTNeuralExport(unittest.TestCase):
    """Test RTNeural export functionality."""

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

    def test_export_produces_valid_json(self):
        """Test that export produces valid JSON files."""
        for model_name, model in self.models.items():
            export_to_rtneural(model, model_name, self.temp_dir)
            json_path = self.temp_dir / f"{model_name.lower()}.json"

            self.assertTrue(json_path.exists(), f"{model_name}: JSON file not created")

            # Try to parse JSON
            with open(json_path, 'r') as f:
                data = json.load(f)

            self.assertIsInstance(data, dict, f"{model_name}: JSON is not a dict")

    def test_export_has_required_keys(self):
        """Test that exported JSON has required keys."""
        for model_name, model in self.models.items():
            export_to_rtneural(model, model_name, self.temp_dir)
            json_path = self.temp_dir / f"{model_name.lower()}.json"

            with open(json_path, 'r') as f:
                data = json.load(f)

            self.assertIn("layers", data, f"{model_name}: Missing 'layers' key")
            self.assertIn("metadata", data, f"{model_name}: Missing 'metadata' key")
            self.assertIsInstance(data["layers"], list, f"{model_name}: 'layers' is not a list")
            self.assertGreater(len(data["layers"]), 0, f"{model_name}: 'layers' is empty")

    def test_metadata_structure(self):
        """Test that metadata has required fields."""
        for model_name, model in self.models.items():
            export_to_rtneural(model, model_name, self.temp_dir)
            json_path = self.temp_dir / f"{model_name.lower()}.json"

            with open(json_path, 'r') as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            required_fields = ["model_name", "framework", "export_version",
                            "parameter_count", "input_size", "output_size"]

            for field in required_fields:
                self.assertIn(field, metadata,
                            f"{model_name}: Missing metadata field '{field}'")

    def test_layer_structure(self):
        """Test that layers have correct structure."""
        for model_name, model in self.models.items():
            export_to_rtneural(model, model_name, self.temp_dir)
            json_path = self.temp_dir / f"{model_name.lower()}.json"

            with open(json_path, 'r') as f:
                data = json.load(f)

            layers = data.get("layers", [])
            for i, layer in enumerate(layers):
                self.assertIn("type", layer,
                            f"{model_name} layer {i}: Missing 'type'")
                self.assertIn(layer["type"], ["dense", "lstm"],
                            f"{model_name} layer {i}: Invalid type '{layer.get('type')}'")

                if layer["type"] == "dense":
                    required = ["in_size", "out_size", "activation", "weights"]
                    for key in required:
                        self.assertIn(key, layer,
                                    f"{model_name} layer {i} (dense): Missing '{key}'")

                elif layer["type"] == "lstm":
                    required = ["in_size", "out_size", "weights_ih", "weights_hh"]
                    for key in required:
                        self.assertIn(key, layer,
                                    f"{model_name} layer {i} (lstm): Missing '{key}'")

    def test_lstm_weight_splitting(self):
        """Test that LSTM weights are properly split into 4 gates."""
        # EmotionRecognizer and MelodyTransformer have LSTM layers
        for model_name in ["EmotionRecognizer", "MelodyTransformer"]:
            model = self.models[model_name]
            export_to_rtneural(model, model_name, self.temp_dir)
            json_path = self.temp_dir / f"{model_name.lower()}.json"

            with open(json_path, 'r') as f:
                data = json.load(f)

            layers = data.get("layers", [])
            lstm_layers = [l for l in layers if l.get("type") == "lstm"]

            for lstm_layer in lstm_layers:
                # Check weights_ih has 4 gates
                weights_ih = lstm_layer.get("weights_ih", [])
                self.assertEqual(len(weights_ih), 4,
                               f"{model_name}: weights_ih should have 4 gates, got {len(weights_ih)}")

                # Check weights_hh has 4 gates
                weights_hh = lstm_layer.get("weights_hh", [])
                self.assertEqual(len(weights_hh), 4,
                               f"{model_name}: weights_hh should have 4 gates, got {len(weights_hh)}")

                # Check bias_ih has 4 gates (if present)
                if "bias_ih" in lstm_layer:
                    bias_ih = lstm_layer["bias_ih"]
                    if isinstance(bias_ih, list) and len(bias_ih) > 0:
                        self.assertEqual(len(bias_ih), 4,
                                       f"{model_name}: bias_ih should have 4 gates, got {len(bias_ih)}")

                # Check bias_hh has 4 gates (if present)
                if "bias_hh" in lstm_layer:
                    bias_hh = lstm_layer["bias_hh"]
                    if isinstance(bias_hh, list) and len(bias_hh) > 0:
                        self.assertEqual(len(bias_hh), 4,
                                       f"{model_name}: bias_hh should have 4 gates, got {len(bias_hh)}")

    def test_dense_layer_weights_shape(self):
        """Test that dense layer weights have correct shape."""
        for model_name, model in self.models.items():
            export_to_rtneural(model, model_name, self.temp_dir)
            json_path = self.temp_dir / f"{model_name.lower()}.json"

            with open(json_path, 'r') as f:
                data = json.load(f)

            layers = data.get("layers", [])
            for i, layer in enumerate(layers):
                if layer.get("type") == "dense":
                    in_size = layer.get("in_size", 0)
                    out_size = layer.get("out_size", 0)
                    weights = layer.get("weights", [])

                    self.assertEqual(len(weights), out_size,
                                   f"{model_name} layer {i}: Weight rows should be {out_size}, got {len(weights)}")

                    if len(weights) > 0:
                        self.assertEqual(len(weights[0]), in_size,
                                       f"{model_name} layer {i}: Weight cols should be {in_size}, got {len(weights[0])}")

    def test_activation_functions(self):
        """Test that activation functions are correctly specified."""
        expected_activations = {
            "EmotionRecognizer": ["tanh", "tanh", None, "tanh"],  # LSTM has no activation
            "MelodyTransformer": ["relu", None, "relu", "sigmoid"],  # LSTM has no activation
            "HarmonyPredictor": ["tanh", "tanh", "softmax"],
            "DynamicsEngine": ["relu", "relu", "sigmoid"],
            "GroovePredictor": ["tanh", "tanh", "tanh"]
        }

        for model_name, model in self.models.items():
            export_to_rtneural(model, model_name, self.temp_dir)
            json_path = self.temp_dir / f"{model_name.lower()}.json"

            with open(json_path, 'r') as f:
                data = json.load(f)

            layers = data.get("layers", [])
            dense_layers = [l for l in layers if l.get("type") == "dense"]
            expected = expected_activations.get(model_name, [])

            for i, dense_layer in enumerate(dense_layers):
                if i < len(expected) and expected[i] is not None:
                    activation = dense_layer.get("activation")
                    self.assertIn(activation, ["linear", "tanh", "relu", "sigmoid", "softmax"],
                                f"{model_name} layer {i}: Invalid activation '{activation}'")

    def test_parameter_count_matches(self):
        """Test that exported parameter count matches model."""
        for model_name, model in self.models.items():
            actual_params = sum(p.numel() for p in model.parameters())

            export_to_rtneural(model, model_name, self.temp_dir)
            json_path = self.temp_dir / f"{model_name.lower()}.json"

            with open(json_path, 'r') as f:
                data = json.load(f)

            exported_params = data.get("metadata", {}).get("parameter_count", 0)

            self.assertEqual(exported_params, actual_params,
                           f"{model_name}: Parameter count mismatch: model={actual_params}, exported={exported_params}")

    def test_input_output_sizes_match(self):
        """Test that input/output sizes in metadata match first/last layers."""
        for model_name, model in self.models.items():
            export_to_rtneural(model, model_name, self.temp_dir)
            json_path = self.temp_dir / f"{model_name.lower()}.json"

            with open(json_path, 'r') as f:
                data = json.load(f)

            layers = data.get("layers", [])
            metadata = data.get("metadata", {})

            if len(layers) > 0:
                first_layer = layers[0]
                last_layer = layers[-1]

                self.assertEqual(metadata.get("input_size"), first_layer.get("in_size"),
                               f"{model_name}: Input size mismatch")
                self.assertEqual(metadata.get("output_size"), last_layer.get("out_size"),
                               f"{model_name}: Output size mismatch")


if __name__ == "__main__":
    unittest.main()
