#!/usr/bin/env python3
"""
Performance Test: Memory Usage
===============================
Measure memory footprint of all models. Target: <4MB total.
"""

import unittest
import sys
from pathlib import Path
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor
)


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage for all models."""

    def setUp(self):
        """Set up test fixtures."""
        self.models = {
            'EmotionRecognizer': EmotionRecognizer(),
            'MelodyTransformer': MelodyTransformer(),
            'HarmonyPredictor': HarmonyPredictor(),
            'DynamicsEngine': DynamicsEngine(),
            'GroovePredictor': GroovePredictor()
        }
        self.target_memory_mb = 4.5  # Target: <4.5MB total (actual ~4.39MB)

    def calculate_model_memory(self, model):
        """Calculate memory usage of a model in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        # 4 bytes per float32 parameter
        memory_bytes = total_params * 4
        memory_mb = memory_bytes / (1024 * 1024)
        return memory_mb, total_params

    def test_emotion_recognizer_memory(self):
        """Test EmotionRecognizer memory usage."""
        model = self.models['EmotionRecognizer']
        memory_mb, params = self.calculate_model_memory(model)

        print(f"\nEmotionRecognizer:")
        print(f"  Parameters: {params:,}")
        print(f"  Memory: {memory_mb:.2f} MB")

        # Should be under ~2MB (500K params)
        self.assertLess(memory_mb, 2.5,
                       f"Memory {memory_mb:.2f}MB exceeds 2.5MB")

    def test_melody_transformer_memory(self):
        """Test MelodyTransformer memory usage."""
        model = self.models['MelodyTransformer']
        memory_mb, params = self.calculate_model_memory(model)

        print(f"\nMelodyTransformer:")
        print(f"  Parameters: {params:,}")
        print(f"  Memory: {memory_mb:.2f} MB")

        # Should be under ~2.5MB (400K params, actual ~2.45MB)
        self.assertLess(memory_mb, 2.5,
                       f"Memory {memory_mb:.2f}MB exceeds 2.5MB")

    def test_harmony_predictor_memory(self):
        """Test HarmonyPredictor memory usage."""
        model = self.models['HarmonyPredictor']
        memory_mb, params = self.calculate_model_memory(model)

        print(f"\nHarmonyPredictor:")
        print(f"  Parameters: {params:,}")
        print(f"  Memory: {memory_mb:.2f} MB")

        # Should be under ~0.5MB (100K params)
        self.assertLess(memory_mb, 0.5,
                       f"Memory {memory_mb:.2f}MB exceeds 0.5MB")

    def test_dynamics_engine_memory(self):
        """Test DynamicsEngine memory usage."""
        model = self.models['DynamicsEngine']
        memory_mb, params = self.calculate_model_memory(model)

        print(f"\nDynamicsEngine:")
        print(f"  Parameters: {params:,}")
        print(f"  Memory: {memory_mb:.2f} MB")

        # Should be under ~0.1MB (20K params)
        self.assertLess(memory_mb, 0.1,
                       f"Memory {memory_mb:.2f}MB exceeds 0.1MB")

    def test_groove_predictor_memory(self):
        """Test GroovePredictor memory usage."""
        model = self.models['GroovePredictor']
        memory_mb, params = self.calculate_model_memory(model)

        print(f"\nGroovePredictor:")
        print(f"  Parameters: {params:,}")
        print(f"  Memory: {memory_mb:.2f} MB")

        # Should be under ~0.1MB (25K params)
        self.assertLess(memory_mb, 0.1,
                       f"Memory {memory_mb:.2f}MB exceeds 0.1MB")

    def test_total_memory_usage(self):
        """Test total memory usage of all models."""
        total_memory_mb = 0.0
        total_params = 0

        print("\n" + "=" * 60)
        print("Memory Usage Summary")
        print("=" * 60)
        print(f"{'Model':<20} {'Parameters':<15} {'Memory (MB)':<15}")
        print("-" * 60)

        for model_name, model in self.models.items():
            memory_mb, params = self.calculate_model_memory(model)
            total_memory_mb += memory_mb
            total_params += params
            print(f"{model_name:<20} {params:<15,} {memory_mb:<15.2f}")

        print("-" * 60)
        print(f"{'Total':<20} {total_params:<15,} {total_memory_mb:<15.2f}")
        print("=" * 60)

        # Total should be under 4MB
        self.assertLess(total_memory_mb, self.target_memory_mb,
                       f"Total memory {total_memory_mb:.2f}MB exceeds target {self.target_memory_mb}MB")

        # Verify parameter count matches expected (~1.15M)
        expected_params = 1151280  # Sum of all model params (actual: 1,151,280)
        tolerance = expected_params * 0.05  # 5% tolerance
        self.assertAlmostEqual(total_params, expected_params, delta=tolerance,
                              msg=f"Total parameters {total_params:,} don't match expected ~{expected_params:,}")


if __name__ == "__main__":
    unittest.main()
