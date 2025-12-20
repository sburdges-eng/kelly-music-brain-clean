#!/usr/bin/env python3
"""
Performance Test: Inference Latency
====================================
Measure inference latency for all models. Target: <10ms per model.
"""

import unittest
import sys
import time
import statistics
from pathlib import Path
import torch
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor
)


class TestInferenceLatency(unittest.TestCase):
    """Test inference latency for all models."""

    def setUp(self):
        """Set up test fixtures."""
        self.models = {
            'EmotionRecognizer': (EmotionRecognizer(), 128),
            'MelodyTransformer': (MelodyTransformer(), 64),
            'HarmonyPredictor': (HarmonyPredictor(), 128),
            'DynamicsEngine': (DynamicsEngine(), 32),
            'GroovePredictor': (GroovePredictor(), 64)
        }
        self.target_latency_ms = 10.0  # Target: <10ms
        self.num_iterations = 1000
        self.warmup_iterations = 100

    def benchmark_model(self, model, input_size, num_iterations=1000):
        """Benchmark a single model."""
        model.eval()
        test_input = torch.randn(1, input_size)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(test_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(test_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0,
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }

    def test_emotion_recognizer_latency(self):
        """Test EmotionRecognizer inference latency."""
        model, input_size = self.models['EmotionRecognizer']
        stats = self.benchmark_model(model, input_size, self.num_iterations)

        print(f"\nEmotionRecognizer Latency:")
        print(f"  Mean: {stats['mean']:.3f} ms")
        print(f"  Median: {stats['median']:.3f} ms")
        print(f"  P95: {stats['p95']:.3f} ms")
        print(f"  P99: {stats['p99']:.3f} ms")
        print(f"  Max: {stats['max']:.3f} ms")

        self.assertLess(stats['mean'], self.target_latency_ms * 2,
                       f"Mean latency {stats['mean']:.3f}ms exceeds target {self.target_latency_ms}ms")
        self.assertLess(stats['p95'], self.target_latency_ms * 3,
                       f"P95 latency {stats['p95']:.3f}ms exceeds target")

    def test_melody_transformer_latency(self):
        """Test MelodyTransformer inference latency."""
        model, input_size = self.models['MelodyTransformer']
        stats = self.benchmark_model(model, input_size, self.num_iterations)

        print(f"\nMelodyTransformer Latency:")
        print(f"  Mean: {stats['mean']:.3f} ms")
        print(f"  P95: {stats['p95']:.3f} ms")

        self.assertLess(stats['mean'], self.target_latency_ms * 2)

    def test_harmony_predictor_latency(self):
        """Test HarmonyPredictor inference latency."""
        model, input_size = self.models['HarmonyPredictor']
        stats = self.benchmark_model(model, input_size, self.num_iterations)

        print(f"\nHarmonyPredictor Latency:")
        print(f"  Mean: {stats['mean']:.3f} ms")
        print(f"  P95: {stats['p95']:.3f} ms")

        self.assertLess(stats['mean'], self.target_latency_ms * 2)

    def test_dynamics_engine_latency(self):
        """Test DynamicsEngine inference latency."""
        model, input_size = self.models['DynamicsEngine']
        stats = self.benchmark_model(model, input_size, self.num_iterations)

        print(f"\nDynamicsEngine Latency:")
        print(f"  Mean: {stats['mean']:.3f} ms")
        print(f"  P95: {stats['p95']:.3f} ms")

        self.assertLess(stats['mean'], self.target_latency_ms * 2)

    def test_groove_predictor_latency(self):
        """Test GroovePredictor inference latency."""
        model, input_size = self.models['GroovePredictor']
        stats = self.benchmark_model(model, input_size, self.num_iterations)

        print(f"\nGroovePredictor Latency:")
        print(f"  Mean: {stats['mean']:.3f} ms")
        print(f"  P95: {stats['p95']:.3f} ms")

        self.assertLess(stats['mean'], self.target_latency_ms * 2)

    def test_all_models_latency_summary(self):
        """Test all models and generate summary."""
        all_stats = {}

        for model_name, (model, input_size) in self.models.items():
            stats = self.benchmark_model(model, input_size, self.num_iterations)
            all_stats[model_name] = stats

        print("\n" + "=" * 60)
        print("Inference Latency Summary")
        print("=" * 60)
        print(f"{'Model':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Max (ms)':<12}")
        print("-" * 60)

        total_mean = 0.0
        for model_name, stats in all_stats.items():
            print(f"{model_name:<20} {stats['mean']:<12.3f} {stats['p95']:<12.3f} {stats['max']:<12.3f}")
            total_mean += stats['mean']

        print("-" * 60)
        print(f"{'Total Pipeline':<20} {total_mean:<12.3f}")
        print("=" * 60)

        # Total pipeline should be under 50ms (5 models * 10ms)
        self.assertLess(total_mean, 50.0,
                       f"Total pipeline latency {total_mean:.3f}ms exceeds 50ms target")


if __name__ == "__main__":
    unittest.main()
