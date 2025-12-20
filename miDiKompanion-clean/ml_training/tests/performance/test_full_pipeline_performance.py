#!/usr/bin/env python3
"""
Performance Test: Full Pipeline
================================
Test end-to-end performance of the complete ML pipeline.
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


class TestFullPipelinePerformance(unittest.TestCase):
    """Test full pipeline performance."""

    def setUp(self):
        """Set up test fixtures."""
        self.models = {
            'EmotionRecognizer': EmotionRecognizer(),
            'MelodyTransformer': MelodyTransformer(),
            'HarmonyPredictor': HarmonyPredictor(),
            'DynamicsEngine': DynamicsEngine(),
            'GroovePredictor': GroovePredictor()
        }
        self.num_iterations = 100
        self.warmup_iterations = 10

    def run_full_pipeline(self, audio_features):
        """Simulate full pipeline execution."""
        # 1. EmotionRecognizer
        emotion = self.models['EmotionRecognizer'](audio_features)

        # 2. MelodyTransformer (uses emotion)
        melody = self.models['MelodyTransformer'](emotion)

        # 3. GroovePredictor (uses emotion)
        groove = self.models['GroovePredictor'](emotion)

        # 4. HarmonyPredictor (uses context: emotion + audio features)
        # HarmonyPredictor expects 128-dim input, so take emotion (64) + first 64 of audio_features
        context = torch.cat([emotion, audio_features[:, :64]], dim=1)
        harmony = self.models['HarmonyPredictor'](context)

        # 5. DynamicsEngine (uses compact context)
        compact_context = torch.cat([emotion[:, :16], audio_features[:, :16]], dim=1)
        dynamics = self.models['DynamicsEngine'](compact_context)

        return {
            'emotion': emotion,
            'melody': melody,
            'harmony': harmony,
            'dynamics': dynamics,
            'groove': groove
        }

    def test_full_pipeline_latency(self):
        """Test full pipeline latency."""
        audio_features = torch.randn(1, 128)

        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.run_full_pipeline(audio_features)

        # Benchmark
        times = []
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.run_full_pipeline(audio_features)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        mean_latency = statistics.mean(times)
        median_latency = statistics.median(times)
        p95_latency = np.percentile(times, 95)
        max_latency = max(times)

        print("\n" + "=" * 60)
        print("Full Pipeline Latency")
        print("=" * 60)
        print(f"  Mean: {mean_latency:.3f} ms")
        print(f"  Median: {median_latency:.3f} ms")
        print(f"  P95: {p95_latency:.3f} ms")
        print(f"  Max: {max_latency:.3f} ms")
        print("=" * 60)

        # Target: <50ms for full pipeline (5 models * 10ms)
        self.assertLess(mean_latency, 50.0,
                       f"Full pipeline latency {mean_latency:.3f}ms exceeds 50ms target")

    def test_pipeline_throughput(self):
        """Test pipeline throughput (samples per second)."""
        audio_features = torch.randn(1, 128)

        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.run_full_pipeline(audio_features)

        # Benchmark
        num_samples = 1000
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_samples):
                _ = self.run_full_pipeline(audio_features)

        elapsed = time.perf_counter() - start
        throughput = num_samples / elapsed

        print(f"\nPipeline Throughput: {throughput:.1f} samples/second")

        # Should be able to process at least 20 samples/second (50ms per sample)
        self.assertGreater(throughput, 20.0,
                          f"Throughput {throughput:.1f} samples/s below 20 samples/s target")

    def test_pipeline_output_validity(self):
        """Test that pipeline produces valid outputs."""
        audio_features = torch.randn(1, 128)

        with torch.no_grad():
            results = self.run_full_pipeline(audio_features)

        # Check all outputs are valid
        self.assertEqual(results['emotion'].shape[-1], 64)
        self.assertEqual(results['melody'].shape[-1], 128)
        self.assertEqual(results['harmony'].shape[-1], 64)
        self.assertEqual(results['dynamics'].shape[-1], 16)
        self.assertEqual(results['groove'].shape[-1], 32)

        # Check no NaN or Inf
        for name, output in results.items():
            self.assertFalse(torch.any(torch.isnan(output)),
                           f"{name}: Output contains NaN")
            self.assertFalse(torch.any(torch.isinf(output)),
                           f"{name}: Output contains Inf")

    def test_pipeline_consistency(self):
        """Test that pipeline produces consistent results."""
        audio_features = torch.randn(1, 128)

        with torch.no_grad():
            results1 = self.run_full_pipeline(audio_features)
            results2 = self.run_full_pipeline(audio_features)

        # Results should be identical (deterministic)
        for name in results1.keys():
            self.assertTrue(torch.allclose(results1[name], results2[name], atol=1e-6),
                          f"{name}: Pipeline not deterministic")


if __name__ == "__main__":
    unittest.main()
