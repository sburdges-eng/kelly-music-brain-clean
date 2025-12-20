#!/usr/bin/env python3
"""
Unit Tests: Dataset Loaders
============================
Test dataset loaders handle various formats, missing files, and edge cases gracefully.
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dataset_loaders import (
        DEAMDataset,
        LakhMIDIDataset,
        MAESTRODataset,
        GrooveMIDIDataset,
        HarmonyDataset,
        create_dataset
    )
    DATASET_LOADERS_AVAILABLE = True
except ImportError:
    DATASET_LOADERS_AVAILABLE = False


class TestDatasetLoaders(unittest.TestCase):
    """Test dataset loader functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @unittest.skipUnless(DATASET_LOADERS_AVAILABLE, "Dataset loaders not available")
    def test_create_dataset_factory(self):
        """Test create_dataset factory function."""
        # Test with non-existent dataset (should raise error or handle gracefully)
        with self.assertRaises((ValueError, FileNotFoundError)):
            create_dataset('deam', self.temp_dir / 'nonexistent')

    @unittest.skipUnless(DATASET_LOADERS_AVAILABLE, "Dataset loaders not available")
    def test_dataset_handles_missing_files(self):
        """Test that datasets handle missing files gracefully."""
        # Create empty directory
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()

        # Should raise ValueError or handle gracefully
        with self.assertRaises((ValueError, FileNotFoundError)):
            create_dataset('deam', empty_dir)

    @unittest.skipUnless(DATASET_LOADERS_AVAILABLE, "Dataset loaders not available")
    def test_dataset_returns_dict_format(self):
        """Test that datasets return dict format with correct keys."""
        # Test with synthetic datasets from train_all_models
        from train_all_models import (
            SyntheticEmotionDataset,
            SyntheticMelodyDataset,
            SyntheticHarmonyDataset,
            SyntheticDynamicsDataset,
            SyntheticGrooveDataset
        )

        # Test EmotionRecognizer dataset format
        emotion_dataset = SyntheticEmotionDataset(num_samples=10)
        sample = emotion_dataset[0]
        self.assertIsInstance(sample, dict, "Sample should be a dictionary")
        self.assertIn('mel_features', sample, "Should have 'mel_features' key")
        self.assertIn('emotion', sample, "Should have 'emotion' key")
        self.assertEqual(sample['mel_features'].shape, torch.Size([128]))
        self.assertEqual(sample['emotion'].shape, torch.Size([64]))

        # Test MelodyTransformer dataset format
        melody_dataset = SyntheticMelodyDataset(num_samples=10)
        sample = melody_dataset[0]
        self.assertIsInstance(sample, dict, "Sample should be a dictionary")
        self.assertIn('emotion', sample, "Should have 'emotion' key")
        self.assertIn('notes', sample, "Should have 'notes' key")
        self.assertEqual(sample['emotion'].shape, torch.Size([64]))
        self.assertEqual(sample['notes'].shape, torch.Size([128]))

        # Test HarmonyPredictor dataset format
        harmony_dataset = SyntheticHarmonyDataset(num_samples=10)
        sample = harmony_dataset[0]
        self.assertIsInstance(sample, dict, "Sample should be a dictionary")
        self.assertIn('context', sample, "Should have 'context' key")
        self.assertIn('chords', sample, "Should have 'chords' key")
        self.assertEqual(sample['context'].shape, torch.Size([128]))
        self.assertEqual(sample['chords'].shape, torch.Size([64]))

        # Test DynamicsEngine dataset format
        dynamics_dataset = SyntheticDynamicsDataset(num_samples=10)
        sample = dynamics_dataset[0]
        self.assertIsInstance(sample, dict, "Sample should be a dictionary")
        self.assertIn('context', sample, "Should have 'context' key")
        # Dataset uses 'expression' key (training supports both 'expression' and 'dynamics')
        self.assertIn('expression', sample, "Should have 'expression' key")
        self.assertEqual(sample['context'].shape, torch.Size([32]))
        self.assertEqual(sample['expression'].shape, torch.Size([16]))

        # Test GroovePredictor dataset format
        groove_dataset = SyntheticGrooveDataset(num_samples=10)
        sample = groove_dataset[0]
        self.assertIsInstance(sample, dict, "Sample should be a dictionary")
        self.assertIn('emotion', sample, "Should have 'emotion' key")
        self.assertIn('groove', sample, "Should have 'groove' key")
        self.assertEqual(sample['emotion'].shape, torch.Size([64]))
        self.assertEqual(sample['groove'].shape, torch.Size([32]))

    def test_synthetic_dataset_creation(self):
        """Test that synthetic datasets can be created."""
        from train_all_models import SyntheticEmotionDataset, SyntheticMelodyDataset

        # Test EmotionDataset
        emotion_dataset = SyntheticEmotionDataset(num_samples=100)
        self.assertEqual(len(emotion_dataset), 100)

        sample = emotion_dataset[0]
        self.assertIn('mel_features', sample)
        self.assertIn('emotion', sample)
        self.assertEqual(sample['mel_features'].shape, torch.Size([128]))
        self.assertEqual(sample['emotion'].shape, torch.Size([64]))

        # Test MelodyDataset
        melody_dataset = SyntheticMelodyDataset(num_samples=50)
        self.assertEqual(len(melody_dataset), 50)

        sample = melody_dataset[0]
        self.assertIn('emotion', sample)
        self.assertIn('notes', sample)
        self.assertEqual(sample['emotion'].shape, torch.Size([64]))
        self.assertEqual(sample['notes'].shape, torch.Size([128]))

    def test_synthetic_dataset_values_in_range(self):
        """Test that synthetic dataset values are in expected ranges."""
        from train_all_models import SyntheticEmotionDataset, SyntheticMelodyDataset

        emotion_dataset = SyntheticEmotionDataset(num_samples=10)
        for i in range(len(emotion_dataset)):
            sample = emotion_dataset[i]
            # Mel features should be reasonable
            self.assertTrue(torch.all(torch.isfinite(sample['mel_features'])))
            # Emotion should be in [-1, 1] (tanh output)
            self.assertTrue(torch.all(sample['emotion'] >= -1.0))
            self.assertTrue(torch.all(sample['emotion'] <= 1.0))

        melody_dataset = SyntheticMelodyDataset(num_samples=10)
        for i in range(len(melody_dataset)):
            sample = melody_dataset[i]
            # Emotion should be in [-1, 1]
            self.assertTrue(torch.all(sample['emotion'] >= -1.0))
            self.assertTrue(torch.all(sample['emotion'] <= 1.0))
            # Notes should be probabilities [0, 1]
            self.assertTrue(torch.all(sample['notes'] >= 0.0))
            self.assertTrue(torch.all(sample['notes'] <= 1.0))
            # Should sum to ~1.0 (probability distribution)
            self.assertAlmostEqual(sample['notes'].sum().item(), 1.0, places=2)

    def test_dataset_loader_integration(self):
        """Test dataset loader with DataLoader."""
        from train_all_models import SyntheticEmotionDataset
        from torch.utils.data import DataLoader

        dataset = SyntheticEmotionDataset(num_samples=100)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Get a batch
        batch = next(iter(loader))

        self.assertIn('mel_features', batch)
        self.assertIn('emotion', batch)
        self.assertEqual(batch['mel_features'].shape[0], 16)
        self.assertEqual(batch['mel_features'].shape[1], 128)
        self.assertEqual(batch['emotion'].shape[0], 16)
        self.assertEqual(batch['emotion'].shape[1], 64)


if __name__ == "__main__":
    unittest.main()
