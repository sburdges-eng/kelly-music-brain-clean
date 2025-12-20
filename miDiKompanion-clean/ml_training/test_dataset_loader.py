#!/usr/bin/env python3
"""
Test script for the EmotionDataset loader.
Tests loading audio files and extracting features.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path to import dataset_loaders
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loaders import EmotionDataset, create_train_val_split

def test_dataset_loader():
    """Test the EmotionDataset loader."""
    print("=" * 60)
    print("Testing EmotionDataset Loader")
    print("=" * 60)

    # Check if librosa is available
    try:
        import librosa
        print("✓ librosa is available")
    except ImportError:
        print("✗ librosa not available. Install with: pip install librosa")
        return False

    # Test with a dataset directory (if provided)
    dataset_path = Path("./datasets/audio") if len(sys.argv) < 2 else Path(sys.argv[1])

    if not dataset_path.exists():
        print(f"\nDataset directory not found: {dataset_path}")
        print("\nTo test with real data:")
        print("  1. Create a directory with audio files")
        print("  2. Create a labels.csv file with: filename,valence,arousal")
        print("  3. Run: python test_dataset_loader.py /path/to/dataset")
        print("\nExample labels.csv:")
        print("  filename,valence,arousal")
        print("  happy_001.wav,0.8,0.9")
        print("  sad_001.wav,-0.6,0.3")
        return False

    print(f"\nLoading dataset from: {dataset_path}")

    try:
        # Try to load dataset
        labels_file = dataset_path / "labels.csv"
        if not labels_file.exists():
            labels_file = None

        dataset = EmotionDataset(
            audio_dir=dataset_path,
            labels_file=labels_file,
            n_mels=128,
            duration=2.0,
            cache_features=True
        )

        print(f"✓ Loaded {len(dataset)} samples")

        # Test getting a sample
        if len(dataset) > 0:
            features, labels = dataset[0]
            print(f"\nSample shape:")
            print(f"  Features: {features.shape} (expected: torch.Size([128]))")
            print(f"  Labels: {labels.shape} (expected: torch.Size([64]))")
            print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"  Label range: [{labels.min():.3f}, {labels.max():.3f}]")

            # Test train/val split
            train_dataset, val_dataset = create_train_val_split(
                dataset,
                val_ratio=0.2,
                random_seed=42
            )
            print(f"\n✓ Train/Val split:")
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Validation samples: {len(val_dataset)}")

            # Test DataLoader
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            batch_features, batch_labels = next(iter(loader))
            print(f"\n✓ DataLoader test:")
            print(f"  Batch features shape: {batch_features.shape}")
            print(f"  Batch labels shape: {batch_labels.shape}")

            print("\n" + "=" * 60)
            print("✓ All tests passed!")
            print("=" * 60)
            return True

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loader()
    sys.exit(0 if success else 1)
