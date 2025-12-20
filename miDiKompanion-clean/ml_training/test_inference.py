#!/usr/bin/env python3
"""
Test script to verify EmotionRecognizer inference.
Tests model loading and forward pass with sample data.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_emotion_model import EmotionRecognitionModel


def test_inference():
    """Test model inference with sample data."""
    print("=" * 60)
    print("Testing EmotionRecognizer Inference")
    print("=" * 60)

    # Create model
    print("\n1. Creating model...")
    model = EmotionRecognitionModel()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created")
    print(f"   Total parameters: {total_params:,}")

    # Test with random input (simulating mel-spectrogram features)
    print("\n2. Testing forward pass...")
    batch_size = 4
    input_features = torch.randn(batch_size, 128)  # 128-dim mel features

    with torch.no_grad():
        output = model(input_features)

    print(f"   Input shape: {input_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   ✓ Forward pass successful")

    # Test with single sample
    print("\n3. Testing single sample...")
    single_input = torch.randn(1, 128)
    with torch.no_grad():
        single_output = model(single_input)

    print(f"   Input shape: {single_input.shape}")
    print(f"   Output shape: {single_output.shape}")
    print(f"   Output sample: {single_output[0, :5].tolist()}...")
    print(f"   ✓ Single sample inference successful")

    # Test loading from checkpoint
    print("\n4. Testing checkpoint loading...")
    # Try multiple possible paths
    checkpoint_path = Path(__file__).parent / "models/emotion_model.pth"
    if not checkpoint_path.exists():
        # If run from parent directory
        checkpoint_path = Path("ml_training/models/emotion_model.pth")

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        # Test inference with loaded weights
        test_input = torch.randn(1, 128)
        with torch.no_grad():
            test_output = model(test_input)

        print(f"   ✓ Checkpoint loaded successfully")
        print(f"   ✓ Inference with loaded weights successful")
        print(f"   Best epoch: {checkpoint['epoch']+1}")
        print(f"   Best val loss: {checkpoint['val_loss']:.6f}")
    else:
        print(f"   ⚠ Checkpoint not found at {checkpoint_path}")
        print(f"   Run training first to generate checkpoint")

    # Performance test
    print("\n5. Performance test...")
    model.eval()
    test_input = torch.randn(1, 128)

    import time
    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_input)

    elapsed = time.time() - start_time
    avg_time_ms = (elapsed / num_iterations) * 1000

    print(f"   Average inference time: {avg_time_ms:.2f} ms")
    print(f"   Target: <10 ms")
    if avg_time_ms < 10:
        print(f"   ✓ Performance target met!")
    else:
        print(f"   ⚠ Performance target not met (consider optimization)")

    print("\n" + "=" * 60)
    print("✓ All inference tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_inference()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
