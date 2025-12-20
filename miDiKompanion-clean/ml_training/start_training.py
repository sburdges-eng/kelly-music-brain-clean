#!/usr/bin/env python3
"""
Quick Start Training Script
===========================
Starts training all 5 ML models with sensible defaults.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Start training with default settings."""

    script_dir = Path(__file__).parent
    train_script = script_dir / "train_all_models.py"

    if not train_script.exists():
        print(f"Error: {train_script} not found")
        sys.exit(1)

    # Default training arguments (synthetic data, quick test)
    args = [
        sys.executable,
        str(train_script),
        "--output", "trained_models",
        "--epochs", "10",  # Quick test
        "--batch-size", "32",
        "--device", "cpu",  # Will auto-detect GPU if available
        "--use-synthetic"  # Use synthetic data for quick start
    ]

    print("=" * 60)
    print("Starting ML Model Training")
    print("=" * 60)
    print(f"Command: {' '.join(args)}")
    print()
    print("This will train all 5 models:")
    print("  1. EmotionRecognizer (128→64)")
    print("  2. MelodyTransformer (64→128)")
    print("  3. HarmonyPredictor (128→64)")
    print("  4. DynamicsEngine (32→16)")
    print("  5. GroovePredictor (64→32)")
    print()
    print("Using synthetic data for quick testing.")
    print("For production training, use: python train_all_models.py --epochs 50 --device cuda")
    print()
    print("=" * 60)
    print()

    # Run training
    result = subprocess.run(args)

    if result.returncode == 0:
        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print("Next steps:")
        print("1. Export to ONNX: python export_to_onnx.py")
        print("2. Deploy: python deploy_models.py")
        print("3. Copy models to plugin Resources directory")
    else:
        print()
        print("Training failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
