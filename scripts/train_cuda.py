#!/usr/bin/env python3
"""
Kelly NVIDIA CUDA Training Pipeline

Master training script that:
1. Downloads audio samples (Freesound)
2. Generates MIDI training data
3. Creates unified dataset
4. Trains all models on CUDA

Usage:
    # Full pipeline (download + generate + train)
    python scripts/train_cuda.py --full

    # Just generate data (no download)
    python scripts/train_cuda.py --generate

    # Just train on existing data
    python scripts/train_cuda.py --train

    # Train specific model
    python scripts/train_cuda.py --train --model emotion_recognizer

    # Check GPU status
    python scripts/train_cuda.py --check-gpu
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_gpu():
    """Check GPU availability and print info."""
    print("=" * 60)
    print("GPU Status Check")
    print("=" * 60)

    try:
        import torch

        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Device count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Memory: {memory_gb:.1f} GB")
                print(f"    Compute: {props.major}.{props.minor}")
                print(f"    Multi-processors: {props.multi_processor_count}")

            # Test CUDA
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x)
            print(f"\n✓ CUDA test passed")
        else:
            print("\n⚠ No CUDA available")

            # Check for MPS (Apple)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("✓ Apple MPS available")
            else:
                print("  Falling back to CPU")

    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install torch")
        return False

    return True


def run_downloaders(args):
    """Run audio sample downloaders."""
    print("\n" + "=" * 60)
    print("Step 1: Downloading Audio Samples")
    print("=" * 60)

    # Check for API key
    config_path = PROJECT_ROOT / "configs" / "freesound_config.json"
    if not config_path.exists():
        print("\n⚠ Freesound config not found!")
        print(f"  1. Copy template: cp {PROJECT_ROOT}/configs/freesound_config.json.template {config_path}")
        print("  2. Add your API key from https://freesound.org/apiv2/apply/")
        print("\n  Skipping download step...")
        return False

    with open(config_path) as f:
        config = json.load(f)

    if config.get("freesound_api_key", "").startswith("YOUR_"):
        print("\n⚠ API key not configured!")
        print(f"  Edit {config_path} and add your Freesound API key")
        print("\n  Skipping download step...")
        return False

    # Run auto emotion sampler
    print("\nRunning emotion sampler...")
    sampler_script = PROJECT_ROOT / "scripts" / "auto_emotion_sampler.py"

    if sampler_script.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(sampler_script), "start", "--limit", str(args.download_limit)],
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            print("✓ Audio samples downloaded")
            return True
        else:
            print("⚠ Sampler had issues, continuing...")
    else:
        print(f"⚠ Sampler script not found: {sampler_script}")

    return False


def generate_datasets(args):
    """Generate unified dataset from all sources."""
    print("\n" + "=" * 60)
    print("Step 2: Generating Training Datasets")
    print("=" * 60)

    # Generate MIDI sequences
    print("\n[2a] Generating MIDI sequences from chord progressions...")
    try:
        from python.penta_core.ml.datasets.midi_generator import MIDIDatasetGenerator

        midi_gen = MIDIDatasetGenerator(
            data_dir=PROJECT_ROOT / "data",
            output_dir=PROJECT_ROOT / "datasets" / "midi_generated"
        )
        midi_path = midi_gen.generate_and_save()
        print(f"  ✓ MIDI dataset: {midi_path}")
    except Exception as e:
        print(f"  ⚠ MIDI generation failed: {e}")

    # Generate unified dataset
    print("\n[2b] Creating unified dataset...")
    try:
        from python.penta_core.ml.datasets.unified_generator import UnifiedDatasetGenerator

        unified_gen = UnifiedDatasetGenerator(
            output_dir=PROJECT_ROOT / "datasets" / "unified"
        )
        dataset = unified_gen.generate(include_audio=True, include_osc=True)
        output_dir = unified_gen.save(dataset)

        print(f"\n  Dataset Statistics:")
        for dtype, count in sorted(dataset.stats.items()):
            print(f"    {dtype}: {count}")
        print(f"    TOTAL: {len(dataset)} samples")
        print(f"\n  ✓ Unified dataset: {output_dir}")

        return output_dir / f"{dataset.name}_v{dataset.version}.json"
    except Exception as e:
        print(f"  ⚠ Unified dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_models(args, dataset_path: Path = None):
    """Train models with CUDA."""
    print("\n" + "=" * 60)
    print("Step 3: Training Models (CUDA)")
    print("=" * 60)

    # Find dataset
    if dataset_path is None or not dataset_path.exists():
        # Look for existing unified dataset
        unified_dir = PROJECT_ROOT / "datasets" / "unified"
        if unified_dir.exists():
            datasets = list(unified_dir.glob("kelly_unified_*.json"))
            if datasets:
                dataset_path = max(datasets, key=lambda p: p.stat().st_mtime)
                print(f"\nUsing existing dataset: {dataset_path}")

    if dataset_path is None or not dataset_path.exists():
        print("\n❌ No dataset found!")
        print("   Run with --generate first to create training data")
        return False

    try:
        from python.penta_core.ml.training.cuda_trainer import (
            CUDATrainer, TrainingConfig, ModelType, train_all_models
        )

        if args.model:
            # Train specific model
            model_type = ModelType(args.model)
            print(f"\nTraining {model_type.value}...")

            config = TrainingConfig(
                model_type=model_type,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                checkpoint_dir=PROJECT_ROOT / "checkpoints" / model_type.value,
                use_cuda=not args.cpu,
                mixed_precision=args.fp16,
            )

            trainer = CUDATrainer(config)
            train_loader, val_loader, test_loader = trainer.prepare_data(dataset_path)

            if train_loader is None:
                print(f"❌ Could not prepare data for {model_type.value}")
                return False

            result = trainer.train(train_loader, val_loader)
            if result:
                test_metrics = trainer.evaluate(test_loader)
                print(f"\n✓ Training complete!")
                print(f"  Best epoch: {result.best_epoch}")
                print(f"  Best val loss: {result.best_val_loss:.4f}")
                print(f"  Test accuracy: {test_metrics.get('test_accuracy', 0):.2f}%")
                print(f"  Checkpoint: {result.checkpoint_path}")
                return True
        else:
            # Train all models
            print("\nTraining all models...")
            results = train_all_models(
                dataset_path,
                output_dir=PROJECT_ROOT / "checkpoints"
            )

            print("\n" + "=" * 60)
            print("Training Results")
            print("=" * 60)
            for name, result in results.items():
                acc = result.final_metrics.get('test_accuracy', 0)
                print(f"  {name}: val_loss={result.best_val_loss:.4f}, test_acc={acc:.2f}%")

            return len(results) > 0

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Install PyTorch: pip install torch")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Kelly NVIDIA CUDA Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_cuda.py --check-gpu
  python scripts/train_cuda.py --full
  python scripts/train_cuda.py --generate
  python scripts/train_cuda.py --train --model emotion_recognizer
  python scripts/train_cuda.py --train --epochs 50 --batch-size 64
        """
    )

    # Pipeline stages
    parser.add_argument("--full", action="store_true",
                       help="Run full pipeline: download + generate + train")
    parser.add_argument("--download", action="store_true",
                       help="Download audio samples")
    parser.add_argument("--generate", action="store_true",
                       help="Generate training datasets")
    parser.add_argument("--train", action="store_true",
                       help="Train models")
    parser.add_argument("--check-gpu", action="store_true",
                       help="Check GPU availability")

    # Training options
    parser.add_argument("--model", type=str, default=None,
                       choices=["emotion_recognizer", "melody_transformer",
                               "harmony_predictor", "dynamics_engine",
                               "groove_predictor", "instrument_recognizer",
                               "osc_pattern_learner"],
                       help="Train specific model (default: all)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU training")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use FP16 mixed precision (default: True)")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16",
                       help="Disable FP16 mixed precision")

    # Download options
    parser.add_argument("--download-limit", type=int, default=100,
                       help="Max samples to download per emotion (default: 100)")

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("Kelly CUDA Training Pipeline")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")

    # Default to full if no stage specified
    if not any([args.full, args.download, args.generate, args.train, args.check_gpu]):
        args.check_gpu = True

    start_time = time.time()

    # Execute stages
    if args.check_gpu:
        check_gpu()
        if not any([args.full, args.download, args.generate, args.train]):
            return

    dataset_path = None

    if args.full or args.download:
        run_downloaders(args)

    if args.full or args.generate:
        dataset_path = generate_datasets(args)

    if args.full or args.train:
        train_models(args, dataset_path)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
