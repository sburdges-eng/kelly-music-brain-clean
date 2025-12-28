#!/usr/bin/env python3
"""
ML Training: Intent Schema Usage Example

Demonstrates how to use CompleteSongIntent schema for ML model targets and validation.

This example shows:
1. Loading intent files as training data
2. Encoding intent schema to model targets
3. Validating dataset for training
4. Using intent data with PyTorch DataLoader
5. Creating multi-task model targets from intent

Key Concepts:
- Intent schema provides rich, structured targets beyond simple labels
- Phase 0 (Core Wound) → guides data augmentation
- Phase 1 (Emotional Intent) → primary model targets
- Phase 2 (Technical Constraints) → secondary/auxiliary targets
- Rule breaking → teaches model to intentionally violate rules

Usage:
    # Basic usage
    python examples/ml_training/intent_schema_usage_example.py
    
    # With custom intent directory
    python examples/ml_training/intent_schema_usage_example.py --intent-dir path/to/intents
    
    # Show statistics only
    python examples/ml_training/intent_schema_usage_example.py --stats-only
"""

import argparse
import sys
from pathlib import Path
import json

# Add music_brain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Note: Direct module import is used here to avoid importing music_brain.audio module,
# which has a pre-existing import issue (uses np.ndarray in signatures without
# guaranteed numpy import). This workaround keeps the example script functional
# without numpy/librosa dependencies.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "intent_schema",
    Path(__file__).parent.parent.parent / "music_brain" / "session" / "intent_schema.py"
)
intent_schema_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intent_schema_module)

CompleteSongIntent = intent_schema_module.CompleteSongIntent
SongRoot = intent_schema_module.SongRoot
SongIntent = intent_schema_module.SongIntent
TechnicalConstraints = intent_schema_module.TechnicalConstraints
validate_intent = intent_schema_module.validate_intent

from python.penta_core.ml.datasets.intent_dataset import (
    IntentDataset,
    IntentEncoder,
    IntentEncodingConfig,
    validate_dataset_for_training,
)


# =============================================================================
# Example 1: Load and Inspect Intent Dataset
# =============================================================================


def example_load_dataset(intent_dir: Path):
    """
    Example: Load intent files and inspect the dataset.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Load and Inspect Intent Dataset")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading intents from: {intent_dir}")
    dataset = IntentDataset(
        intent_dir=intent_dir,
        validate_on_load=True
    )
    
    print(f"✓ Loaded {len(dataset)} intents")
    
    # Show first sample
    if len(dataset) > 0:
        print("\nFirst sample:")
        sample = dataset[0]
        intent = sample["intent"]
        targets = sample["targets"]
        
        print(f"  Title: {intent.title}")
        print(f"  Path: {sample['path']}")
        print(f"\n  Phase 1 (Emotional Intent):")
        print(f"    Primary Emotion: {intent.song_intent.mood_primary}")
        print(f"    Tension: {intent.song_intent.mood_secondary_tension}")
        print(f"    Vulnerability: {intent.song_intent.vulnerability_scale}")
        print(f"    Narrative Arc: {intent.song_intent.narrative_arc}")
        
        print(f"\n  Encoded Targets:")
        print(f"    Emotion ID: {targets['emotion_label']}")
        print(f"    Tension: {targets['tension']:.3f}")
        print(f"    Vulnerability: {targets['vulnerability']}")
        print(f"    Rule Break: {targets['rule_break_str']}")
        print(f"    Valid: {targets['intent_valid']}")


# =============================================================================
# Example 2: Get Dataset Statistics
# =============================================================================


def example_dataset_statistics(intent_dir: Path):
    """
    Example: Compute and display dataset statistics.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Dataset Statistics")
    print("=" * 80)
    
    dataset = IntentDataset(intent_dir=intent_dir)
    stats = dataset.get_statistics()
    
    print(f"\nTotal Samples: {stats['total_samples']}")
    
    print("\nEmotion Distribution:")
    for emotion, count in sorted(stats['emotion_distribution'].items()):
        print(f"  {emotion:15s}: {count:3d} samples")
    
    print("\nNarrative Arc Distribution:")
    for arc, count in sorted(stats['narrative_arc_distribution'].items()):
        print(f"  {arc:20s}: {count:3d} samples")
    
    print("\nRule Break Distribution:")
    for rule, count in sorted(stats['rule_break_distribution'].items()):
        print(f"  {rule:35s}: {count:3d} samples")
    
    print("\nTension Statistics:")
    for key, value in stats['tension_stats'].items():
        print(f"  {key:5s}: {value:.3f}")
    
    print("\nTempo Statistics (BPM):")
    for key, value in stats['tempo_stats'].items():
        print(f"  {key:5s}: {value:.1f}")


# =============================================================================
# Example 3: Validate Dataset for Training
# =============================================================================


def example_validate_dataset(intent_dir: Path):
    """
    Example: Validate dataset is suitable for training.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Validate Dataset for Training")
    print("=" * 80)
    
    dataset = IntentDataset(intent_dir=intent_dir)
    
    print(f"\nValidating dataset with {len(dataset)} samples...")
    is_valid, issues = validate_dataset_for_training(
        dataset,
        min_samples_per_emotion=2  # Lenient for demo
    )
    
    if is_valid:
        print("\n✓ Dataset is valid for training!")
    else:
        print(f"\n✗ Dataset has {len(issues)} validation issues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")


# =============================================================================
# Example 4: Batch Processing
# =============================================================================


def example_batch_processing(intent_dir: Path):
    """
    Example: Get batches of encoded targets.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 80)
    
    dataset = IntentDataset(intent_dir=intent_dir)
    
    if len(dataset) < 2:
        print("\nNeed at least 2 samples for batch demo")
        return
    
    # Get batch of first 2 samples (or all if less than 2)
    batch_size = min(2, len(dataset))
    indices = list(range(batch_size))
    
    print(f"\nGetting batch of {batch_size} samples...")
    batch = dataset.get_batch(indices)
    
    print("\nBatch shapes:")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"  {key:25s}: {value.shape} ({value.dtype})")
    
    print("\nBatch values:")
    print(f"  Emotions: {batch['emotion_labels']}")
    print(f"  Tensions: {batch['tensions']}")
    print(f"  Tempos (BPM): {batch['tempo_bpm']}")
    print(f"  Rule breaks: {batch['rule_break_ids']}")


# =============================================================================
# Example 5: PyTorch Integration (Optional)
# =============================================================================


def example_pytorch_integration(intent_dir: Path):
    """
    Example: Use IntentDataset with PyTorch DataLoader.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: PyTorch Integration (Optional)")
    print("=" * 80)
    
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("\nPyTorch not available - skipping this example")
        return
    
    # Create dataset
    dataset = IntentDataset(intent_dir=intent_dir)
    
    if len(dataset) == 0:
        print("\nNo samples in dataset")
        return
    
    # Create DataLoader
    print("\nCreating PyTorch DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Single thread for demo
    )
    
    print(f"✓ DataLoader created with {len(dataset)} samples")
    
    # Get one batch
    print("\nIterating through DataLoader...")
    for batch_idx, samples in enumerate(loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Number of samples: {len(samples['intent'])}")
        
        # Access encoded targets from first sample
        first_sample = samples['targets'][0]
        print(f"  First sample emotion: {first_sample['emotion_str']}")
        print(f"  First sample tension: {first_sample['tension']:.3f}")
        
        # Only show first batch
        break


# =============================================================================
# Example 6: Multi-Task Learning Targets
# =============================================================================


def example_multitask_targets(intent_dir: Path):
    """
    Example: Extract different target types for multi-task learning.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Multi-Task Learning Targets")
    print("=" * 80)
    
    dataset = IntentDataset(intent_dir=intent_dir)
    
    if len(dataset) == 0:
        print("\nNo samples in dataset")
        return
    
    print("\nIntent schema provides multiple target types:")
    print("\n1. Classification Tasks:")
    print("   - Emotion recognition (10 classes)")
    print("   - Narrative arc prediction (8 classes)")
    print("   - Vulnerability level (3 classes)")
    print("   - Rule break category (5 classes)")
    
    print("\n2. Regression Tasks:")
    print("   - Tension prediction (0.0-1.0)")
    print("   - Tempo prediction (normalized)")
    
    print("\n3. Binary Tasks:")
    print("   - Has rule break justification (yes/no)")
    print("   - Intent is valid (yes/no)")
    
    # Show example targets from first sample
    sample = dataset[0]
    targets = sample["targets"]
    
    print("\nExample from first sample:")
    print(f"  Classification: emotion={targets['emotion_str']}, arc={targets['narrative_arc_str']}")
    print(f"  Regression: tension={targets['tension']:.3f}, tempo_norm={targets['tempo_normalized']:.3f}")
    print(f"  Binary: has_justification={targets['has_justification']}, valid={targets['intent_valid']}")


# =============================================================================
# Example 7: Custom Encoding Configuration
# =============================================================================


def example_custom_encoding(intent_dir: Path):
    """
    Example: Use custom encoding configuration.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Custom Encoding Configuration")
    print("=" * 80)
    
    # Create custom encoding config
    print("\nCreating custom encoding configuration...")
    custom_config = IntentEncodingConfig(
        emotion_labels=["grief", "joy", "anger"],  # Subset of emotions
        embed_dim_emotion=128,  # Larger embedding
        tempo_range=(60, 140),  # Narrower tempo range
    )
    
    print(f"  Emotion labels: {custom_config.emotion_labels}")
    print(f"  Emotion embedding dim: {custom_config.embed_dim_emotion}")
    print(f"  Tempo range: {custom_config.tempo_range}")
    
    # Create dataset with custom config
    dataset = IntentDataset(
        intent_dir=intent_dir,
        encoding_config=custom_config
    )
    
    print(f"\n✓ Dataset created with custom encoding")
    print(f"  Loaded {len(dataset)} samples")


# =============================================================================
# Example 8: Intent-Driven Data Augmentation Strategy
# =============================================================================


def example_augmentation_strategy(intent_dir: Path):
    """
    Example: Show how intent schema guides data augmentation.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Intent-Driven Data Augmentation")
    print("=" * 80)
    
    dataset = IntentDataset(intent_dir=intent_dir)
    
    if len(dataset) == 0:
        print("\nNo samples in dataset")
        return
    
    print("\nIntent schema guides augmentation decisions:")
    print("\n> Phase 0 (Core Wound/Desire)")
    print("  - core_stakes → determines what can be modified")
    print("  - core_transformation → guides augmentation direction")
    
    print("\n> Phase 1 (Emotional Intent)")
    print("  - mood_primary → constrains key/mode changes")
    print("  - vulnerability_scale → limits dynamic changes")
    print("  - narrative_arc → preserves structural meaning")
    
    print("\n> Phase 2 (Technical Constraints)")
    print("  - technical_rule_to_break → MUST be preserved")
    print("  - rule_breaking_justification → validates augmentation")
    
    # Show example
    sample = dataset[0]
    intent = sample["intent"]
    
    print(f"\nExample from '{intent.title}':")
    print(f"  Primary emotion: {intent.song_intent.mood_primary}")
    print(f"  Rule to break: {intent.technical_constraints.technical_rule_to_break}")
    
    print("\n  Safe augmentations:")
    print("    ✓ Tempo shift ±10% (preserves emotion)")
    print("    ✓ Transpose to nearby keys (preserves mode relationships)")
    print("    ✓ Add instrumentation (preserves core harmony)")
    
    print("\n  Unsafe augmentations:")
    print("    ✗ Change mode (breaks emotional intent)")
    print("    ✗ Remove rule break (loses creative intent)")
    print("    ✗ Extreme tempo change (changes emotion category)")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="ML Training: Intent Schema Usage Example"
    )
    parser.add_argument(
        "--intent-dir",
        type=Path,
        default=Path(__file__).parent.parent / "music_brain" / "intents",
        help="Directory containing intent JSON files"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics"
    )
    
    args = parser.parse_args()
    
    # Check intent directory exists
    if not args.intent_dir.exists():
        print(f"Error: Intent directory not found: {args.intent_dir}")
        print("\nTry creating some intent files first:")
        print("  python -m music_brain.cli intent new --save my_intent.json")
        return 1
    
    print("\n" + "=" * 80)
    print("ML Training: Intent Schema Usage Example")
    print("=" * 80)
    print(f"\nIntent directory: {args.intent_dir}")
    
    if args.stats_only:
        example_dataset_statistics(args.intent_dir)
        return 0
    
    # Run all examples
    example_load_dataset(args.intent_dir)
    example_dataset_statistics(args.intent_dir)
    example_validate_dataset(args.intent_dir)
    example_batch_processing(args.intent_dir)
    example_pytorch_integration(args.intent_dir)
    example_multitask_targets(args.intent_dir)
    example_custom_encoding(args.intent_dir)
    example_augmentation_strategy(args.intent_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary: Intent Schema for ML Training")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Intent schema provides rich, structured training targets")
    print("  2. Multiple task types: classification, regression, binary")
    print("  3. Validation ensures dataset quality")
    print("  4. Compatible with PyTorch/TensorFlow workflows")
    print("  5. Intent guides augmentation strategies")
    print("  6. Rule-breaking teaches creative decisions")
    
    print("\nNext Steps:")
    print("  - Create more intent files with diverse emotions")
    print("  - Train multi-task model with intent targets")
    print("  - Use intent to validate generated music")
    print("  - Implement intent-aware augmentation pipeline")
    
    print("\n" + "=" * 80)
    print("'Interrogate Before Generate' - Intent-Driven ML")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
