#!/usr/bin/env python3
"""
Kelly ML Dataset Preparation Script

Complete pipeline for preparing training datasets for Kelly's 5-model architecture.

Workflow:
1. Create dataset structure
2. Import raw files (MIDI/audio)
3. Annotate samples (interactive or auto)
4. Extract features
5. Augment data (expand 100 → 1000+)
6. Generate synthetic data (from music theory)
7. Validate dataset
8. Export for training

Target: 1000 samples per category, <3TB total

Usage:
    # Full pipeline
    python scripts/prepare_datasets.py --all --target-model emotion_recognizer
    
    # Individual steps
    python scripts/prepare_datasets.py --create --dataset emotion_dataset_v1
    python scripts/prepare_datasets.py --import-dir /path/to/midi --dataset emotion_dataset_v1
    python scripts/prepare_datasets.py --annotate --dataset emotion_dataset_v1
    python scripts/prepare_datasets.py --extract-features --dataset emotion_dataset_v1
    python scripts/prepare_datasets.py --augment --multiplier 10 --dataset emotion_dataset_v1
    python scripts/prepare_datasets.py --synthesize --count 5000 --dataset emotion_dataset_v1
    python scripts/prepare_datasets.py --validate --dataset emotion_dataset_v1
    
    # Quick stats
    python scripts/prepare_datasets.py --stats --dataset emotion_dataset_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
DATASETS_DIR = ROOT / "datasets"
DATA_DIR = ROOT / "data"


# =============================================================================
# Dataset Creation
# =============================================================================


def create_dataset(
    dataset_id: str,
    target_model: str = "emotionrecognizer",
    description: str = "",
) -> Path:
    """Create a new dataset with standard structure."""
    from python.penta_core.ml.datasets.base import (
        DatasetConfig,
        create_dataset_structure,
    )
    
    config = DatasetConfig(
        dataset_id=dataset_id,
        target_model=target_model,
        description=description or f"Dataset for {target_model}",
        created_at=datetime.now().isoformat(),
    )
    
    # Model-specific settings
    if target_model == "emotionrecognizer":
        config.sample_type = "audio"
        config.emotion_labels = ["happy", "sad", "angry", "peaceful", "tense", "melancholic", "energetic", "neutral"]
    elif target_model == "melodytransformer":
        config.sample_type = "midi"
    elif target_model == "harmonypredictor":
        config.sample_type = "midi"
    elif target_model == "dynamicsengine":
        config.sample_type = "midi"
    elif target_model == "groovepredictor":
        config.sample_type = "midi"
        config.groove_labels = ["straight", "swing", "shuffle", "laid_back", "rushed"]
    
    manifest = create_dataset_structure(DATASETS_DIR, config)
    dataset_path = DATASETS_DIR / dataset_id
    
    logger.info(f"Created dataset: {dataset_path}")
    return dataset_path


# =============================================================================
# File Import
# =============================================================================


def import_files(
    source_dir: Path,
    dataset_id: str,
    file_types: List[str] = None,
    recursive: bool = True,
) -> int:
    """Import files from a directory into the dataset."""
    from python.penta_core.ml.datasets.base import (
        load_manifest,
        save_manifest,
        create_sample_from_file,
    )
    
    dataset_path = DATASETS_DIR / dataset_id
    manifest_path = dataset_path / "manifest.json"
    
    if not manifest_path.exists():
        logger.error(f"Dataset not found: {dataset_id}")
        return 0
    
    manifest = load_manifest(manifest_path)
    source_dir = Path(source_dir)
    
    file_types = file_types or ['midi', 'audio']
    patterns = []
    if 'midi' in file_types:
        patterns.extend(['*.mid', '*.midi'])
    if 'audio' in file_types:
        patterns.extend(['*.wav', '*.mp3', '*.flac', '*.ogg'])
    
    imported = 0
    glob_fn = source_dir.rglob if recursive else source_dir.glob
    
    for pattern in patterns:
        for file_path in glob_fn(pattern):
            # Determine destination
            if file_path.suffix.lower() in ['.mid', '.midi']:
                dest_dir = dataset_path / "raw" / "midi"
            else:
                dest_dir = dataset_path / "raw" / "audio"
            
            dest_path = dest_dir / file_path.name
            
            # Copy file
            shutil.copy2(file_path, dest_path)
            
            # Create sample entry
            sample = create_sample_from_file(dest_path, dataset_path)
            manifest.add_sample(sample)
            
            imported += 1
            if imported % 100 == 0:
                logger.info(f"Imported {imported} files...")
    
    # Save manifest
    save_manifest(manifest, manifest_path)
    logger.info(f"Imported {imported} files into {dataset_id}")
    
    return imported


# =============================================================================
# Annotation
# =============================================================================


def auto_annotate(
    dataset_id: str,
    use_directory_labels: bool = True,
    use_filename_labels: bool = True,
) -> int:
    """Auto-annotate samples based on directory structure and filenames."""
    from python.penta_core.ml.datasets.base import (
        load_manifest,
        save_manifest,
        SampleAnnotation,
    )
    
    dataset_path = DATASETS_DIR / dataset_id
    manifest = load_manifest(dataset_path / "manifest.json")
    
    annotated = 0
    
    for sample in manifest.samples:
        if sample.annotations and sample.annotations.emotion:
            continue  # Already annotated
        
        file_path = Path(sample.file_path)
        emotion = None
        
        # Try directory name
        if use_directory_labels:
            parent = file_path.parent.name.lower()
            emotions = ["happy", "sad", "angry", "peaceful", "tense", "melancholic", "energetic", "neutral"]
            for e in emotions:
                if e in parent:
                    emotion = e
                    break
        
        # Try filename
        if not emotion and use_filename_labels:
            name = file_path.stem.lower()
            for e in ["happy", "sad", "angry", "peaceful", "tense", "melancholic", "energetic", "neutral"]:
                if e in name:
                    emotion = e
                    break
        
        if emotion:
            if not sample.annotations:
                sample.annotations = SampleAnnotation()
            sample.annotations.emotion = emotion
            annotated += 1
    
    save_manifest(manifest, dataset_path / "manifest.json")
    logger.info(f"Auto-annotated {annotated} samples in {dataset_id}")
    
    return annotated


def interactive_annotate(dataset_id: str, limit: int = None) -> int:
    """Interactively annotate samples."""
    from python.penta_core.ml.datasets.base import (
        load_manifest,
        save_manifest,
        SampleAnnotation,
    )
    
    dataset_path = DATASETS_DIR / dataset_id
    manifest = load_manifest(dataset_path / "manifest.json")
    
    # Find unannotated samples
    unannotated = [s for s in manifest.samples 
                   if not s.annotations or not s.annotations.emotion]
    
    if limit:
        unannotated = unannotated[:limit]
    
    if not unannotated:
        logger.info("All samples are annotated!")
        return 0
    
    emotions = ["happy", "sad", "angry", "peaceful", "tense", "melancholic", "energetic", "neutral", "skip"]
    
    print("\n" + "=" * 50)
    print("Interactive Annotation")
    print("=" * 50)
    print(f"Samples to annotate: {len(unannotated)}")
    print(f"Options: {', '.join(f'{i}={e}' for i, e in enumerate(emotions))}")
    print("Enter 'q' to quit and save")
    print("=" * 50 + "\n")
    
    annotated = 0
    
    for i, sample in enumerate(unannotated):
        print(f"\n[{i+1}/{len(unannotated)}] {sample.file_path}")
        
        # Try to play audio if possible
        # (Would need audio playback library)
        
        choice = input(f"Emotion [0-{len(emotions)-1}]: ").strip()
        
        if choice.lower() == 'q':
            break
        
        try:
            idx = int(choice)
            if 0 <= idx < len(emotions) - 1:  # Exclude 'skip'
                if not sample.annotations:
                    sample.annotations = SampleAnnotation()
                sample.annotations.emotion = emotions[idx]
                annotated += 1
        except ValueError:
            print("Invalid input, skipping")
    
    save_manifest(manifest, dataset_path / "manifest.json")
    logger.info(f"Annotated {annotated} samples")
    
    return annotated


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_features(dataset_id: str, batch_size: int = 100) -> int:
    """Extract features from all samples in a dataset."""
    from python.penta_core.ml.datasets.base import load_manifest, save_manifest
    from python.penta_core.ml.datasets.midi_features import MIDIFeatureExtractor
    from python.penta_core.ml.datasets.audio_features import AudioFeatureExtractor
    
    dataset_path = DATASETS_DIR / dataset_id
    manifest = load_manifest(dataset_path / "manifest.json")
    
    features_dir = dataset_path / "processed" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    midi_extractor = MIDIFeatureExtractor()
    audio_extractor = AudioFeatureExtractor()
    
    extracted = 0
    
    try:
        from tqdm import tqdm
        samples = tqdm(manifest.samples, desc="Extracting features")
    except ImportError:
        samples = manifest.samples
    
    for sample in samples:
        file_path = dataset_path / sample.file_path
        
        if not file_path.exists():
            logger.warning(f"File not found: {sample.file_path}")
            continue
        
        try:
            if sample.file_type == 'midi':
                features = midi_extractor.extract(file_path)
                features_path = features_dir / f"{sample.sample_id}_features.json"
                features.save(features_path)
            elif sample.file_type == 'audio':
                features = audio_extractor.extract(file_path)
                features_path = features_dir / f"{sample.sample_id}_features.npz"
                features.save_npz(features_path)
            
            sample.features_path = str(features_path.relative_to(dataset_path))
            extracted += 1
            
        except Exception as e:
            logger.warning(f"Failed to extract features from {sample.file_path}: {e}")
    
    save_manifest(manifest, dataset_path / "manifest.json")
    logger.info(f"Extracted features for {extracted} samples")
    
    return extracted


# =============================================================================
# Augmentation
# =============================================================================


def augment_dataset(
    dataset_id: str,
    multiplier: int = 10,
    include_original: bool = True,
) -> int:
    """Augment all samples in a dataset."""
    from python.penta_core.ml.datasets.base import (
        load_manifest,
        save_manifest,
        create_sample_from_file,
        SampleAnnotation,
    )
    from python.penta_core.ml.datasets.augmentation import (
        MIDIAugmenter,
        AudioAugmenter,
        AugmentationConfig,
    )
    
    dataset_path = DATASETS_DIR / dataset_id
    manifest = load_manifest(dataset_path / "manifest.json")
    
    augmented_dir = dataset_path / "processed" / "augmented"
    
    config = AugmentationConfig(
        preserve_original=include_original,
        max_augmentations_per_sample=multiplier,
    )
    
    midi_augmenter = MIDIAugmenter(config)
    audio_augmenter = AudioAugmenter(config)
    
    total_generated = 0
    original_samples = list(manifest.samples)  # Copy to avoid modifying during iteration
    
    try:
        from tqdm import tqdm
        samples = tqdm(original_samples, desc="Augmenting")
    except ImportError:
        samples = original_samples
    
    for sample in samples:
        file_path = dataset_path / sample.file_path
        
        if not file_path.exists():
            continue
        
        try:
            if sample.file_type == 'midi':
                output_paths = midi_augmenter.augment(
                    file_path,
                    augmented_dir / "midi",
                    num_variations=multiplier,
                )
            elif sample.file_type == 'audio':
                output_paths = audio_augmenter.augment(
                    file_path,
                    augmented_dir / "audio",
                    num_variations=multiplier // 2,  # Audio augmentation is slower
                )
            else:
                continue
            
            # Add augmented samples to manifest
            for aug_path in output_paths:
                if aug_path == file_path:
                    continue  # Skip original
                
                aug_sample = create_sample_from_file(aug_path, dataset_path)
                aug_sample.is_synthetic = True
                aug_sample.parent_id = sample.sample_id
                aug_sample.augmentation_type = "augmented"
                
                # Copy annotations from original
                if sample.annotations:
                    aug_sample.annotations = SampleAnnotation(**sample.annotations.to_dict())
                
                manifest.add_sample(aug_sample)
                total_generated += 1
                
        except Exception as e:
            logger.warning(f"Failed to augment {sample.file_path}: {e}")
    
    save_manifest(manifest, dataset_path / "manifest.json")
    logger.info(f"Generated {total_generated} augmented samples")
    
    return total_generated


# =============================================================================
# Synthetic Data Generation
# =============================================================================


def generate_synthetic(
    dataset_id: str,
    target_model: str,
    num_samples: int = 5000,
) -> int:
    """Generate synthetic training data."""
    from python.penta_core.ml.datasets.base import (
        load_manifest,
        save_manifest,
        Sample,
        SampleAnnotation,
    )
    from python.penta_core.ml.datasets.synthetic import SyntheticGenerator
    
    dataset_path = DATASETS_DIR / dataset_id
    manifest = load_manifest(dataset_path / "manifest.json")
    
    synthetic_dir = dataset_path / "raw" / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticGenerator()
    generated = 0
    
    logger.info(f"Generating {num_samples} synthetic samples for {target_model}...")
    
    if target_model in ["emotionrecognizer", "emotion_recognizer"]:
        samples = generator.generate_emotion_samples(num_samples, synthetic_dir)
    elif target_model in ["melodytransformer", "melody_transformer"]:
        samples = generator.generate_melody_samples(num_samples, synthetic_dir)
    elif target_model in ["harmonypredictor", "harmony_predictor"]:
        samples = generator.generate_harmony_samples(num_samples, synthetic_dir)
    elif target_model in ["groovepredictor", "groove_predictor"]:
        samples = generator.generate_groove_samples(num_samples, synthetic_dir)
    elif target_model in ["dynamicsengine", "dynamics_engine"]:
        # Use emotion samples with dynamics focus
        samples = generator.generate_emotion_samples(num_samples, synthetic_dir)
    else:
        logger.error(f"Unknown target model: {target_model}")
        return 0
    
    # Add synthetic samples to manifest
    for syn_data in samples:
        sample = Sample(
            sample_id=syn_data['id'],
            file_path=f"raw/synthetic/{syn_data.get('emotion', target_model)}/{syn_data['id']}.mid",
            file_type='midi',
            split='train',
            is_synthetic=True,
            annotations=SampleAnnotation(
                emotion=syn_data.get('emotion', ''),
                valence=syn_data.get('valence', 0.0),
                arousal=syn_data.get('arousal', 0.0),
                key=syn_data.get('key', ''),
                mode=syn_data.get('mode', ''),
                tempo_bpm=syn_data.get('tempo_bpm', 0.0),
                groove_type=syn_data.get('groove_type', ''),
                swing_ratio=syn_data.get('swing_ratio', 0.5),
            ),
        )
        manifest.add_sample(sample)
        generated += 1
    
    save_manifest(manifest, dataset_path / "manifest.json")
    logger.info(f"Generated {generated} synthetic samples")
    
    return generated


# =============================================================================
# Validation
# =============================================================================


def validate_dataset_cmd(dataset_id: str) -> bool:
    """Validate a dataset and print report."""
    from python.penta_core.ml.datasets.validation import validate_dataset
    
    dataset_path = DATASETS_DIR / dataset_id
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_id}")
        return False
    
    report = validate_dataset(dataset_path)
    report.print_summary()
    
    # Save report
    report_path = dataset_path / "validation_report.json"
    report.save(report_path)
    logger.info(f"Report saved to: {report_path}")
    
    return report.is_valid


# =============================================================================
# Statistics
# =============================================================================


def print_stats(dataset_id: str):
    """Print dataset statistics."""
    from python.penta_core.ml.datasets.base import load_manifest
    from collections import Counter
    
    dataset_path = DATASETS_DIR / dataset_id
    manifest_path = dataset_path / "manifest.json"
    
    if not manifest_path.exists():
        logger.error(f"Dataset not found: {dataset_id}")
        return
    
    manifest = load_manifest(manifest_path)
    
    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_id}")
    print("=" * 60)
    
    print(f"\nTotal samples: {len(manifest.samples)}")
    print(f"  Train: {sum(1 for s in manifest.samples if s.split == 'train')}")
    print(f"  Val:   {sum(1 for s in manifest.samples if s.split == 'val')}")
    print(f"  Test:  {sum(1 for s in manifest.samples if s.split == 'test')}")
    
    # File types
    types = Counter(s.file_type for s in manifest.samples)
    print(f"\nFile types:")
    for t, c in types.items():
        print(f"  {t}: {c}")
    
    # Synthetic vs real
    synthetic = sum(1 for s in manifest.samples if s.is_synthetic)
    print(f"\nSynthetic: {synthetic} ({100*synthetic/len(manifest.samples):.1f}%)")
    
    # Categories
    emotions = Counter()
    for s in manifest.samples:
        if s.annotations and s.annotations.emotion:
            emotions[s.annotations.emotion] += 1
    
    if emotions:
        print(f"\nEmotion distribution:")
        for e, c in sorted(emotions.items(), key=lambda x: -x[1]):
            bar = "█" * min(40, int(c / max(emotions.values()) * 40))
            print(f"  {e:12} {c:5} {bar}")
    
    # Duration
    durations = [s.duration_sec for s in manifest.samples if s.duration_sec > 0]
    if durations:
        import numpy as np
        print(f"\nDuration:")
        print(f"  Total:   {sum(durations)/3600:.2f} hours")
        print(f"  Average: {np.mean(durations):.1f} sec")
        print(f"  Range:   {min(durations):.1f} - {max(durations):.1f} sec")
    
    # Size
    sizes = [s.file_size_bytes for s in manifest.samples if s.file_size_bytes > 0]
    if sizes:
        total_gb = sum(sizes) / (1024**3)
        print(f"\nSize: {total_gb:.2f} GB")
    
    print("\n" + "=" * 60)


# =============================================================================
# Full Pipeline
# =============================================================================


def run_full_pipeline(
    dataset_id: str,
    target_model: str,
    source_dir: Optional[Path] = None,
    target_samples: int = 1000,
    augment_multiplier: int = 10,
    synthetic_ratio: float = 0.5,
):
    """Run the complete dataset preparation pipeline."""
    logger.info("=" * 60)
    logger.info("Kelly ML Dataset Preparation Pipeline")
    logger.info("=" * 60)
    
    # 1. Create dataset
    logger.info("\n[1/7] Creating dataset structure...")
    create_dataset(dataset_id, target_model)
    
    # 2. Import files if source provided
    if source_dir:
        logger.info("\n[2/7] Importing files...")
        import_files(source_dir, dataset_id)
    else:
        logger.info("\n[2/7] No source directory provided, skipping import")
    
    # 3. Auto-annotate
    logger.info("\n[3/7] Auto-annotating samples...")
    auto_annotate(dataset_id)
    
    # 4. Extract features
    logger.info("\n[4/7] Extracting features...")
    extract_features(dataset_id)
    
    # 5. Augment
    logger.info("\n[5/7] Augmenting data...")
    augment_dataset(dataset_id, multiplier=augment_multiplier)
    
    # 6. Generate synthetic
    synthetic_count = int(target_samples * synthetic_ratio)
    logger.info(f"\n[6/7] Generating {synthetic_count} synthetic samples...")
    generate_synthetic(dataset_id, target_model, num_samples=synthetic_count)
    
    # 7. Validate
    logger.info("\n[7/7] Validating dataset...")
    is_valid = validate_dataset_cmd(dataset_id)
    
    # Summary
    print_stats(dataset_id)
    
    if is_valid:
        logger.info("✅ Dataset preparation complete!")
    else:
        logger.warning("⚠️ Dataset has validation issues - review before training")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Kelly ML Dataset Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create new dataset
    python scripts/prepare_datasets.py --create --dataset emotion_dataset_v1 --target-model emotionrecognizer
    
    # Import files
    python scripts/prepare_datasets.py --import-dir /path/to/midi --dataset emotion_dataset_v1
    
    # Run full pipeline
    python scripts/prepare_datasets.py --all --dataset emotion_dataset_v1 --target-model emotionrecognizer
    
    # Generate synthetic data only
    python scripts/prepare_datasets.py --synthesize --count 5000 --dataset emotion_dataset_v1 --target-model emotionrecognizer
    
    # Validate and show stats
    python scripts/prepare_datasets.py --validate --stats --dataset emotion_dataset_v1
        """,
    )
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="emotion_dataset_v1",
                       help="Dataset ID")
    parser.add_argument("--target-model", type=str, default="emotionrecognizer",
                       choices=["emotionrecognizer", "melodytransformer", "harmonypredictor", 
                               "dynamicsengine", "groovepredictor"],
                       help="Target model for the dataset")
    
    # Actions
    parser.add_argument("--all", action="store_true",
                       help="Run full pipeline")
    parser.add_argument("--create", action="store_true",
                       help="Create dataset structure")
    parser.add_argument("--import-dir", type=str,
                       help="Import files from directory")
    parser.add_argument("--annotate", action="store_true",
                       help="Auto-annotate samples")
    parser.add_argument("--annotate-interactive", action="store_true",
                       help="Interactively annotate samples")
    parser.add_argument("--extract-features", action="store_true",
                       help="Extract features")
    parser.add_argument("--augment", action="store_true",
                       help="Augment data")
    parser.add_argument("--synthesize", action="store_true",
                       help="Generate synthetic data")
    parser.add_argument("--validate", action="store_true",
                       help="Validate dataset")
    parser.add_argument("--stats", action="store_true",
                       help="Print statistics")
    
    # Options
    parser.add_argument("--multiplier", type=int, default=10,
                       help="Augmentation multiplier")
    parser.add_argument("--count", type=int, default=5000,
                       help="Number of synthetic samples")
    parser.add_argument("--target-samples", type=int, default=1000,
                       help="Target samples per category")
    
    args = parser.parse_args()
    
    # Ensure datasets directory exists
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run requested actions
    if args.all:
        source_dir = Path(args.import_dir) if args.import_dir else None
        run_full_pipeline(
            args.dataset,
            args.target_model,
            source_dir=source_dir,
            target_samples=args.target_samples,
            augment_multiplier=args.multiplier,
        )
    else:
        if args.create:
            create_dataset(args.dataset, args.target_model)
        
        if args.import_dir:
            import_files(Path(args.import_dir), args.dataset)
        
        if args.annotate:
            auto_annotate(args.dataset)
        
        if args.annotate_interactive:
            interactive_annotate(args.dataset)
        
        if args.extract_features:
            extract_features(args.dataset)
        
        if args.augment:
            augment_dataset(args.dataset, multiplier=args.multiplier)
        
        if args.synthesize:
            generate_synthetic(args.dataset, args.target_model, num_samples=args.count)
        
        if args.validate:
            validate_dataset_cmd(args.dataset)
        
        if args.stats:
            print_stats(args.dataset)


if __name__ == "__main__":
    main()

