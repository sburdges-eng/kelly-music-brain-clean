#!/usr/bin/env python3
"""
Example: Integrating Auto Emotion Sampler with ML Training Workflow

This script demonstrates how to use the auto_emotion_sampler.py output
for training Kelly's emotion recognition models.

Workflow:
1. Download samples using auto_emotion_sampler.py
2. Process samples for ML training
3. Generate training datasets
4. Train emotion recognition models
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add scripts directory to path
BRAIN_PYTHON_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BRAIN_PYTHON_DIR / "scripts"))

import auto_emotion_sampler


def load_sampler_metadata() -> Dict[str, Any]:
    """Load metadata from auto_emotion_sampler downloads."""
    download_log = BRAIN_PYTHON_DIR / "scripts" / "emotion_instrument_downloads.json"
    
    if not download_log.exists():
        print(f"⚠ No download log found at {download_log}")
        print("Run: python scripts/auto_emotion_sampler.py start")
        return {}
    
    with open(download_log, 'r') as f:
        return json.load(f)


def get_samples_for_emotion(emotion: str, instrument: str = None) -> List[Path]:
    """
    Get all downloaded samples for a specific emotion.
    
    Args:
        emotion: Base emotion name (happy, sad, angry, fear, surprise, disgust)
        instrument: Optional instrument filter (piano, guitar, drums, vocals)
        
    Returns:
        List of sample file paths
    """
    staging_dir = BRAIN_PYTHON_DIR / "scripts" / "emotion_instrument_staging"
    
    if not staging_dir.exists():
        print(f"⚠ Staging directory not found: {staging_dir}")
        return []
    
    samples = []
    
    # Check base emotions
    base_dir = staging_dir / "base" / emotion.lower()
    if base_dir.exists():
        if instrument:
            # Specific instrument
            instrument_dir = base_dir / instrument.lower()
            if instrument_dir.exists():
                samples.extend(instrument_dir.glob("*.mp3"))
        else:
            # All instruments
            for inst_dir in base_dir.iterdir():
                if inst_dir.is_dir():
                    samples.extend(inst_dir.glob("*.mp3"))
    
    # Check sub-emotions
    sub_dir = staging_dir / "sub" / emotion.lower()
    if sub_dir.exists():
        if instrument:
            instrument_dir = sub_dir / instrument.lower()
            if instrument_dir.exists():
                samples.extend(instrument_dir.glob("*.mp3"))
        else:
            for inst_dir in sub_dir.iterdir():
                if inst_dir.is_dir():
                    samples.extend(inst_dir.glob("*.mp3"))
    
    return samples


def prepare_training_dataset():
    """
    Prepare a training dataset from downloaded samples.
    
    Creates a dataset manifest that can be used by ML training scripts.
    """
    # Import from nested music_brain package
    sys.path.insert(0, str(BRAIN_PYTHON_DIR / "music_brain"))
    from music_brain.emotion_thesaurus import EmotionThesaurus
    
    thesaurus = EmotionThesaurus()
    metadata = load_sampler_metadata()
    
    if not metadata:
        print("No samples downloaded yet. Run the sampler first.")
        return
    
    dataset = {
        "name": "Kelly Emotion Recognition Dataset",
        "version": "1.0",
        "created_from": "auto_emotion_sampler",
        "base_emotions": thesaurus.BASE_EMOTIONS,
        "samples": []
    }
    
    # Process each emotion
    for emotion in thesaurus.BASE_EMOTIONS:
        samples = get_samples_for_emotion(emotion)
        
        print(f"\n{emotion.upper()}: {len(samples)} samples")
        
        for sample_path in samples:
            dataset["samples"].append({
                "path": str(sample_path),
                "emotion": emotion,
                "instrument": sample_path.parent.name,
                "level": sample_path.parent.parent.parent.name,  # base or sub
                "filename": sample_path.name
            })
    
    # Save dataset manifest
    output_path = BRAIN_PYTHON_DIR / "data" / "emotion_dataset_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Dataset manifest saved to: {output_path}")
    print(f"  Total samples: {len(dataset['samples'])}")
    print(f"  Base emotions: {len(thesaurus.BASE_EMOTIONS)}")


def show_sample_statistics():
    """Show statistics about downloaded samples."""
    # Import from nested music_brain package
    sys.path.insert(0, str(BRAIN_PYTHON_DIR / "music_brain"))
    from music_brain.emotion_thesaurus import EmotionThesaurus
    
    thesaurus = EmotionThesaurus()
    
    print("="*70)
    print("SAMPLE STATISTICS FOR ML TRAINING")
    print("="*70)
    
    total_samples = 0
    
    for emotion in thesaurus.BASE_EMOTIONS:
        samples = get_samples_for_emotion(emotion)
        total_samples += len(samples)
        
        print(f"\n{emotion.upper()}: {len(samples)} samples")
        
        # Breakdown by instrument
        for instrument in ["piano", "guitar", "drums", "vocals"]:
            inst_samples = get_samples_for_emotion(emotion, instrument)
            if inst_samples:
                print(f"  {instrument}: {len(inst_samples)}")
    
    print(f"\n{'='*70}")
    print(f"Total samples: {total_samples}")
    print(f"Base emotions: {len(thesaurus.BASE_EMOTIONS)}")
    print(f"Target instruments: 4 (piano, guitar, drums, vocals)")
    
    if total_samples > 0:
        avg_per_emotion = total_samples / len(thesaurus.BASE_EMOTIONS)
        print(f"Average per emotion: {avg_per_emotion:.1f}")


def example_training_integration():
    """
    Example of how to integrate samples into model training.
    
    This is a simplified example - actual training would use
    librosa for audio feature extraction and PyTorch/ONNX for training.
    """
    print("\n" + "="*70)
    print("EXAMPLE: ML TRAINING INTEGRATION")
    print("="*70)
    
    print("\n1. Download samples:")
    print("   python scripts/auto_emotion_sampler.py start")
    
    print("\n2. Prepare dataset manifest:")
    print("   python scripts/ml_integration_example.py prepare")
    
    print("\n3. Extract audio features:")
    print("   # Use librosa to extract features from each sample")
    print("   import librosa")
    print("   y, sr = librosa.load('sample.mp3')")
    print("   mfcc = librosa.feature.mfcc(y=y, sr=sr)")
    print("   chroma = librosa.feature.chroma_stft(y=y, sr=sr)")
    
    print("\n4. Train emotion classifier:")
    print("   # Train a neural network to classify emotions")
    print("   # Input: Audio features (MFCC, chroma, etc.)")
    print("   # Output: Emotion class (6 base emotions)")
    
    print("\n5. Export trained model:")
    print("   # Export to ONNX for use in production")
    print("   torch.onnx.export(model, 'emotion_classifier.onnx')")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("="*70)
        print("ML TRAINING INTEGRATION - AUTO EMOTION SAMPLER")
        print("="*70)
        print("\nIntegrates auto_emotion_sampler.py with Kelly ML training workflow")
        print("\nCOMMANDS:")
        print("  stats    - Show sample statistics")
        print("  prepare  - Prepare training dataset manifest")
        print("  example  - Show training integration example")
        print("\nEXAMPLE:")
        print("  python scripts/ml_integration_example.py stats")
        return
    
    command = sys.argv[1].lower()
    
    if command == "stats":
        show_sample_statistics()
    elif command == "prepare":
        prepare_training_dataset()
    elif command == "example":
        example_training_integration()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
