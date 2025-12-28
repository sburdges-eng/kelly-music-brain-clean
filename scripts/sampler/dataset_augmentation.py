#!/usr/bin/env python3
"""
Dataset Augmentation Workflow (Integration Example 3)
====================================================
Augment downloaded samples for ML training.

Augmentation techniques:
- Time stretching (tempo changes)
- Pitch shifting
- Adding noise
- Volume normalization
- Mixing samples
"""

import sys
from pathlib import Path
import json
from datetime import datetime

BRAIN_PYTHON_DIR = Path(__file__).parent.parent


class DatasetAugmenter:
    """Dataset augmentation for emotion samples."""
    
    def __init__(self):
        self.augmented_count = 0
        self.augmentation_log = []
    
    def time_stretch(self, audio_file, rate=1.2):
        """Mock time stretching (would use librosa)."""
        print(f"  Time stretch: {rate}x")
        return f"{audio_file}_stretch_{rate}.mp3"
    
    def pitch_shift(self, audio_file, steps=2):
        """Mock pitch shifting (would use librosa)."""
        print(f"  Pitch shift: {steps} steps")
        return f"{audio_file}_pitch_{steps}.mp3"
    
    def add_noise(self, audio_file, noise_level=0.05):
        """Mock noise addition (would use librosa)."""
        print(f"  Add noise: {noise_level}")
        return f"{audio_file}_noise_{noise_level}.mp3"
    
    def normalize_volume(self, audio_file):
        """Mock volume normalization (would use librosa)."""
        print(f"  Normalize volume")
        return f"{audio_file}_normalized.mp3"
    
    def mix_samples(self, audio_file1, audio_file2, mix_ratio=0.5):
        """Mock sample mixing (would use librosa)."""
        print(f"  Mix samples: {mix_ratio}")
        return f"mix_{Path(audio_file1).stem}_{Path(audio_file2).stem}.mp3"
    
    def augment_sample(self, sample_path, techniques=None):
        """Apply augmentation techniques to a sample."""
        if techniques is None:
            techniques = ["time_stretch", "pitch_shift", "normalize"]
        
        print(f"\nAugmenting: {Path(sample_path).name}")
        
        augmented_files = []
        
        for technique in techniques:
            if technique == "time_stretch":
                for rate in [0.8, 1.2]:
                    aug_file = self.time_stretch(sample_path, rate)
                    augmented_files.append({
                        "original": str(sample_path),
                        "augmented": aug_file,
                        "technique": f"time_stretch_{rate}"
                    })
                    self.augmented_count += 1
            
            elif technique == "pitch_shift":
                for steps in [-2, 2]:
                    aug_file = self.pitch_shift(sample_path, steps)
                    augmented_files.append({
                        "original": str(sample_path),
                        "augmented": aug_file,
                        "technique": f"pitch_shift_{steps}"
                    })
                    self.augmented_count += 1
            
            elif technique == "normalize":
                aug_file = self.normalize_volume(sample_path)
                augmented_files.append({
                    "original": str(sample_path),
                    "augmented": aug_file,
                    "technique": "normalize"
                })
                self.augmented_count += 1
        
        self.augmentation_log.extend(augmented_files)
        return augmented_files
    
    def augment_emotion_dataset(self, emotion, max_samples=5):
        """Augment all samples for a specific emotion."""
        staging_dir = BRAIN_PYTHON_DIR / "scripts" / "emotion_instrument_staging"
        emotion_dir = staging_dir / "base" / emotion.lower()
        
        if not emotion_dir.exists():
            print(f"⚠ No samples found for {emotion}")
            return
        
        print(f"="*70)
        print(f"AUGMENTING {emotion.upper()} SAMPLES")
        print(f"="*70)
        
        samples_processed = 0
        
        for instrument_dir in emotion_dir.iterdir():
            if not instrument_dir.is_dir():
                continue
            
            samples = list(instrument_dir.glob("*.mp3"))[:max_samples]
            
            for sample in samples:
                self.augment_sample(sample)
                samples_processed += 1
        
        print(f"\n✓ Processed {samples_processed} samples")
        print(f"✓ Created {self.augmented_count} augmented versions")
    
    def save_augmentation_log(self, output_file="augmentation_log.json"):
        """Save augmentation log."""
        log_data = {
            "augmentations": self.augmentation_log,
            "summary": {
                "total_augmented": self.augmented_count,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n✓ Log saved to {output_file}")
    
    def generate_augmentation_report(self):
        """Generate augmentation report."""
        print(f"\n{'='*70}")
        print("AUGMENTATION REPORT")
        print(f"{'='*70}")
        
        # Technique distribution
        techniques = {}
        for aug in self.augmentation_log:
            tech = aug['technique']
            techniques[tech] = techniques.get(tech, 0) + 1
        
        print("\nTechniques Applied:")
        for tech, count in sorted(techniques.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tech}: {count}")
        
        print(f"\nTotal Augmented Files: {self.augmented_count}")
        print(f"Original Files: {len(set(aug['original'] for aug in self.augmentation_log))}")
        print(f"Multiplication Factor: {self.augmented_count / max(len(set(aug['original'] for aug in self.augmentation_log)), 1):.1f}x")


def main():
    if len(sys.argv) < 2:
        print("DATASET AUGMENTATION WORKFLOW (Integration Example 3)")
        print("\nUSAGE:")
        print("  python dataset_augmentation.py <emotion> [max_samples]")
        print("\nEXAMPLE:")
        print("  python dataset_augmentation.py happy 5")
        print("  python dataset_augmentation.py sad 10")
        return
    
    emotion = sys.argv[1]
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    augmenter = DatasetAugmenter()
    augmenter.augment_emotion_dataset(emotion, max_samples)
    augmenter.save_augmentation_log()
    augmenter.generate_augmentation_report()


if __name__ == "__main__":
    main()
