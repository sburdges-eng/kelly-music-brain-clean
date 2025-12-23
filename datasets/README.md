# Kelly ML Training Datasets

This directory contains training datasets for Kelly's 5-model ML pipeline.

## Target: 1000 samples per category, <3TB total

## Dataset Structure

```
datasets/
├── emotion_dataset_v1/          # EmotionRecognizer training
│   ├── manifest.json            # Dataset index
│   ├── config.json              # Dataset configuration
│   ├── raw/
│   │   ├── midi/                # Original MIDI files
│   │   ├── audio/               # Original audio files
│   │   └── synthetic/           # Generated samples
│   ├── processed/
│   │   ├── features/            # Extracted features (.json, .npz)
│   │   └── augmented/           # Augmented samples
│   ├── annotations/             # Per-file annotation JSONs
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── melody_dataset_v1/           # MelodyTransformer training
├── harmony_dataset_v1/          # HarmonyPredictor training
├── dynamics_dataset_v1/         # DynamicsEngine training
└── groove_dataset_v1/           # GroovePredictor training
```

## Quick Start

### 1. Create a Dataset

```bash
python scripts/prepare_datasets.py --create \
    --dataset emotion_dataset_v1 \
    --target-model emotionrecognizer
```

### 2. Import Files

```bash
# From your MIDI/audio collection
python scripts/prepare_datasets.py --import-dir /path/to/music \
    --dataset emotion_dataset_v1
```

### 3. Annotate

```bash
# Auto-annotate from directory structure (happy/, sad/, etc.)
python scripts/prepare_datasets.py --annotate --dataset emotion_dataset_v1

# Or interactive annotation
python scripts/prepare_datasets.py --annotate-interactive --dataset emotion_dataset_v1
```

### 4. Extract Features

```bash
python scripts/prepare_datasets.py --extract-features --dataset emotion_dataset_v1
```

### 5. Augment (100 files → 1000+)

```bash
python scripts/prepare_datasets.py --augment --multiplier 10 --dataset emotion_dataset_v1
```

### 6. Generate Synthetic Data

```bash
python scripts/prepare_datasets.py --synthesize --count 5000 \
    --dataset emotion_dataset_v1 --target-model emotionrecognizer
```

### 7. Validate

```bash
python scripts/prepare_datasets.py --validate --stats --dataset emotion_dataset_v1
```

### Full Pipeline (All at Once)

```bash
python scripts/prepare_datasets.py --all \
    --dataset emotion_dataset_v1 \
    --target-model emotionrecognizer \
    --import-dir /path/to/music \
    --multiplier 10 \
    --count 5000
```

## Sample Annotation Format

Each sample can have annotations in `manifest.json`:

```json
{
  "sample_id": "abc12345",
  "file_path": "raw/midi/song.mid",
  "annotations": {
    "emotion": "happy",
    "valence": 0.8,
    "arousal": 0.6,
    "key": "C",
    "mode": "major",
    "tempo_bpm": 120,
    "groove_type": "straight",
    "swing_ratio": 0.5,
    "articulation": "staccato",
    "chord_progression": ["C", "G", "Am", "F"],
    "quality_score": 0.9,
    "is_verified": true,
    "notes": "Upbeat pop feel"
  }
}
```

## What Each Model Needs

| Model | Primary Data | Key Labels | Target Count |
|-------|-------------|------------|--------------|
| EmotionRecognizer | Audio clips | emotion, valence, arousal | 1000/emotion |
| MelodyTransformer | MIDI melodies | emotion, contour, key | 5000+ |
| HarmonyPredictor | Chord progressions | emotion, key, chords | 2000+ |
| DynamicsEngine | Velocity curves | dynamics, articulation | 1000+ |
| GroovePredictor | Timing data | groove_type, swing_ratio | 1000/groove |

## Augmentation Types

### MIDI Augmentations
- **Transpose**: Shift by -6 to +6 semitones
- **Time Stretch**: 0.9x to 1.1x speed
- **Velocity Scale**: 0.8x to 1.2x dynamics
- **Timing Jitter**: ±5ms humanization
- **Note Dropout**: Random 2% note removal

### Audio Augmentations
- **Pitch Shift**: -2 to +2 semitones
- **Time Stretch**: 0.9x to 1.1x speed
- **Noise Addition**: 0-0.5% Gaussian noise
- **Gain Adjustment**: 0.8x to 1.2x volume

## Synthetic Data Generation

Generates training data from music theory rules:

### Emotion Samples
- Chord progressions with emotional mappings
- Major → happy, Minor → sad
- Tempo and dynamics aligned with emotion

### Melody Samples  
- Scale-based note sequences
- Proper voice leading
- Varied contours and ranges

### Harmony Samples
- Circle of fifths relationships
- Common progressions per emotion
- Proper voice leading

### Groove Samples
- Straight (swing_ratio=0.5)
- Swing (swing_ratio=0.67)
- Shuffle, Laid-back, Rushed

## Validation Checks

The validator checks:

1. **Balance**: Equal samples per category
2. **Diversity**: Varied keys, tempos, modes
3. **Quality**: File integrity, proper annotations
4. **Structure**: Correct directory layout

```bash
# Run validation
python scripts/prepare_datasets.py --validate --dataset emotion_dataset_v1

# Output:
# ✅ VALID
# Total Samples: 8500
# Balance Score: 92%
# Diversity Score: 78%
# Quality Score: 99%
```

## Python API

```python
from python.penta_core.ml.datasets import (
    DatasetConfig,
    create_dataset_structure,
    load_manifest,
    extract_midi_features,
    extract_audio_features,
    augment_midi,
    generate_emotion_samples,
    validate_dataset,
)

# Create dataset
config = DatasetConfig(
    dataset_id="my_dataset",
    target_model="emotionrecognizer",
)
manifest = create_dataset_structure("datasets", config)

# Extract features
features = extract_midi_features("song.mid")
print(f"Detected key: {features.key_signature}")
print(f"Groove type: {features.groove_type}")

# Generate synthetic data
samples = generate_emotion_samples(1000)

# Validate
report = validate_dataset("datasets/my_dataset")
report.print_summary()
```

## Size Estimation

| Samples | MIDI | Audio (22kHz) | Features |
|---------|------|---------------|----------|
| 1,000 | ~50MB | ~2GB | ~100MB |
| 10,000 | ~500MB | ~20GB | ~1GB |
| 100,000 | ~5GB | ~200GB | ~10GB |

**Your 3TB Budget:**
- ~150,000 audio samples at 30s each
- Or ~1M+ MIDI samples
- Plus features and augmentations

## Next Steps

1. Gather your raw music files (MIDI preferred)
2. Organize by emotion/category in directories
3. Run the full pipeline
4. Review validation report
5. Iterate until targets are met
6. Train with `python scripts/train.py`

See [MK_TRAINING_GUIDELINES.md](../docs/MK_TRAINING_GUIDELINES.md) for the full workflow.

