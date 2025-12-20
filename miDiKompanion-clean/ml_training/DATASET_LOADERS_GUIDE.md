# Real Dataset Loaders Guide

This guide explains how to use the real dataset loaders for training the Kelly ML models.

## Overview

The dataset loaders replace synthetic data with real datasets:

- **DEAM**: Emotion recognition (audio + valence/arousal labels)
- **Lakh MIDI**: Melody generation (176K MIDI files)
- **MAESTRO**: Dynamics engine (piano with velocity data)
- **Groove MIDI**: Groove prediction (drum patterns)
- **Harmony**: Chord progressions (JSON format)

## Installation

Install required dependencies:

```bash
pip install librosa soundfile mido numpy torch
```

## Dataset Structure

### DEAM Dataset

```
datasets/deam/
├── audio/
│   ├── audio_001.wav
│   └── ...
└── annotations/
    └── annotations.csv  (filename,valence,arousal)
```

### Lakh MIDI Dataset

```
datasets/lakh_midi/
├── midi_files/
│   ├── *.mid
│   └── ...
└── emotion_labels.json  (optional, {filename: {valence, arousal}})
```

### MAESTRO Dataset

```
datasets/maestro/
└── *.midi
```

### Groove MIDI Dataset

```
datasets/groove/
└── *.midi
```

### Harmony Dataset

```
datasets/harmony/
└── chord_progressions.json
```

## Usage

### Basic Training with Real Data

```bash
# Train with real datasets
python ml_training/train_all_models.py \
    --datasets-dir ./datasets \
    --output ./trained_models \
    --epochs 50
```

### Force Synthetic Data

```bash
# Use synthetic data even if real datasets available
python ml_training/train_all_models.py \
    --use-synthetic \
    --output ./trained_models
```

### Test Dataset Loaders

```bash
# Test individual dataset loaders
python ml_training/dataset_loaders.py deam ./datasets/deam
python ml_training/dataset_loaders.py lakh ./datasets/lakh_midi
python ml_training/dataset_loaders.py maestro ./datasets/maestro
python ml_training/dataset_loaders.py groove ./datasets/groove
python ml_training/dataset_loaders.py harmony ./datasets/harmony
```

## Dataset Loader Details

### DEAMDataset

- **Input**: Audio files + CSV annotations
- **Output**: Mel-spectrogram features (128 dims) + emotion embedding (64 dims)
- **Features**: Extracts mel-spectrogram from audio, normalizes to [-1, 1]
- **Emotion**: Creates 64-dim embedding from valence/arousal labels

### LakhMIDIDataset

- **Input**: MIDI files + optional emotion labels
- **Output**: Emotion embedding (64 dims) + MIDI note probabilities (128 dims)
- **Features**: Extracts note patterns from MIDI, converts to probability distribution
- **Emotion**: Uses provided labels or generates from filename

### MAESTRODataset

- **Input**: MIDI files with velocity data
- **Output**: Compact context (32 dims) + expression parameters (16 dims)
- **Features**: Extracts velocity and timing patterns
- **Context**: Combines dynamics features with emotion-like features

### GrooveMIDIDataset

- **Input**: Drum MIDI files
- **Output**: Emotion embedding (64 dims) + groove parameters (32 dims)
- **Features**: Extracts timing deviations and velocity patterns
- **Groove**: 32-dim vector with timing/velocity histograms

### HarmonyDataset

- **Input**: JSON file with chord progressions
- **Output**: Context (128 dims) + chord probabilities (64 dims)
- **Features**: Maps chord names to indices, creates probability distribution
- **Context**: Combines emotion with audio features

## Fallback Behavior

If real datasets are not available or fail to load, the training script automatically falls back to synthetic data:

1. Tries to load real dataset
2. If error occurs, prints warning
3. Falls back to synthetic dataset
4. Continues training

## Performance Considerations

- **DEAM**: Audio processing can be slow. Consider limiting duration or using pre-extracted features
- **Lakh MIDI**: Very large dataset. Use `max_files` parameter to limit
- **MAESTRO**: MIDI parsing is fast, but large datasets may take time
- **Groove MIDI**: Small dataset, loads quickly
- **Harmony**: JSON parsing is very fast

## Troubleshooting

### "librosa not available"

Install librosa:

```bash
pip install librosa soundfile
```

### "mido not available"

Install mido:

```bash
pip install mido
```

### "No MIDI files found"

Check that:

1. Dataset directory path is correct
2. MIDI files have `.mid` or `.midi` extension
3. Files are not in subdirectories (loaders search recursively)

### "Annotations file not found"

For DEAM:

- Check that `annotations.csv` exists
- Try different common locations (see DEAMDataset code)
- Verify CSV format: `filename,valence,arousal`

### "No valid samples found"

- Check file paths match between annotations and audio files
- Verify file extensions match
- Check for permission issues

## Next Steps

1. **Download Datasets**: Use `training_pipe/scripts/download_datasets.py`
2. **Prepare Data**: Run `training_pipe/scripts/prepare_datasets.py`
3. **Train Models**: Use `train_all_models.py` with `--datasets-dir`
4. **Validate**: Check training metrics and model exports

## References

- [DEAM Dataset](https://cvml.unige.ch/databases/DEAM/)
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
- [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove)
