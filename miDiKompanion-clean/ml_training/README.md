# EmotionRecognizer Training - Real Dataset Integration

This directory contains the training infrastructure for the EmotionRecognizer model with **real dataset support**.

## What's New

✅ **Real Dataset Loader** (`dataset_loaders.py`)

- Supports DEAM, PMEmo, and custom CSV/JSON formats
- Automatic mel-spectrogram feature extraction
- Emotion label loading (valence/arousal → 64-dim embeddings)
- Optional feature caching for faster training

✅ **Enhanced Training Script** (`train_emotion_model.py`)

- Real dataset integration
- Validation split with early stopping
- Learning rate scheduling
- Gradient clipping
- Training loss curve tracking
- Checkpoint saving/loading
- Model export to RTNeural JSON format

✅ **Test Script** (`test_dataset_loader.py`)

- Validates dataset loader functionality
- Tests feature extraction
- Verifies train/val splits

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchaudio librosa numpy
```

### 2. Prepare Your Dataset

Create a directory structure:

```
datasets/
└── audio/
    ├── audio_001.wav
    ├── audio_002.wav
    └── labels.csv
```

**labels.csv format:**

```csv
filename,valence,arousal
audio_001.wav,0.8,0.9
audio_002.wav,-0.6,0.3
calm_001.wav,0.2,-0.5
```

**Or use JSON format (emotion_labels.json):**

```json
{
  "audio_001.wav": {"valence": 0.8, "arousal": 0.9},
  "audio_002.wav": {"valence": -0.6, "arousal": 0.3}
}
```

### 3. Test Dataset Loader

```bash
python test_dataset_loader.py datasets/audio
```

### 4. Train the Model

**Basic training:**

```bash
python train_emotion_model.py --dataset datasets/audio --epochs 50
```

**Advanced training with all features:**

```bash
python train_emotion_model.py \
    --dataset datasets/audio \
    --labels datasets/audio/labels.csv \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --val-ratio 0.2 \
    --patience 15 \
    --use-scheduler \
    --save-curves \
    --cache-features \
    --checkpoint models/emotion_model.pth \
    --output models/emotion_model.json
```

## Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Path to audio dataset directory | Required |
| `--labels` | Path to labels file (auto-detected if not provided) | Auto |
| `--epochs` | Number of training epochs | 50 |
| `--batch-size` | Batch size | 32 |
| `--learning-rate` | Learning rate | 0.001 |
| `--val-ratio` | Validation set ratio | 0.2 |
| `--patience` | Early stopping patience (epochs) | 10 |
| `--duration` | Audio duration to analyze (seconds) | 2.0 |
| `--use-scheduler` | Use learning rate scheduler | False |
| `--max-grad-norm` | Gradient clipping threshold (0 to disable) | 1.0 |
| `--cache-features` | Cache extracted features in memory | False |
| `--save-curves` | Save training loss curves to JSON | False |
| `--num-workers` | Data loader workers | 0 |
| `--seed` | Random seed | 42 |

## Model Architecture

The EmotionRecognizer model architecture matches `train_all_models.py`:

- **Input**: 128-dim mel-spectrogram features
- **Dense**: 128 → 512 (tanh)
- **Dense**: 512 → 256 (tanh)
- **LSTM**: 256 → 128
- **Dense**: 128 → 64 (tanh)
- **Output**: 64-dim emotion embedding

**Total**: ~500K parameters, ~2MB memory

## Output Files

After training, you'll get:

1. **Checkpoint** (`models/emotion_model.pth`)
   - PyTorch state dict with best model
   - Can be loaded to continue training

2. **RTNeural JSON** (`models/emotion_model.json`)
   - Export for C++ plugin integration
   - Copy to `Resources/emotionrecognizer.json`

3. **Training Curves** (`models/training_curves.json`) - if `--save-curves` used
   - Train/val loss history
   - Best epoch info

## Integration with Plugin

1. **Copy model to plugin:**

   ```bash
   cp models/emotion_model.json "Kelly MIDI Companion.app/Contents/Resources/models/emotionrecognizer.json"
   ```

2. **Rebuild plugin** (if needed):

   ```bash
   cmake --build build --config Release
   ```

3. **Verify in plugin logs:**

   ```
   MultiModelProcessor initialized:
     EmotionRecognizer: 497,664 params
   ```

## Dataset Format Details

### Supported Formats

1. **DEAM Format**
   - CSV with `filename,valence,arousal`
   - Valence: -1.0 (negative) to 1.0 (positive)
   - Arousal: 0.0 (calm) to 1.0 (excited)

2. **PMEmo Format**
   - JSON with nested emotion labels
   - Same valence/arousal range

3. **Custom Format**
   - Any CSV with `filename,valence,arousal` columns
   - Or JSON with `{filename: {valence, arousal}}` structure

### Label Conversion

The loader automatically converts 2D valence/arousal labels to 64-dim embeddings:

- First 32 dimensions: valence-related features
- Last 32 dimensions: arousal-related features

## Example Training Session

```
================================================================================
Kelly MIDI Companion - Emotion Recognition Model Training
================================================================================

Using device: cuda

Model architecture:
  Input: 128 mel-spectrogram features
  Dense: 128 → 512 (tanh)
  Dense: 512 → 256 (tanh)
  LSTM: 256 → 128
  Dense: 128 → 64 (tanh)
  Output: 64-dimensional emotion embedding

Total parameters: 497,664 (~1,946.0 KB)

Loading dataset...
  Dataset path: datasets/audio
  ✓ Loaded 1802 samples
  Training samples: 1441
  Validation samples: 361

Training for 100 epochs...
Early stopping patience: 15 epochs

Epoch [1/100] Train Loss: 0.452301 | Val Loss: 0.389421 | LR: 0.001000
  ✓ Saved best model (val_loss: 0.389421)
Epoch [2/100] Train Loss: 0.345123 | Val Loss: 0.312456 | LR: 0.001000
  ✓ Saved best model (val_loss: 0.312456)
...
Epoch [45/100] Train Loss: 0.098234 | Val Loss: 0.105678 | LR: 0.000500
  ✓ Saved best model (val_loss: 0.105678)

Early stopping triggered after 60 epochs
Best validation loss: 0.104523

Training complete!
Best validation loss: 0.104523
Model checkpoint saved to: models/emotion_model.pth
Loaded best model from epoch 45

Exporting to RTNeural JSON format...
Model exported to models/emotion_model.json
  Architecture: 128 → 512 → 256 → 128 → 64
```

## Next Steps

1. **Test with your dataset** - Use `test_dataset_loader.py` first
2. **Train the model** - Run with your audio files and labels
3. **Export and integrate** - Copy JSON to plugin resources
4. **Repeat for other models** - Use this as a template for MelodyTransformer, etc.

## Troubleshooting

### "No audio files found"

- Check that your dataset directory contains `.wav`, `.mp3`, `.flac`, `.m4a`, or `.ogg` files
- Verify the path is correct

### "No labels found"

- Create a `labels.csv` file in your audio directory
- Or provide path with `--labels` option
- The loader will use placeholder labels if none found (for testing)

### "CUDA out of memory"

- Reduce batch size: `--batch-size 16`
- Don't cache features: remove `--cache-features`
- Reduce audio duration: `--duration 1.0`

### "Training too slow"

- Use GPU: Ensure CUDA is available
- Enable feature caching: `--cache-features`
- Increase batch size (if memory allows)
- Reduce number of workers if causing issues: `--num-workers 0`

## Notes

- The model architecture matches `train_all_models.py` for consistency
- Feature extraction uses librosa's mel-spectrogram (128 bands)
- Labels are automatically converted from 2D (valence, arousal) to 64-dim embeddings
- Early stopping saves the best model based on validation loss
- Checkpoints include full training state for resuming
