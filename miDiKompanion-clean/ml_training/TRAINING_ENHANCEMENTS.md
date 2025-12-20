# Training Pipeline Enhancements

## Overview

The training pipeline has been enhanced with comprehensive validation, early stopping, metrics tracking, checkpoint management, and evaluation capabilities.

## New Features

### 1. Early Stopping

Prevents overfitting by monitoring validation loss and stopping training when no improvement is detected.

**Usage:**

```python
from training_utils import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,              # Wait 10 epochs before stopping
    min_delta=0.001,         # Minimum improvement required
    mode='min',               # 'min' for loss, 'max' for accuracy
    restore_best_weights=True # Restore best weights when stopping
)
```

### 2. Validation Split

Automatically splits datasets into training and validation sets.

**Usage:**

```bash
python train_all_models.py --validation-split 0.2
```

This uses 80% for training and 20% for validation.

### 3. Training Metrics

Comprehensive metrics tracking including:

- Training loss
- Validation loss
- Training metrics (e.g., cosine similarity, accuracy)
- Validation metrics
- Custom metrics

**Metrics are automatically saved to:**

- JSON format: `history/{model_name}_history.json`
- CSV format: `history/{model_name}_history.csv`
- Plots: `plots/{model_name}_loss.png` and `{model_name}_metric.png`

### 4. Checkpoint Management

Automatic checkpoint saving:

- **Latest checkpoint**: Saved every epoch (`{model_name}_latest.pt`)
- **Best checkpoint**: Saved when validation loss improves (`{model_name}_best.pt`)

**Checkpoints include:**

- Model state dict
- Optimizer state dict
- Training metrics history
- Epoch number

### 5. Resume Training

Resume training from a checkpoint:

```bash
python train_all_models.py --resume EmotionRecognizer
```

### 6. Model Evaluation

Comprehensive evaluation script with multiple metrics:

```bash
python evaluate_models.py \
    --checkpoint-dir ./trained_models/checkpoints \
    --output ./evaluation_results
```

**Evaluation Metrics:**

**EmotionRecognizer:**

- Loss (MSE)
- Cosine Similarity
- Mean Absolute Error (MAE)
- Mean Correlation

**MelodyTransformer:**

- Loss (BCE)
- Top-K Accuracy
- Precision
- Recall
- F1 Score

## Enhanced Training Script

### Basic Usage

```bash
python train_all_models.py \
    --output ./trained_models \
    --epochs 100 \
    --batch-size 64 \
    --device mps \
    --validation-split 0.2 \
    --early-stopping-patience 10 \
    --learning-rate 0.001
```

### Advanced Options

```bash
python train_all_models.py \
    --output ./trained_models \
    --epochs 200 \
    --batch-size 128 \
    --device cuda \
    --validation-split 0.2 \
    --early-stopping-patience 15 \
    --early-stopping-min-delta 0.0001 \
    --learning-rate 0.0005 \
    --resume EmotionRecognizer \
    --no-history \
    --no-plots
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--output` | `-o` | `./trained_models` | Output directory |
| `--epochs` | `-e` | `50` | Number of epochs |
| `--batch-size` | `-b` | `64` | Batch size |
| `--device` | `-d` | `cpu` | Device (cpu/cuda/mps) |
| `--validation-split` | `-v` | `0.2` | Validation split ratio |
| `--early-stopping-patience` | | `10` | Early stopping patience |
| `--early-stopping-min-delta` | | `0.001` | Early stopping min delta |
| `--learning-rate` | `-lr` | `0.001` | Learning rate |
| `--resume` | | `None` | Resume from checkpoint |
| `--no-history` | | | Don't save training history |
| `--no-plots` | | | Don't generate plots |

## Output Structure

After training, the output directory will contain:

```
trained_models/
├── emotionrecognizer.json          # RTNeural export
├── melodytransformer.json
├── harmonypredictor.json
├── dynamicsengine.json
├── groovepredictor.json
├── checkpoints/
│   ├── emotionrecognizer_latest.pt
│   ├── emotionrecognizer_best.pt
│   ├── melodytransformer_latest.pt
│   └── melodytransformer_best.pt
├── history/
│   ├── emotionrecognizer_history.json
│   ├── emotionrecognizer_history.csv
│   ├── melodytransformer_history.json
│   └── melodytransformer_history.csv
└── plots/
    ├── emotionrecognizer_loss.png
    ├── emotionrecognizer_metric.png
    ├── melodytransformer_loss.png
    └── melodytransformer_metric.png
```

## Training History Format

### JSON Format

```json
{
  "epoch": [1, 2, 3, ...],
  "train_loss": [0.523, 0.456, 0.412, ...],
  "val_loss": [0.567, 0.489, 0.445, ...],
  "train_metric": [0.234, 0.312, 0.378, ...],
  "val_metric": [0.198, 0.287, 0.345, ...]
}
```

### CSV Format

| epoch | train_loss | val_loss | train_metric | val_metric |
|-------|------------|----------|--------------|------------|
| 1 | 0.523 | 0.567 | 0.234 | 0.198 |
| 2 | 0.456 | 0.489 | 0.312 | 0.287 |
| ... | ... | ... | ... | ... |

## Example Training Session

```bash
# Train with validation and early stopping
$ python train_all_models.py \
    --output ./trained_models \
    --epochs 100 \
    --validation-split 0.2 \
    --early-stopping-patience 10 \
    --device mps

# Output:
# ============================================================
# Kelly MIDI Companion - Multi-Model Training
# ============================================================
# Output directory: ./trained_models
# Validation split: 20.0%
# Early stopping: patience=10, min_delta=0.001
# Device: mps
#
# Dataset splits:
#   Emotion: 8000 train, 2000 validation
#   Melody: 8000 train, 2000 validation
#
# [1/5] Training EmotionRecognizer...
# Epoch 1/100 | Train Loss: 0.523456 | Val Loss: 0.567890 | Train Sim: 0.2345 | Val Sim: 0.1987
# Epoch 2/100 | Train Loss: 0.456789 | Val Loss: 0.489012 | Train Sim: 0.3123 | Val Sim: 0.2876
# ...
# Early stopping triggered at epoch 45
# Best validation loss: 0.123456
#
# Training Summary
# ============================================================
# EmotionRecognizer: Best epoch 35, Val Loss: 0.123456
# MelodyTransformer: Best epoch 42, Val Loss: 0.234567
```

## Evaluation Example

```bash
# Evaluate trained models
$ python evaluate_models.py \
    --checkpoint-dir ./trained_models/checkpoints \
    --output ./evaluation_results \
    --device mps

# Output:
# ============================================================
# Kelly MIDI Companion - Model Evaluation
# ============================================================
# [1/2] Evaluating EmotionRecognizer...
#   Loaded checkpoint from epoch 35
#   Loss: 0.123456
#   Cosine Similarity: 0.8765
#   MAE: 0.098765
#   Mean Correlation: 0.9123
#
# [2/2] Evaluating MelodyTransformer...
#   Loaded checkpoint from epoch 42
#   Loss: 0.234567
#   Top-10 Accuracy: 0.7890
#   F1 Score: 0.6543
```

## Best Practices

1. **Use validation split**: Always use `--validation-split 0.2` to monitor overfitting
2. **Enable early stopping**: Use `--early-stopping-patience 10` to prevent overfitting
3. **Save history**: Keep `--no-history` and `--no-plots` disabled to track training
4. **Monitor metrics**: Check the plots and CSV files to understand model behavior
5. **Use best checkpoints**: The evaluation script automatically uses `_best.pt` checkpoints
6. **Resume training**: Use `--resume` to continue training from a checkpoint

## Troubleshooting

### Early Stopping Too Early

If training stops too early, increase patience:

```bash
--early-stopping-patience 20
```

### No Validation Improvement

If validation loss doesn't improve:

- Check if learning rate is too high
- Try reducing batch size
- Check if dataset is too small
- Verify data quality

### Checkpoint Not Found

If resume fails:

- Check checkpoint directory path
- Verify checkpoint file exists
- Ensure model architecture matches

## Integration with Existing Code

The enhanced training functions are backward compatible. Existing code will work, but you can now add:

```python
from training_utils import EarlyStopping, TrainingMetrics, CheckpointManager

# Add to training function
early_stopping = EarlyStopping(patience=10)
metrics = TrainingMetrics()
checkpoint_manager = CheckpointManager(checkpoint_dir)

# Use in training loop
train_emotion_recognizer(
    model, train_loader,
    val_loader=val_loader,
    early_stopping=early_stopping,
    metrics=metrics,
    checkpoint_manager=checkpoint_manager
)
```

## Next Steps

1. **Real Dataset Integration**: Replace synthetic datasets with real data loaders
2. **Hyperparameter Tuning**: Add grid search or random search
3. **Cross-Validation**: Implement k-fold cross-validation
4. **Model Ensembling**: Combine multiple model checkpoints
5. **Distributed Training**: Add multi-GPU support
