# ML Training Guide

This guide covers training the 5-model ML pipeline for MidiKompanion.

## Overview

MidiKompanion uses 5 neural network models for emotion-to-music generation:

| Model | Input | Output | Purpose |
|-------|-------|--------|---------|
| **EmotionRecognizer** | 128-dim audio features | 64-dim emotion embedding | Audio → Emotion |
| **MelodyTransformer** | 64-dim emotion embedding | 128-dim note probabilities | Emotion → Notes |
| **HarmonyPredictor** | 128-dim context | 64-dim chord probabilities | Context → Chords |
| **DynamicsEngine** | 32-dim intensity | 16-dim expression | Intensity → Expression |
| **GroovePredictor** | 64-dim arousal | 32-dim groove params | Arousal → Groove |

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: 8+ cores for data processing
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for datasets and models

### Software
- Python 3.9+
- PyTorch 2.1+
- CUDA 12.1+ (for GPU training)
- Docker (optional, for reproducible environment)

## Quick Start

### Option 1: Docker (Recommended)

```bash
cd ml_training

# Build and train
docker-compose up training

# This will:
# 1. Build the training environment
# 2. Generate synthetic training data
# 3. Train all 5 models
# 4. Export to ONNX format
# 5. Save models to ./models/
```

### Option 2: Local Python

```bash
cd ml_training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Prepare datasets
python prepare_datasets.py --output ./data

# Train models
python train_all_models.py --epochs 100 --output-dir ./models

# Validate models
python validate_models.py --models-dir ./models
```

### Option 3: Google Colab (Free GPU)

1. Upload `Train_MidiKompanion_Models.ipynb` to Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells
4. Download trained models from `./models/`

## Training Pipeline

### 1. Data Preparation

```bash
python prepare_datasets.py --input ./raw_data --output ./data
```

**Input formats:**
- Audio: WAV, MP3 (for emotion recognition)
- MIDI: .mid, .midi (for melody/harmony training)
- Labels: JSON with emotion annotations

**Output:**
- `train.json`: 80% of data
- `val.json`: 10% of data
- `test.json`: 10% of data
- `thesaurus.json`: 216-node emotion structure

### 2. Model Training

```bash
python train_all_models.py \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --output-dir ./models \
    --device cuda  # or cpu
```

**Training parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--learning-rate` | 0.001 | Initial learning rate |
| `--num-samples` | 10000 | Synthetic training samples |
| `--device` | auto | cuda/cpu/auto |

**Training time:**
- GPU (RTX 3080): ~30 minutes
- CPU (8 cores): ~4-6 hours

### 3. Model Export

```bash
python export_to_onnx.py --input ./models --output ./models
```

Converts PyTorch models to ONNX format for C++ inference.

### 4. Validation

```bash
python validate_models.py --models-dir ./models
```

Checks:
- Model structure and I/O shapes
- Inference correctness
- Performance (<10ms per inference)
- File size (<5MB per model)

## Integration with 216-Node Thesaurus

### Node-Aware Training

The training data is labeled with emotion node IDs (0-215). Each node has:

- **VAD coordinates**: Valence, Arousal, Dominance, Intensity
- **Musical attributes**: Mode, tempo, dynamics, etc.
- **Relationships**: Related emotion nodes

### Data Augmentation

Use node relationships for augmentation:

```python
# Example: augment with related emotions
for sample in training_data:
    node_id = sample['node_id']
    related_ids = thesaurus[node_id]['relatedNodes']
    
    for related_id in related_ids:
        # Create augmented sample with related emotion
        augmented = augment_sample(sample, related_id)
        training_data.append(augmented)
```

## Model Architecture

### EmotionRecognizer

```
Input: 128-dim audio features
    ↓
Linear(128 → 256) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Linear(256 → 128) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Linear(128 → 64) + Tanh
    ↓
Output: 64-dim emotion embedding
```

### MelodyTransformer

```
Input: 64-dim emotion embedding
    ↓
Linear(64 → 128) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Linear(128 → 256) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Linear(256 → 128) + Sigmoid
    ↓
Output: 128-dim note probabilities
```

## Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Inference time | <10ms | ~2-5ms |
| Total model size | <5MB | ~3MB |
| Validation loss | <0.1 | varies |
| GPU memory | <2GB | ~1GB |

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 16`
- Use gradient checkpointing
- Try CPU training for smaller datasets

### Training Loss Not Decreasing
- Increase learning rate: `--learning-rate 0.01`
- Check data preprocessing
- Verify label quality

### Poor Inference Performance
- Check model complexity
- Verify ONNX export optimization
- Profile inference bottlenecks

### ONNX Export Fails
- Ensure model is in eval mode
- Check for unsupported operations
- Try different opset version

## Custom Training

### Using Your Own Data

1. Create emotion-labeled dataset:
```json
{
    "audio_path": "path/to/audio.wav",
    "node_id": 42,
    "valence": 0.7,
    "arousal": 0.5,
    "dominance": 0.6,
    "intensity": 0.8
}
```

2. Run data preparation with custom input:
```bash
python prepare_datasets.py --input ./your_data --output ./data
```

3. Train with custom epochs:
```bash
python train_all_models.py --epochs 200 --output-dir ./models
```

### Fine-Tuning Pre-Trained Models

```python
# Load pre-trained model
model.load_state_dict(torch.load('pretrained.pt'))

# Freeze early layers
for param in model.network[:4].parameters():
    param.requires_grad = False

# Train only later layers
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
```

## Monitoring

### TensorBoard

```bash
# Start TensorBoard
docker-compose up tensorboard

# Or locally
tensorboard --logdir=./logs
```

Access at: http://localhost:6006

### Training Metrics

- Loss curves
- Validation accuracy
- Learning rate schedule
- Gradient norms

## Next Steps

After training:

1. Copy models to plugin directory:
   ```bash
   cp models/*.onnx ../resources/models/
   ```

2. Enable ONNX in CMake:
   ```bash
   cmake -DMIDIKOMPANION_HAS_ONNX=ON ..
   ```

3. Test in plugin:
   - Load models
   - Run inference
   - Compare with stub mode
