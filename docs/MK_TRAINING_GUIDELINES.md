# MK Training Guidelines

> Mac-optimized ML training workflow for Kelly's 5-model architecture

## Overview

This document covers the complete ML training workflow for Kelly, optimized for macOS development environments (Intel and Apple Silicon).

### The 5-Model Architecture

| Model | Task | Input | Output | Target Latency |
|-------|------|-------|--------|----------------|
| **EmotionRecognizer** | Audio emotion classification | Mel spectrogram | 7 emotion classes | <5ms |
| **MelodyTransformer** | Melodic sequence generation | Note context | Next note prediction | <5ms |
| **HarmonyPredictor** | Chord progression prediction | Melody + key | Chord embedding | <3ms |
| **DynamicsEngine** | Expression parameter mapping | Note context | Velocity/timing | <1ms |
| **GroovePredictor** | Groove/timing patterns | Position + style | Timing offsets | <2ms |

---

## 1. Environment Setup

### Prerequisites

```bash
# Install Python via Homebrew
brew install python@3.11

# Navigate to project
cd /path/to/kelly-clean

# Run setup script
chmod +x scripts/setup_ml_env.sh
./scripts/setup_ml_env.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check PyTorch and device
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Device: {\"mps\" if torch.backends.mps.is_available() else \"cpu\"}')
"
```

---

## 2. Hardware Considerations

### Apple Silicon (M1/M2/M3)

- **Backend**: Metal Performance Shaders (MPS)
- **Memory**: Unified memory helps with larger models
- **Performance**: ~2-5x faster than Intel Mac CPU
- **Usage**: `device = torch.device("mps")`

### Intel Mac

- **Backend**: CPU only (no CUDA)
- **Memory**: System RAM only
- **Performance**: Slower, use smaller batch sizes
- **Usage**: `device = torch.device("cpu")`

### Recommended Settings

| Hardware | Batch Size | Workers | Pin Memory |
|----------|------------|---------|------------|
| Apple Silicon | 16-32 | 0 | False |
| Intel Mac | 8-16 | 0 | False |

> **Note**: `num_workers=0` is required on macOS to avoid multiprocessing issues.

---

## 3. Dataset Guidelines

### Storage Location

**All audio data MUST be stored on the external SSD:**

```
/Volumes/Extreme SSD/kelly-audio-data/
```

This is required for:
- Sufficient storage space (audio datasets can be large)
- Better I/O performance during training
- Separation of code and data

### Directory Structure

```
/Volumes/Extreme SSD/kelly-audio-data/
├── raw/                    # Original audio files
│   ├── emotions/
│   │   ├── happy/
│   │   │   ├── audio_001.wav
│   │   │   └── ...
│   │   ├── sad/
│   │   └── ...
│   ├── melodies/
│   │   ├── melody_001.mid
│   │   └── ...
│   └── grooves/
│       └── ...
├── processed/              # Pre-computed features
│   └── mel_spectrograms/
├── downloads/              # Downloaded archives
├── cache/                  # Temporary cache
└── metadata.csv

# Symlink in project for convenience:
kelly-clean/data/audio → /Volumes/Extreme SSD/kelly-audio-data/
```

### Audio Requirements

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Sample Rate | 16 kHz | Lower = faster processing |
| Channels | Mono | Convert stereo to mono |
| Format | WAV | Lossless, fast loading |
| Duration | 1-10 sec | Truncate/pad as needed |
| Total Size | <2 GB | For comfortable training |

### Metadata Format (CSV)

```csv
file,label,duration,emotion,genre,tempo,key
happy/audio_001.wav,happy,3.5,happy,pop,120,C
sad/audio_002.wav,sad,4.2,sad,ballad,80,Am
```

### Metadata Format (JSON)

```json
{
  "samples": [
    {
      "file": "happy/audio_001.wav",
      "label": "happy",
      "duration": 3.5,
      "emotion": "happy",
      "tags": ["upbeat", "major"]
    }
  ]
}
```

---

## 4. Training Workflow

### Quick Start

```bash
# List available models
python scripts/train.py --list

# Train with defaults
python scripts/train.py --model emotion_recognizer --epochs 50

# Train with config file
python scripts/train.py --config configs/emotion_recognizer.yaml

# Dry run (show config without training)
python scripts/train.py --model melody_transformer --dry-run
```

### Training Options

```bash
python scripts/train.py \
    --model emotion_recognizer \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --data data/raw/emotions \
    --export-onnx \
    --export-coreml
```

### Config Files

Training configs are in `configs/`:

- `emotion_recognizer.yaml`
- `melody_transformer.yaml`
- `harmony_predictor.yaml`
- `dynamics_engine.yaml`
- `groove_predictor.yaml`

Example config:

```yaml
# configs/emotion_recognizer.yaml
model_id: emotionrecognizer
task: emotion_embedding
architecture_type: cnn

# Architecture
input_size: 128
output_size: 7
hidden_layers: [512, 256, 128]

# Training (Mac-optimized)
epochs: 100
batch_size: 16
learning_rate: 0.001
device: auto  # Uses MPS on Apple Silicon

# Export
export_onnx: true
export_coreml: true
```

---

## 5. Training Best Practices

### Memory Management

```python
# Avoid loading entire dataset
# ✗ Bad
all_data = [load_audio(f) for f in files]

# ✓ Good - lazy loading
class AudioDataset:
    def __getitem__(self, idx):
        return load_audio(self.files[idx])
```

### Pre-compute Features

```python
# Pre-extract mel spectrograms for faster training
from python.penta_core.ml.audio_dataset import AudioDataset

dataset = AudioDataset(
    data_dir="data/raw/emotions",
    precompute_features=True,
    feature_cache_dir="data/processed/mel_cache"
)
```

### Training Loop Tips

```python
# Small batch sizes for Mac
batch_size = 16  # Not 64+

# Move data to device in batches
for x, y in dataloader:
    x = x.to(device)  # Don't preload everything
    y = y.to(device)
    
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

# Log audio samples periodically
if epoch % 10 == 0:
    save_sample_predictions(model, val_loader)
```

### Checkpointing

```python
# Save regularly
if epoch % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, f'checkpoints/{model_id}/epoch_{epoch}.pt')

# Save best model
if val_loss < best_val_loss:
    torch.save(model.state_dict(), f'checkpoints/{model_id}/best.pt')
```

---

## 6. Model Export

### Supported Formats

| Format | Use Case | Tool |
|--------|----------|------|
| `.onnx` | Cross-platform inference | `torch.onnx.export()` |
| `.mlmodel` | Native macOS/iOS | `coremltools` |
| `.json` | RTNeural (audio plugins) | Custom export |
| `.pt` | PyTorch/TorchScript | `torch.jit.trace()` |

### Export Commands

```bash
# Export during training
python scripts/train.py --model emotion_recognizer --export-onnx --export-coreml

# Export existing model
python -c "
from python.penta_core.ml.export import ModelExporter, ExportConfig
import torch

# Load model
model = torch.load('checkpoints/emotionrecognizer/best.pt')

# Configure export
config = ExportConfig(
    model_id='emotionrecognizer',
    input_size=128,
    output_size=7,
    architecture_type='cnn'
)

# Export all formats
exporter = ModelExporter(model, config)
exporter.export_all('models/')
"
```

### Core ML Export

```python
import coremltools as ct
import torch

# Trace model
traced = torch.jit.trace(model, dummy_input)

# Convert
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=(1, 1, 64, 128))],
    minimum_deployment_target=ct.target.macOS13,
)

# Save
mlmodel.save("models/emotionrecognizer.mlmodel")
```

### ONNX Export

```python
import torch

torch.onnx.export(
    model,
    dummy_input,
    "models/emotionrecognizer.onnx",
    export_params=True,
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
```

---

## 7. Model Registry

After training, models are registered in `models/registry.json`:

```json
{
  "registry_version": "1.1",
  "models": [
    {
      "id": "emotionrecognizer",
      "file": "emotionrecognizer.json",
      "format": "rtneural-json",
      "task": "emotion_embedding",
      "status": "trained",
      "training": {
        "epochs": 100,
        "train_loss": 0.05,
        "val_loss": 0.08,
        "git_commit": "abc123",
        "training_date": "2025-12-23T10:00:00"
      },
      "exports": {
        "onnx_path": "emotionrecognizer.onnx",
        "coreml_path": "emotionrecognizer.mlmodel"
      }
    }
  ]
}
```

### Validate Registry

```bash
# Validate against schema
python -c "
import json
import jsonschema

with open('models/registry.json') as f:
    registry = json.load(f)
with open('models/registry.schema.json') as f:
    schema = json.load(f)

jsonschema.validate(registry, schema)
print('Registry is valid!')
"
```

---

## 8. Model Cards

Create a model card for each trained model in `docs/model_cards/`:

```markdown
# Model Card: emotionrecognizer

## Overview
| Field | Value |
|-------|-------|
| **Name** | emotionrecognizer |
| **Task** | emotion_embedding |
| **Status** | trained |

## Training Data
- Dataset: emotion_dataset_v1
- Size: 10,000 samples
- Split: 80/10/10

## Metrics
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Loss | 0.05 | 0.08 | 0.09 |
| Accuracy | 95% | 92% | 91% |

## Known Limitations
- Trained on Western music only
- May underperform on ambient/noise genres
```

Use the template: `docs/MODEL_CARD_TEMPLATE.md`

---

## 9. Integration

### Python API

```python
from python.penta_core.ml.inference import create_engine_by_name

# Load and run inference
engine = create_engine_by_name("emotionrecognizer")
engine.load()

result = engine.infer({"input": mel_spectrogram})
emotion = result.get_top_k(k=1)[0]
```

### C++ Integration

Models exported to ONNX/RTNeural JSON can be loaded by:
- `src/ml/ONNXInference.cpp`
- `src/ml/RTNeuralProcessor.cpp`

### Swift Integration

Core ML models can be used directly in Swift:

```swift
import CoreML

let model = try emotionrecognizer()
let prediction = try model.prediction(input: melSpectrogram)
```

---

## 10. Troubleshooting

### Common Issues

**MPS not available on Apple Silicon:**
```bash
# Check PyTorch version supports MPS
pip install torch>=2.0.0
```

**Out of memory:**
```bash
# Reduce batch size
python scripts/train.py --model emotion_recognizer --batch-size 8
```

**Slow training on Intel Mac:**
- Use smaller models
- Pre-compute features
- Reduce dataset size for prototyping

**Core ML export fails:**
```bash
# Ensure coremltools is installed
pip install coremltools>=7.0
```

**ONNX export fails:**
```bash
# Check for unsupported ops
pip install onnx onnxruntime
python -c "import onnx; onnx.checker.check_model('model.onnx')"
```

---

## 11. Quick Reference

### Commands

```bash
# Setup
./scripts/setup_ml_env.sh
source venv/bin/activate

# Training
python scripts/train.py --list
python scripts/train.py --model <name> --epochs 50
python scripts/train.py --config configs/<name>.yaml

# Export
python scripts/train.py --model <name> --export-onnx --export-coreml

# Verify
python -c "from python.penta_core.ml.export import verify_onnx_model; verify_onnx_model('models/<name>.onnx')"
```

### File Locations

| Type | Location |
|------|----------|
| **Audio Data** | `/Volumes/Extreme SSD/kelly-audio-data/` |
| Training configs | `configs/*.yaml` |
| Training script | `scripts/train.py` |
| Checkpoints | `checkpoints/<model_id>/` |
| Exported models | `models/` |
| Training logs | `logs/training/` |
| Model cards | `docs/model_cards/` |
| Registry | `models/registry.json` |

> ⚠️ **Important**: All audio data must be stored on `/Volumes/Extreme SSD/` for space and performance.

### Model Status

| Status | Meaning |
|--------|---------|
| `stub` | Placeholder, not trained |
| `training` | Currently being trained |
| `trained` | Training complete, ready for testing |
| `production` | Verified and deployed |
| `deprecated` | Replaced by newer version |

---

## Appendix: Recommended Models for Mac

### Trainable Locally

| Model Type | Description | Training Time |
|------------|-------------|---------------|
| Tiny CNN | Audio classifier | <1 hour |
| Small LSTM | Sequence prediction | 1-2 hours |
| MLP | Feature mapping | <30 min |
| Autoencoder | Audio embedding | 1-2 hours |

### Pretrained (Inference Only)

| Model | Task | Size |
|-------|------|------|
| Whisper Tiny | Transcription | 39M params |
| CLAP | Audio-text embedding | Use features |

### Avoid on Mac

- GPT-like text models (too large)
- MusicLM/Jukebox (requires GPU cluster)
- Full Whisper Large (>16GB RAM needed)
- Diffusion models (very slow on CPU)

