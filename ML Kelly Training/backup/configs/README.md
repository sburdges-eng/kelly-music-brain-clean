# Kelly ML Training Configs Backup

**Backup Date:** 2025-12-23  
**Total Configs:** 18

## Config Index

### Core Models (from registry.json)

| Config | Model ID | Task | Status |
|--------|----------|------|--------|
| `emotion_recognizer.yaml` | emotionrecognizer | emotion_embedding | ✅ Active |
| `melody_transformer.yaml` | melodytransformer | melody_generation | ✅ Active |
| `harmony_predictor.yaml` | harmonypredictor | harmony_prediction | ✅ Active |
| `dynamics_engine.yaml` | dynamicsengine | dynamics_mapping | ✅ Active |
| `groove_predictor.yaml` | groovepredictor | groove_prediction | ✅ Active |
| `instrument_recognizer.yaml` | instrumentrecognizer | dual_instrument_recognition | ✅ Active |
| `emotion_node_classifier.yaml` | emotionnodeclassifier | emotion_node_classification | ✅ Active |

### Training Presets

| Config | Purpose | Device |
|--------|---------|--------|
| `laptop_m4_small.yaml` | M4 MacBook training preset | MPS |

### Additional Audio Analysis Models

| Config | Model ID | Task | Inference Target |
|--------|----------|------|------------------|
| `audio_classifier.yaml` | audioclassifier | Multi-label classification | <10ms |
| `voice_activity_detector.yaml` | voiceactivitydetector | VAD | <1ms |
| `tempo_estimator.yaml` | tempoestimator | BPM estimation | <5ms |
| `key_detector.yaml` | keydetector | Key detection | <5ms |
| `beat_tracker.yaml` | beattracker | Beat tracking | <10ms |
| `onset_detector.yaml` | onsetdetector | Onset detection | <2ms |
| `pitch_tracker.yaml` | pitchtracker | F0 tracking | <5ms |
| `chord_recognizer.yaml` | chordrecognizer | Chord recognition | <10ms |
| `source_separator.yaml` | sourceseparator | Stem separation | <100ms |
| `timbre_encoder.yaml` | timbreencoder | Timbre VAE | <5ms |

## Quick Reference

### Mac-Optimized Settings (All Configs)

```yaml
device: auto  # Uses MPS on Apple Silicon, CPU on Intel
num_workers: 0  # Required for Mac compatibility
pin_memory: false  # MPS doesn't support pinned memory
batch_size: 8-32  # Keep small for RAM constraints
```

### Common Data Paths

```
/Volumes/Extreme SSD/kelly-audio-data/
├── raw/           # Original audio files
│   ├── emotions/
│   ├── melodies/
│   ├── chords/
│   └── ...
├── processed/     # Pre-computed features
├── downloads/     # Downloaded datasets
└── cache/         # Temporary cache
```

### Training Commands

```bash
# Train with specific config
python scripts/train_model.py --config configs/emotion_recognizer.yaml

# Quick test with synthetic data
python scripts/train_model.py --model emotion --synthetic --epochs 5

# MPS-optimized lightweight training
python scripts/train_mps_stub.py --config configs/laptop_m4_small.yaml
```

## Config Schema

Each config follows this structure:

```yaml
# Model Identity
model_id: string
model_type: RTNeural | PyTorch
task: string

# Architecture
input_size: int
output_size: int
hidden_layers: [int, ...]
activation: relu | gelu | tanh
dropout: float

# Data
data_path: string
sample_rate: int
...

# Training
epochs: int
batch_size: int
learning_rate: float
optimizer: adam | adamw
loss: string
scheduler: cosine | step | cosine_warmup

# Mac-specific
device: auto | mps | cpu
num_workers: 0
pin_memory: false

# Output
output_dir: string
export_onnx: bool
export_coreml: bool
```

## Restore Instructions

To restore configs to main project:

```bash
cp -v "ML Kelly Training/backup/configs/"*.yaml configs/
```

## Version History

| Date | Changes |
|------|---------|
| 2025-12-23 | Initial backup with 18 configs |

