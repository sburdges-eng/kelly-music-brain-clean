# Model Card: EmotionRecognizer

## Overview

| Field | Value |
|-------|-------|
| **Name** | `emotionrecognizer` |
| **Version** | `1.0.0` |
| **Task** | `emotion_embedding` |
| **Format** | RTNeural JSON / ONNX / Core ML |
| **Status** | `stub` |
| **Created** | 2025-12-23 |
| **Author** | Kelly ML Pipeline |

---

## Training Data

| Field | Value |
|-------|-------|
| **Dataset Name** | `emotion_dataset_v1` |
| **Source(s)** | TBD (internal, Freesound, licensed) |
| **Size** | TBD |
| **Split** | Train: 80% / Val: 10% / Test: 10% |
| **Sample Rate** | 16000 Hz |
| **Preprocessing** | Mel spectrogram (64 bands) |
| **License** | TBD |

**Provenance Notes:**

```
Dataset not yet collected. Will include:
- Emotion-labeled audio clips (1-10 seconds)
- 7 emotion categories: happy, sad, angry, fear, surprise, disgust, neutral
- Balanced across genres and tempos
```

---

## Architecture

| Field | Value |
|-------|-------|
| **Input Size** | 64 × 128 (mel × time frames) |
| **Output Size** | 7 (emotion classes) |
| **Layers** | CNN: 32→64→128 + FC: 256→128→7 |
| **Parameters** | ~500K |
| **Inference Time** | <5 ms (target) |

**Architecture Notes:**

```
2D CNN for mel spectrogram classification:
- Conv2d(1, 32, 3) + BN + ReLU + MaxPool
- Conv2d(32, 64, 3) + BN + ReLU + MaxPool
- Conv2d(64, 128, 3) + BN + ReLU + AdaptiveAvgPool
- Flatten + FC(2048, 256) + ReLU + Dropout
- FC(256, 128) + ReLU + Dropout
- FC(128, 7)
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Batch Size | 16 |
| Epochs | 100 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Regularization | Dropout 0.3, Weight Decay 1e-4 |

**Training Notes:**

```
- Mac-optimized: batch_size=16 for memory efficiency
- Early stopping with patience=15
- Cosine annealing learning rate schedule
- Data augmentation: time stretch, pitch shift, noise injection
```

---

## Evaluation Metrics

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Loss | - | - | - |
| Accuracy | - | - | - |
| F1 (macro) | - | - | - |

**Qualitative Review:**

```
Model not yet trained. Evaluation pending.
```

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Git Commit** | - |
| **Data Version** | v1 |
| **Training Date** | - |
| **Hardware** | - |
| **Duration** | - |
| **Logs** | `logs/training/emotionrecognizer_*.json` |

---

## Known Limitations

- Not yet trained (stub model)
- Training data not yet collected
- May underperform on non-Western music
- May struggle with subtle emotional nuances
- Trained on short clips only (1-10 sec)

---

## Intended Usage

**Primary Use Cases:**

- Real-time emotion detection in UnifiedHub
- Emotion-driven parameter modulation
- Mood-based music recommendations

**Out-of-Scope Uses:**

- Clinical emotion assessment
- Long-form audio analysis (>10 sec)
- Production-critical decisions without human review

---

## Integration

| Target | Status | Notes |
|--------|--------|-------|
| Python API (`music_brain/`) | ⬜ | Pending training |
| C++ MLInterface | ⬜ | Pending ONNX export |
| Tauri UI | ⬜ | Pending integration |
| ONNX Export | ⬜ | Will export after training |
| Core ML Export | ⬜ | Will export after training |

---

## Files

| File | Description |
|------|-------------|
| `models/emotionrecognizer.json` | RTNeural weights (stub) |
| `models/emotionrecognizer.onnx` | ONNX export (pending) |
| `models/emotionrecognizer.mlmodel` | Core ML export (pending) |
| `docs/model_cards/emotionrecognizer.md` | This model card |
| `configs/emotion_recognizer.yaml` | Training configuration |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-23 | Initial stub model |

---

## Quick Checklist

Before marking as "production":

- [ ] Model card completed
- [ ] Metrics reviewed and acceptable
- [ ] Qualitative samples reviewed
- [ ] Integration verified in Python API
- [ ] Integration verified in C++ MLInterface
- [ ] Fallback behavior documented
- [ ] Artifacts exported (ONNX/Core ML)
- [ ] Registry updated with SHA256 and paths

