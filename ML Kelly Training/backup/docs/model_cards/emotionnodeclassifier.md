# Model Card: EmotionNodeClassifier

## Overview

| Field | Value |
|-------|-------|
| **Name** | `emotionnodeclassifier` |
| **Version** | `0.1.0` |
| **Task** | `emotion_node_classification` (6×6×6 × 24 keys) |
| **Format** | RTNeural JSON / ONNX / Core ML |
| **Status** | `stub` |
| **Created** | 2025-12-23 |
| **Author** | Kelly ML Pipeline |
| **Purpose** | **Validation test** for entire DAiW emotion thesaurus |

---

## 🎯 Purpose: System Validation Test

This model serves as a **comprehensive test harness** for the DAiW emotion system:

1. **Validates all 216 emotion nodes** can be distinguished from audio
2. **Tests key-invariance** (same emotion regardless of musical key)
3. **Validates hierarchical consistency** (base → sub → sub-sub)
4. **Measures intensity tier discrimination** (6 levels per node)

### What We're Testing

```
6×6×6 Emotion Thesaurus × 24 Keys × 6 Intensity Tiers
= 216 nodes × 24 keys × 6 tiers
= 31,104 possible classifications
```

---

## 🏗️ Architecture: Multi-Head Hierarchical CNN

### Thesaurus Structure

```
6 BASE EMOTIONS
├── HAPPY (I)
│   ├── CONTENTMENT (Ii)
│   │   ├── satisfied (Iia)
│   │   ├── comfortable (Iib)
│   │   ├── peaceful (Iic)
│   │   ├── secure (Iid)
│   │   ├── fulfilled (Iie)
│   │   └── grateful (Iif)
│   ├── JOY (Ij)
│   │   ├── cheerful (Ija)
│   │   ├── delighted (Ijb)
│   │   ├── elated (Ijc)
│   │   ├── blissful (Ijd)
│   │   ├── amused (Ije)
│   │   └── playful (Ijf)
│   ├── EXCITEMENT (Ik) ... [6 sub-sub]
│   ├── PRIDE (Il) ... [6 sub-sub]
│   ├── LOVE (Im) ... [6 sub-sub]
│   └── HOPE (In) ... [6 sub-sub]
├── SAD (II) ... [6 sub × 6 sub-sub = 36 nodes]
├── ANGRY (III) ... [36 nodes]
├── FEAR (IV) ... [36 nodes]
├── SURPRISE (V) ... [36 nodes]
└── DISGUST (VI) ... [36 nodes]

TOTAL: 6 × 6 × 6 = 216 emotion nodes
+ 6 intensity tiers per node = 1,296 intensity-aware states
× 24 keys = full key-invariant coverage
```

---

## Training Data

| Field | Value |
|-------|-------|
| **Dataset Name** | `emotion_thesaurus_dataset_v1` |
| **Target Size** | 216 nodes × 200 samples × 24 keys = ~1M samples |
| **Source(s)** | DAiW thesaurus, synthetic, licensed audio |
| **Split** | Train: 75% / Val: 15% / Test: 10% |
| **Sample Rate** | 44100 Hz |
| **Preprocessing** | Mel spectrogram (64 bands × 128 frames) |
| **Key Augmentation** | All samples transposed to all 12 keys |
| **License** | TBD |

**Data Requirements:**

```
Minimum viable:
- 50 samples per node × 216 nodes = 10,800 samples
- Augmented across 24 keys = 259,200 training samples

Target:
- 200 samples per node × 216 nodes = 43,200 samples
- Augmented across 24 keys = 1,036,800 training samples
- 6 intensity variations each = 6,220,800 intensity-aware samples
```

---

## Architecture

| Field | Value |
|-------|-------|
| **Input Size** | 64 × 128 (mel × time frames) |
| **Output Size** | 258 total (multi-head) |
| **CNN Backbone** | 64→128→256→512 channels |
| **Shared FC** | 384→256 |
| **Parameters** | ~3M |
| **Inference Time** | <15 ms (target) |

### Classification Heads

| Head | Output Size | Purpose | Weight |
|------|-------------|---------|--------|
| `emotion_node` | **216** | Full 6×6×6 classification | 1.0 |
| `base_emotion` | 6 | Coarse (HAPPY, SAD, etc.) | 0.5 |
| `sub_emotion` | 36 | Medium (JOY, CONTENTMENT, etc.) | 0.3 |
| `intensity_tier` | 6 | subtle → overwhelming | 0.3 |
| `key_detection` | 24 | 12 major + 12 minor keys | 0.2 |

**Architecture Notes:**

```
Input: Mel Spectrogram (64 × 128)
    ↓
[CNN Backbone]
Conv2d(1, 64, 3) + BN + GELU + MaxPool
Conv2d(64, 128, 3, stride=2) + BN + GELU + MaxPool
Conv2d(128, 256, 3, stride=2) + BN + GELU + MaxPool
Conv2d(256, 512, 3, stride=2) + BN + GELU + AdaptiveAvgPool
    ↓
[Shared FC]
Flatten → FC(512, 384) → GELU → Dropout(0.35)
FC(384, 256) → GELU → Dropout(0.35)
    ↓
[Classification Heads]
├── emotion_node: FC(256, 216) + Softmax
├── base_emotion: FC(256, 6) + Softmax
├── sub_emotion: FC(256, 36) + Softmax
├── intensity_tier: FC(256, 6) + Softmax
└── key_detection: FC(256, 24) + Softmax
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0005 |
| Batch Size | 8 |
| Epochs | 200 |
| Optimizer | AdamW |
| Loss Function | Hierarchical CrossEntropy (multi-task) |
| Label Smoothing | 0.1 |
| Regularization | Dropout 0.35, Weight Decay 1e-4 |

**Training Notes:**

```
- Hierarchical loss ensures consistency (predict "cheerful" → "JOY" → "HAPPY")
- Key transposition augmentation: each sample trained in all 12 keys
- Mixup within same base emotion for better generalization
- Balanced sampling ensures all 216 nodes represented
- Early stopping with patience=25 for 216-class problem
```

---

## Evaluation Metrics

| Metric | Train | Val | Test | Description |
|--------|-------|-----|------|-------------|
| Node Accuracy (216) | - | - | - | Primary: correct 216-class prediction |
| Base Accuracy (6) | - | - | - | Coarse emotion accuracy |
| Sub Accuracy (36) | - | - | - | Medium granularity accuracy |
| Intensity Accuracy | - | - | - | Intensity tier accuracy |
| Key Accuracy | - | - | - | Key detection accuracy |
| Top-3 Accuracy | - | - | - | Top-3 for 216 classes |
| Hierarchical F1 | - | - | - | Weighted F1 across hierarchy |
| Key Invariance | - | - | - | Same prediction across keys |

**Validation Test Suite:**

```python
# 1. Exhaustive key test: all 216 nodes × 24 keys
for node in range(216):
    for key in range(24):
        assert model.predict(sample[node][key]) == node

# 2. Cross-key consistency: same emotion across transpositions
for node in range(216):
    predictions = [model.predict(transpose(sample, k)) for k in range(12)]
    assert all(p == predictions[0] for p in predictions)

# 3. Hierarchical consistency
pred_node = model.predict(sample)
pred_sub = pred_node // 6
pred_base = pred_node // 36
assert model.heads['sub_emotion'].predict(sample) == pred_sub
assert model.heads['base_emotion'].predict(sample) == pred_base

# 4. Intensity gradient: ordered samples should show ordered intensity
for node in range(216):
    intensities = [model.heads['intensity'].predict(sample[node][tier]) 
                   for tier in range(6)]
    assert intensities == sorted(intensities)
```

---

## Run Metadata

| Field | Value |
|-------|-------|
| **Git Commit** | - |
| **Data Version** | v1 |
| **Training Date** | - |
| **Hardware** | - (target: Apple Silicon M-series) |
| **Duration** | - |
| **Logs** | `logs/training/emotionnodeclassifier_*.json` |

---

## Known Limitations

- **Not yet trained** (stub model)
- **Training data not yet collected** for all 216 nodes
- **High class imbalance risk** (216 classes)
- **May struggle with subtle distinctions** (e.g., "satisfied" vs "content")
- **Key-invariance requires extensive augmentation**
- **Blend emotions** (between nodes) may be misclassified
- **Intensity tiers are subjective** and hard to label consistently
- **Non-Western music** may not fit the emotion taxonomy

---

## Intended Usage

**Primary Use Case: System Validation**

This model validates the entire DAiW emotion thesaurus by testing:

1. ✅ Can all 216 emotion nodes be distinguished?
2. ✅ Is the model key-invariant (same emotion in any key)?
3. ✅ Is the hierarchical structure learnable?
4. ✅ Can intensity tiers be discriminated?

**Secondary Use Cases:**

- Fine-grained emotion detection in audio
- Emotion-aware music recommendations
- Therapeutic music generation with precise emotion targeting
- Music analysis and annotation

**Out-of-Scope Uses:**

- Clinical diagnosis
- High-stakes emotion assessment
- Real-time classification without verification
- Non-music audio (speech, environmental sounds)

---

## Integration

| Target | Status | Notes |
|--------|--------|-------|
| Python API (`music_brain/`) | ⬜ | Pending training |
| C++ MLInterface | ⬜ | Pending ONNX export |
| Tauri UI | ⬜ | Pending integration |
| EmotionWheel component | ⬜ | Will map predictions to wheel |
| IntentProcessor | ⬜ | Will use for emotion detection |
| ONNX Export | ⬜ | Will export after training |
| Core ML Export | ⬜ | Will export after training |

---

## Files

| File | Description |
|------|-------------|
| `models/emotionnodeclassifier.json` | RTNeural weights (stub) |
| `models/emotionnodeclassifier.onnx` | ONNX export (pending) |
| `models/emotionnodeclassifier.mlmodel` | Core ML export (pending) |
| `docs/model_cards/emotionnodeclassifier.md` | This model card |
| `configs/emotion_node_classifier.yaml` | Training configuration |
| `data/emotion_thesaurus/` | Source thesaurus JSON files |

---

## Thesaurus Reference

### Node ID Calculation

```python
# Node ID = base_idx * 36 + sub_idx * 6 + subsub_idx
# where base_idx, sub_idx, subsub_idx ∈ [0, 5]

def get_node_id(base_idx: int, sub_idx: int, subsub_idx: int) -> int:
    return base_idx * 36 + sub_idx * 6 + subsub_idx

def get_node_indices(node_id: int) -> tuple[int, int, int]:
    base_idx = node_id // 36
    sub_idx = (node_id % 36) // 6
    subsub_idx = node_id % 6
    return base_idx, sub_idx, subsub_idx

# Examples:
# HAPPY/CONTENTMENT/satisfied = 0*36 + 0*6 + 0 = 0
# HAPPY/JOY/cheerful = 0*36 + 1*6 + 0 = 6
# HAPPY/JOY/playful = 0*36 + 1*6 + 5 = 11
# SAD/GRIEF/devastated = 1*36 + 0*6 + 0 = 36
# DISGUST/REVULSION/nauseated = 5*36 + 5*6 + 5 = 215
```

### Key Encoding

```python
# Key ID = root * 2 + mode
# where root ∈ [0, 11] (C, C#, D, ..., B)
# and mode ∈ [0, 1] (major, minor)

KEYS = [
    "C_major", "C_minor",
    "Db_major", "Db_minor",
    "D_major", "D_minor",
    "Eb_major", "Eb_minor",
    "E_major", "E_minor",
    "F_major", "F_minor",
    "Gb_major", "Gb_minor",
    "G_major", "G_minor",
    "Ab_major", "Ab_minor",
    "A_major", "A_minor",
    "Bb_major", "Bb_minor",
    "B_major", "B_minor",
]
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-23 | Initial stub model for validation test |

---

## Quick Checklist

Before marking as "production":

- [ ] Model card completed
- [ ] All 216 nodes tested
- [ ] Key-invariance validated
- [ ] Hierarchical consistency verified
- [ ] Intensity tier discrimination measured
- [ ] Integration verified in Python API
- [ ] Integration verified in C++ MLInterface
- [ ] Confusion matrix reviewed per base emotion
- [ ] Fallback behavior documented
- [ ] Artifacts exported (ONNX/Core ML)
- [ ] Registry updated with SHA256 and paths

---

## Success Criteria

For this validation model to pass, we need:

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Base Emotion Accuracy | 70% | 85% | 95% |
| Sub-Emotion Accuracy | 50% | 70% | 85% |
| Node Accuracy (216) | 30% | 50% | 70% |
| Top-3 Node Accuracy | 50% | 70% | 85% |
| Key Invariance Score | 80% | 90% | 98% |
| Hierarchical Consistency | 85% | 95% | 99% |

**If these metrics are achieved, the emotion thesaurus is validated as learnable and discriminable!**

