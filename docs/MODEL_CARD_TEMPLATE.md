# Model Card Template

Use this template to document each trained model per [MK Training Guidelines](./MK_TRAINING_GUIDELINES.md) Section 6.

---

## Model Card: `<model_name>`

### Overview

| Field | Value |
|-------|-------|
| **Name** | `<model_id>` |
| **Version** | `<version>` (e.g., `1.0.0`) |
| **Task** | `<task_type>` (e.g., emotion_embedding, melody_generation) |
| **Format** | `<format>` (RTNeural JSON / ONNX / Core ML) |
| **Status** | `<status>` (stub / trained / production) |
| **Created** | `<date>` |
| **Author** | `<author>` |

---

### Training Data

| Field | Value |
|-------|-------|
| **Dataset Name** | `<dataset_id>` |
| **Source(s)** | `<sources>` (e.g., internal, Freesound, licensed) |
| **Size** | `<size>` (e.g., 10K samples, 50 hours) |
| **Split** | Train: `<train_pct>`% / Val: `<val_pct>`% / Test: `<test_pct>`% |
| **Sample Rate** | `<sample_rate>` Hz (if audio) |
| **Preprocessing** | `<preprocessing>` (e.g., MFCC, normalization) |
| **License** | `<data_license>` |

**Provenance Notes:**

```
<Brief description of data collection, consent, or licensing considerations.>
```

---

### Architecture

| Field | Value |
|-------|-------|
| **Input Size** | `<input_size>` floats |
| **Output Size** | `<output_size>` floats |
| **Layers** | `<arch_hint>` (e.g., 128→512→256→128→64) |
| **Parameters** | ~`<param_count>` |
| **Inference Time** | <`<latency_ms>` ms (target device) |

**Architecture Notes:**

```
<Brief description of layer types, activations, or design rationale.>
```

---

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | `<lr>` |
| Batch Size | `<batch_size>` |
| Epochs | `<epochs>` |
| Optimizer | `<optimizer>` |
| Loss Function | `<loss>` |
| Regularization | `<regularization>` |

**Training Notes:**

```
<Any special training considerations, schedules, or augmentation strategies.>
```

---

### Evaluation Metrics

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Loss | `<train_loss>` | `<val_loss>` | `<test_loss>` |
| Accuracy / F1 | `<train_acc>` | `<val_acc>` | `<test_acc>` |
| `<custom_metric>` | `<train_val>` | `<val_val>` | `<test_val>` |

**Qualitative Review:**

```
<Summary of listening tests, A/B comparisons, or subjective evaluation.>
```

---

### Run Metadata

| Field | Value |
|-------|-------|
| **Git Commit** | `<commit_sha>` |
| **Data Version** | `<data_version>` |
| **Training Date** | `<training_date>` |
| **Hardware** | `<hardware>` (e.g., M1 Pro, RTX 4090) |
| **Duration** | `<training_duration>` |
| **Logs** | `<path_to_logs>` |

---

### Known Limitations

- `<limitation_1>` (e.g., underperforms on genre X)
- `<limitation_2>` (e.g., trained only on 44.1kHz, untested at 48kHz)
- `<limitation_3>` (e.g., may overfit on small emotion categories)

---

### Intended Usage

**Primary Use Cases:**

- `<use_case_1>` (e.g., real-time emotion embedding in UnifiedHub)
- `<use_case_2>` (e.g., offline batch processing via Python API)

**Out-of-Scope Uses:**

- `<out_of_scope_1>` (e.g., production mastering decisions)
- `<out_of_scope_2>` (e.g., standalone music generation without review)

---

### Integration

| Target | Status | Notes |
|--------|--------|-------|
| Python API (`music_brain/`) | ✅ / ⬜ | `<notes>` |
| C++ MLInterface | ✅ / ⬜ | `<notes>` |
| Tauri UI | ✅ / ⬜ | `<notes>` |
| ONNX Export | ✅ / ⬜ | `<onnx_path>` |
| Core ML Export | ✅ / ⬜ | `<coreml_path>` |

---

### Files

| File | Description |
|------|-------------|
| `models/<model_id>.json` | RTNeural weights |
| `models/<model_id>.onnx` | ONNX export (optional) |
| `models/<model_id>.mlmodel` | Core ML export (optional) |
| `docs/model_cards/<model_id>.md` | This model card |

---

### Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | `<date>` | Initial trained release |

---

## Quick Checklist

Before marking a model as "production":

- [ ] Model card completed
- [ ] Metrics reviewed and acceptable
- [ ] Qualitative samples reviewed
- [ ] Integration verified in Python API
- [ ] Integration verified in C++ MLInterface
- [ ] Fallback behavior documented
- [ ] Artifacts exported (ONNX/Core ML if needed)
- [ ] Registry updated with SHA256 and paths

