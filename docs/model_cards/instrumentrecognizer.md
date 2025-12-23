# Model Card: InstrumentRecognizer

## Overview

| Field | Value |
|-------|-------|
| **Name** | `instrumentrecognizer` |
| **Version** | `1.0.0` |
| **Task** | `dual_instrument_recognition` |
| **Format** | RTNeural JSON / ONNX / Core ML |
| **Status** | `stub` |
| **Created** | 2025-12-23 |
| **Author** | Kelly ML Pipeline |

---

## What This Model Does

**This is a dual-head model** that understands instruments in two dimensions:

### 🔧 Technical Recognition (HEAD 1)
*"What instrument is this and how is it being played?"*

- **Instrument Family** (16 classes): Piano, guitar, strings, brass, woodwind, synth...
- **Specific Instrument** (32 classes): Acoustic guitar, electric piano, violin...
- **Playing Technique** (16 classes): Bowed, plucked, sustained, staccato...
- **Articulation** (8 classes): Legato, marcato, pizzicato...
- **Register** (4 classes): Bass, mid, treble

### 💫 Emotional Recognition (HEAD 2)
*"How does this instrument sound emotionally?"*

- **Expression Style** (12 classes): Aggressive, gentle, melancholic, joyful...
- **Energy Level** (8 levels): Calm → Intense
- **Musical Role** (8 classes): Lead, rhythm, pad, bass, accent...
- **Sentiment** (8+8 dims): Valence (sad↔happy) + Arousal (calm↔excited)
- **Human Feel** (8 dims): Mechanical → Highly expressive

---

## Training Data

| Field | Value |
|-------|-------|
| **Dataset Name** | `instrument_dataset_v1` |
| **Source(s)** | TBD (internal recordings, sample libraries, Freesound) |
| **Size** | Target: 50K samples (5K per family × 10 families) |
| **Split** | Train: 80% / Val: 10% / Test: 10% |
| **Sample Rate** | 22050 Hz |
| **Preprocessing** | Mel spectrogram (128 bands) |
| **License** | TBD |

**Provenance Notes:**

```
Dataset will include:
- Professional sample library recordings
- Field recordings of acoustic instruments
- Synthesizer presets across emotional ranges
- Varied playing styles per instrument
- Multiple performers for natural variation
```

---

## Architecture

| Field | Value |
|-------|-------|
| **Input Size** | 128 × 128 (mel × time frames) |
| **Output Size** | 160 (80 technical + 80 emotional) |
| **Backbone** | CNN: 64→128→256→512 |
| **Parameters** | ~2M |
| **Inference Time** | <10 ms (target) |

**Architecture Notes:**

```
Multi-Head CNN Architecture:

SHARED BACKBONE (extracts timbral features):
├── Conv2d(1, 64, 7×7, stride=2)  — Wide kernel for timbre
├── Conv2d(64, 128, 3×3, stride=2)
├── Conv2d(128, 256, 3×3, stride=2)
├── Conv2d(256, 512, 3×3, stride=2)
└── AdaptiveAvgPool → FC(512)

TECHNICAL HEAD:
├── FC(512, 256, relu)
├── Dropout(0.3)
├── FC(256, 128, relu)
└── FC(128, 80) → [instrument, technique, articulation, register]

EMOTIONAL HEAD:
├── FC(512, 256, gelu)  — Smoother activation for emotion
├── Dropout(0.25)
├── FC(256, 128, gelu)
└── FC(128, 80) → [expression, energy, role, sentiment]
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0005 |
| Batch Size | 12 |
| Epochs | 150 |
| Optimizer | AdamW |
| Loss Function | Multi-task (weighted CE + MSE) |
| Regularization | Dropout 0.25-0.3, Weight Decay 0.01 |

**Training Notes:**

```
Multi-task learning with loss balancing:
- Technical head weight: 0.5
- Emotional head weight: 0.5
- Label smoothing: 0.1

Data augmentation:
- Pitch shift: ±3 semitones (small to preserve timbre)
- Time stretch: 0.9x - 1.1x
- EQ variation: ±3dB bands
- Mixup within instrument families
```

---

## What It Learns

### Technical Understanding
| Feature | What the Model Learns |
|---------|----------------------|
| **Spectral Envelope** | Each instrument has a unique "fingerprint" |
| **Harmonics** | Brass is bright, woodwinds are warm |
| **Attack/Decay** | Piano has fast attack, pads have slow attack |
| **Vibrato** | Strings use vibrato, piano doesn't |
| **Register** | Bass notes vs melody notes |

### Emotional Understanding
| Feature | What the Model Learns |
|---------|----------------------|
| **Dynamics** | Loud + sudden = aggressive, soft + gradual = gentle |
| **Brightness** | Bright spectrum = happy, dark = sad |
| **Tension** | Dissonance = tension, consonance = resolution |
| **Humanization** | Perfect timing = mechanical, slight variation = expressive |
| **Energy** | RMS + tempo = overall energy level |

---

## Example Output

```json
{
  "technical": {
    "instrument_family": "strings_bowed",
    "instrument_family_confidence": 0.94,
    "instrument_specific": "violin",
    "technique": "legato",
    "articulation": "normal",
    "register": "mid"
  },
  "emotional": {
    "expression_style": "melancholic",
    "expression_confidence": 0.87,
    "energy_level": 3,
    "musical_role": "lead_melody",
    "sentiment": {
      "valence": -0.6,
      "arousal": 0.3
    },
    "human_feel": 0.85
  }
}
```

---

## Evaluation Metrics

| Metric | Technical Head | Emotional Head |
|--------|----------------|----------------|
| Family Accuracy | - | N/A |
| Technique Accuracy | - | N/A |
| Expression Accuracy | N/A | - |
| Energy MAE | N/A | - |
| Overall Loss | - | - |

**Qualitative Review:**

```
Model not yet trained. Evaluation will include:
- Blind listening tests with musicians
- A/B comparisons with ground truth
- Edge case analysis (ambiguous instruments)
- Cross-genre evaluation
```

---

## Known Limitations

- Not yet trained (stub model)
- Training data not yet collected
- May confuse similar instruments (viola/violin, oboe/clarinet)
- Emotional labels are subjective (trained on consensus)
- Single-instrument detection only (no polyphonic separation)
- Requires 0.5-5 second audio clips

---

## Intended Usage

**Primary Use Cases:**

- 🎸 Intelligent instrument tagging in DAWs
- 🎭 Emotion-aware mixing suggestions ("this violin sounds sad, try boosting presence")
- 🎹 Live performance analysis
- 🎼 Automatic orchestration assistance
- 🎧 Music recommendation based on instrumental mood

**Out-of-Scope Uses:**

- Polyphonic instrument separation
- Full arrangement analysis
- Production mastering decisions
- Real-time pitch correction

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
| `models/instrumentrecognizer.json` | RTNeural weights (stub) |
| `models/instrumentrecognizer.onnx` | ONNX export (pending) |
| `configs/instrument_recognizer.yaml` | Training configuration |
| `python/penta_core/ml/datasets/instrument_features.py` | Feature extraction |
| `python/penta_core/ml/datasets/instrument_synthetic.py` | Synthetic data generation |
| `docs/model_cards/instrumentrecognizer.md` | This model card |

---

## Data Collection Checklist

Before training, collect samples covering:

### Instrument Diversity
- [ ] Piano (grand, upright, electric)
- [ ] Guitars (acoustic, electric, bass)
- [ ] Strings (violin, viola, cello, bass, ensembles)
- [ ] Brass (trumpet, trombone, horn, tuba)
- [ ] Woodwinds (flute, clarinet, sax, oboe)
- [ ] Synths (leads, pads, bass)
- [ ] Drums/Percussion
- [ ] Voice (solo, choir)

### Expression Diversity
- [ ] Aggressive playing (all instruments)
- [ ] Gentle playing (all instruments)
- [ ] Melancholic expression
- [ ] Joyful expression
- [ ] Full dynamic range (pp to ff)
- [ ] Various articulations per instrument

### Context Diversity
- [ ] Solo recordings
- [ ] In-mix recordings
- [ ] Different rooms/reverbs
- [ ] Various microphone positions
- [ ] Different playing styles per instrument

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-23 | Initial stub model and architecture design |

---

## Quick Checklist

Before marking as "production":

- [ ] Model card completed
- [ ] Technical accuracy > 85%
- [ ] Emotional accuracy > 75%
- [ ] Qualitative review by musicians
- [ ] Integration verified in Python API
- [ ] Integration verified in C++ MLInterface
- [ ] Fallback behavior documented
- [ ] Artifacts exported (ONNX/Core ML)
- [ ] Registry updated with SHA256 and paths

