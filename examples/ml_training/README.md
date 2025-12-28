# ML Training: Intent Schema Usage Examples

This directory demonstrates how to use the **CompleteSongIntent schema** for machine learning model training and validation.

## Overview

The intent schema provides **rich, structured training targets** beyond simple emotion labels. It captures:

- **Phase 0 (Core Wound/Desire)**: Deep emotional context that guides data augmentation
- **Phase 1 (Emotional Intent)**: Primary model targets (emotion, tension, vulnerability, narrative arc)
- **Phase 2 (Technical Constraints)**: Secondary targets (tempo, key, rule-breaking decisions)

## Files

### `intent_schema_usage_example.py`

Comprehensive example demonstrating:

1. **Loading Intent Files**: Create dataset from JSON intent files
2. **Encoding for ML**: Convert intent schema to model-ready targets
3. **Dataset Validation**: Ensure data quality for training
4. **Batch Processing**: Get batches of encoded targets
5. **PyTorch Integration**: Use with PyTorch DataLoader
6. **Multi-Task Learning**: Extract multiple target types from single intent
7. **Custom Encoding**: Configure target encoding for specific models
8. **Augmentation Strategy**: How intent guides data augmentation decisions

## Quick Start

### 1. Create Intent Files

First, create some intent JSON files:

```bash
# Using the CLI
cd examples/music_brain/intents

# Files already exist there:
# - kelly_when_i_found_you_sleeping.json
```

### 2. Run the Example

```bash
# Basic usage (uses examples/music_brain/intents by default)
python examples/ml_training/intent_schema_usage_example.py

# With custom intent directory
python examples/ml_training/intent_schema_usage_example.py --intent-dir path/to/intents

# Show statistics only
python examples/ml_training/intent_schema_usage_example.py --stats-only
```

## Usage in Training Code

### Basic: Load Intent Dataset

```python
from python.penta_core.ml.datasets.intent_dataset import IntentDataset

# Load intents
dataset = IntentDataset(
    intent_dir="examples/music_brain/intents",
    validate_on_load=True
)

print(f"Loaded {len(dataset)} intents")

# Get sample
sample = dataset[0]
intent = sample["intent"]  # CompleteSongIntent
targets = sample["targets"]  # Encoded targets dict
```

### Advanced: PyTorch Training Loop

```python
import torch
from torch.utils.data import DataLoader
from python.penta_core.ml.datasets.intent_dataset import IntentDataset

# Create dataset
dataset = IntentDataset(intent_dir="path/to/intents")

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch in loader:
    intents = batch["intent"]
    targets = batch["targets"]
    
    # Get specific targets
    emotion_labels = [t["emotion_label"] for t in targets]
    tensions = [t["tension"] for t in targets]
    
    # Forward pass
    # ...
```

### Multi-Task Model Example

```python
from python.penta_core.ml.datasets.intent_dataset import IntentDataset

dataset = IntentDataset(intent_dir="path/to/intents")

# Get batch
batch = dataset.get_batch([0, 1, 2])

# Classification tasks
emotion_labels = batch["emotion_labels"]  # Shape: (batch_size,)
arc_labels = batch["narrative_arc_labels"]
vulnerability = batch["vulnerabilities"]

# Regression tasks
tensions = batch["tensions"]  # Shape: (batch_size,)
tempos = batch["tempo_normalized"]

# Binary tasks
has_justification = batch["has_justification"]
intent_valid = batch["intent_valid"]
```

## Intent Schema → Model Targets

### Phase 1: Emotional Intent (Primary Targets)

| Intent Field | Target Type | Model Use |
|--------------|-------------|-----------|
| `mood_primary` | Classification (10 classes) | Emotion recognition |
| `mood_secondary_tension` | Regression (0.0-1.0) | Tension prediction |
| `vulnerability_scale` | Classification (3 classes) | Vulnerability level |
| `narrative_arc` | Classification (8 classes) | Structural prediction |
| `imagery_texture` | Text (optional) | Semantic embedding |

### Phase 2: Technical Constraints (Secondary Targets)

| Intent Field | Target Type | Model Use |
|--------------|-------------|-----------|
| `technical_tempo_range` | Regression | Tempo prediction |
| `technical_key` | Classification | Key detection |
| `technical_mode` | Classification | Mode classification |
| `technical_rule_to_break` | Classification (21+ classes) | Rule-breaking category |
| `rule_breaking_justification` | Binary | Has justification? |

### Encoded Target Dictionary

Each intent is encoded to:

```python
{
    # Emotion
    "emotion_label": int,          # 0-9
    "emotion_onehot": np.ndarray,  # (10,)
    "emotion_str": str,
    
    # Continuous values
    "tension": float,              # 0.0-1.0
    "vulnerability": int,          # 0-2
    "tempo_normalized": float,     # 0.0-1.0
    "tempo_bpm": float,
    
    # Structural
    "narrative_arc_label": int,    # 0-7
    "narrative_arc_str": str,
    
    # Rule breaking
    "rule_break_id": int,
    "rule_break_str": str,
    "rule_break_category": int,    # HARMONY, RHYTHM, etc.
    "has_justification": bool,
    
    # Validation
    "intent_valid": bool,
    "validation_issues": int,
}
```

## Intent-Driven Data Augmentation

The intent schema **guides safe augmentation**:

### Phase 0: Core Wound/Desire

- `core_stakes` → Determines what can be modified
- `core_transformation` → Guides augmentation direction

### Phase 1: Emotional Intent

- `mood_primary` → Constrains key/mode changes
- `vulnerability_scale` → Limits dynamic changes
- `narrative_arc` → Preserves structural meaning

### Phase 2: Technical Constraints

- `technical_rule_to_break` → **MUST be preserved**
- `rule_breaking_justification` → Validates augmentation

### Safe Augmentations

✓ **Tempo shift** ±10% (preserves emotion)  
✓ **Transpose** to nearby keys (preserves mode relationships)  
✓ **Add instrumentation** (preserves core harmony)  
✓ **Dynamics variation** within vulnerability_scale limits

### Unsafe Augmentations

✗ **Change mode** (breaks emotional intent)  
✗ **Remove rule break** (loses creative intent)  
✗ **Extreme tempo change** (changes emotion category)  
✗ **Violate narrative_arc** (breaks structural meaning)

## Multi-Task Learning Architecture

Intent schema enables **multi-task learning** with shared representations:

```
┌─────────────────┐
│  Audio/MIDI     │
│  Input          │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Shared  │
    │ Encoder  │
    └────┬─────┘
         │
    ┌────▼───────────────────────┐
    │  Intent Representation     │
    └────┬───────────────────────┘
         │
    ┌────┴────────────────────────────┐
    │                                  │
┌───▼────────┐              ┌─────────▼──────┐
│ Emotion    │              │ Rule Breaking  │
│ Classifier │              │ Classifier     │
└───┬────────┘              └─────────┬──────┘
    │                                  │
┌───▼────────┐              ┌─────────▼──────┐
│ Tension    │              │ Tempo          │
│ Regressor  │              │ Regressor      │
└────────────┘              └────────────────┘
```

## Dataset Validation

Validate before training:

```python
from python.penta_core.ml.datasets.intent_dataset import (
    IntentDataset,
    validate_dataset_for_training
)

dataset = IntentDataset(intent_dir="path/to/intents")

is_valid, issues = validate_dataset_for_training(
    dataset,
    min_samples_per_emotion=10
)

if not is_valid:
    print("Dataset issues:")
    for issue in issues:
        print(f"  - {issue}")
```

Checks:
- Minimum samples per emotion class
- Emotion distribution balance
- Intent validation status
- Tension value variance
- Tempo distribution

## Custom Encoding Configuration

Customize encoding for specific models:

```python
from python.penta_core.ml.datasets.intent_dataset import (
    IntentDataset,
    IntentEncodingConfig
)

# Custom config
config = IntentEncodingConfig(
    emotion_labels=["grief", "joy", "anger"],  # Subset
    embed_dim_emotion=128,                      # Larger embedding
    tempo_range=(60, 140),                      # Narrower range
)

# Create dataset with custom encoding
dataset = IntentDataset(
    intent_dir="path/to/intents",
    encoding_config=config
)
```

## Philosophy: "Interrogate Before Generate"

Traditional music generation:
- **Input**: "Generate sad music in Am at 80 BPM"
- **Problem**: No emotional context, no creative intent

Intent-driven generation:
- **Input**: CompleteSongIntent with deep interrogation
  - Why are you making this?
  - What do you want to feel?
  - What rules should be broken and why?
- **Benefit**: Model learns **meaningful emotional relationships**, not just patterns

## Key Principles

1. **Emotional intent drives technical decisions** - Never generate without understanding the "why"
2. **Rules are broken intentionally** - Every rule break requires emotional justification
3. **Multi-task targets** - Single intent provides multiple learning signals
4. **Validation matters** - Ensure intent quality before training
5. **Augmentation respects intent** - Don't break what the artist carefully crafted

## Related Documentation

- [Intent Schema Guide](../../Songwriting_Guides/song_intent_schema.md) - Complete schema documentation
- [Intent Examples](../music_brain/intents/README.md) - Example intent files
- [Rule Breaking](../../Songwriting_Guides/rule_breaking_practical.md) - Rule-breaking guide
- [Intent Schema Code](../../music_brain/session/intent_schema.py) - Schema implementation

## References

Intent schema implementation:
- `music_brain/session/intent_schema.py` - Core schema definition
- `python/penta_core/ml/datasets/intent_dataset.py` - Dataset implementation
- `examples/music_brain/intents/` - Example intent files

---

*"The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"*

**Intent-driven ML teaches models to understand emotion, not just imitate patterns.**
