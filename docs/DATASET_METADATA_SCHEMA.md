# AudioDataset Metadata Schema Documentation

## Overview

The AudioDataset class in `scripts/train_model.py` supports rich metadata for ML training that includes:

- **Emotional attributes**: Basic emotion labels, dimensional emotion model (valence/arousal), intensity tiers
- **DAiW Thesaurus integration**: 216-node emotion hierarchy (6×6×6 structure)
- **Musical intent**: Key, tempo, mode, chord progressions, groove/feel
- **Performance attributes**: Articulation, dynamics, swing ratio
- **Quality metadata**: Quality scores, intentional rule breaks, tags, notes

## Metadata File Format

The dataset expects a `metadata.json` file in the dataset directory with the following structure:

```json
{
  "dataset_id": "emotion_audio_v1",
  "version": "1.0.0",
  "description": "Audio dataset with emotional and intent metadata",
  "samples": [
    {
      "file": "path/to/audio.wav",
      "emotion": "happy",
      "valence": 0.8,
      "arousal": 0.6,
      "intensity_tier": 3,
      "node_id": 42,
      "key": "C major",
      "tempo_bpm": 120.0,
      "mode": "major",
      "chord_progression": ["C", "Am", "F", "G"],
      "groove_type": "straight",
      "swing_ratio": 0.5,
      "articulation": "legato",
      "dynamic_range": "mf",
      "tags": ["uplifting", "energetic"],
      "quality_score": 0.9,
      "rule_breaks": [],
      "notes": "Optional description"
    }
  ]
}
```

## Field Definitions

### Required Fields

- **`file`** (string): Path to audio file, relative to dataset directory

### Emotional Attributes

- **`emotion`** (string): Basic emotion label
  - Examples: "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral", "grief", "peaceful", "anxiety"
  - Default: "neutral"

- **`valence`** (float): Emotional valence (positivity/negativity)
  - Range: -1.0 (very negative) to +1.0 (very positive)
  - Default: 0.0
  - Values outside range are automatically clamped

- **`arousal`** (float): Emotional arousal (energy level)
  - Range: 0.0 (very calm) to 1.0 (very energetic)
  - Default: 0.0
  - Values outside range are automatically clamped

- **`intensity_tier`** (int): Intensity tier (0-5)
  - 0: subtle
  - 1: mild
  - 2: moderate
  - 3: strong
  - 4: intense
  - 5: overwhelming
  - Default: 3
  - Values outside range are automatically clamped

- **`node_id`** (int, optional): DAiW emotion thesaurus node ID
  - Range: 0-215 (maps to 6×6×6 emotion hierarchy)
  - See `python/penta_core/ml/datasets/thesaurus_loader.py` for node structure
  - Each node represents: base_emotion → sub_emotion → sub_sub_emotion

### Musical Intent

- **`key`** (string): Musical key
  - Examples: "C major", "A minor", "F lydian", "E phrygian"
  - Default: ""

- **`tempo_bpm`** (float): Tempo in beats per minute
  - Default: 0.0

- **`mode`** (string): Musical mode
  - Examples: "major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian"
  - Default: ""

- **`chord_progression`** (array of strings): Chord progression
  - Examples: ["C", "Am", "F", "G"], ["F", "C", "Dm", "Bbm"]
  - Default: []

- **`groove_type`** (string): Groove/feel type
  - Options: "straight", "swing", "shuffle", "laid_back", "rushed"
  - Default: "straight"

### Performance Attributes

- **`swing_ratio`** (float, optional): Swing ratio
  - 0.5 = straight (no swing)
  - 0.58-0.62 = funk/soul
  - 0.66 = jazz triplet swing
  - Default: 0.5

- **`articulation`** (string, optional): Articulation type
  - Options: "legato", "staccato", "accent", "marcato", "tenuto", "normal"
  - Default: "normal"

- **`dynamic_range`** (string, optional): Dynamic marking
  - Options: "pp", "p", "mp", "mf", "f", "ff"
  - Default: "mf"

### Quality and Annotations

- **`tags`** (array of strings): Free-form tags
  - Examples: ["uplifting", "energetic", "bright"], ["melancholic", "introspective"]
  - Default: []

- **`quality_score`** (float): Human-assessed quality score
  - Range: 0.0 to 1.0
  - Default: 0.0

- **`rule_breaks`** (array of strings): Intentional music theory rule violations
  - Examples: ["HARMONY_ModalInterchange"], ["RHYTHM_ConstantDisplacement"]
  - See `music_brain/session/intent_schema.py` for complete list
  - Default: []

- **`notes`** (string, optional): Free-form notes about the sample

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from scripts.train_model import AudioDataset

# Create dataset
dataset = AudioDataset(
    data_dir=Path("data/my_emotion_dataset"),
    sample_rate=16000,
    n_mels=64,
    max_duration=5.0,
)

# Get sample metadata
metadata = dataset.get_sample_metadata(0)
print(f"Emotion: {metadata['emotion']}")
print(f"Valence: {metadata['valence']}, Arousal: {metadata['arousal']}")

# Get emotion labels for training
emotion_labels = dataset.get_emotion_labels(0)
print(emotion_labels)
# Output: {
#   'emotion': 'happy',
#   'valence': 0.8,
#   'arousal': 0.6,
#   'intensity_tier': 3,
#   'node_id': 42
# }

# Get musical metadata
musical = dataset.get_musical_metadata(0)
print(f"Key: {musical['key']}, Tempo: {musical['tempo_bpm']} BPM")
print(f"Progression: {musical['chord_progression']}")
```

### With Thesaurus Integration

```python
# Enable thesaurus integration
dataset = AudioDataset(
    data_dir=Path("data/my_emotion_dataset"),
    use_thesaurus=True,  # Load DAiW emotion thesaurus
)

# Thesaurus will map emotion names to node IDs
emotion_labels = dataset.get_emotion_labels(0)
if 'node_id' in emotion_labels:
    print(f"Emotion node: {emotion_labels['node_id']}")
```

### With PyTorch DataLoader

```python
from torch.utils.data import DataLoader

dataset = AudioDataset(data_dir=Path("data/my_emotion_dataset"))

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
)

for inputs, labels in loader:
    # inputs: [batch_size, 1, n_mels, time_steps]
    # labels: [batch_size] (basic emotion indices)
    train_step(inputs, labels)
```

## Fallback Behavior

If `metadata.json` is not found, the dataset will:

1. Scan the `processed/` directory for emotion-labeled subdirectories
2. Create basic metadata with:
   - Emotion label from directory name
   - Default values for all other fields (valence=0, arousal=0, intensity_tier=3)

## Validation

The dataset automatically validates metadata:

- **Range clamping**: Valence, arousal, and intensity_tier values outside valid ranges are clamped
- **Default values**: Missing optional fields are filled with sensible defaults
- **Required fields**: Raises `ValueError` if required `file` field is missing

## Integration with DAiW Philosophy

The metadata schema aligns with DAiW's "Interrogate Before Generate" philosophy:

1. **Emotional intent** (valence, arousal, intensity) drives technical decisions
2. **Musical metadata** (key, tempo, mode) follows from emotional intent
3. **Rule breaks** are intentional and emotionally justified
4. **Thesaurus integration** provides hierarchical emotion classification (216 nodes)

### Example: Grief with Modal Interchange

```json
{
  "file": "grief_piano.wav",
  "emotion": "grief",
  "valence": -0.9,
  "arousal": 0.2,
  "intensity_tier": 5,
  "key": "F major",
  "chord_progression": ["F", "C", "Dm", "Bbm"],
  "rule_breaks": ["HARMONY_ModalInterchange"],
  "notes": "Bbm in F major = borrowed sadness. Makes hope feel earned and bittersweet."
}
```

This captures:
- **Emotional state**: High negative valence, low arousal = grief
- **Musical choice**: Bbm (borrowed from F minor) creates bittersweet color
- **Intentional rule break**: Modal interchange is emotionally justified
- **Teaching moment**: Notes explain the "why" behind the technique

## See Also

- `examples/metadata_example.json` - Complete example with 6 samples
- `python/penta_core/ml/datasets/thesaurus_loader.py` - DAiW emotion thesaurus
- `python/penta_core/ml/datasets/base.py` - Base dataset structures
- `music_brain/session/intent_schema.py` - Rule-breaking categories
- `music_brain/data/emotional_mapping.py` - Emotion → musical parameter mapping
