# Implementation Summary: ML Training Data Loader with Emotional and Intent Metadata

## Issue
Implement a data loader for ML training that includes emotional and intent metadata.

Reference: Issue "ML Training: Data Loader and Metadata Extraction Example"

## Implementation

### Overview
Enhanced the `AudioDataset` class in `scripts/train_model.py` to support rich metadata for ML training, including:
- Emotional attributes (valence, arousal, intensity)
- DAiW thesaurus integration (216 emotion nodes)
- Musical intent (key, tempo, mode, chord progressions)
- Quality annotations (tags, rule breaks, notes)

### Files Modified

#### `scripts/train_model.py`
- Enhanced `AudioDataset` class (lines 158-340)
- Added `use_thesaurus` parameter to enable DAiW emotion thesaurus integration
- Enhanced `_load_samples()` to support rich metadata schema
- Added `_validate_sample_metadata()` for validation and defaults
- Added `get_emotion_labels()` to extract emotion training labels
- Added `get_musical_metadata()` to extract musical intent metadata
- Added `get_sample_metadata()` to get complete sample metadata

### Files Created

#### Documentation
- **`docs/DATASET_METADATA_SCHEMA.md`** (7,990 bytes)
  - Complete schema reference
  - Field definitions and ranges
  - Usage examples
  - Integration with DAiW philosophy

#### Examples
- **`examples/metadata_example.json`** (5,921 bytes)
  - 6 example samples with complete metadata
  - Demonstrates emotional attributes (happy, sad, grief, angry, peaceful, anxiety)
  - Shows intentional rule-breaking (modal interchange, rhythm displacement)
  - Includes musical intent (key, tempo, mode, chord progressions)

- **`examples/demo_metadata_loading.py`** (6,367 bytes)
  - Interactive demonstration script
  - Shows emotional attributes extraction
  - Shows musical metadata extraction
  - Shows rule-breaking examples
  - Shows thesaurus node mapping

- **`examples/validate_enhancements.py`** (7,680 bytes)
  - Comprehensive validation script
  - Tests metadata loading logic
  - Validates code structure
  - Checks documentation completeness
  - Runs standalone tests

#### Tests
- **`tests/ml/test_metadata_standalone.py`** (7,640 bytes)
  - 6 standalone tests (no PyTorch required)
  - Tests metadata loading
  - Tests emotional attributes
  - Tests musical metadata
  - Tests thesaurus integration
  - Tests rule-breaking metadata
  - Tests validation ranges
  - ✓ All tests passing

- **`tests/ml/test_audio_dataset_metadata.py`** (8,769 bytes)
  - Full PyTorch-based tests (requires PyTorch)
  - Tests with DataLoader integration
  - Tests metadata validation
  - Tests fallback directory scanning

## Metadata Schema

### Required Fields
- `file`: Path to audio file (relative to dataset directory)

### Emotional Attributes
- `emotion`: Basic emotion label (happy, sad, grief, etc.)
- `valence`: float [-1.0, 1.0] - negative to positive
- `arousal`: float [0.0, 1.0] - calm to energetic
- `intensity_tier`: int [0-5] - subtle to overwhelming
- `node_id`: int [0-215] - DAiW thesaurus node (optional)

### Musical Intent
- `key`: Musical key (e.g., "C major", "F lydian")
- `tempo_bpm`: Tempo in beats per minute
- `mode`: Musical mode (major, minor, dorian, etc.)
- `chord_progression`: Array of chord symbols
- `groove_type`: Groove/feel (straight, swing, laid_back, etc.)
- `swing_ratio`: Swing ratio (0.5 = straight, 0.66 = jazz)
- `articulation`: Articulation type (legato, staccato, etc.)
- `dynamic_range`: Dynamic marking (pp, p, mp, mf, f, ff)

### Quality & Annotations
- `tags`: Array of free-form tags
- `quality_score`: float [0.0, 1.0] - human-assessed quality
- `rule_breaks`: Array of intentional rule violations
- `notes`: Free-form description

## Example Usage

```python
from pathlib import Path
from scripts.train_model import AudioDataset

# Create dataset
dataset = AudioDataset(
    data_dir=Path("data/my_emotion_dataset"),
    use_thesaurus=True,  # Enable DAiW thesaurus
)

# Get emotion labels for multi-head training
labels = dataset.get_emotion_labels(0)
# {'emotion': 'grief', 'valence': -0.9, 'arousal': 0.2, 'intensity_tier': 5, 'node_id': 156}

# Get musical metadata
musical = dataset.get_musical_metadata(0)
# {'key': 'F major', 'tempo_bpm': 68.0, 'chord_progression': ['F', 'C', 'Dm', 'Bbm'], ...}
```

## DAiW Philosophy Integration

The metadata schema aligns with DAiW's core principles:

1. **"Interrogate Before Generate"**
   - Emotional intent (valence, arousal, intensity) is defined before technical choices
   - Musical metadata follows from emotional intent

2. **Intentional Rule Breaking**
   - `rule_breaks` field captures emotionally-justified violations
   - Example: Modal interchange (Bbm in F major) for bittersweet grief

3. **Hierarchical Emotion Model**
   - Integration with 6×6×6 thesaurus (216 emotion nodes)
   - base_emotion → sub_emotion → sub_sub_emotion

4. **Emotional State Drives Musical Choices**
   - High negative valence + low arousal = grief → slow tempo, minor or modal interchange
   - High arousal + negative valence = anger → fast tempo, dissonant modes

## Validation

All validation checks pass:
- ✓ Metadata loading logic
- ✓ Code structure complete
- ✓ Documentation complete
- ✓ Test coverage (6/6 standalone tests passing)
- ✓ Example files present
- ✓ Demo scripts functional

## Testing

Run validation:
```bash
python examples/validate_enhancements.py
```

Run standalone tests:
```bash
python tests/ml/test_metadata_standalone.py
```

Run demo:
```bash
python examples/demo_metadata_loading.py
```

## Key Features

### 1. Flexible Metadata Loading
- Supports JSON metadata files with rich attributes
- Falls back to directory scanning if metadata.json not found
- Automatic validation and range clamping

### 2. Multi-Head Training Support
- `get_emotion_labels()` returns labels for multi-head models
- Supports basic emotion (7 classes), valence/arousal (regression), intensity (6 classes), and node_id (216 classes)

### 3. DAiW Thesaurus Integration
- Optional integration with ThesaurusLoader
- Maps emotion names to hierarchical node IDs
- Supports 6×6×6 emotion hierarchy (216 nodes)

### 4. Musical Intent Metadata
- Captures key, tempo, mode, chord progressions
- Supports groove/feel attributes
- Includes articulation and dynamics

### 5. Quality Annotations
- Human-assessed quality scores
- Intentional rule violations with emotional justification
- Free-form tags and notes

## Example: Grief with Modal Interchange

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
- Emotional state: High negative valence, low arousal
- Musical choice: Bbm (borrowed from F minor) creates bittersweet color
- Intentional rule break: Modal interchange with emotional justification
- Teaching moment: Notes explain the "why"

## Impact

This implementation enables:
1. **Rich training data** - Beyond basic emotion labels to dimensional models
2. **Multi-task learning** - Train models on emotion, intensity, musical attributes simultaneously
3. **Interpretable models** - Metadata provides context for model decisions
4. **DAiW philosophy** - "Interrogate before generate" embedded in the data

## References

- `examples/metadata_example.json` - Complete example with 6 samples
- `docs/DATASET_METADATA_SCHEMA.md` - Full schema documentation
- `python/penta_core/ml/datasets/thesaurus_loader.py` - DAiW emotion thesaurus
- `music_brain/session/intent_schema.py` - Rule-breaking categories
- `music_brain/data/emotional_mapping.py` - Emotion → musical parameter mapping

## Conclusion

The AudioDataset has been successfully enhanced with comprehensive metadata support for emotional and intent-based ML training. The implementation is minimal, surgical, and fully validated with tests and documentation.
