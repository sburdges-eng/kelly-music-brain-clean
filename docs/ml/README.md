# ML Documentation

This directory contains documentation for machine learning components in the Kelly project.

## Contents

### Guides

- **[Emotional Mapping for ML Labeling](./EMOTIONAL_MAPPING_FOR_ML_LABELING.md)** - Comprehensive guide on using `EmotionalState` from `data/emotional_mapping.py` for creating ML training labels
  - Basic emotional state labeling
  - Multi-label classification
  - Musical parameter integration
  - Dataset manifest creation
  - PyTorch integration examples
  - Validation and testing

### Related Documentation

- **[ML Frameworks Evaluation](./ML_FRAMEWORKS_EVALUATION.md)** - Evaluation of ML frameworks for the project
- **[Model Cards](../model_cards/)** - Specifications for trained models
  - [EmotionRecognizer](../model_cards/emotionrecognizer.md)
  - [EmotionNodeClassifier](../model_cards/emotionnodeclassifier.md)

## Quick Start

### Using EmotionalState for Labeling

```python
from data.emotional_mapping import EmotionalState, get_parameters_for_state

# Create emotional label
emotion = EmotionalState(
    valence=0.8,      # Positive
    arousal=0.75,     # High energy
    primary_emotion="joy"
)

# Use in training
label = {
    "class_name": emotion.primary_emotion,
    "valence": emotion.valence,
    "arousal": emotion.arousal
}
```

See [EMOTIONAL_MAPPING_FOR_ML_LABELING.md](./EMOTIONAL_MAPPING_FOR_ML_LABELING.md) for complete examples.

### Running Examples

```bash
# Run emotional labeling example
python examples/ml/emotional_labeling_example.py

# Run tests
python -m pytest tests/ml/test_emotional_labeling.py
```

## Core Components

### EmotionalState Class

Defined in `data/emotional_mapping.py`:

```python
@dataclass
class EmotionalState:
    valence: float              # -1 (negative) to +1 (positive)
    arousal: float              # 0 (calm) to 1 (energetic)
    primary_emotion: str
    secondary_emotions: List[str] = field(default_factory=list)
    has_intrusions: bool = False
    intrusion_probability: float = 0.0
```

### Available Presets

- `profound_grief` - Deep grief with negative valence, low arousal
- `ptsd_anxiety` - Anxiety with intrusions
- `bittersweet_nostalgia` - Nostalgia with mixed valence
- `suppressed_anger` - Anger held back

## Training Pipeline

1. **Label Creation** - Use `EmotionalState` to create rich emotional labels
2. **Manifest Generation** - Create JSONL manifests with emotional metadata
3. **Dataset Loading** - Use PyTorch Dataset/DataLoader with emotional labels
4. **Multi-Task Training** - Train models on both discrete emotions and continuous valence/arousal
5. **Validation** - Ensure labels are consistent and within valid ranges

## Best Practices

1. **Use Presets** - Start with `EMOTIONAL_STATE_PRESETS` for common emotions
2. **Validate Ranges** - Always validate valence, arousal, and probability ranges
3. **Multi-Label Support** - Use `secondary_emotions` for complex emotional states
4. **Musical Integration** - Use `get_parameters_for_state()` to convert to musical parameters
5. **JSONL Format** - Store training manifests in JSONL format for easy streaming

## References

- Source: [`data/emotional_mapping.py`](../../data/emotional_mapping.py)
- Examples: [`examples/ml/emotional_labeling_example.py`](../../examples/ml/emotional_labeling_example.py)
- Tests: [`tests/ml/test_emotional_labeling.py`](../../tests/ml/test_emotional_labeling.py)
- Training Stub: [`ML Kelly Training/train_mps_stub.py`](../../ML%20Kelly%20Training/train_mps_stub.py)

## Contributing

When adding new emotional labels or presets:

1. Ensure valence is in [-1, 1]
2. Ensure arousal is in [0, 1]
3. Ensure intrusion_probability is in [0, 1]
4. Add corresponding tests
5. Update documentation with examples
6. Consider musical parameter mappings

---

**Last Updated**: December 2024  
**Maintainer**: Kelly Development Team
