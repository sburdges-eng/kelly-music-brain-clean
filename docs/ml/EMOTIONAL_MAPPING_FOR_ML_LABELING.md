# Emotional Mapping for ML Training Labels

## Overview

This guide demonstrates how to use the `EmotionalState` class from `data/emotional_mapping.py` for labeling data in ML training pipelines. The emotional mapping system translates emotional states (valence, arousal, primary/secondary emotions) into musical parameters, providing rich labels for training emotion-aware models.

## Core Class: EmotionalState

Located in [`data/emotional_mapping.py`](../../data/emotional_mapping.py)

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class EmotionalState:
    """
    Represents an emotional state for mapping to musical parameters.

    Attributes:
        valence: Negative (-1) to positive (+1) emotional tone
        arousal: Calm (0) to energetic (1) activation level
        primary_emotion: Main emotion (grief, anger, anxiety, etc.)
        secondary_emotions: Supporting emotions
        has_intrusions: PTSD/trauma intrusions present
        intrusion_probability: Likelihood of intrusion events (0-1)
    """
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (energetic)
    primary_emotion: str
    secondary_emotions: List[str] = field(default_factory=list)
    has_intrusions: bool = False
    intrusion_probability: float = 0.0
```

### Key Features

- **Valence-Arousal Model**: Two-dimensional emotional space
- **Multi-label Support**: Primary + secondary emotions
- **Trauma-Aware**: PTSD intrusion tracking
- **Validation**: Built-in range checks via `__post_init__`

## Usage in ML Training

### 1. Basic Labeling Example

```python
from data.emotional_mapping import EmotionalState, get_parameters_for_state

# Create an emotional state label
emotion = EmotionalState(
    valence=0.8,      # Positive
    arousal=0.75,     # High energy
    primary_emotion="joy"
)

# Use for model training
label = {
    "emotion_state": emotion,
    "valence": emotion.valence,
    "arousal": emotion.arousal,
    "primary": emotion.primary_emotion
}
```

### 2. Multi-Label Classification

```python
from data.emotional_mapping import EmotionalState

# Complex emotional state with multiple labels
emotion = EmotionalState(
    valence=-0.6,
    arousal=0.4,
    primary_emotion="grief",
    secondary_emotions=["nostalgia", "loss"]
)

# Generate multi-hot encoding for secondary emotions
all_emotions = ["grief", "nostalgia", "loss", "anger", "joy"]
multi_label = [
    1 if e == emotion.primary_emotion or e in emotion.secondary_emotions else 0
    for e in all_emotions
]
# Result: [1, 1, 1, 0, 0]
```

### 3. Integration with Musical Parameters

The `EmotionalState` can be converted to musical parameters for training models that generate or modify music:

```python
from data.emotional_mapping import EmotionalState, get_parameters_for_state

# Create emotional state
emotion = EmotionalState(
    valence=-0.8,
    arousal=0.3,
    primary_emotion="grief"
)

# Get corresponding musical parameters
params = get_parameters_for_state(emotion)

# Use as training targets
training_sample = {
    "input_audio": "path/to/audio.wav",
    "emotion_label": emotion,
    "target_tempo": params.tempo_suggested,      # 72 BPM for grief
    "target_mode": params.mode_weights,           # {"minor": 0.6, "dorian": 0.3}
    "target_dissonance": params.dissonance,       # 0.3
    "target_timing_feel": params.timing_feel.value  # "behind"
}
```

### 4. Preset Emotional States

Use predefined presets for common emotional states:

```python
from data.emotional_mapping import EMOTIONAL_STATE_PRESETS

# Use preset states
profound_grief = EMOTIONAL_STATE_PRESETS["profound_grief"]
# EmotionalState(valence=-0.8, arousal=0.3, primary_emotion="grief", ...)

ptsd_anxiety = EMOTIONAL_STATE_PRESETS["ptsd_anxiety"]
# EmotionalState(valence=-0.6, arousal=0.8, primary_emotion="anxiety",
#                has_intrusions=True, intrusion_probability=0.2)

bittersweet = EMOTIONAL_STATE_PRESETS["bittersweet_nostalgia"]
# EmotionalState(valence=0.2, arousal=0.4, primary_emotion="nostalgia", ...)
```

## Dataset Integration

### Creating Training Manifests with Emotional Labels

```python
import json
from pathlib import Path
from data.emotional_mapping import EmotionalState

def create_labeled_manifest(audio_files, output_path):
    """Create a training manifest with emotional labels."""
    manifest = []
    
    for audio_file, emotion_data in audio_files:
        # Create emotional state
        emotion = EmotionalState(
            valence=emotion_data["valence"],
            arousal=emotion_data["arousal"],
            primary_emotion=emotion_data["primary_emotion"],
            secondary_emotions=emotion_data.get("secondary_emotions", [])
        )
        
        # Create manifest entry
        entry = {
            "audio": str(audio_file),
            "label": emotion.primary_emotion,
            "valence": emotion.valence,
            "arousal": emotion.arousal,
            "secondary_emotions": emotion.secondary_emotions,
            "has_intrusions": emotion.has_intrusions
        }
        manifest.append(entry)
    
    # Save as JSONL
    with open(output_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

# Usage
audio_files = [
    (Path("audio/sad_01.wav"), {
        "valence": -0.7,
        "arousal": 0.3,
        "primary_emotion": "grief",
        "secondary_emotions": ["loss"]
    }),
    (Path("audio/happy_01.wav"), {
        "valence": 0.8,
        "arousal": 0.7,
        "primary_emotion": "joy",
        "secondary_emotions": ["excitement"]
    })
]

create_labeled_manifest(audio_files, "manifests/emotion_train.jsonl")
```

### Using in PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset
from data.emotional_mapping import EmotionalState

class EmotionalAudioDataset(Dataset):
    """Dataset that uses EmotionalState for labels."""
    
    def __init__(self, manifest_path, emotion_to_idx):
        with open(manifest_path) as f:
            self.samples = [json.loads(line) for line in f]
        self.emotion_to_idx = emotion_to_idx
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Reconstruct EmotionalState
        emotion = EmotionalState(
            valence=sample["valence"],
            arousal=sample["arousal"],
            primary_emotion=sample["label"],
            secondary_emotions=sample.get("secondary_emotions", []),
            has_intrusions=sample.get("has_intrusions", False)
        )
        
        # Load audio (simplified)
        audio = self._load_audio(sample["audio"])
        
        # Create labels
        label_idx = self.emotion_to_idx[emotion.primary_emotion]
        
        return audio, {
            "class_idx": label_idx,
            "valence": torch.tensor(emotion.valence, dtype=torch.float32),
            "arousal": torch.tensor(emotion.arousal, dtype=torch.float32),
            "emotion_state": emotion
        }
    
    def _load_audio(self, path):
        # Implement audio loading
        pass

# Usage
emotion_to_idx = {"grief": 0, "joy": 1, "anger": 2, "anxiety": 3, "calm": 4}
dataset = EmotionalAudioDataset("manifests/emotion_train.jsonl", emotion_to_idx)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
```

## Training Examples

### 1. Multi-Task Emotion Model

Train a model to predict both discrete emotions and continuous valence/arousal:

```python
import torch
import torch.nn as nn
from data.emotional_mapping import EmotionalState

class EmotionMultiTaskModel(nn.Module):
    """Predicts both emotion class and valence/arousal."""
    
    def __init__(self, num_emotions):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Discrete emotion classification
        self.emotion_head = nn.Linear(64, num_emotions)
        
        # Continuous valence/arousal regression
        self.valence_head = nn.Linear(64, 1)
        self.arousal_head = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        return {
            "emotion_logits": self.emotion_head(features),
            "valence": torch.tanh(self.valence_head(features)),  # [-1, 1]
            "arousal": torch.sigmoid(self.arousal_head(features))  # [0, 1]
        }

# Training loop
model = EmotionMultiTaskModel(num_emotions=5)
optimizer = torch.optim.Adam(model.parameters())

for audio, labels in dataloader:
    optimizer.zero_grad()
    
    outputs = model(audio)
    
    # Multi-task loss
    emotion_loss = nn.functional.cross_entropy(
        outputs["emotion_logits"], labels["class_idx"]
    )
    valence_loss = nn.functional.mse_loss(
        outputs["valence"], labels["valence"]
    )
    arousal_loss = nn.functional.mse_loss(
        outputs["arousal"], labels["arousal"]
    )
    
    total_loss = emotion_loss + 0.5 * valence_loss + 0.5 * arousal_loss
    total_loss.backward()
    optimizer.step()
```

### 2. Generating Labels from Audio Analysis

```python
from data.emotional_mapping import EmotionalState, get_parameters_for_state
import librosa

def analyze_audio_emotion(audio_path):
    """Analyze audio and create EmotionalState label."""
    # Load audio
    y, sr = librosa.load(audio_path)
    
    # Extract features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    
    # Map to emotional dimensions
    # (These are simplified heuristics; real analysis would be more complex)
    
    # Arousal: Higher tempo + spectral brightness → higher arousal
    arousal = min(1.0, (tempo / 200.0) + (spectral_centroid / 5000.0))
    
    # Valence: Major chroma (C, E, G) → positive, minor → negative
    major_strength = chroma[[0, 4, 7], :].mean()
    minor_strength = chroma[[0, 3, 7], :].mean()
    valence = (major_strength - minor_strength)
    
    # Determine primary emotion
    if valence < -0.5 and arousal < 0.5:
        primary = "grief"
    elif valence < -0.5 and arousal > 0.5:
        primary = "anxiety"
    elif valence > 0.5 and arousal > 0.5:
        primary = "joy"
    elif valence > 0.5 and arousal < 0.5:
        primary = "calm"
    else:
        primary = "nostalgia"
    
    return EmotionalState(
        valence=float(valence),
        arousal=float(arousal),
        primary_emotion=primary
    )

# Usage
emotion = analyze_audio_emotion("audio/track.wav")
params = get_parameters_for_state(emotion)
print(f"Detected: {emotion.primary_emotion}")
print(f"Suggested tempo: {params.tempo_suggested} BPM")
```

### 3. Augmentation with Emotional Context

```python
from data.emotional_mapping import EmotionalState, EMOTIONAL_PRESETS

def augment_with_emotion(audio_path, target_emotion):
    """Augment audio to match target emotional state."""
    # Get target parameters
    emotion = EmotionalState(
        valence=-0.7 if "sad" in target_emotion else 0.7,
        arousal=0.3 if "calm" in target_emotion else 0.8,
        primary_emotion=target_emotion
    )
    params = get_parameters_for_state(emotion)
    
    # Apply transformations based on emotional parameters
    augmentations = {
        "tempo_shift": params.tempo_suggested / 120.0,  # Relative to base 120 BPM
        "pitch_shift": 0 if params.register.value == "mid" else (
            -3 if params.register.value == "low" else 3
        ),
        "reverb_amount": params.space_probability,
        "compression": 0.8 if params.density.value == "dense" else 0.3
    }
    
    return augmentations

# Usage
aug = augment_with_emotion("track.wav", "grief")
# Returns: {"tempo_shift": 0.6, "pitch_shift": 0, "reverb_amount": 0.3, ...}
```

## Validation and Testing

### Validating Emotional State Labels

```python
from data.emotional_mapping import EmotionalState
import pytest

def test_emotional_state_validation():
    """Test EmotionalState validates ranges."""
    
    # Valid state
    emotion = EmotionalState(
        valence=0.5,
        arousal=0.5,
        primary_emotion="calm"
    )
    assert -1 <= emotion.valence <= 1
    assert 0 <= emotion.arousal <= 1
    
    # Invalid valence
    with pytest.raises(AssertionError):
        EmotionalState(valence=1.5, arousal=0.5, primary_emotion="joy")
    
    # Invalid arousal
    with pytest.raises(AssertionError):
        EmotionalState(valence=0.5, arousal=-0.1, primary_emotion="calm")

def test_preset_consistency():
    """Test preset emotional states are valid."""
    from data.emotional_mapping import EMOTIONAL_STATE_PRESETS
    
    for name, emotion in EMOTIONAL_STATE_PRESETS.items():
        assert -1 <= emotion.valence <= 1
        assert 0 <= emotion.arousal <= 1
        assert 0 <= emotion.intrusion_probability <= 1
```

## Best Practices

### 1. Label Consistency

- **Use Presets**: Start with `EMOTIONAL_STATE_PRESETS` for common emotions
- **Document Mappings**: Keep a mapping file of emotion_name → EmotionalState
- **Validate Ranges**: Always validate valence, arousal, and probability ranges

### 2. Multi-Label Strategies

- **Primary + Secondary**: Use primary_emotion for main label, secondary_emotions for multi-hot encoding
- **Hierarchical**: Map to emotion taxonomy (base → sub → sub-sub)
- **Continuous**: Use valence/arousal for regression tasks

### 3. Integration with Existing Systems

- **Model Cards**: Document emotional labels in model cards (see `docs/model_cards/`)
- **Manifests**: Use JSONL format for training manifests with emotional labels
- **Config Files**: Store emotion-to-index mappings in YAML configs

### 4. Handling Edge Cases

```python
from data.emotional_mapping import EmotionalState, EMOTIONAL_PRESETS

def handle_unknown_emotion(emotion_name, fallback="calm"):
    """Safely handle unknown emotions."""
    # Try presets first
    for preset_name, preset in EMOTIONAL_STATE_PRESETS.items():
        if preset.primary_emotion == emotion_name:
            return preset
    
    # Try emotional presets
    if emotion_name in EMOTIONAL_PRESETS:
        params = EMOTIONAL_PRESETS[emotion_name]
        # Infer valence/arousal from parameters
        valence = -0.5 if "minor" in params.mode_weights else 0.5
        arousal = (params.tempo_suggested - 60) / 100.0
        return EmotionalState(
            valence=valence,
            arousal=min(1.0, max(0.0, arousal)),
            primary_emotion=emotion_name
        )
    
    # Fallback
    return EMOTIONAL_STATE_PRESETS.get(fallback)
```

## References

- **Source Code**: [`data/emotional_mapping.py`](../../data/emotional_mapping.py)
- **Model Cards**: [`docs/model_cards/emotionrecognizer.md`](../model_cards/emotionrecognizer.md)
- **Training Stub**: [`ML Kelly Training/train_mps_stub.py`](../../ML%20Kelly%20Training/train_mps_stub.py)
- **Dataset Base**: [`python/penta_core/ml/datasets/base.py`](../../python/penta_core/ml/datasets/base.py)

## Example Workflow

Complete end-to-end example:

```python
#!/usr/bin/env python3
"""
Complete workflow: Audio labeling → Training → Inference
"""

from pathlib import Path
import json
import torch
from data.emotional_mapping import EmotionalState, get_parameters_for_state

# Step 1: Create labeled dataset
def create_dataset():
    labels = [
        ("audio/grief_1.wav", EmotionalState(-0.8, 0.3, "grief")),
        ("audio/joy_1.wav", EmotionalState(0.8, 0.7, "joy")),
        ("audio/calm_1.wav", EmotionalState(0.5, 0.2, "calm"))
    ]
    
    with open("train_manifest.jsonl", "w") as f:
        for audio, emotion in labels:
            entry = {
                "audio": audio,
                "label": emotion.primary_emotion,
                "valence": emotion.valence,
                "arousal": emotion.arousal
            }
            f.write(json.dumps(entry) + "\n")

# Step 2: Train model (simplified)
def train_model():
    model = EmotionMultiTaskModel(num_emotions=3)
    # ... training code ...
    torch.save(model.state_dict(), "emotion_model.pt")

# Step 3: Inference
def predict_emotion(audio_path):
    model = EmotionMultiTaskModel(num_emotions=3)
    model.load_state_dict(torch.load("emotion_model.pt"))
    model.eval()
    
    # Process audio
    audio = load_audio(audio_path)
    
    with torch.no_grad():
        outputs = model(audio)
        emotion_idx = outputs["emotion_logits"].argmax()
        valence = outputs["valence"].item()
        arousal = outputs["arousal"].item()
    
    emotions = ["grief", "joy", "calm"]
    return EmotionalState(
        valence=valence,
        arousal=arousal,
        primary_emotion=emotions[emotion_idx]
    )

if __name__ == "__main__":
    create_dataset()
    train_model()
    emotion = predict_emotion("test_audio.wav")
    print(f"Predicted: {emotion.primary_emotion} "
          f"(valence={emotion.valence:.2f}, arousal={emotion.arousal:.2f})")
```

## Troubleshooting

### Common Issues

1. **Invalid Range Errors**
   ```python
   # ❌ Wrong
   emotion = EmotionalState(valence=2.0, arousal=0.5, primary_emotion="joy")
   # AssertionError: Valence must be in [-1, 1]
   
   # ✅ Correct
   emotion = EmotionalState(valence=1.0, arousal=0.5, primary_emotion="joy")
   ```

2. **Missing Secondary Emotions**
   ```python
   # ❌ Wrong (causes error if not careful)
   emotion = EmotionalState(valence=0.5, arousal=0.5, primary_emotion="joy",
                           secondary_emotions="happiness")  # String, not list
   
   # ✅ Correct
   emotion = EmotionalState(valence=0.5, arousal=0.5, primary_emotion="joy",
                           secondary_emotions=["happiness", "excitement"])
   ```

3. **Inconsistent Emotion Names**
   ```python
   # Use consistent naming
   VALID_EMOTIONS = ["grief", "joy", "anxiety", "anger", "calm", "nostalgia"]
   
   def validate_emotion(emotion_name):
       if emotion_name not in VALID_EMOTIONS:
           raise ValueError(f"Unknown emotion: {emotion_name}")
   ```

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Production Ready
