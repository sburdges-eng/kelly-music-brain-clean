# Usage Guide: Emotion Detection Integration Examples

Quick guide for using the emotion detection integration examples.

## Installation

First, install the required dependencies:

```bash
# Install all research dependencies
pip install -e ".[research]"

# Or install individually:
pip install -U funasr modelscope  # for emotion2vec
pip install speechbrain            # for SpeechBrain
pip install opensmile              # for OpenSMILE
```

## Quick Usage Examples

### emotion2vec

**Command-line usage:**
```bash
# Basic emotion recognition
python research/examples/use_emotion2vec.py path/to/audio.wav

# Extract embeddings instead
python research/examples/use_emotion2vec.py path/to/audio.wav --embeddings --model base

# Convert audio and integrate with miDiKompanion
python research/examples/use_emotion2vec.py path/to/audio.mp3 --convert --integrate
```

**Python code:**
```python
from research.examples.emotion2vec_integration import (
    Emotion2VecExtractor,
    create_emotional_state_from_emotion2vec,
)

# Initialize extractor
extractor = Emotion2VecExtractor(model_type="plus")

# Recognize emotion
result = extractor.recognize_emotion("path/to/audio.wav")
print(f"Emotion: {result['emotion']}")
print(f"Valence: {result['valence']:.2f}, Arousal: {result['arousal']:.2f}")

# Integrate with existing system
state = create_emotional_state_from_emotion2vec(extractor, "path/to/audio.wav")
from data.emotional_mapping import get_parameters_for_state
params = get_parameters_for_state(state)
```

### SpeechBrain

**Python code:**
```python
from research.examples.speechbrain_integration import (
    SpeechBrainEmotionDetector,
    create_emotional_state_from_speechbrain,
)

# Initialize detector
detector = SpeechBrainEmotionDetector()

# Detect emotion
result = detector.detect_emotion("path/to/audio.wav")
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2f}")

# Real-time processing from array
import librosa
audio, sr = librosa.load("path/to/audio.wav", sr=16000)
result = detector.detect_emotion_from_array(audio, sr)

# Integrate with existing system
state = create_emotional_state_from_speechbrain(detector, "path/to/audio.wav")
```

### OpenSMILE

**Python code:**
```python
from research.examples.opensmile_integration import (
    OpenSMILEFeatureExtractor,
    EnhancedFeatureExtractor,
    create_emotional_state_from_opensmile,
)

# Extract features
extractor = OpenSMILEFeatureExtractor(feature_set="eGeMAPSv02")
features = extractor.extract_features("path/to/audio.wav")
print(f"Extracted {len(features)} features")

# Enhanced features (OpenSMILE + librosa)
enhanced = EnhancedFeatureExtractor(
    opensmile_set="eGeMAPSv02",
    include_librosa=True
)
result = enhanced.extract_all_features("path/to/audio.wav")

# Integrate with existing system
state = create_emotional_state_from_opensmile(extractor, "path/to/audio.wav")
```

## Comparing Libraries

Use the comparison script to test all libraries on the same audio file:

```bash
python research/examples/compare_libraries.py path/to/audio.wav
```

## Integration with miDiKompanion

All three integration examples provide helper functions that create `EmotionalState` objects compatible with your existing system:

```python
from data.emotional_mapping import EmotionalState, get_parameters_for_state

# Using emotion2vec
from research.examples.emotion2vec_integration import (
    create_emotional_state_from_emotion2vec,
)
extractor = Emotion2VecExtractor(model_type="plus")
state = create_emotional_state_from_emotion2vec(extractor, "audio.wav")

# Using SpeechBrain
from research.examples.speechbrain_integration import (
    create_emotional_state_from_speechbrain,
)
detector = SpeechBrainEmotionDetector()
state = create_emotional_state_from_speechbrain(detector, "audio.wav")

# Using OpenSMILE
from research.examples.opensmile_integration import (
    create_emotional_state_from_opensmile,
)
extractor = OpenSMILEFeatureExtractor()
state = create_emotional_state_from_opensmile(extractor, "audio.wav")

# Get musical parameters
params = get_parameters_for_state(state)
print(f"Tempo: {params.tempo_suggested} BPM")
print(f"Mode: {params.mode}")
```

## Requirements

- **emotion2vec**: Requires 16kHz mono WAV files
- **SpeechBrain**: Works with various formats (auto-converts)
- **OpenSMILE**: Works with various formats

## Troubleshooting

### emotion2vec
- **Error**: "emotion2vec not installed"
  - Solution: `pip install -U funasr modelscope`
- **Error**: Audio format issues
  - Solution: Use `--convert` flag or `convert_to_16khz_mono()` method

### SpeechBrain
- **Error**: Model loading fails
  - Solution: Check internet connection (downloads models on first use)
  - Solution: Ensure torch/torchaudio are installed

### OpenSMILE
- **Error**: Feature extraction fails
  - Solution: Ensure audio file is valid and readable
  - Solution: Check that opensmile is properly installed

## Next Steps

1. Test on your audio files
2. Compare results between libraries
3. Choose the best library for your use case
4. Integrate into your production code
5. Fine-tune models on your specific data (optional)
