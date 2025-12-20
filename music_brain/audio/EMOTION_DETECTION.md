# Emotion Detection Module

State-of-the-art audio emotion detection using SpeechBrain pre-trained models.

## Quick Start

### Installation

```bash
pip install speechbrain
pip install torch torchaudio
```

### Basic Usage

```python
from music_brain.audio import detect_emotion_from_audio, get_emotional_state_from_audio
from data.emotional_mapping import get_parameters_for_state

# Quick detection
result = detect_emotion_from_audio("path/to/audio.wav")
print(f"Emotion: {result['emotion']}, Confidence: {result['confidence']:.2%}")

# Get EmotionalState (compatible with existing system)
state = get_emotional_state_from_audio("path/to/audio.wav")
params = get_parameters_for_state(state)
print(f"Suggested tempo: {params.tempo_suggested} BPM")
```

## Features

- ✅ Pre-trained models (70-75% accuracy)
- ✅ Real-time capable
- ✅ Seamless integration with existing `emotional_mapping.py`
- ✅ Maps to valence-arousal model
- ✅ Batch processing support
- ✅ Apache 2.0 license (commercial-friendly)

## API Reference

### Quick Functions

#### `detect_emotion_from_audio(audio_path) -> Dict`
Detect emotion from audio file. Returns dictionary with:
- `emotion`: Detected emotion label
- `confidence`: Confidence score (0-1)
- `valence`: Valence value (-1 to 1)
- `arousal`: Arousal value (0 to 1)
- `primary_emotion`: Mapped primary emotion

#### `get_emotional_state_from_audio(audio_path) -> EmotionalState`
Get `EmotionalState` instance compatible with existing system.

### Class-Based API

#### `EmotionDetector`

```python
from music_brain.audio import EmotionDetector

detector = EmotionDetector()

# Detect emotion
result = detector.detect_emotion("audio.wav")

# Real-time processing
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)
result = detector.detect_emotion_from_array(audio, sr)

# Get EmotionalState
state = detector.get_emotional_state("audio.wav")

# Batch processing
results = detector.batch_detect(["file1.wav", "file2.wav"])
```

## Integration Examples

### Example 1: Replace librosa emotion mapping

```python
from music_brain.audio import get_emotional_state_from_audio
from data.emotional_mapping import get_parameters_for_state

def analyze_audio_emotion(audio_path: str):
    """Analyze audio and get musical parameters."""
    # Detect emotion using SpeechBrain
    state = get_emotional_state_from_audio(audio_path)
    
    # Get musical parameters (existing function)
    params = get_parameters_for_state(state)
    
    return {
        "emotion": state.primary_emotion,
        "valence": state.valence,
        "arousal": state.arousal,
        "tempo": params.tempo_suggested,
        "mode": max(params.mode_weights, key=params.mode_weights.get),
    }
```

### Example 2: Combine with existing analysis

```python
from music_brain.audio import detect_emotion_from_audio, analyze_feel
from data.emotional_mapping import get_parameters_for_state, EmotionalState

def comprehensive_audio_analysis(audio_path: str):
    """Combine emotion detection with feel analysis."""
    # Existing feel analysis
    feel = analyze_feel(audio_path)
    
    # New emotion detection
    emotion_result = detect_emotion_from_audio(audio_path)
    
    # Create emotional state
    state = EmotionalState(
        valence=emotion_result["valence"],
        arousal=emotion_result["arousal"],
        primary_emotion=emotion_result["primary_emotion"],
    )
    
    # Get musical parameters
    from data.emotional_mapping import get_parameters_for_state
    params = get_parameters_for_state(state)
    
    return {
        "tempo": feel.tempo_bpm,
        "emotion": emotion_result["emotion"],
        "valence": emotion_result["valence"],
        "arousal": emotion_result["arousal"],
        "suggested_tempo": params.tempo_suggested,
        "mode": max(params.mode_weights, key=params.mode_weights.get),
    }
```

### Example 3: Real-time processing

```python
import librosa
from music_brain.audio import EmotionDetector

detector = EmotionDetector()

def process_audio_stream(audio_chunk, sample_rate=44100):
    """Process audio chunk in real-time."""
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=16000)
    
    # Detect emotion
    result = detector.detect_emotion_from_array(audio_chunk, sample_rate=16000)
    return result
```

## Testing

Test the integration:

```bash
python -m music_brain.audio.test_emotion_detection path/to/audio.wav
```

## Performance

- **Accuracy:** 70-75% on speech emotion recognition
- **Speed:** 10-50ms per 3-second clip (GPU), 50-200ms (CPU)
- **Model Size:** ~300MB (downloads automatically on first use)
- **Memory:** ~500MB RAM during inference

## Supported Emotions

The model detects 4 emotions:
- **happy**: Positive valence, high arousal
- **sad**: Negative valence, low arousal
- **angry**: Negative valence, high arousal
- **neutral**: Neutral valence, moderate arousal

These are automatically mapped to your system's emotion categories:
- `happy` → `calm`
- `sad` → `grief`
- `angry` → `anger`
- `neutral` → `calm`

## Troubleshooting

### Model not loading
- Check internet connection (first run downloads model)
- Try: `pip install --upgrade speechbrain`
- Check HuggingFace access: https://huggingface.co/speechbrain

### Audio format issues
- Works with most formats (wav, mp3, etc.)
- For best results, use 16kHz mono WAV
- Automatic resampling is handled internally

### Performance issues
- Use GPU if available: `torch.cuda.is_available()`
- For batch processing, reuse `EmotionDetector` instance
- Consider caching model (don't reload for each file)

## License

SpeechBrain uses Apache 2.0 license - fully compatible with commercial use.

## References

- [SpeechBrain Documentation](https://speechbrain.readthedocs.io/)
- [Research Report](../../research/audio-emotion-detection-research.md)
- [Quick Start Guide](../../research/QUICK_START.md)
