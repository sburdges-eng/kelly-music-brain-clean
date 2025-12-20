# Quick Start Guide: Audio Emotion Detection Integration

This guide will help you quickly integrate state-of-the-art emotion detection into your miDiKompanion system.

## Recommended Starting Point: SpeechBrain

**Why SpeechBrain?**

- ✅ Pre-trained models ready to use
- ✅ Easy integration (1-2 hours)
- ✅ Good accuracy (70-75%)
- ✅ Real-time capable
- ✅ Apache 2.0 license (commercial-friendly)

---

## Step 1: Installation

### Option A: Install from pyproject.toml (Recommended)

```bash
# Install project with research dependencies (includes SpeechBrain, OpenSMILE, emotion2vec)
pip install -e ".[research]"
```

This installs all research libraries:
- SpeechBrain (with torch, torchaudio)
- OpenSMILE
- emotion2vec (funasr, modelscope)
- librosa, numpy, soundfile

### Option B: Install SpeechBrain only

```bash
# Install SpeechBrain
pip install speechbrain

# Also install torch and torchaudio if not already installed
pip install torch torchaudio
```

---

## Step 2: Quick Test

Create a test script `test_emotion.py`:

```python
from research.examples.speechbrain_integration import SpeechBrainEmotionDetector

# Initialize detector
detector = SpeechBrainEmotionDetector()

# Test on an audio file
result = detector.detect_emotion("path/to/your/audio.wav")

print(f"Detected emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Valence: {result['valence']:.2f}")
print(f"Arousal: {result['arousal']:.2f}")
```

Run it:

```bash
python test_emotion.py
```

---

## Step 3: Integrate with Existing System

### Option A: Replace librosa emotion mapping

Update `music_brain/audio/feel.py` or create a new module:

```python
from research.examples.speechbrain_integration import SpeechBrainEmotionDetector
from data.emotional_mapping import EmotionalState, get_parameters_for_state

# Initialize detector (can be a singleton)
detector = SpeechBrainEmotionDetector()

def get_emotion_from_audio(audio_path: str) -> EmotionalState:
    """Extract emotion from audio using SpeechBrain."""
    result = detector.detect_emotion(audio_path)
    
    return EmotionalState(
        valence=result["valence"],
        arousal=result["arousal"],
        primary_emotion=result["primary_emotion"],
        secondary_emotions=[],
        has_intrusions=False,
        intrusion_probability=0.0,
    )

# Use it
state = get_emotion_from_audio("path/to/audio.wav")
params = get_parameters_for_state(state)
print(f"Suggested tempo: {params.tempo_suggested} BPM")
```

### Option B: Add as enhancement layer

Keep your existing librosa features, but add SpeechBrain as a validation/enhancement:

```python
from research.examples.speechbrain_integration import SpeechBrainEmotionDetector
import librosa

def analyze_audio_enhanced(audio_path: str):
    """Combine librosa and SpeechBrain analysis."""
    # Your existing librosa analysis
    y, sr = librosa.load(audio_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # ... other librosa features ...
    
    # Add SpeechBrain emotion detection
    detector = SpeechBrainEmotionDetector()
    emotion_result = detector.detect_emotion(audio_path)
    
    return {
        "tempo": tempo,
        # ... other librosa features ...
        "emotion": emotion_result["emotion"],
        "valence": emotion_result["valence"],
        "arousal": emotion_result["arousal"],
    }
```

---

## Step 4: Real-time Processing

For real-time audio processing:

```python
import librosa
from research.examples.speechbrain_integration import SpeechBrainEmotionDetector

detector = SpeechBrainEmotionDetector()

def process_audio_chunk(audio_chunk, sample_rate=44100):
    """Process audio chunk in real-time."""
    # Resample to 16kHz if needed (SpeechBrain requirement)
    if sample_rate != 16000:
        audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=16000)
    
    # Detect emotion
    result = detector.detect_emotion_from_array(audio_chunk, sample_rate=16000)
    return result
```

---

## Step 5: Compare Libraries (Optional)

Test multiple libraries on the same audio file:

```bash
python research/examples/compare_libraries.py path/to/audio.wav
```

This will show you:

- Which libraries are available
- Processing times
- Detected emotions
- Confidence scores

---

## Next Steps

### If SpeechBrain works well

1. ✅ You're done! Use it in production
2. Consider adding OpenSMILE features for even better accuracy (see Phase 2 in research doc)

### If you need better accuracy

1. Add OpenSMILE features (see `research/examples/opensmile_integration.py`)
2. Combine SpeechBrain + OpenSMILE + librosa features
3. Train a custom classifier on combined features

### If you need music-specific emotions

1. Use emotion2vec for embeddings (see `research/examples/emotion2vec_integration.py`)
2. Fine-tune on music emotion datasets (PMEmo, AMG1608)
3. Train custom music emotion model

---

## Troubleshooting

### SpeechBrain model not loading

- Check internet connection (first run downloads model)
- Try: `pip install --upgrade speechbrain`
- Check HuggingFace access: <https://huggingface.co/speechbrain>

### Audio format issues

- SpeechBrain works with most formats (wav, mp3, etc.)
- For best results, use 16kHz mono WAV
- Use librosa to convert: `librosa.load(path, sr=16000, mono=True)`

### Performance issues

- Use GPU if available: `torch.cuda.is_available()`
- For batch processing, use `batch_detect()` method
- Consider caching model instance (don't reload for each file)

---

## Example: Complete Integration

```python
"""
Complete example: Audio -> Emotion -> Musical Parameters
"""

from research.examples.speechbrain_integration import SpeechBrainEmotionDetector
from data.emotional_mapping import EmotionalState, get_parameters_for_state

class AudioEmotionPipeline:
    def __init__(self):
        self.detector = SpeechBrainEmotionDetector()
    
    def process(self, audio_path: str):
        """Complete pipeline: audio -> emotion -> musical parameters."""
        # 1. Detect emotion
        emotion_result = self.detector.detect_emotion(audio_path)
        
        # 2. Create emotional state
        state = EmotionalState(
            valence=emotion_result["valence"],
            arousal=emotion_result["arousal"],
            primary_emotion=emotion_result["primary_emotion"],
            secondary_emotions=[],
            has_intrusions=False,
            intrusion_probability=0.0,
        )
        
        # 3. Get musical parameters
        params = get_parameters_for_state(state)
        
        return {
            "emotion": emotion_result["emotion"],
            "confidence": emotion_result["confidence"],
            "valence": emotion_result["valence"],
            "arousal": emotion_result["arousal"],
            "tempo_suggested": params.tempo_suggested,
            "mode": max(params.mode_weights, key=params.mode_weights.get),
            "dissonance": params.dissonance,
            "timing_feel": params.timing_feel.value,
        }

# Use it
pipeline = AudioEmotionPipeline()
result = pipeline.process("path/to/audio.wav")
print(result)
```

---

## Performance Expectations

- **Accuracy:** 70-75% on speech emotion recognition
- **Speed:** 10-50ms per 3-second clip (GPU), 50-200ms (CPU)
- **Model Size:** ~300MB (downloads automatically)
- **Memory:** ~500MB RAM during inference

---

## Support

- **SpeechBrain Docs:** <https://speechbrain.readthedocs.io/>
- **Research Report:** `research/audio-emotion-detection-research.md`
- **Code Examples:** `research/examples/`

---

## License

All recommended libraries use Apache 2.0 or similar open-source licenses, making them suitable for commercial use.
