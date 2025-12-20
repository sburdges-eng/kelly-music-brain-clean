# Research Directory

This directory contains research findings, implementation examples, and infrastructure for using Cursor K2 as a research assistant to improve the miDiKompanion codebase.

## 🎯 Current Research Status

### ✅ Completed Research

- **Audio Emotion Detection** - Comprehensive research complete with integration examples

### 🔄 Research Infrastructure

- Templates and prompts for future research domains
- Automation tools for generating research prompts
- Index system for tracking findings

---

## Audio Emotion Detection Research

**Status:** ✅ Research Complete - Ready for Integration

This section contains research and implementation examples for integrating state-of-the-art audio emotion detection into the miDiKompanion music production AI system.

## 📚 Documents

### Main Research Report

- **[audio-emotion-detection-research.md](./audio-emotion-detection-research.md)** - Comprehensive research report with:
  - Top 5 Python libraries for audio emotion detection
  - Detailed comparisons (accuracy, speed, features)
  - Integration plan with 4 phases
  - Code examples and usage instructions

### Quick Start Guide

- **[QUICK_START.md](./QUICK_START.md)** - Get started in 5 minutes:
  - Step-by-step installation
  - Quick test script
  - Integration examples
  - Troubleshooting tips

## 💻 Code Examples

All examples are in the `examples/` directory:

### Integration Examples

- **[speechbrain_integration.py](./examples/speechbrain_integration.py)**
  - SpeechBrain emotion detector class
  - Maps to valence-arousal model
  - Real-time processing support
  - Batch processing

- **[opensmile_integration.py](./examples/opensmile_integration.py)**
  - OpenSMILE feature extraction
  - eGeMAPS (88 features) and ComParE (6,373 features)
  - Combined with librosa features
  - Industry-standard acoustic features

- **[emotion2vec_integration.py](./examples/emotion2vec_integration.py)**
  - emotion2vec embeddings extraction
  - Direct emotion recognition (9-class)
  - Self-supervised learning models
  - Multilingual support

### Utility Scripts

- **[compare_libraries.py](./examples/compare_libraries.py)**
  - Compare all libraries on same audio file
  - Performance benchmarking
  - Results export to JSON

## 🎯 Quick Recommendations

### For Immediate Use

**Start with SpeechBrain** - Pre-trained models, easy integration, good accuracy (70-75%)

```bash
pip install speechbrain
python research/examples/speechbrain_integration.py
```

### For Best Accuracy

**Combine SpeechBrain + OpenSMILE** - Pre-trained models + industry-standard features

### For Music-Specific

**emotion2vec + custom training** - Fine-tune on music emotion datasets

## 📊 Library Comparison Summary

| Library | Accuracy | Speed | Pre-trained | Real-time | Best For |
|---------|----------|-------|-------------|-----------|----------|
| SpeechBrain | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | Production use |
| emotion2vec | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ | Embeddings |
| OpenSMILE | ⭐⭐⭐* | ⭐⭐⭐⭐⭐ | ❌ | ✅ | Feature extraction |
| VANPY | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | ⚠️ | Voice analysis |
| emotion_detective | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | ⚠️ | Sentence-level |

*OpenSMILE accuracy depends on classifier used

## 🚀 Integration Phases

### Phase 1: Quick Win (Week 1) ✅ Recommended Start

- Integrate SpeechBrain
- Replace basic librosa emotion mapping
- Immediate accuracy improvement

### Phase 2: Enhanced Features (Week 2-3)

- Add OpenSMILE features
- Combine with librosa features
- Train classifier on combined features

### Phase 3: Advanced Embeddings (Month 2)

- Integrate emotion2vec
- Extract emotion embeddings
- Fine-tune for music-specific emotions

### Phase 4: Custom Model (Month 3+)

- Train on music emotion datasets
- Create ensemble model
- Optimize for your use case

## 📦 Installation

### Recommended: Install from pyproject.toml

```bash
# Install project with all research dependencies
pip install -e ".[research]"
```

This installs:
- SpeechBrain (with torch, torchaudio)
- OpenSMILE
- emotion2vec (funasr, modelscope)
- librosa, numpy, soundfile

### Alternative: Manual Installation

#### Minimal (SpeechBrain only)

```bash
pip install speechbrain torch torchaudio
```

#### Complete (All libraries)

```bash
pip install speechbrain opensmile funasr modelscope
pip install torch torchaudio librosa soundfile numpy
```

## 🔗 Resources

### Official Documentation

- [SpeechBrain](https://speechbrain.readthedocs.io/)
- [emotion2vec](https://github.com/ddlBoJack/emotion2vec)
- [OpenSMILE](https://audeering.github.io/opensmile-python/)

### Datasets

- **IEMOCAP** - Speech emotion (4 emotions)
- **CREMA-D** - 7,442 speech clips
- **PMEmo** - 794 music tracks with emotions
- **AMG1608** - 1,608 music clips
- **EmoMusic** - 1,000 songs from FMA

### Research Papers

- SpeechBrain: <https://arxiv.org/abs/2106.04624>
- emotion2vec: <https://arxiv.org/abs/2312.15185>
- OpenSMILE: Industry standard for acoustic features

## 📝 Usage Examples

### Basic Emotion Detection

```python
from research.examples.speechbrain_integration import SpeechBrainEmotionDetector

detector = SpeechBrainEmotionDetector()
result = detector.detect_emotion("audio.wav")
print(f"Emotion: {result['emotion']}, Confidence: {result['confidence']}")
```

### Integration with Existing System

```python
from research.examples.speechbrain_integration import create_emotional_state_from_speechbrain
from data.emotional_mapping import get_parameters_for_state

detector = SpeechBrainEmotionDetector()
state = create_emotional_state_from_speechbrain(detector, "audio.wav")
params = get_parameters_for_state(state)
print(f"Suggested tempo: {params.tempo_suggested} BPM")
```

### Compare Libraries

```bash
python research/examples/compare_libraries.py path/to/audio.wav
```

## 🎵 Current System Integration

Your current system uses:

- `data/emotional_mapping.py` - Maps emotions to musical parameters
- `src/core/emotion_engine.cpp` - C++ emotion engine
- `music_brain/audio/feel.py` - Librosa feature extraction

**Integration Points:**

1. Replace librosa emotion mapping with SpeechBrain
2. Add OpenSMILE features to enhance accuracy
3. Use emotion2vec embeddings for music-specific emotions
4. Train custom model on music datasets

## 📄 License Compatibility

All recommended libraries use Apache 2.0 or similar open-source licenses:

- ✅ SpeechBrain: Apache 2.0
- ✅ emotion2vec: Apache 2.0
- ✅ OpenSMILE: OpenSMILE License (commercial use allowed)
- ⚠️ VANPY: Check PyPI
- ⚠️ emotion_detective: Check PyPI

## 🤝 Contributing

When integrating these libraries:

1. Test on your audio files first
2. Compare results with existing system
3. Measure performance impact
4. Document any custom modifications

## 📧 Support

For questions or issues:

1. Check the [Quick Start Guide](./QUICK_START.md)
2. Review code examples in `examples/`
3. Consult the [main research report](./audio-emotion-detection-research.md)

---

## 🔬 Research Infrastructure

For conducting future research using Cursor K2:

## Quick Start

1. **Use Pre-generated Prompts** - Browse `prompts/` directory
2. **Generate Custom Prompts** - Run `python3 k2_research_automation.py`
3. **Use Template** - Copy `.cursor/research-template.md` and customize

## Available Research Domains

| Domain | Status | Prompt File |
|--------|--------|-------------|
| Audio Emotion Detection | ✅ Complete | `prompts/01-audio-emotion-detection.md` |
| JUCE Plugin Architecture | 🔄 Ready | `prompts/02-juce-plugin-architecture.md` |
| DAW Integration | 🔄 Ready | `prompts/03-daw-integration-patterns.md` |
| Audio Analysis | 🔄 Ready | `prompts/audio-analysis-20251220.md` |
| Music Theory | 🔄 Ready | `prompts/music-theory-20251220.md` |

## Research Workflow

1. Choose a prompt from `prompts/`
2. Open Cursor Composer (Cmd+I)
3. Paste prompt and let K2 research
4. Save findings to `findings/`
5. Extract patterns to `patterns/`
6. Update `research-index.md`

## Documentation

- **Quick Start Guide**: `QUICKSTART.md` or `QUICK_START.md`
- **Research Template**: `.cursor/research-template.md`
- **Automation Script**: `k2_research_automation.py`

---

**Last Updated:** December 2024  
**Audio Emotion Detection:** ✅ Complete  
**Research Infrastructure:** ✅ Ready
