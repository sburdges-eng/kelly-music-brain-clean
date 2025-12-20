# Repository Research Index

## Audio Analysis
### Top Repos
- [librosa](https://github.com/librosa/librosa) ⭐ 6.8k
  - **Use for:** Spectral analysis, beat tracking
  - **Pattern adopted:** Caching strategy for STFT
  - **Code ref:** `research/patterns/librosa-caching.py`
  
- [essentia](https://github.com/MTG/essentia) ⭐ 2.5k
  - **Use for:** Music emotion detection, genre classification
  - **Pattern adopted:** Feature extractor pipeline
  - **Code ref:** `research/patterns/essentia-pipeline.py`

### Patterns Extracted
1. **Lazy STFT Computation** (from librosa)
   - File: `research/patterns/lazy-stft.md`
   - Applied to: `music_brain/analyzer.py`
   
2. **Feature Vector Normalization** (from essentia)
   - File: `research/patterns/feature-norm.md`
   - Applied to: `music_brain/emotional_mapping.py`

## DAW Integration
### Top Repos
- [python-osc](https://github.com/attwad/python-osc) ⭐ 500+
  - **Use for:** OSC communication with DAWs
  - **Pattern adopted:** [To be filled]
  - **Code ref:** [To be filled]

### Patterns Extracted
[To be filled]

## JUCE/C++ Audio
### Top Repos
- [Surge Synthesizer](https://github.com/surge-synthesizer/surge) ⭐ 2.5k+
  - **Use for:** Professional VST3 plugin architecture
  - **Pattern adopted:** [To be filled]
  - **Code ref:** [To be filled]

### Patterns Extracted
[To be filled]

## Music Theory Engine
### Top Repos
[To be filled]

## Emotion → Music Mapping
### Top Repos
- [SpeechBrain](https://github.com/speechbrain/speechbrain) ⭐ 7.5k+
  - **Use for:** Pre-trained emotion detection models
  - **Pattern adopted:** Emotion detector class with valence-arousal mapping
  - **Code ref:** `research/examples/speechbrain_integration.py`
  - **Status:** ✅ Integrated - Ready for use

- [emotion2vec](https://github.com/ddlBoJack/emotion2vec) ⭐ 500+
  - **Use for:** Emotion embeddings and 9-class emotion recognition
  - **Pattern adopted:** Embedding extraction for music-specific emotions
  - **Code ref:** `research/examples/emotion2vec_integration.py`
  - **Status:** ✅ Integrated - Ready for use

- [OpenSMILE](https://github.com/audeering/opensmile-python) ⭐ 1.2k+
  - **Use for:** Industry-standard acoustic feature extraction
  - **Pattern adopted:** eGeMAPS (88) and ComParE (6,373) features
  - **Code ref:** `research/examples/opensmile_integration.py`
  - **Status:** ✅ Integrated - Ready for use

### Patterns Extracted
1. **SpeechBrain Emotion Detection** (from SpeechBrain)
   - File: `research/examples/speechbrain_integration.py`
   - Applied to: Replace librosa emotion mapping
   - Accuracy: 70-75% on speech emotion datasets

2. **OpenSMILE Feature Pipeline** (from OpenSMILE)
   - File: `research/examples/opensmile_integration.py`
   - Applied to: Enhance emotion detection with acoustic features
   - Features: 88 (eGeMAPS) or 6,373 (ComParE) features

3. **emotion2vec Embeddings** (from emotion2vec)
   - File: `research/examples/emotion2vec_integration.py`
   - Applied to: Music-specific emotion classification
   - Model: Self-supervised learning, multilingual support

### Research Completed
- **Date:** December 2024
- **Report:** `research/audio-emotion-detection-research.md`
- **Quick Start:** `research/QUICK_START.md`
- **Status:** ✅ Ready for Phase 1 integration (SpeechBrain)

## Python ↔ C++ Bridges
### Top Repos
[To be filled]

## Sample Library Management
### Top Repos
[To be filled]
