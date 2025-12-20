# Kelly Architecture

## Overview

Kelly is designed as a dual-language therapeutic iDAW with Python handling high-level logic and C++ managing real-time audio processing.

## Component Architecture

### Python Brain
- **Emotion Processing**: 216-node thesaurus with valence, arousal, and intensity dimensions
- **Intent Pipeline**: Three-phase processing (Wound → Emotion → Rule-breaks)
- **MIDI Generation**: Pattern-based MIDI creation with groove templates
- **CLI Interface**: User-friendly command-line tool with Typer

### C++ Body
- **Audio Engine**: JUCE-based real-time audio processing
- **GUI**: Qt 6 interface for visual interaction
- **Plugins**: VST3 and CLAP plugin formats
- **Core Library**: Shared emotion and MIDI processing logic

## Data Flow

```
User Input (Wound)
    ↓
Intent Processor (Python)
    ↓
Emotion Thesaurus Mapping
    ↓
Rule-Break Generation
    ↓
Musical Parameter Compilation
    ↓
MIDI Generator
    ↓
Audio Output (C++ / Plugins)
```

## Emotion Thesaurus

The 216-node emotion thesaurus is organized around three dimensions:

1. **Valence**: Negative (-1.0) to Positive (+1.0)
2. **Arousal**: Calm (0.0) to Excited (1.0)
3. **Intensity**: Subtle (0.0) to Extreme (1.0)

Each emotion maps to musical attributes:
- Tempo modifier
- Mode (major/minor)
- Dynamic range
- Harmonic complexity

## Rule-Breaking System

Emotions trigger intentional violations of musical conventions:

1. **Dynamics Rule-Breaks**: Extreme contrasts for high intensity
2. **Harmony Rule-Breaks**: Dissonance for negative valence
3. **Rhythm Rule-Breaks**: Irregular patterns for high arousal

## Plugin Architecture

Plugins implement the emotion processing engine in real-time:

- VST3: Industry-standard plugin format
- CLAP: Modern cross-platform format
- Audio I/O: Stereo input/output
- MIDI I/O: Bidirectional MIDI processing

## Testing Strategy

- **Python**: pytest for unit and integration tests
- **C++**: Catch2 for core logic testing
- **CI/CD**: GitHub Actions for multi-platform validation
