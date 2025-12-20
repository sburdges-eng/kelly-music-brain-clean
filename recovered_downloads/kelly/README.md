# Kelly - Therapeutic iDAW

> "Interrogate Before Generate" — The tool shouldn't finish art for people; it should make them braver.

Kelly is a therapeutic interactive Digital Audio Workstation (iDAW) that translates emotional intent into authentic musical compositions. Named in memory of a friend lost to suicide, Kelly bridges the gap between emotional expression and musical creation.

## Philosophy

Kelly operates on a three-phase intent system:

1. **Wound** → Identify the emotional trigger
2. **Emotion** → Map to the 216-node emotion thesaurus
3. **Rule-breaks** → Express through intentional musical violations

Unlike AI music generators that produce finished compositions, Kelly helps users become braver in their artistic expression by understanding and translating their emotional intent into musical parameters.

## Features

- **216-Node Emotion Thesaurus**: Comprehensive mapping of emotional states to musical parameters
- **Intent Processing Pipeline**: Three-phase emotional interrogation (Wound → Emotion → Rule-breaks)
- **Emotion-Driven Generation**: Bass, melody, rhythm, pads, strings, dynamics all respond to emotional intent
- **Intentional Rule-Breaking**: Music theory violations that serve emotional authenticity
- **Humanization Engine**: Groove, timing feel, and velocity curves for authentic expression
- **Full Arrangement Support**: Complete song structure generation

## Installation

```bash
# Basic installation
pip install kelly

# With CLI support
pip install kelly[cli]

# Full installation with all dependencies
pip install kelly[full]

# Development installation
pip install kelly[dev]
```

## Quick Start

### As a Library

```python
from kelly import EmotionThesaurus, IntentProcessor, Wound, MidiGenerator

# Process emotional intent
processor = IntentProcessor()
wound = Wound("the loss of my best friend", intensity=0.9)
result = processor.process_intent(wound)

print(f"Mapped to: {result.emotion.name}")
print(f"Mode: {result.emotion.musical_mapping.mode}")
print(f"Rule breaks: {[rb.rule_type for rb in result.rule_breaks]}")

# Generate MIDI
generator = MidiGenerator(tempo=82, key="F", mode="minor")
progression = generator.generate_chord_progression(bars=4)
generator.create_full_arrangement(bars=8, output_path="grief_expression.mid")
```

### Using Engines

```python
from kelly.engines import (
    BassEngine, BassConfig,
    MelodyEngine, MelodyConfig,
    RhythmEngine, RhythmConfig,
    GrooveEngine,
)

# Generate emotion-driven bass line
bass = BassEngine()
config = BassConfig(
    emotion="grief",
    chord_progression=["F", "C", "Dm", "Bbm"],
    key="F",
    bars=4,
    tempo_bpm=82
)
bass_output = bass.generate(config)

# Generate melody
melody = MelodyEngine()
melody_config = MelodyConfig(emotion="grief", key="F", mode="minor", bars=4)
melody_output = melody.generate(melody_config)

# Apply humanization
groove = GrooveEngine()
humanized = groove.apply_groove(notes, emotion="grief")
```

### CLI Usage

```bash
# List available emotions
kelly list-emotions

# Analyze emotional wound
kelly analyze "feeling overwhelmed by loss"

# Process wound and generate MIDI
kelly process "the loss of my best friend" --intensity 0.9 --output grief.mid

# Quick generate from emotion
kelly generate grief --output grief_output.mid --bars 8 --tempo 82
```

## Architecture

```
Kelly/
├── src/kelly/
│   ├── core/                    # Core processing
│   │   ├── emotion_thesaurus.py # 216-node emotion mapping
│   │   ├── intent_processor.py  # Three-phase pipeline
│   │   ├── intent_schema.py     # Rule-breaking definitions
│   │   ├── emotional_mapping.py # Valence/arousal mapping
│   │   └── midi_generator.py    # MIDI output
│   ├── engines/                 # Generation engines
│   │   ├── groove_engine.py     # Humanization
│   │   ├── bass_engine.py       # Bass lines
│   │   ├── melody_engine.py     # Melodic lines
│   │   ├── rhythm_engine.py     # Drum patterns
│   │   ├── pad_engine.py        # Atmospheric pads
│   │   ├── string_engine.py     # Orchestral strings
│   │   ├── dynamics_engine.py   # Dynamic curves
│   │   ├── arrangement_engine.py# Song structure
│   │   └── ...                  # More engines
│   └── cli.py                   # Command-line interface
└── tests/                       # Test suite
```

## Emotion Categories

Kelly's thesaurus covers six primary emotion categories (Plutchik's wheel):

| Category | Example Emotions |
|----------|-----------------|
| Joy | serenity, happiness, ecstasy |
| Sadness | pensiveness, grief, despair |
| Anger | annoyance, rage, hatred |
| Fear | apprehension, terror, paranoia |
| Surprise | interest, amazement, shock |
| Disgust | discomfort, loathing, detestation |

Each emotion maps to specific musical parameters including mode, tempo, dynamics, articulation, and rule-breaking directives.

## Rule-Breaking System

Kelly embraces intentional violations of musical conventions to express authentic emotion:

- **Harmony**: Unresolved dissonance, modal interchange, avoiding tonic
- **Rhythm**: Tempo fluctuation, dropped beats, metric modulation
- **Arrangement**: Buried vocals, extreme dynamics, structural mismatch
- **Production**: Lo-fi degradation, room noise, silence as instrument
- **Melody**: Avoided resolution, angular intervals, fragmented phrases

## Development

```bash
# Clone repository
git clone https://github.com/kelly-project/kelly
cd kelly

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/kelly tests
ruff check src/kelly tests
```

## License

MIT License - See LICENSE file for details.

## In Memory

Kelly is dedicated to the memory of Kelly, a bright light taken too soon. This project exists to help others process difficult emotions through the healing power of music creation.

---

*"Every rule-break needs justification. The imperfection IS the authenticity."*
