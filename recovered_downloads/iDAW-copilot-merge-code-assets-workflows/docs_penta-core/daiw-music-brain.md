# DAiW Music Brain Project Architecture

A comprehensive project structure for the DAiW (Digital Audio Intelligence Workstation) Music Brain system - an AI-powered music composition and production assistant.

---

## Project Overview

The DAiW Music Brain is a modular AI system designed to:
- Analyze and apply musical grooves and feel
- Understand and generate emotional musical content
- Integrate with major DAWs (Logic Pro, Ableton, Pro Tools, Reaper)
- Teach and apply rule-breaking compositional techniques
- Process user intent and generate musical arrangements

---

## 🧠 Core Music Brain (1-15)

The heart of the system - groove analysis, structure understanding, and session processing.

### Groove Analysis

```
groove/
├── extractor.py      # Extract groove patterns from audio/MIDI
├── applicator.py     # Apply groove templates to new content
└── templates.py      # Pre-defined groove templates (swing, shuffle, etc.)
```

| Module | Purpose |
|--------|---------|
| `extractor.py` | Analyze timing, velocity, and articulation patterns from source material |
| `applicator.py` | Transfer extracted grooves to target MIDI/audio with configurable strength |
| `templates.py` | Library of genre-specific groove templates (jazz swing, hip-hop bounce, etc.) |

### Structure Analysis

```
structure/
├── chord.py          # Chord detection and representation
├── progression.py    # Chord progression analysis and generation
└── sections.py       # Song section detection (verse, chorus, bridge)
```

| Module | Purpose |
|--------|---------|
| `chord.py` | Chord voicing analysis, inversion detection, extension identification |
| `progression.py` | Harmonic rhythm analysis, common progression patterns, reharmonization |
| `sections.py` | Structural segmentation, form detection, arrangement suggestions |

### Session Processing

```
session/
├── intent_schema.py      # Define user intent data structures
├── intent_processor.py   # Parse and validate user musical intentions
├── interrogator.py       # Ask clarifying questions about user goals
├── teaching.py           # Educational content delivery system
└── generator.py          # Generate arrangements from processed intent
```

| Module | Purpose |
|--------|---------|
| `intent_schema.py` | Pydantic models for song intent, emotional targets, style parameters |
| `intent_processor.py` | Convert natural language into structured musical parameters |
| `interrogator.py` | Socratic questioning to refine vague musical ideas |
| `teaching.py` | Deliver context-appropriate music theory and production tips |
| `generator.py` | Orchestrate generation pipeline from intent to final arrangement |

### Audio & Utilities

```
audio/
└── feel.py           # Audio-level feel analysis (dynamics, timing, space)

utils/
├── midi_io.py        # MIDI file reading/writing utilities
├── instruments.py    # Instrument definitions and mappings
└── ppq.py            # Pulses-per-quarter-note timing utilities
```

| Module | Purpose |
|--------|---------|
| `feel.py` | Analyze "human feel" in audio: micro-timing, velocity curves, dynamics |
| `midi_io.py` | Read/write Standard MIDI Files, convert to/from internal format |
| `instruments.py` | GM/custom instrument mappings, range definitions, articulation support |
| `ppq.py` | High-resolution timing math, tempo conversions, quantization utilities |

---

## 🧠 Emotional & Rule-Breaking Engine (16-25)

The creative intelligence layer - emotional mapping and unconventional compositional techniques.

### Models

```
models/
├── emotional_mapping.py    # Map emotions to musical parameters
└── musical_parameters.py   # Define controllable musical dimensions
```

| Module | Purpose |
|--------|---------|
| `emotional_mapping.py` | Valence/arousal to key, tempo, rhythm, harmony mappings |
| `musical_parameters.py` | Tension, energy, complexity, density parameter definitions |

### Data Resources

```
data/
├── emotional_presets.json        # Pre-defined emotional configurations
├── rule_breaking_database.json   # Catalog of effective rule violations
└── interval_emotions.json        # Emotional associations per interval
```

| Data File | Contents |
|-----------|----------|
| `emotional_presets.json` | "Melancholic" → minor key, slow tempo, sparse arrangement |
| `rule_breaking_database.json` | Tritone substitutions, parallel fifths that work, deceptive resolutions |
| `interval_emotions.json` | Minor 2nd → tension; Perfect 5th → stability; etc. |

### Core Modules

```
RuleBreakingTeacher     # Teach unconventional techniques with examples
SongInterrogator        # Deep-dive questioning about song meaning
EmotionalArrangement    # Auto-arrange based on emotional arc
GrooveTrainer           # Learn and apply custom grooves
MisdirectionModifier    # Create unexpected harmonic movements
```

| Module | Purpose |
|--------|---------|
| `RuleBreakingTeacher` | Present "forbidden" techniques with successful examples from music history |
| `SongInterrogator` | Ask "what story does this song tell?" and derive musical parameters |
| `EmotionalArrangement` | Generate arrangements that follow emotional journey curves |
| `GrooveTrainer` | Learn grooves from user examples, create custom templates |
| `MisdirectionModifier` | Apply chord substitutions, unexpected modulations, deceptive cadences |

---

## 🧩 DAW Integration Layer (26-35)

Bridges between the Music Brain and professional DAW software.

### DAW Adapters

```
daw/
├── logic.py          # Logic Pro X integration
├── ableton.py        # Ableton Live integration
├── protools.py       # Pro Tools integration
├── reaper.py         # Reaper integration
└── common_io.py      # Shared DAW I/O abstractions
```

| Module | Purpose |
|--------|---------|
| `logic.py` | Logic Pro X scripting, AppleScript integration, project parsing |
| `ableton.py` | Ableton Live Set parsing, Max for Live communication, clip launching |
| `protools.py` | Pro Tools session reading, AAF/OMF export, automation |
| `reaper.py` | Reaper API integration, ReaScript support, project manipulation |
| `common_io.py` | Abstract DAW interface, common track/clip/automation models |

### Session Management

```
daw/
├── session_exporter.py   # Export sessions to various DAW formats
├── automation_map.py     # Automation curve translation
├── tempo_sync.py         # Tempo and time signature sync
├── render_queue.py       # Batch rendering management
└── plugin_bridge.py      # Plugin parameter control bridge
```

| Module | Purpose |
|--------|---------|
| `session_exporter.py` | Convert internal session to Logic/Ableton/PT/Reaper format |
| `automation_map.py` | Translate automation between DAW-specific formats |
| `tempo_sync.py` | Sync tempo maps, time signatures, click tracks |
| `render_queue.py` | Queue stems, masters, previews for background rendering |
| `plugin_bridge.py` | Control VST/AU parameters, preset management, state recall |

---

## 🎹 User Interface / CLI / Workflow (36-45)

Multiple interface options for different user preferences.

### Application Entry Points

```
cli.py                    # Command-line interface
app.py                    # Streamlit web UI
launcher.py               # PyWebView desktop wrapper
daiw.spec                 # PyInstaller build configuration
```

| File | Purpose |
|------|---------|
| `cli.py` | Full CLI with subcommands: `daiw analyze`, `daiw generate`, `daiw teach` |
| `app.py` | Streamlit-based web UI for visual interaction and real-time preview |
| `launcher.py` | Desktop app wrapper using PyWebView for native feel |
| `daiw.spec` | PyInstaller spec for building standalone executables |

### Workflow Templates

```
templates/
├── workflow_logic.json         # Logic Pro workflow presets
├── workflow_ableton.json       # Ableton Live workflow presets
├── workflow_protools.json      # Pro Tools workflow presets
├── song_intent_template.json   # Template for capturing song intent
└── emotional_prompt_template.json  # Template for emotional descriptions
```

| Template | Purpose |
|----------|---------|
| `workflow_logic.json` | Logic-specific track templates, routing presets, key commands |
| `workflow_ableton.json` | Ableton clip arrangements, chain presets, device racks |
| `workflow_protools.json` | PT session templates, routing configurations, I/O setups |
| `song_intent_template.json` | Structured questionnaire for song vision capture |
| `emotional_prompt_template.json` | Guided prompts for emotional content definition |

### UI Configuration

```
ui/
└── style_theme.json      # Visual theme configuration
```

---

## 🧪 Testing and Quality Modules (46-55)

Comprehensive test suite ensuring reliability and correctness.

### Unit Tests

```
tests/
├── test_basic.py              # Basic functionality tests
├── test_emotional_mapping.py  # Emotional engine tests
├── test_groove_applicator.py  # Groove system tests
├── test_midi_io.py            # MIDI I/O tests
├── test_reharmonization.py    # Chord/progression tests
└── test_session_intent.py     # Intent processing tests
```

| Test File | Coverage |
|-----------|----------|
| `test_basic.py` | Smoke tests, import validation, configuration loading |
| `test_emotional_mapping.py` | Emotion → parameter mappings, preset loading |
| `test_groove_applicator.py` | Groove extraction accuracy, application fidelity |
| `test_midi_io.py` | MIDI read/write round-trip, edge cases |
| `test_reharmonization.py` | Chord detection, progression analysis, substitutions |
| `test_session_intent.py` | Intent parsing, validation, generation pipeline |

### Integration & System Tests

```
tests/
├── test_rule_breaking.py      # Rule-breaking engine tests
├── test_audio_features.py     # Audio analysis tests
├── test_integration_end2end.py # Full pipeline tests
└── test_cli_commands.py       # CLI interface tests
```

| Test File | Coverage |
|-----------|----------|
| `test_rule_breaking.py` | Rule database loading, technique application |
| `test_audio_features.py` | Feel extraction, dynamics analysis |
| `test_integration_end2end.py` | Intent → DAW session full pipeline |
| `test_cli_commands.py` | All CLI subcommands with various flags |

---

## 📂 Vault + Knowledge System (56-65)

Obsidian-compatible knowledge base for reference and learning.

### Songwriting Guides

```
vault/Songwriting_Guides/
├── rule_breaking_practical.md    # Practical rule-breaking techniques
├── rule_breaking_masterpieces.md # Analysis of famous rule-breakers
└── ...
```

| Document | Content |
|----------|---------|
| `rule_breaking_practical.md` | Step-by-step techniques for breaking conventions effectively |
| `rule_breaking_masterpieces.md` | Analysis of Beatles, Radiohead, Debussy rule violations |

### Theory Reference

```
vault/Theory_Reference/
├── chord_extensions.md       # Extended chord voicings
├── modal_interchange.md      # Borrowing from parallel modes
├── voice_leading.md          # Smooth voice motion techniques
├── counterpoint_basics.md    # Contrapuntal writing fundamentals
└── orchestration_tips.md     # Instrument combination guidelines
```

### Production Reference

```
vault/Production_Reference/
├── mixing_fundamentals.md    # EQ, compression, panning basics
├── arrangement_density.md    # Managing sonic space over time
├── genre_conventions.md      # Genre-specific production norms
└── reference_tracks.md       # How to use reference material
```

### Emotional Mapping

```
vault/Emotional_Mapping/
├── valence_arousal_model.md  # Psychological basis
├── key_emotions.md           # Key signature emotional associations
├── tempo_emotions.md         # BPM and energy relationships
└── timbre_emotions.md        # Instrument tone and feeling
```

---

## 🔧 Configuration & Dependencies

### Project Configuration

```
pyproject.toml          # Modern Python project configuration
requirements.txt        # Pip dependencies
requirements-dev.txt    # Development dependencies
.env.example           # Environment variable template
config/
├── default.yaml       # Default configuration
├── production.yaml    # Production overrides
└── development.yaml   # Development overrides
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `mido` | MIDI file handling |
| `librosa` | Audio analysis |
| `music21` | Music theory computations |
| `pretty_midi` | MIDI manipulation |
| `streamlit` | Web UI framework |
| `typer` | CLI framework |
| `pydantic` | Data validation |
| `pytest` | Testing framework |

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone <repository-url>
cd daiw-music-brain

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### Running the Application

```bash
# CLI usage
python cli.py analyze input.mid --output analysis.json
python cli.py generate --intent intent.json --output session/
python cli.py teach "What is a tritone substitution?"

# Web UI
streamlit run app.py

# Desktop app
python launcher.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_groove_applicator.py

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │   CLI    │  │  Streamlit   │  │  PyWebView Desktop App    │ │
│  └────┬─────┘  └──────┬───────┘  └─────────────┬─────────────┘ │
└───────┼───────────────┼────────────────────────┼────────────────┘
        │               │                        │
        └───────────────┴────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │      Session Processing       │
        │  ┌──────────────────────────┐ │
        │  │   Intent Processor       │ │
        │  │   Interrogator           │ │
        │  │   Generator              │ │
        │  └──────────────────────────┘ │
        └───────────────┬───────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
┌─────────┐      ┌─────────────┐      ┌─────────────┐
│ Groove  │      │  Structure  │      │  Emotional  │
│ Engine  │      │  Analysis   │      │  Mapping    │
└────┬────┘      └──────┬──────┘      └──────┬──────┘
     │                  │                    │
     └──────────────────┼────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │       DAW Integration         │
        │  ┌─────┐ ┌─────┐ ┌─────────┐ │
        │  │Logic│ │Able-│ │Pro Tools│ │
        │  │ Pro │ │ ton │ │ /Reaper │ │
        │  └─────┘ └─────┘ └─────────┘ │
        └───────────────────────────────┘
```

---

## 🔮 Future Roadmap

1. **Real-time collaboration** - Multi-user session editing
2. **Cloud sync** - Cross-device project synchronization
3. **Plugin ecosystem** - Third-party module support
4. **AI model improvements** - Fine-tuned models for specific genres
5. **Hardware integration** - MIDI controller mappings
6. **Mobile companion** - iOS/Android idea capture app

---

## 📚 Resources

- [Mido Documentation](https://mido.readthedocs.io/)
- [Librosa Documentation](https://librosa.org/doc/)
- [Music21 Documentation](https://web.mit.edu/music21/doc/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Typer Documentation](https://typer.tiangolo.com/)
