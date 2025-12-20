# Kelly MIDI Companion - Complete Workspace

**Version:** v3.0.00+ (Development)
**Philosophy:** *"Interrogate Before Generate"* — The tool shouldn't finish art for people; it should make them braver.

---

## Overview

Kelly MIDI Companion is a therapeutic music generation system that translates emotions into music through a 216-node emotion thesaurus and intentional rule-breaking system. This workspace contains the complete implementation including:

- **C++/JUCE Plugin** (`src/`) - Real-time audio plugin for DAWs
- **Python ML Framework** (`ml_framework/`) - CIF/LAS/QEF conscious AI systems
- **Python Reference** (`python/`, `reference/`) - Reference implementations
- **Data Resources** (`data/`) - Emotion mappings, chord progressions, music theory
- **Visualization Tools** (`CODE/PYTHON CODE/`) - 3D emotion visualization
- **Phase 2 Implementation** (`phase2/`) - Advanced features

---

## Project Structure

```
final kel/
├── src/                    # C++/JUCE plugin implementation
│   ├── core/              # EmotionThesaurus, WoundProcessor, IntentPipeline
│   ├── engines/           # 14 music generation engines
│   ├── midi/              # MIDI generation and export
│   ├── ui/                # Plugin UI components
│   ├── plugin/            # JUCE AudioProcessor/Editor
│   ├── voice/             # Voice synthesis (v2.0+)
│   └── biometric/         # Biometric input (v2.0+)
│
├── ml_framework/          # CIF/LAS/QEF ML Framework
│   ├── cif_las_qef/      # Core framework modules
│   │   ├── cif/          # Conscious Integration Framework
│   │   ├── las/          # Living Art Systems
│   │   ├── qef/          # Quantum Emotional Field
│   │   └── ethics/       # Resonant Ethics
│   └── examples/         # Usage examples
│
├── python/                # Python wrapper and utilities
├── reference/             # Reference implementations
│   ├── daiw_music_brain/ # DAiW Music Brain reference
│   └── python_kelly/     # Python Kelly implementation
│
├── data/                  # Data resources
│   ├── emotions/         # Emotion JSON files
│   ├── progressions/     # Chord progression databases
│   ├── scales/           # Scale emotional mappings
│   ├── grooves/          # Genre groove patterns
│   └── rules/            # Rule-breaking database
│
├── CODE/                  # Python visualization code
│   └── PYTHON CODE/      # 3D emotion visualization tools
│
├── phase2/                # Phase 2 advanced features
│   ├── kelly_phase2_core.py
│   └── kelly_phase2_agents.py
│
├── tests/                 # Test suite
├── MARKDOWN/              # Documentation
└── H T M L /              # HTML documentation

```

---

## Quick Start

### Prerequisites

**C++ Build:**

- CMake 3.22+
- C++20 compiler (Clang 14+, GCC 11+, or MSVC 2022)
- JUCE 7.0+ (downloaded automatically via CMake)
- macOS 11+ / Windows 10+ / Ubuntu 22.04+

**Python:**

- Python 3.8+
- See `ml_framework/requirements.txt` and `python/requirements.txt`

### Building the C++ Plugin

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Install (macOS)
./build_and_install.sh Release
```

Plugins will be in:

- `build/KellyMidiCompanion_artefacts/Release/VST3/`
- `build/KellyMidiCompanion_artefacts/Release/AU/` (macOS only)

### Python Setup

```bash
# Install ML framework dependencies
cd ml_framework
pip install -r requirements.txt

# Install Python utilities
cd ../python
pip install -r requirements.txt
```

### Running Examples

```bash
# ML Framework examples
cd ml_framework/examples
python basic_usage.py
python emotion_models_demo.py

# Visualization
cd ../../CODE/PYTHON\ CODE
python visualize_3d_emotion_wheel.py
python visualize_quantum_emotional_field.py
```

---

## Core Concepts

### The Three-Phase Intent System

1. **Phase 0: Wound** - "What hurts?"
   - Describe your current emotional state
   - System identifies the core emotional trigger

2. **Phase 1: Emotion** - Map to the 216-node thesaurus
   - Wound maps to emotions with valence, arousal, and intensity
   - Related emotions form a network of musical possibilities

3. **Phase 2: Rule-Breaks** - "What rules to break and why"
   - Intense emotions trigger intentional music theory violations
   - Dissonance, syncopation, extreme dynamics serve emotional truth

### Emotion Thesaurus

The 216-node thesaurus organizes emotions in 3D space:

| Dimension | Range | Effect |
|-----------|-------|--------|
| **Valence** | -1.0 to +1.0 | Mode (minor ↔ major) |
| **Arousal** | 0.0 to 1.0 | Tempo, rhythm complexity |
| **Intensity** | 0.0 to 1.0 | Dynamic range, rule-breaking |

### ML Framework Components

- **CIF (Conscious Integration Framework)**: Human-AI consciousness bridge
- **LAS (Living Art Systems)**: Self-evolving creative AI systems
- **QEF (Quantum Emotional Field)**: Network-based collective emotion synchronization
- **Resonant Ethics**: Ethical framework for conscious AI

---

## Documentation

### Main Documentation

- **Project Overview**: `MARKDOWN/README.md`
- **Build Guide**: `MARKDOWN/BUILD_PYTHON_BRIDGE.md`
- **Integration Status**: `MARKDOWN/INTEGRATION_COMPLETE.md`
- **Phase 2 Guide**: `phase2/KELLY_PHASE2_IMPLEMENTATION_GUIDE.md`

### ML Framework

- **Framework Overview**: `ml_framework/README.md`
- **Architecture**: `ml_framework/FRAMEWORK_SUMMARY.md`
- **Emotion Models**: `ml_framework/EMOTION_MODELS.md`

### Reference

- **DAiW Music Brain**: `reference/daiw_music_brain/README.md`
- **Python Kelly**: `reference/python_kelly/` (see `__init__.py`)

---

## Development

### Running Tests

```bash
# C++ tests
cmake -B build -DBUILD_TESTING=ON
cmake --build build
cd build && ctest --output-on-failure
```

### Debug Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
# Standalone app for testing without DAW:
./build/KellyMidiCompanion_artefacts/Debug/Standalone/Kelly\ MIDI\ Companion
```

### Code Style

- **C++**: Follow JUCE coding standards
- **Python**: PEP 8, type hints encouraged
- See `.cursorrules` for AI assistant guidelines

---

## Key Features

### v2.0.0+ Features

- ✅ Full cassette visual design with animated tape reels
- ✅ Emotion wheel selector with visual emotion mapping
- ✅ Voice synthesis integration framework
- ✅ Biometric input support framework
- ✅ Enhanced UI with toggleable visual components

### Phase 2 Features

- Advanced groove engine
- Humanization presets
- Mix parameter automation
- Agent-based generation
- ORP (Organic Rhythm Protocol)
- Biometric integration

### ML Framework Features

- Conscious Integration Framework (CIF)
- Living Art Systems (LAS)
- Quantum Emotional Field (QEF)
- Resonant Ethics protocols

---

## Philosophy

> *"Interrogate Before Generate"* — The tool shouldn't finish art for people; it should make them braver.

Kelly exists because technical perfection is not the same as emotional truth. Sometimes the "wrong" note is exactly right.

This project is dedicated to Kelly, whose memory inspires us to create tools that help people express what words cannot.

---

## License

MIT License - See LICENSE file (if present)

---

## Contributing

This is an active development project. Key areas for contribution:

- **Testing**: Expand test coverage
- **Documentation**: Improve guides and examples
- **Integration**: Bridge Python-C++ more seamlessly
- **Performance**: Optimize real-time audio processing
- **Features**: Implement Phase 2+ features

---

## Resources

- **Emotion Data**: `data/emotions/`
- **Music Theory**: `data/progressions/`, `data/scales/`
- **Visualization**: `CODE/PYTHON CODE/`
- **Reference Code**: `reference/`

---

*Built with love, grief, and JUCE.*
