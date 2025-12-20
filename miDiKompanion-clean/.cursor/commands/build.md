# Kelly MIDI Companion - Complete Build Guide

**Focus: AI/ML Features and Learning Systems**

This guide covers building all components of Kelly MIDI Companion with emphasis on AI/ML features including CIF (Conscious Integration Framework), LAS (Living Art Systems), QEF (Quantum Emotional Field), and emotion learning models.

---

## Quick Start

```bash
# 1. Setup Python environments and dependencies
./setup_workspace.sh

# 2. Build plugin with Python bridge (enables AI integration)
cmake -B build -DBUILD_PYTHON_BRIDGE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# 3. Verify AI/ML features
cd ml_framework && python examples/basic_usage.py
cd ml_framework && python examples/emotion_models_demo.py

# 4. Verify Python bridge
cd python && python -c "import kelly_bridge; print('Bridge OK')"
```

---

## Prerequisites

### System Requirements

- **macOS**: 11+ (Big Sur or later)
- **Windows**: 10+ (64-bit)
- **Linux**: Ubuntu 22.04+ or equivalent

### Required Tools

#### C++ Build Tools

- **CMake**: 3.22 or later

  ```bash
  # macOS
  brew install cmake

  # Linux
  sudo apt-get install cmake

  # Windows
  # Download from https://cmake.org/download/
  ```

- **C++ Compiler**: C++20 compatible
  - **macOS**: Clang 14+ (Xcode Command Line Tools)
  - **Linux**: GCC 11+ or Clang 14+
  - **Windows**: MSVC 2022+ or MinGW-w64

- **JUCE**: 8.0.4 (automatically fetched by CMake)

#### Python (for AI/ML Features)

- **Python**: 3.8 or later

  ```bash
  # Verify installation
  python3 --version
  ```

- **pip**: Latest version

  ```bash
  python3 -m pip install --upgrade pip
  ```

### Optional (for Full AI Features)

- **Python development headers** (required for Python bridge)

  ```bash
  # macOS
  brew install python3

  # Linux
  sudo apt-get install python3-dev python3-pip
  ```

---

## Build Components

### 1. Python ML Framework (Primary AI Focus)

The ML framework implements advanced AI systems for emotion-driven music generation:

- **CIF (Conscious Integration Framework)**: Human-AI consciousness bridge
- **LAS (Living Art Systems)**: Self-evolving creative AI systems
- **QEF (Quantum Emotional Field)**: Network-based collective emotion synchronization
- **Emotion Models**: Classical, quantum, hybrid, voice synthesis models
- **Resonant Ethics**: Ethical framework for conscious AI

#### Setup ML Framework

```bash
# Navigate to ML framework directory
cd ml_framework

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Verify ML Framework

```bash
# Test basic usage
python examples/basic_usage.py

# Test emotion models
python examples/emotion_models_demo.py

# Test advanced field features
python examples/advanced_field_demo.py

# Test music/voice generation
python examples/music_voice_demo.py
```

#### ML Framework Dependencies

The framework requires:

- `numpy>=1.21.0` - Numerical computations
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Visualization

See `ml_framework/requirements.txt` for complete list.

### 2. C++ Plugin with Python Bridge

The JUCE audio plugin with Python bridge enables real-time AI integration between C++ emotion processing and Python ML systems.

#### Build Plugin (Standard)

```bash
# Configure build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release
```

#### Build Plugin with Python Bridge (AI Integration)

```bash
# Configure with Python bridge enabled
cmake -B build \
    -DBUILD_PYTHON_BRIDGE=ON \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release
```

**Note**: The Python bridge connects C++ emotion engines with Python ML framework, enabling:

- Real-time emotion processing in C++
- Advanced AI learning in Python
- Bidirectional data flow for consciousness integration

#### Plugin Output Locations

After building, plugins are located in:

- **VST3**: `build/KellyMidiCompanion_artefacts/Release/VST3/`
- **AU** (macOS only): `build/KellyMidiCompanion_artefacts/Release/AU/`
- **Standalone**: `build/KellyMidiCompanion_artefacts/Release/Standalone/`
- **Python Bridge**: `python/kelly_bridge.*` (`.so` on Linux/macOS, `.pyd` on Windows)

#### Verify Python Bridge

```bash
# Navigate to python directory
cd python

# Test import
python3 -c "import kelly_bridge; print('Python bridge loaded successfully')"

# Run example
python examples/basic_usage.py
python examples/advanced_features.py
```

### 3. Python Utilities

Python utilities provide MIDI processing and wrapper functions for the C++ bridge.

#### Setup Python Utilities

```bash
# Navigate to python directory
cd python

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install as package
pip install -e .
```

#### Python Utilities Dependencies

- `mido>=1.2.10` - MIDI file I/O
- `python-rtmidi>=1.4.9` - Real-time MIDI (optional)

See `python/requirements.txt` for complete list.

### 4. Test Suite

Build and run tests to verify all components work correctly.

#### Build Tests

```bash
# Configure with tests enabled (default)
cmake -B build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release
```

#### Run Tests

```bash
# Using ctest
cd build
ctest --output-on-failure

# Or using test script
cd tests
./run_tests.sh
```

---

## Build Configurations

### Debug Build

For development and debugging:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

**Use Debug when**:

- Developing new features
- Debugging issues
- Testing with standalone app

### Release Build

For production use:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Use Release when**:

- Final deployment
- Performance testing
- Distribution

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_PYTHON_BRIDGE` | `OFF` | Enable Python-C++ bridge for AI integration |
| `BUILD_TESTS` | `ON` | Build unit tests |
| `CMAKE_BUILD_TYPE` | `Debug` | Build type: `Debug` or `Release` |

Example with all options:

```bash
cmake -B build \
    -DBUILD_PYTHON_BRIDGE=ON \
    -DBUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release
```

---

## Complete Build Process

### Phase 1: Python Environment Setup

```bash
# Run workspace setup script
./setup_workspace.sh

# This will:
# - Check Python, CMake, and compiler versions
# - Create virtual environments for ML framework and Python utilities
# - Install all Python dependencies
```

### Phase 2: C++ Build with AI Integration

```bash
# Configure with Python bridge enabled
cmake -B build \
    -DBUILD_PYTHON_BRIDGE=ON \
    -DBUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release

# Build (use -j flag for parallel builds)
cmake --build build --config Release -j$(nproc 2>/dev/null || echo 4)

# Verify Python bridge module exists
ls python/kelly_bridge.*
```

### Phase 3: Verification

```bash
# 1. Test ML framework
cd ml_framework
source venv/bin/activate
python examples/basic_usage.py
python examples/emotion_models_demo.py
deactivate
cd ..

# 2. Verify Python bridge
cd python
source venv/bin/activate
python -c "import kelly_bridge; print('Bridge OK')"
python examples/basic_usage.py
deactivate
cd ..

# 3. Run tests
cd build
ctest --output-on-failure
cd ..
```

---

## AI/ML Features Verification

### Verify CIF (Conscious Integration Framework)

```python
from cif_las_qef import CIF

cif = CIF()
result = cif.integrate(
    human_bio_data={"heart_rate": 75, "eeg_alpha": 0.6},
    las_emotional_state={"esv": {"valence": 0.3, "arousal": 0.5}}
)
print(cif.get_status())
```

### Verify LAS (Living Art Systems)

```python
from cif_las_qef import LAS

las = LAS()
result = las.generate(
    emotional_input={
        "biofeedback": {"heart_rate": 75},
        "voice": {"tone": 0.3}
    },
    creative_goal={"style": "ambient"}
)
evolution = las.evolve({
    "aesthetic_rating": 0.8,
    "emotional_resonance": 0.7
})
```

### Verify QEF (Quantum Emotional Field)

```python
from cif_las_qef import QEF

qef = QEF(node_id="my_node")
qef.activate()
qas = qef.emit_emotional_state(
    esv={"valence": 0.3, "arousal": 0.5}
)
collective = qef.receive_collective_resonance()
```

### Verify Emotion Models

```bash
cd ml_framework
source venv/bin/activate
python examples/emotion_models_demo.py
```

This demonstrates:

- Classical emotion models
- Quantum emotion models
- Hybrid emotion models
- Voice synthesis models
- Field dynamics

### Verify Python Bridge Integration

```python
import kelly_bridge

# Test emotion processing
emotion = kelly_bridge.process_emotion(
    valence=0.3,
    arousal=0.5,
    intensity=0.7
)

# Test MIDI generation
midi_data = kelly_bridge.generate_midi(
    emotion=emotion,
    length=16
)
```

---

## Installation

### macOS Plugin Installation

```bash
# VST3
cp -r build/KellyMidiCompanion_artefacts/Release/VST3/KellyMidiCompanion.vst3 \
   ~/Library/Audio/Plug-Ins/VST3/

# AU
cp -r build/KellyMidiCompanion_artefacts/Release/AU/KellyMidiCompanion.component \
   ~/Library/Audio/Plug-Ins/Components/

# Standalone
cp -r build/KellyMidiCompanion_artefacts/Release/Standalone/Kelly\ MIDI\ Companion.app \
   /Applications/
```

**Note**: On macOS, you may need to remove quarantine attributes:

```bash
xattr -d com.apple.quarantine ~/Library/Audio/Plug-Ins/VST3/KellyMidiCompanion.vst3
```

### Linux Plugin Installation

```bash
# VST3
mkdir -p ~/.vst3
cp -r build/KellyMidiCompanion_artefacts/Release/VST3/KellyMidiCompanion.vst3 \
   ~/.vst3/
```

### Windows Plugin Installation

```powershell
# VST3
Copy-Item -Recurse build\KellyMidiCompanion_artefacts\Release\VST3\KellyMidiCompanion.vst3 `
   "$env:ProgramFiles\Common Files\VST3\"
```

---

## Troubleshooting

### Python Bridge Issues

#### "Python not found" during CMake configuration

**Solution**: Install Python development headers

```bash
# macOS
brew install python3

# Linux
sudo apt-get install python3-dev
```

#### "ImportError: No module named 'kelly_bridge'"

**Solutions**:

1. Verify module exists: `ls python/kelly_bridge.*`
2. Add to PYTHONPATH: `export PYTHONPATH=$PWD/python:$PYTHONPATH`
3. Install package: `cd python && pip install -e .`

#### "pybind11 not found"

**Solution**: CMake should fetch automatically. If not:

```bash
pip install pybind11
```

### ML Framework Issues

#### "ModuleNotFoundError: No module named 'cif_las_qef'"

**Solution**: Ensure you're in the ML framework directory and virtual environment is activated:

```bash
cd ml_framework
source venv/bin/activate
pip install -r requirements.txt
```

#### Import errors with emotion models

**Solution**: Verify all dependencies are installed:

```bash
pip install numpy scipy matplotlib
```

### Build Issues

#### CMake can't find JUCE

**Solution**: JUCE is automatically fetched. If issues occur:

```bash
# Clean build directory
rm -rf build
cmake -B build
```

#### Build fails with compiler errors

**Solution**:

1. Ensure C++20 compatible compiler
2. Check CMake version (3.22+)
3. Try clean build: `rm -rf build && cmake -B build`

#### Plugin doesn't appear in DAW

**Solutions**:

1. Verify plugin is in correct location (see Installation section)
2. Rescan plugins in your DAW
3. Check plugin format compatibility (VST3, AU)
4. On macOS, check Gatekeeper: `xattr -d com.apple.quarantine <plugin_path>`

### Test Issues

#### Tests fail to build

**Solution**: Ensure BUILD_TESTS is enabled:

```bash
cmake -B build -DBUILD_TESTS=ON
```

#### Tests fail to run

**Solution**:

1. Verify build completed successfully
2. Check test executable exists: `ls build/tests/`
3. Run with verbose output: `ctest --verbose --output-on-failure`

---

## AI/ML Learning Systems

### Recursive Memory (LAS)

The LAS system includes recursive memory that learns from feedback:

```python
from cif_las_qef import LAS

las = LAS()

# Generate initial output
output = las.generate(emotional_input={...}, creative_goal={...})

# Provide feedback
feedback = {
    "aesthetic_rating": 0.8,
    "emotional_resonance": 0.7,
    "engagement": 0.75
}

# System evolves from feedback
evolution = las.evolve(feedback)
```

### Aesthetic DNA Evolution

LAS systems evolve their creative identity (aesthetic DNA) over time:

```python
# Get current aesthetic DNA
aesthetic_dna = las.get_aesthetic_dna()

# Evolve based on successful patterns
las.evolve_aesthetic_dna(successful_patterns)
```

### Collective Learning (QEF)

QEF enables collective learning across network nodes:

```python
from cif_las_qef import QEF

qef = QEF(node_id="learning_node")
qef.activate()

# Emit learning signal
qef.emit_learning_signal(learned_pattern)

# Receive collective knowledge
collective_knowledge = qef.receive_collective_resonance()
```

---

## Development Workflow

### Working with AI Features

1. **Start ML Framework**: Activate virtual environment and test examples
2. **Develop in Python**: Use ML framework for AI/ML development
3. **Integrate with C++**: Use Python bridge to connect with plugin
4. **Test in Plugin**: Load plugin in DAW and test real-time processing

### Testing AI Integration

```bash
# 1. Test ML framework standalone
cd ml_framework && python examples/basic_usage.py

# 2. Test Python bridge
cd python && python examples/basic_usage.py

# 3. Test plugin with AI features
# Load plugin in DAW and use AI generation features
```

### Debugging AI Systems

```python
# Enable verbose logging
from cif_las_qef import UnifiedFramework, FrameworkConfig

config = FrameworkConfig(
    enable_cif=True,
    enable_las=True,
    verbose=True  # Enable debug output
)

framework = UnifiedFramework(config)
# Framework will output detailed debug information
```

---

## Next Steps

After building:

1. **Explore ML Framework**: See `ml_framework/examples/` for AI/ML demos
2. **Use Python Bridge**: See `python/examples/` for integration examples
3. **Read Documentation**:
   - `ml_framework/README.md` - ML framework overview
   - `ml_framework/EMOTION_MODELS.md` - Emotion model details
   - `MARKDOWN/BUILD_PYTHON_BRIDGE.md` - Python bridge details
4. **Test in DAW**: Load plugin and test emotion-driven MIDI generation

---

## Resources

- **ML Framework**: `ml_framework/` - CIF/LAS/QEF implementations
- **Python Bridge**: `python/` - C++ to Python integration
- **Emotion Data**: `data/emotions/` - Emotion mappings and thesaurus
- **Documentation**: `MARKDOWN/` - Detailed guides and references
- **Examples**: `ml_framework/examples/` - AI/ML demonstration scripts

---

*Built with focus on AI/ML features and learning systems. The Kelly MIDI Companion bridges emotional expression with advanced machine learning for therapeutic music generation.*
