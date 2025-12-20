# Quick Start Guide - Penta Core Phase 3

## 🚀 5-Minute Setup

### 1. Prerequisites Check
```bash
# Verify you have required tools
cmake --version    # Need 3.20+
python3 --version  # Need 3.8+
c++ --version      # Need C++20 support
```

### 2. Clone & Build
```bash
# Clone repository
git clone https://github.com/yourusername/penta-core.git
cd penta-core

# Quick build (Release mode)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 3. Test the Build
```bash
# Run C++ tests
cd build && ctest --output-on-failure

# Try Python examples (Note: implementations are stubs)
cd ..
python3 examples/harmony_example.py
python3 examples/groove_example.py
```

## 📁 What You Get

```
penta-core/
├── C++ Core Library
│   ├── 18 header files (complete API)
│   ├── 4 implementation stubs
│   └── Real-time safe infrastructure
│
├── Python Bindings
│   ├── pybind11 integration
│   └── High-level Python API
│
├── JUCE Plugin
│   ├── VST3 / AU targets
│   └── Real-time GUI
│
├── Examples
│   ├── Harmony analysis
│   ├── Groove detection
│   └── Full integration
│
└── Documentation
    ├── Architecture guide (14K+ words)
    ├── Build instructions
    └── Implementation summary
```

## 🎯 Try It Out

### Harmony Analysis (Python)
```python
from penta_core import HarmonyEngine

# Create engine
harmony = HarmonyEngine(sample_rate=48000.0)

# Process MIDI notes (C major chord)
notes = [(60, 80), (64, 75), (67, 70)]
harmony.process_midi_notes(notes)

# Get detected chord
chord = harmony.get_current_chord()
print(f"Detected: {chord['name']}")
# Output: Detected: C (confidence: 0.XX)
```

### Groove Analysis (Python)
```python
import numpy as np
from penta_core import GrooveEngine

# Create engine
groove = GrooveEngine(sample_rate=48000.0)

# Process audio buffer
audio = np.random.randn(512).astype(np.float32)
groove.process_audio(audio)

# Get tempo estimate
analysis = groove.get_analysis()
print(f"Tempo: {analysis['tempo']:.1f} BPM")
```

### Full Integration
```python
from penta_core import PentaCore
import numpy as np

# All-in-one engine
penta = PentaCore(sample_rate=48000.0)
penta.start_osc()  # Start OSC communication

# Process audio + MIDI together
audio = np.random.randn(512).astype(np.float32)
midi = [(60, 80), (64, 75), (67, 70)]
penta.process(audio, midi)

# Get complete state
state = penta.get_state()
print(f"Chord: {state['chord']['name']}")
print(f"Tempo: {state['groove']['tempo']:.1f} BPM")
print(f"CPU: {state['diagnostics']['cpu_usage']:.1f}%")
```

## 🔧 Build Options

### Development Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
```

### Release with Optimizations
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPENTA_ENABLE_SIMD=ON \
  -DPENTA_ENABLE_LTO=ON
```

### Python Module Only
```bash
cmake -B build \
  -DPENTA_BUILD_PYTHON_BINDINGS=ON \
  -DPENTA_BUILD_JUCE_PLUGIN=OFF
```

### JUCE Plugin Only
```bash
cmake -B build \
  -DPENTA_BUILD_PYTHON_BINDINGS=OFF \
  -DPENTA_BUILD_JUCE_PLUGIN=ON
```

## 🎵 JUCE Plugin Usage

### Build Plugin
```bash
cmake -B build -DPENTA_BUILD_JUCE_PLUGIN=ON
cmake --build build --target PentaCorePlugin
```

### Install (macOS)
```bash
# VST3
cp -r build/plugins/PentaCorePlugin_artefacts/Release/VST3/*.vst3 \
      ~/Library/Audio/Plug-Ins/VST3/

# AU
cp -r build/plugins/PentaCorePlugin_artefacts/Release/AU/*.component \
      ~/Library/Audio/Plug-Ins/Components/
```

### Use in DAW
1. Open your DAW (Ableton, Logic, Reaper, etc.)
2. Load "Penta Core" as a MIDI effect
3. Send MIDI notes to see harmony analysis
4. View real-time chord/scale detection in plugin GUI

## 📊 Architecture Overview

```
┌─────────────────────────────────────┐
│     Python "Brain"                  │
│  • High-level logic                 │
│  • AI integration                   │
│  • Experimentation                  │
└──────────────┬──────────────────────┘
               │ pybind11
┌──────────────┴──────────────────────┐
│     C++ "Engine"                    │
│  ┌──────────┐  ┌──────────┐        │
│  │ Harmony  │  │  Groove  │        │
│  └──────────┘  └──────────┘        │
│  ┌──────────┐  ┌──────────┐        │
│  │ Diag     │  │   OSC    │        │
│  └──────────┘  └──────────┘        │
└──────────────┬──────────────────────┘
               │ JUCE
┌──────────────┴──────────────────────┐
│     DAW Integration                 │
│  VST3 • AU • Standalone             │
└─────────────────────────────────────┘
```

## ⚡ Performance Goals

| Component | Target | Status |
|-----------|--------|--------|
| Harmony Analysis | < 100μs | API Complete ✅ |
| Groove Analysis | < 200μs | API Complete ✅ |
| Diagnostics | < 10μs | API Complete ✅ |
| OSC Messaging | < 50μs | API Complete ✅ |

## 🐛 Current Status

### ✅ Completed (Phase 3.1)
- Complete C++ API (headers)
- CMake build system
- pybind11 Python bindings
- JUCE plugin scaffold
- Example code
- Comprehensive documentation

### ⏳ In Progress (Phase 3.2+)
- Algorithm implementations (stubs present)
- DSP routines (FFT, autocorrelation)
- SIMD optimizations
- Full test coverage

### 📝 Note
Most implementations are currently **stubs** that demonstrate the API but don't perform actual analysis. The architecture and interfaces are complete and ready for implementation.

## 📚 Documentation

- **[README.md](../README.md)** - Project overview
- **[BUILD.md](BUILD.md)** - Detailed build instructions
- **[PHASE3_DESIGN.md](PHASE3_DESIGN.md)** - Architecture deep-dive
- **[PHASE3_SUMMARY.md](PHASE3_SUMMARY.md)** - Implementation summary

## 🤝 Next Steps

1. **For Users**: Try the examples, explore the Python API
2. **For Contributors**: Implement the DSP algorithms (see PHASE3_DESIGN.md)
3. **For Integrators**: Build the JUCE plugin, test in your DAW

## 💡 Key Features

- **Real-Time Safe**: Lock-free, wait-free algorithms
- **Zero Allocations**: Pre-allocated memory pools
- **Cross-Platform**: macOS, Linux, Windows
- **Flexible**: Python brain, C++ engine
- **Professional**: DAW-ready with JUCE

## 🆘 Getting Help

- **Build Issues**: Check [BUILD.md](BUILD.md)
- **Architecture Questions**: See [PHASE3_DESIGN.md](PHASE3_DESIGN.md)
- **API Reference**: Read header files in `include/penta/`
- **Examples**: Check `examples/` directory

## 🎉 Success!

If you made it here and everything built successfully, you're ready to explore Penta Core Phase 3!

**What's working:**
- ✅ Build system compiles
- ✅ Python module imports
- ✅ Examples run (with stub implementations)
- ✅ Architecture is solid

**What's next:**
- Implement the actual DSP algorithms
- Add SIMD optimizations
- Comprehensive testing
- Performance validation

Happy coding! 🎵🚀
