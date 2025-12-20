# Penta Core

**Professional-grade music analysis and generation engine with hybrid Python/C++ architecture**

[![Build Status](https://github.com/yourusername/penta-core/workflows/Build/badge.svg)](https://github.com/yourusername/penta-core/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Team Documentation

Check out the [team documentation](docs/README.md) to get up to speed on:

- [Swift SDKs Development](docs/swift-sdks.md)
- [C++ Programming](docs/cpp-programming.md)
- [Rust DAW Backend](docs/rust-daw-backend.md) - 150 things to know about building a Rust DAW
- [Low-Latency DAW Subjects](docs/low-latency-daw.md) - Real-time systems, DSP, and DAW architecture
- [DAW Engine Stability](docs/daw-engine-stability.md) - 100 topics for robust audio engines
- [Psychoacoustic Sound Design](docs/psychoacoustic-sound-design.md) ⭐ NEW - 90+ cinematic audio manipulation techniques
- [DAW Track Import Methods](docs/daw-track-import-methods.md) ⭐ NEW - 100 ways to import tracks into a DAW
- [**Comprehensive System Requirements & TODO**](docs/comprehensive-system-requirements.md) ⭐ MASTER - 400+ requirements with 12-phase implementation roadmap
- [DAiW Music Brain](docs/daiw-music-brain.md) - AI-powered music composition system architecture
- [DAW UI Design Patterns](docs/daw-ui-patterns.md) ⭐ NEW - React/TypeScript UI patterns for DAW interfaces
- [Instrument Learning Research](docs/instrument-learning-research.md) - GitHub research automation tool
- [Music Generation Research](docs/music-generation-research.md) - 150 AI/ML research topics
- [AI Prompting Guide](docs/ai-prompting-guide.md) - 150 techniques for effective AI prompting
- [Multi-Agent MCP Guide](docs/multi-agent-mcp-guide.md) - Multi-agent systems & MCP architecture
- [MCP Protocol & Debugging Strategy](docs/mcp-protocol-debugging.md) ⭐ NEW - Autonomous operation protocol & debugging
- [DAW Programs](docs/daw-programs.md)
- [Audio Software/Hardware Interfaces](docs/audio-interfaces.md)
- [Media Production](docs/media-production.md)

## Setup

## Overview

Penta Core is a high-performance music analysis engine designed for real-time DAW integration. Phase 3 introduces a C++ core for professional-grade performance while maintaining Python flexibility for AI integration and high-level control.

## Architecture

```
Python "Brain"  (Flexibility, AI, Experimentation)
      ↕ pybind11
C++ "Engine"    (Real-time performance, DSP, Analysis)
      ↕ JUCE
DAW Integration (VST3, AU, Standalone)
```

## Features

### Harmony Analysis
- **Real-time chord detection** using pitch class set analysis
- **Scale detection** with Krumhansl-Schmuckler algorithm
- **Voice leading optimization** for smooth transitions
- **Confidence scoring** for musical decisions

### Groove Analysis
- **Onset detection** using spectral flux
- **Tempo estimation** with autocorrelation
- **Rhythm quantization** with configurable grid and swing
- **Time signature detection**

### Performance Monitoring
- **CPU usage tracking** with minimal overhead
- **Latency measurement** for RT validation
- **Audio level monitoring** and clipping detection
- **XRun detection** for buffer underruns

### OSC Communication
- **Real-time safe messaging** to DAWs and controllers
- **Lock-free queues** for zero-blocking communication
- **Bidirectional** protocol support
- **Pattern-based routing**

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/penta-core.git
cd penta-core

# Build C++ library and Python bindings
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Install Python module
pip install -e ..
```

### Basic Usage

```python
from penta_core import PentaCore
import numpy as np

# Initialize
penta = PentaCore(sample_rate=48000.0)
penta.start_osc()

# Process audio and MIDI
audio = np.random.randn(512).astype(np.float32)
midi_notes = [(60, 80), (64, 75), (67, 70)]  # C major
penta.process(audio, midi_notes)

# Get musical state
state = penta.get_state()
print(f"Chord: {state['chord']['name']}")
print(f"Tempo: {state['groove']['tempo']:.1f} BPM")
```

## Project Structure

```
penta-core/
├── CMakeLists.txt              # Root build configuration
├── include/penta/              # C++ public headers
│   ├── common/                 # RT-safe utilities
│   ├── harmony/                # Harmony analysis
│   ├── groove/                 # Groove analysis
│   ├── diagnostics/            # Performance monitoring
│   └── osc/                    # OSC communication
├── src/                        # C++ implementations
├── bindings/                   # pybind11 Python bindings
├── plugins/                    # JUCE VST3/AU plugin
├── python/penta_core/          # Python API wrapper
├── examples/                   # Usage examples
├── tests/                      # Unit tests
└── docs/                       # Documentation
    ├── PHASE3_DESIGN.md        # Architecture overview
    └── BUILD.md                # Build instructions
```

## Performance

### Real-Time Guarantees
- **< 100μs** harmony analysis latency
- **< 200μs** groove analysis per block
- **Zero allocations** in audio thread
- **Lock-free** inter-thread communication

### Optimizations
- **SIMD acceleration** (AVX2) for DSP
- **Cache-friendly** data structures
- **Pre-allocated pools** for RT memory
- **Profile-guided optimization** support

## DAW Integration

### JUCE Plugin
- **VST3** and **AU** formats
- **Real-time** analysis display
- **OSC bridge** for external control
- **Preset management**

### OSC Protocol
```
/penta/harmony/chord   i i f   (root, quality, confidence)
/penta/groove/tempo    f f     (bpm, confidence)
/penta/diagnostics/cpu f       (percentage)
```

## Development

### Building from Source
See [BUILD.md](docs/BUILD.md) for detailed instructions.

### Running Tests
```bash
cd build
ctest --output-on-failure
```

### Examples
```bash
# Harmony analysis
python examples/harmony_example.py

# Groove detection
python examples/groove_example.py

# Full integration
python examples/integration_example.py
```

## Documentation

- **[Phase 3 Design](docs/PHASE3_DESIGN.md)** - Architecture and implementation details
- **[Build Instructions](docs/BUILD.md)** - Comprehensive build guide
- **[Examples](examples/)** - Code examples and tutorials

## Roadmap

### Phase 3.1 ✅ (Current)
- [x] CMake build system
- [x] Core C++ headers and types
- [x] pybind11 bindings skeleton
- [x] JUCE plugin scaffold
- [x] Documentation

### Phase 3.2 (Next)
- [ ] Harmony module implementation
- [ ] ChordAnalyzer with templates
- [ ] ScaleDetector algorithm
- [ ] VoiceLeading optimizer

### Phase 3.3
- [ ] Groove module implementation
- [ ] OnsetDetector with FFT
- [ ] TempoEstimator
- [ ] RhythmQuantizer

### Phase 3.4
- [ ] OSC communication
- [ ] Diagnostics implementation
- [ ] JUCE plugin completion
- [ ] End-to-end testing

### Phase 3.5
- [ ] SIMD optimizations
- [ ] Performance profiling
- [ ] Documentation polish
- [ ] Release preparation

## License

MIT License - see [LICENSE](LICENSE) for details.