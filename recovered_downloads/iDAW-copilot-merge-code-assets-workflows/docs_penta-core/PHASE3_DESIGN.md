# Phase 3: C++ Transition Design & Implementation Guide

## Overview

Phase 3 represents the transition of Penta Core's performance-critical components from Python to C++, while maintaining Python as the high-level "brain" that orchestrates the optimized C++ "engine". This architecture provides professional-grade performance suitable for DAW integration while preserving Python's flexibility for experimentation and AI integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Python "Brain"                          │
│  • High-level logic & AI integration                        │
│  • User interface & DAW communication                       │
│  • Experimentation & prototyping                            │
└────────────────────┬────────────────────────────────────────┘
                     │ pybind11
┌────────────────────┴────────────────────────────────────────┐
│                    C++ "Engine"                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Harmony    │  │    Groove    │  │ Diagnostics  │      │
│  │   Analysis   │  │   Analysis   │  │  Monitoring  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │         Real-Time Safe Infrastructure            │      │
│  │  • Lock-free queues  • RT memory pools           │      │
│  │  • SIMD optimizations • OSC communication        │      │
│  └──────────────────────────────────────────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │ JUCE
┌────────────────────┴────────────────────────────────────────┐
│                  DAW/Plugin Layer                           │
│                VST3 • AU • Standalone                       │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. Harmony Engine (`penta/harmony/`)

**Purpose**: Real-time harmonic analysis and voice leading

**Key Components**:
- `ChordAnalyzer`: Pitch class set analysis using template matching
- `ScaleDetector`: Krumhansl-Schmuckler key detection algorithm
- `VoiceLeading`: Minimal motion voice leading optimizer
- `HarmonyEngine`: Integrated harmony analysis coordinator

**RT-Safety Features**:
- Pre-allocated chord template database
- Lock-free pitch class set updates
- SIMD-optimized pattern matching (AVX2)
- No dynamic allocation in audio thread

**Performance Targets**:
- < 100μs per analysis update
- < 1% CPU at 48kHz/512 samples
- Zero allocations in RT path

### 2. Groove Engine (`penta/groove/`)

**Purpose**: Real-time rhythm analysis and quantization

**Key Components**:
- `OnsetDetector`: Spectral flux-based onset detection
- `TempoEstimator`: Autocorrelation-based tempo tracking
- `RhythmQuantizer`: Grid-based quantization with swing
- `GrooveEngine`: Integrated groove analysis coordinator

**RT-Safety Features**:
- Pre-allocated FFT buffers
- Circular buffer for history
- Lock-free onset queue
- Optimized autocorrelation with SIMD

**Performance Targets**:
- < 200μs per audio block (512 samples)
- Real-time tempo tracking (< 2 bar latency)
- < 2% CPU at 48kHz

### 3. Diagnostics Engine (`penta/diagnostics/`)

**Purpose**: Performance monitoring and audio analysis

**Key Components**:
- `PerformanceMonitor`: CPU usage, latency, xrun tracking
- `AudioAnalyzer`: Level metering, clipping detection
- `DiagnosticsEngine`: Integrated monitoring coordinator

**RT-Safety Features**:
- Atomic statistics updates
- Lock-free circular buffers
- Minimal overhead measurement (<1%)
- Deferred reporting to non-RT thread

### 4. OSC Communication (`penta/osc/`)

**Purpose**: Real-time safe OSC messaging for DAW integration

**Key Components**:
- `OSCServer`: Lock-free message reception
- `OSCClient`: RT-safe message sending
- `RTMessageQueue`: Single-producer/single-consumer queue
- `OSCHub`: Bidirectional OSC coordinator

**RT-Safety Features**:
- Lock-free message queues (cameron314/readerwriterqueue)
- Pre-allocated message pools
- Zero-copy message construction
- Deferred string formatting

## Real-Time Safety Principles

### Memory Management

1. **Pre-allocation**: All buffers allocated during initialization
2. **Object Pools**: RT-safe memory pools for dynamic needs
3. **Stack Allocation**: Prefer stack over heap in RT code
4. **Placement New**: Use when objects needed in RT path

Example:
```cpp
// ✅ GOOD: Pre-allocated pool
RTMemoryPool messagePool(sizeof(Message), 1024);
auto msg = RTPoolPtr<Message>(messagePool);

// ❌ BAD: Dynamic allocation
auto msg = std::make_unique<Message>();
```

### Threading

1. **Lock-Free Primitives**: Use std::atomic, lock-free queues
2. **Wait-Free When Possible**: Avoid blocking operations
3. **Thread Affinity**: Pin audio thread to dedicated core
4. **Priority Boost**: Elevate audio thread priority

Example:
```cpp
// ✅ GOOD: Lock-free queue
readerwriterqueue::ReaderWriterQueue<Message> queue;

// ❌ BAD: Mutex-protected queue
std::mutex mtx;
std::queue<Message> queue;
```

### Algorithms

1. **Bounded Execution**: Guarantee upper bounds on time
2. **SIMD Optimization**: Use AVX2/NEON for DSP
3. **Cache-Friendly**: Minimize cache misses
4. **Branch Prediction**: Write predictable code

## Python/C++ Bridge

### pybind11 Integration

The Python bridge provides:
- **Zero-Copy Arrays**: NumPy arrays map directly to C++ buffers
- **Exception Safety**: C++ exceptions converted to Python
- **Object Lifetime**: Smart pointers manage C++ objects
- **GIL Management**: Release GIL during long operations

Example usage:
```python
from penta_core import PentaCore

# Initialize (creates C++ engines)
penta = PentaCore(sample_rate=48000.0)

# Process audio (zero-copy if contiguous)
audio = np.random.randn(512).astype(np.float32)
midi = [(60, 80), (64, 75), (67, 70)]
penta.process(audio, midi)

# Get results (copies small structs)
state = penta.get_state()
print(f"Chord: {state['chord']['name']}")
```

### Design Patterns

1. **RAII Wrappers**: Python classes manage C++ object lifecycle
2. **Property Exposure**: Read-only access to internal state
3. **Callback Support**: Register Python callbacks for events
4. **Error Handling**: Translate C++ exceptions to Python

## JUCE Plugin Architecture

### Plugin Structure

```
PentaCorePlugin (JUCE AudioProcessor)
├── processBlock()           # RT audio callback
│   ├── diagnostics.beginMeasurement()
│   ├── processMidiForHarmony()
│   ├── processAudioForGroove()
│   ├── oscHub.sendUpdates()
│   └── diagnostics.endMeasurement()
├── createEditor()           # GUI creation
└── getStateInformation()    # Preset management
```

### Parameter System

JUCE `AudioProcessorValueTreeState` provides:
- Thread-safe parameter access
- DAW automation support
- Preset saving/loading
- Undo/redo support

### GUI Architecture

- **Timer-based Updates**: 30Hz refresh (non-RT)
- **Custom Components**: JUCE graphics for displays
- **Modular Panels**: Harmony, Groove, Diagnostics views
- **Accessibility**: Keyboard navigation, screen reader support

## Build System

### CMake Structure

```
CMakeLists.txt              # Root configuration
├── external/               # FetchContent dependencies
│   ├── pybind11
│   ├── JUCE
│   ├── oscpack
│   └── googletest
├── src/                    # C++ core library
├── bindings/               # pybind11 Python module
├── plugins/                # JUCE plugin targets
└── tests/                  # Unit tests
```

### Build Options

- `PENTA_BUILD_PYTHON_BINDINGS`: Build pybind11 module (default: ON)
- `PENTA_BUILD_JUCE_PLUGIN`: Build VST3/AU plugins (default: ON)
- `PENTA_BUILD_TESTS`: Build unit tests (default: ON)
- `PENTA_ENABLE_SIMD`: Enable AVX2 optimizations (default: ON)
- `PENTA_ENABLE_LTO`: Link-time optimization (default: OFF)

### Build Commands

```bash
# Configure with all features
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPENTA_ENABLE_SIMD=ON \
      -DPENTA_ENABLE_LTO=ON

# Build
cmake --build build --config Release -j

# Install Python module
cmake --install build --prefix ~/.local

# Run tests
cd build && ctest --output-on-failure
```

## Performance Optimization

### Profiling Strategy

1. **Instrumentation**: Use PerformanceMonitor in debug builds
2. **Sampling**: Profile with Instruments (macOS) or perf (Linux)
3. **Micro-benchmarks**: Google Benchmark for algorithms
4. **Memory Profiling**: Valgrind for allocation tracking

### Optimization Checklist

- [ ] SIMD intrinsics for DSP loops
- [ ] Cache-line alignment for buffers
- [ ] Profile-guided optimization (PGO)
- [ ] Link-time optimization (LTO)
- [ ] Minimize virtual function calls in RT path
- [ ] Use `inline` and `constexpr` liberally
- [ ] Avoid exceptions in RT code
- [ ] Pre-compute lookup tables

## Testing Strategy

### Unit Tests (Google Test)

```cpp
TEST(HarmonyEngineTest, DetectsCMajorChord) {
    HarmonyEngine engine;
    std::array<bool, 12> pitchClasses = {
        true, false, false, false, true,  // C, E
        false, false, true, false, false, // G
        false, false
    };
    
    ChordAnalyzer analyzer;
    auto chord = analyzer.analyze(pitchClasses);
    
    EXPECT_EQ(chord.root, 0);  // C
    EXPECT_GT(chord.confidence, 0.8f);
}
```

### Integration Tests

- Python scripts that exercise full pipeline
- Audio file regression tests
- OSC communication tests
- JUCE plugin validation

### Performance Tests

- Latency benchmarks (must be < target)
- CPU usage tests (sustained and peak)
- Memory allocation detection (zero in RT)
- Thread safety validation (ThreadSanitizer)

## Migration Path

### Phase 3.1: Foundation (Weeks 1-2)
- ✅ CMake build system setup
- ✅ Core types and RT infrastructure
- ✅ pybind11 bindings skeleton
- ✅ JUCE plugin scaffold

### Phase 3.2: Harmony Module (Weeks 3-4)
- Implement ChordAnalyzer
- Implement ScaleDetector
- Implement VoiceLeading
- Python bindings and tests

### Phase 3.3: Groove Module (Weeks 5-6)
- Implement OnsetDetector (FFT required)
- Implement TempoEstimator
- Implement RhythmQuantizer
- Python bindings and tests

### Phase 3.4: Integration (Weeks 7-8)
- OSC communication implementation
- Diagnostics implementation
- JUCE plugin completion
- End-to-end testing

### Phase 3.5: Optimization (Weeks 9-10)
- SIMD optimization passes
- Profiling and bottleneck elimination
- Memory usage optimization
- Documentation and examples

## OSC Protocol Specification

### Outgoing Messages (Penta → DAW)

```
/penta/harmony/chord      i i f     (root, quality, confidence)
/penta/harmony/scale      i i f     (tonic, mode, confidence)
/penta/groove/tempo       f f       (bpm, confidence)
/penta/groove/onset       i f       (position, strength)
/penta/diagnostics/cpu    f         (percentage)
/penta/diagnostics/xrun   i         (count)
```

### Incoming Messages (DAW → Penta)

```
/penta/config/harmony     f         (confidence_threshold)
/penta/config/groove      f f       (min_tempo, max_tempo)
/penta/config/quantize    i f       (resolution, strength)
/penta/control/reset                (reset all engines)
```

## Dependencies

### Required

- **CMake** >= 3.20: Build system
- **C++20 Compiler**: GCC 11+, Clang 13+, MSVC 2019+
- **Python** >= 3.8: For bindings
- **pybind11**: Python/C++ bridge

### Optional

- **JUCE** 7.0+: For plugin builds
- **Google Test**: For unit tests
- **oscpack**: OSC communication (auto-fetched)
- **readerwriterqueue**: Lock-free queues (auto-fetched)

### System Libraries

- **pthread**: Threading (Linux/macOS)
- **FFTW3** or **Accelerate**: FFT (consider adding)

## Deployment

### Python Package

```bash
# Install from source
pip install -e .

# Or build wheel
python setup.py bdist_wheel
pip install dist/penta_core-1.0.0-*.whl
```

### JUCE Plugin

- **VST3**: Copy to `~/Library/Audio/Plug-Ins/VST3/` (macOS)
- **AU**: Copy to `~/Library/Audio/Plug-Ins/Components/` (macOS)
- **Standalone**: App bundle in build directory

## Future Enhancements

1. **GPU Acceleration**: CUDA/Metal for spectrogram analysis
2. **ML Integration**: TorchScript models in C++
3. **Distributed Processing**: Network-based compute offload
4. **Advanced DSP**: Polyphonic pitch detection, source separation
5. **Mobile Support**: iOS/Android audio engines

## Support & Resources

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **API Reference**: Run `doxygen Doxyfile`
- **Benchmarks**: See `benchmarks/` directory

## License

MIT License - See LICENSE file for details
