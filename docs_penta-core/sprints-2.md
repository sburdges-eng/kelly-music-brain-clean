# Sprint 2 – Core Integration

## Overview

Sprint 2 focuses on building and testing the C++ core engine integration, ensuring the Python-C++ bridge works correctly and the C++ modules compile and run on all platforms.

## Objectives

1. **C++ Build System**: Validate CMake builds correctly
2. **Cross-Platform Compilation**: Ensure code compiles on Linux, macOS, Windows
3. **Core C++ Tests**: Validate C++ modules work as expected
4. **Python-C++ Bridge**: Test pybind11 bindings

## Build Scope

### CMake Configuration

- **Build Type**: Release (optimized for performance)
- **Compiler**: g++, clang, or MSVC
- **Dependencies**:
  - pybind11 (Python bindings)
  - JUCE (audio framework)
  - oscpack (OSC communication)
  - googletest (C++ testing)

### C++ Modules

- **Harmony Engine** (`src_penta-core/harmony/`)
  - ChordAnalyzer (pitch class set analysis)
  - ScaleDetector (Krumhansl-Schmuckler)
  - VoiceLeading (minimal motion optimizer)
  - HarmonyEngine (coordinator)

- **Groove Engine** (`src_penta-core/groove/`)
  - OnsetDetector (spectral flux)
  - TempoEstimator (autocorrelation)
  - RhythmQuantizer (grid quantization)
  - GrooveEngine (coordinator)

- **Diagnostics** (`src_penta-core/diagnostics/`)
  - PerformanceMonitor (CPU, latency, xruns)
  - AudioAnalyzer (level metering, clipping)
  - DiagnosticsEngine (coordinator)

- **OSC Communication** (`src_penta-core/osc/`)
  - OSCHub (bidirectional coordinator)
  - OSCServer (lock-free reception)
  - OSCClient (RT-safe sending)
  - RTMessageQueue (SPSC queue)

- **Common Infrastructure** (`src_penta-core/common/`)
  - RTTypes (real-time safe types)
  - RTMemoryPool (lock-free allocation)
  - RTLogger (deferred logging)

## Test Scope

### C++ Unit Tests (`tests_penta-core/`)

- **Harmony Tests** (`harmony_test.cpp`)
  - Chord detection accuracy
  - Scale detection validation
  - Voice leading optimization

- **Groove Tests** (`groove_test.cpp`)
  - Onset detection precision
  - Tempo estimation accuracy
  - Quantization correctness

- **OSC Tests** (`osc_test.cpp`)
  - Message serialization/deserialization
  - Lock-free queue correctness
  - RT-safety validation

- **RT Memory Tests** (`rt_memory_test.cpp`)
  - Memory pool allocation/deallocation
  - RT-safety guarantees
  - Performance benchmarks

## Success Criteria

- ✅ CMake configuration successful
- ✅ C++ code compiles without warnings (with `-Wall -Wextra`)
- ✅ All C++ unit tests pass via `ctest`
- ✅ Build completes in < 5 minutes
- ✅ pybind11 bindings compile and load in Python

## Workflow Configuration

```yaml
sprint2_core_integration:
  name: "Sprint 2 – Core Integration"
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: C++ Build & Test
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake g++
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        cmake --build . -j
        ctest --output-on-failure
```

## Key Deliverables

1. **Working C++ build system** on all platforms
2. **Passing C++ unit tests** validating core functionality
3. **Python bindings** that can be imported and used
4. **Build documentation** explaining compilation steps

## Dependencies

- CMake 3.15+
- C++17 compatible compiler
  - Linux: g++ 7+ or clang 5+
  - macOS: Xcode 10+ (clang)
  - Windows: Visual Studio 2017+ (MSVC)
- pybind11
- JUCE framework
- oscpack
- googletest

## Build Options

```bash
# Debug build with all features
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_HARMONY=ON \
      -DBUILD_GROOVE=ON \
      -DBUILD_DIAGNOSTICS=ON \
      -DBUILD_OSC=ON \
      -DBUILD_TESTS=ON \
      ..

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## Related Documentation

- [Phase 3 Design](PHASE3_DESIGN.md)
- [Phase 3 Summary](PHASE3_SUMMARY.md)
- [C++ Programming Guide](cpp-programming.md)
- [Build Instructions](BUILD.md)

## Performance Targets

- Chord analysis: < 1ms for 12-note chord
- Scale detection: < 5ms for 88-key piano
- Onset detection: Real-time (< buffer size latency)
- OSC message handling: < 100μs per message

## Notes

- All C++ code must be RT-safe (no allocations, locks, or blocking in audio callback)
- Use SIMD optimizations where possible (SSE2/AVX on x86, NEON on ARM)
- Memory pools must be pre-allocated on initialization
- OSC communication uses lock-free queues for thread safety
