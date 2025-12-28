# Neural DSP Performance Research

## Overview

This document outlines performance targets and architectural patterns inspired by Neural DSP plugins and the Montserrat Professional research (https://github.com/JulietaUla/Montserratprofessional) to ensure Kelly's audio engine meets or exceeds professional-grade performance standards.

## Performance Targets

### Latency Requirements
- **Target**: < 100μs per processing block (512 samples @ 48kHz)
- **Current Kelly Performance** (from audio-engine-cpp/README.md):
  - Chord Analysis: ~2.1 μs per frame (✅ EXCELLENT)
  - Onset Detection: ~140 μs per 512-sample block (⚠️ NEEDS OPTIMIZATION)
  - Tempo Estimation: ~180 μs per update (⚠️ NEEDS OPTIMIZATION)
  - RMS Calculation: ~0.8 μs per block (✅ EXCELLENT)

### Neural DSP Benchmarks (Industry Standard)
- Plugin latency: < 3ms total (buffer + processing)
- CPU usage: < 5% per instance on modern processors
- Memory footprint: < 50MB per instance
- Real-time safety: Zero allocations in audio thread
- SIMD utilization: 90%+ for critical paths

## Architectural Patterns from Neural DSP

### 1. Lock-Free Communication
- Use ring buffers for UI ↔ Audio thread communication
- Current Kelly status: ✅ Implemented (uses readerwriterqueue)
- Enhancement: Add lock-free parameter smoothing

### 2. SIMD-First Design
- Process 4-8 samples simultaneously using AVX2/NEON
- Current Kelly status: ✅ Partial (ChordAnalyzerSIMD.cpp exists)
- Enhancement needed: Extend to all DSP kernels (groove, onset detection)

### 3. Pre-allocated Memory Pools
- All buffers allocated at initialization
- Current Kelly status: ✅ Implemented (Side A/Side B architecture)
- Enhancement: Add buffer reuse tracking

### 4. Oversampling Strategy
- Critical for high-quality nonlinear processing
- Current Kelly status: ❌ Not implemented
- Enhancement needed: Add 2x/4x oversampling for distortion/saturation

### 5. ML Inference Optimization
- Use quantized models (INT8/INT16)
- Batch predictions when possible
- Current Kelly status: ⚠️ ONNX/RTNeural exist but not optimized
- Enhancement needed: Model quantization pipeline

## Implementation Roadmap

### Phase 1: Profiling Infrastructure (Week 1)
1. Add Tracy profiler integration (already has ENABLE_TRACY flag)
2. Create benchmark suite for all DSP modules
3. Establish performance baselines
4. Identify bottlenecks

### Phase 2: SIMD Optimization (Week 2-3)
1. Vectorize onset detection (spectral flux computation)
2. Vectorize tempo estimation (autocorrelation)
3. Add SIMD versions of filter banks
4. Benchmark improvements

### Phase 3: ML Inference Pipeline (Week 4-5)
1. Add model quantization tools
2. Implement batched inference queue
3. Add fallback mechanisms for RT safety
4. Profile ML overhead

### Phase 4: Advanced DSP (Week 6)
1. Implement oversampling framework
2. Add antialiasing filters
3. Optimize voice synthesis pipeline

## Key Metrics to Track

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Harmony Analysis | 2.1 μs | < 5 μs | ✅ |
| Onset Detection | 140 μs | < 80 μs | 🔄 |
| Tempo Estimation | 180 μs | < 100 μs | 🔄 |
| RMS/Peak | 0.8 μs | < 2 μs | ✅ |
| Memory Allocations (audio thread) | 0 | 0 | ✅ |
| SIMD Coverage | ~30% | > 80% | 🔄 |

## References

- Neural DSP plugin architecture principles
- Montserrat Professional research: https://github.com/JulietaUla/Montserratprofessional
- Kelly audio-engine-cpp: `/audio-engine-cpp/README.md`
- JUCE DSP best practices
- Intel AVX2 optimization guide

## Next Steps

1. Enable Tracy profiler in build system
2. Run comprehensive benchmarks
3. Create optimization tickets for critical paths
4. Document SIMD patterns for team
