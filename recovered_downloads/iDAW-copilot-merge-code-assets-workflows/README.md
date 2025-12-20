# iDAW - Integrated Digital Audio Workstation

> **Merged Repository**: This repository combines code, assets, workflows, agents, and documentation from **sburdges-eng/penta-core** and **sburdges-eng/DAiW-Music-Brain** into a unified iDAW project.

## Repository Merge Overview

This repository was created by merging two complementary music production projects:

1. **penta-core**: Professional-grade music analysis and generation engine with hybrid Python/C++ architecture
2. **DAiW-Music-Brain**: Python toolkit for music production intelligence with intent-based composition

**Merge Date**: December 3, 2025  
**Merge Strategy**: Preserve all code with conflict resolution via naming suffixes

---

## 📁 Repository Structure

### Original README Files
- [README_penta-core.md](README_penta-core.md) - Original penta-core documentation
- [README_music-brain.md](README_music-brain.md) - Original DAiW-Music-Brain documentation

### Development Roadmaps
- [ROADMAP_penta-core.md](ROADMAP_penta-core.md) - Penta Core Phase 3 implementation roadmap
- [DEVELOPMENT_ROADMAP_music-brain.md](DEVELOPMENT_ROADMAP_music-brain.md) - DAiW-Music-Brain development queue and phases

### Core Components

#### From penta-core:
- **C++ Core Engine** (`include/`, `src_penta-core/`, `external/`)
  - Harmony analysis (chord detection, scale detection, voice leading)
  - Groove analysis (onset detection, tempo estimation, rhythm quantization)
  - Real-time DSP with SIMD optimization
  - Performance monitoring and diagnostics
  
- **Python Bindings** (`bindings/`, `python/`)
  - pybind11 integration
  - High-level Python API
  
- **JUCE Plugin** (`plugins/`)
  - VST3 and AU formats
  - Real-time analysis display
  - OSC communication bridge
  
- **Build System**
  - `CMakeLists.txt` - CMake build configuration
  - `pyproject_penta-core.toml` - Python package configuration
  
- **Examples & Tests**
  - `examples_penta-core/` - Usage examples
  - `tests_penta-core/` - Unit and integration tests
  
- **Documentation** (`docs_penta-core/`)
  - Architecture design docs
  - Build instructions
  - Team documentation (Swift SDKs, C++, Rust DAW, etc.)

#### From DAiW-Music-Brain:
- **Music Brain Engine** (`music_brain/`)
  - Groove extraction and application
  - Chord and harmony analysis
  - Intent-based composition system
  - Audio feel analysis
  - Session management
  
- **Intent Schema System** (`music_brain/session/`)
  - Three-phase deep interrogation (Core Wound → Emotional Intent → Technical Implementation)
  - Rule-breaking engine for emotional impact
  - Teaching module
  
- **Knowledge Vault** (`vault/`)
  - Songwriting guides
  - Production workflows
  - Theory references (Obsidian-compatible)
  
- **CLI Tools**
  - `app.py` - Main application
  - `launcher.py` - Application launcher
  - `setup.py` - Package setup
  
- **MCP Integration** (`mcp_todo/`, `mcp_workstation/`)
  - MCP server implementations
  - Todo management tools
  
- **Examples & Tests**
  - `examples_music-brain/` - Music Brain examples
  - `tests_music-brain/` - Test suites
  
- **Tools** (`tools/`)
  - Audio cataloger
  - Utility scripts
  
- **Documentation** (`docs_music-brain/`)
  - Session summaries
  - Integration guides

#### GitHub Workflows & Agents
- `.github/workflows/` - CI/CD workflows from DAiW-Music-Brain
  - `ci.yml` - Continuous integration workflow
- `.github/agents/` - Automation agents from DAiW-Music-Brain
  - `my-agent.agent.md` - Custom agent configuration
- `.github/copilot-instructions.md` - GitHub Copilot configuration

#### Additional Assets
- **Emotion Data**: `angry.json`, `disgust.json`, `fear.json`, `happy.json`, `sad.json`, `surprise.json`, `blends.json`
- **Data Directory**: `data/` - Genre templates, scales database, etc.
- **macOS Support**: `macos/` - macOS-specific files
- **Nested Projects**: 
  - `DAiW-Music-Brain/` - Nested copy of original repo
  - `DAiW-Music-Brain 2/` - Secondary copy
  - `iDAW_Core/` - Core components
  - `penta_core_music-brain/` - Penta core from music-brain repo

---

## 🔀 File Conflicts & Resolution

### Configuration Files
| Original File | penta-core | DAiW-Music-Brain |
|--------------|------------|------------------|
| LICENSE | `LICENSE_penta-core` | `LICENSE_music-brain` |
| pyproject.toml | `pyproject_penta-core.toml` | `pyproject_music-brain.toml` |
| requirements.txt | `requirements_penta-core.txt` | `requirements_music-brain.txt` |
| .gitignore | `.gitignore_penta-core` | `.gitignore_music-brain` |

### Directory Conflicts
| Directory | penta-core | DAiW-Music-Brain |
|-----------|------------|------------------|
| docs/ | `docs_penta-core/` | `docs_music-brain/` |
| examples/ | `examples_penta-core/` | `examples_music-brain/` |
| tests/ | `tests_penta-core/` | `tests_music-brain/` |
| src/ | `src_penta-core/` | (N/A - uses `music_brain/`) |

### Unique Files
Files that existed in only one repository were copied without suffix modification.

---

## 🎯 Integration Notes

### Architectural Complementarity

**penta-core** provides:
- High-performance C++ DSP engine
- Real-time audio processing
- Low-latency analysis
- DAW plugin integration (VST3/AU)
- SIMD-optimized algorithms

**DAiW-Music-Brain** provides:
- High-level Python API
- Intent-based composition
- Emotional mapping system
- CLI tools for music production
- Knowledge base and teaching modules

### Integration Opportunities

1. **Hybrid Architecture**: Use penta-core's C++ engine as the performance layer for DAiW-Music-Brain's Python tools
2. **Real-time Analysis**: Connect penta-core's real-time chord/groove detection to DAiW's intent processing
3. **Plugin Integration**: Package DAiW's intent system as presets for penta-core's JUCE plugin
4. **Teaching System**: Combine penta-core's technical capabilities with DAiW's teaching module
5. **MCP Integration**: Expose penta-core's C++ engine via MCP tools

### Potential Conflicts to Resolve

1. **Build System**: Need to unify CMake (penta-core) with Python setuptools (DAiW)
2. **Python Package Structure**: Both have Python components that may need namespace coordination
3. **Testing**: Two separate test suites need integration strategy
4. **Documentation**: Consolidate overlapping documentation topics

---

## 📋 Combined TODO Summary

### From penta-core (Phase 3 Roadmap):

**Current Status**: Phase 3.1 Complete ✅

**Next Phases**:
- **Phase 3.2** (2-3 weeks): Harmony module implementation
  - [ ] Chord analysis with 30+ templates
  - [ ] Scale detection (Krumhansl-Schmuckler algorithm)
  - [ ] Voice leading optimization
  - [ ] Target: <100μs latency, 90%+ accuracy

- **Phase 3.3** (2-3 weeks): Groove module implementation
  - [ ] Onset detection with FFT
  - [ ] Tempo estimation
  - [ ] Rhythm quantization
  - [ ] Target: <200μs latency, <2 BPM error

- **Phase 3.4** (1-2 weeks): Diagnostics & OSC
  - [ ] Performance monitoring
  - [ ] OSC communication
  - [ ] JUCE plugin completion

- **Phase 3.5** (2 weeks): Optimization & polish
  - [ ] SIMD optimizations
  - [ ] Comprehensive testing
  - [ ] Documentation

**Success Metrics**:
- Harmony: < 100μs @ 48kHz/512 samples
- Groove: < 200μs @ 48kHz/512 samples
- Total CPU: < 5% on modern hardware
- Zero RT allocations
- > 90% chord detection accuracy

### From DAiW-Music-Brain:

**Current Status**: Phase 1 at 92%

**Priority Queue**:

1. **Priority 1**: Finish CLI Implementation (2 hours)
   - [ ] Complete CLI command wrappers
   - [ ] Add missing commands (generate, diagnose, reharm, intent, teach)
   - [ ] Comprehensive test suite
   - [ ] Target: Phase 1 → 100%

2. **Priority 2**: Expand MCP Tool Coverage (1 week)
   - [ ] Scale from 3 tools to 22+ tools
   - [ ] Add harmony tools (6 total)
   - [ ] Add groove tools (5 total)
   - [ ] Add intent tools (4 total)
   - [ ] Add audio analysis tools (4 total)
   - [ ] Add teaching tools (3 total)

3. **Priority 3**: Audio Analysis Implementation (1 week)
   - [ ] Complete AudioAnalyzer class
   - [ ] Implement ChordDetector
   - [ ] Implement FrequencyAnalyzer
   - [ ] Integrate with librosa
   - [ ] CLI command: `daiw analyze-audio`

**Phase Targets**:
- **Phase 1**: Complete by end of week (100%)
- **Phase 2**: Complete by next month (0% → 50%)
  - MCP tool expansion
  - Audio analysis
  - Desktop app integration
- **Phase 3**: Q1 2026
  - Real-time MIDI processing
  - DAW plugin integration
  - Machine learning integration

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- CMake 3.15+ (for penta-core C++ build)
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- JUCE framework (for plugin build)

### Installation

#### Option 1: Python-only (DAiW-Music-Brain)
```bash
# Clone repository
git clone https://github.com/sburdges-eng/iDAW.git
cd iDAW

# Install Python package
pip install -e .
```

#### Option 2: Full Build (penta-core + DAiW-Music-Brain)
```bash
# Clone repository
git clone https://github.com/sburdges-eng/iDAW.git
cd iDAW

# Build C++ library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Install Python modules
cd ..
pip install -e .
```

### Quick Test
```bash
# Test penta-core Python bindings
python demo_penta-core.py

# Test DAiW CLI
daiw --help

# Test intent-based generation
daiw intent new --title "My Song" --output my_intent.json
```

---

## 📚 Documentation

### Core Documentation
- [Penta Core README](README_penta-core.md) - C++ engine architecture
- [DAiW-Music-Brain README](README_music-brain.md) - Intent system and CLI
- [QUICKSTART_penta-core.md](QUICKSTART_penta-core.md) - Penta Core quick start
- [CLAUDE.md](CLAUDE.md) - DAiW-Music-Brain AI integration

### Technical Guides (penta-core)
- `docs_penta-core/PHASE3_DESIGN.md` - Phase 3 architecture
- `docs_penta-core/BUILD.md` - Build instructions
- `docs_penta-core/comprehensive-system-requirements.md` - System requirements (400+)
- `docs_penta-core/multi-agent-mcp-guide.md` - MCP architecture

### Songwriting & Production (DAiW-Music-Brain)
- `vault/Songwriting_Guides/song_intent_schema.md` - Intent schema guide
- `vault/Songwriting_Guides/rule_breaking_practical.md` - Rule-breaking techniques
- `vault/Songwriting_Guides/rule_breaking_masterpieces.md` - Examples from masters
- `vault/Production_Workflows/` - Production workflow guides

---

## 🔧 Development

### Building penta-core
```bash
cd iDAW
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest  # Run tests
```

### Running Tests
```bash
# penta-core tests
cd build && ctest --output-on-failure

# DAiW-Music-Brain tests
pytest tests_music-brain/ -v
```

### Development Workflow
1. Make changes to code
2. Run appropriate tests
3. Build if C++ changes
4. Test integration between components

---

## 🤝 Contributing

When contributing to this merged repository:

1. **Identify component**: Specify if changes affect penta-core, DAiW-Music-Brain, or integration
2. **Follow conventions**: Use existing code style from respective component
3. **Test thoroughly**: Run tests for affected components
4. **Document**: Update relevant README or documentation files
5. **Consider integration**: Think about how changes affect the merged architecture

---

## 📄 License

- **penta-core**: MIT License - see [LICENSE_penta-core](LICENSE_penta-core)
- **DAiW-Music-Brain**: MIT License - see [LICENSE_music-brain](LICENSE_music-brain)

Both projects are MIT licensed, allowing free use, modification, and distribution.

---

## 🎵 Philosophy

**penta-core**: "Professional-grade real-time music analysis with zero-latency guarantees"

**DAiW-Music-Brain**: "Interrogate Before Generate — The tool shouldn't finish art for people. It should make them braver."

**iDAW Combined**: Powerful real-time analysis meets intent-driven creativity for fearless music production.

---

## 📞 Support & Resources

### Original Repositories
- penta-core: https://github.com/sburdges-eng/penta-core
- DAiW-Music-Brain: https://github.com/sburdges-eng/DAiW-Music-Brain

### Documentation Locations
- Technical docs: `docs_penta-core/`
- Creative docs: `docs_music-brain/`, `vault/`
- Examples: `examples_penta-core/`, `examples_music-brain/`
- Tests: `tests_penta-core/`, `tests_music-brain/`

---

*Merged: December 3, 2025*  
*Last Updated: December 3, 2025*
