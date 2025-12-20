# Integration Guide - iDAW Merged Repository

This guide provides instructions for working with the merged iDAW repository that combines penta-core and DAiW-Music-Brain.

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sburdges-eng/iDAW.git
cd iDAW
```

### 2. Validate the Merge
```bash
python3 validate_merge.py
```
Expected output: `✅ All validation checks passed!`

## Installation Options

### Option A: Python-Only (DAiW-Music-Brain)

For music production, intent-based composition, and MCP servers without C++ compilation:

```bash
# Install core dependencies
pip install -e .

# Or install with optional features
pip install -e ".[audio,theory,ui,mcp,all]"
```

**Available CLI commands:**
- `daiw` - Main DAiW-Music-Brain CLI
- `mcp-todo` - MCP TODO management
- `mcp-workstation` - MCP workstation tools

### Option B: Full Build (penta-core + DAiW-Music-Brain)

For real-time C++ DSP engine with Python bindings:

```bash
# 1. Build C++ library with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 2. Run C++ tests
ctest --output-on-failure

# 3. Install Python packages
cd ..
pip install -e .
```

**Prerequisites:**
- CMake 3.15+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- JUCE framework (for plugin build)
- pybind11 (for Python bindings)

## Repository Organization

### Understanding the Structure

The repository preserves both original codebases with clear naming:

```
iDAW/
├── README.md                              # Main merged documentation
├── README_penta-core.md                   # Original penta-core README
├── README_music-brain.md                  # Original DAiW-Music-Brain README
├── MERGE_SUMMARY.md                       # Detailed merge documentation
│
├── Configuration Files (Merged)
│   ├── pyproject.toml                     # Unified Python package config
│   ├── requirements.txt                   # Combined dependencies
│   ├── pyproject_penta-core.toml          # Original penta-core config
│   ├── pyproject_music-brain.toml         # Original music-brain config
│   ├── requirements_penta-core.txt        # Original penta-core deps
│   └── requirements_music-brain.txt       # Original music-brain deps
│
├── Penta Core (C++ Engine)
│   ├── include/                           # C++ public headers
│   ├── src_penta-core/                    # C++ implementation
│   ├── bindings/                          # pybind11 Python bindings
│   ├── python/penta_core/                 # Python package
│   ├── plugins/                           # JUCE VST3/AU plugins
│   ├── external/                          # External dependencies
│   ├── docs_penta-core/                   # Technical documentation
│   ├── examples_penta-core/               # C++ examples
│   └── tests_penta-core/                  # C++ tests
│
├── DAiW-Music-Brain (Python Engine)
│   ├── music_brain/                       # Main Python package
│   ├── mcp_todo/                          # MCP TODO server
│   ├── mcp_workstation/                   # MCP workstation
│   ├── vault/                             # Knowledge base
│   ├── tools/                             # Utility tools
│   ├── data/                              # Data files
│   ├── docs_music-brain/                  # Documentation
│   ├── examples_music-brain/              # Python examples
│   └── tests_music-brain/                 # Python tests
│
└── GitHub Configuration
    └── .github/
        ├── workflows/ci.yml               # CI/CD workflow
        ├── agents/my-agent.agent.md       # Custom agent
        └── copilot-instructions.md        # Copilot config
```

## Working with Both Systems

### Scenario 1: Using DAiW-Music-Brain Only

```bash
# Install without C++ build
pip install -e .

# Use intent-based composition
daiw intent new --title "My Song"

# Extract groove from MIDI
daiw extract drums.mid

# Run MCP server
mcp-todo-server
```

### Scenario 2: Using Penta Core Only

```bash
# Build C++ library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Use Python bindings
python3
>>> from penta_core import PentaCore
>>> penta = PentaCore(sample_rate=48000.0)
>>> # Use real-time analysis...
```

### Scenario 3: Integrated Workflow

```bash
# Build everything
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..
pip install -e ".[all]"

# Use both systems together
python3
>>> from penta_core import PentaCore
>>> from music_brain.session import process_intent
>>> # Combine real-time C++ analysis with intent-based composition
```

## Development Workflows

### For Penta Core Development

```bash
# Build in debug mode
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON
cmake --build . -j

# Run specific tests
ctest -R harmony_test -V

# Profile performance
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build . -j
# Use Instruments (macOS) or perf (Linux) for profiling
```

### For DAiW-Music-Brain Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests_music-brain/ -v

# Format code
black music_brain/ --line-length 100

# Type checking
mypy music_brain/
```

### For Integration Development

```bash
# Build C++ and install Python
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j
cd ..
pip install -e ".[all,dev]"

# Run all tests
ctest --test-dir build
pytest tests_music-brain/ tests_penta-core/
```

## Common Tasks

### Adding Dependencies

**Python dependencies:**
Edit `requirements.txt` and `pyproject.toml`

**C++ dependencies:**
Edit `CMakeLists.txt` and update `external/` directory

### Building Documentation

```bash
# Penta Core (Doxygen)
cd docs_penta-core
doxygen Doxyfile

# DAiW-Music-Brain (Sphinx or MkDocs)
cd docs_music-brain
# Follow existing documentation build process
```

### Running CI Locally

```bash
# The GitHub workflow is in .github/workflows/ci.yml
# To test locally, run the same commands:

# Python tests
pytest tests_music-brain/ -v

# C++ build and tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure
```

## Integration Opportunities

### 1. Hybrid Real-Time Analysis

**Goal:** Use penta-core's C++ engine for real-time chord/groove detection, feed results to DAiW's intent processor.

**Implementation:**
```python
from penta_core import PentaCore
from music_brain.session import process_intent, analyze_emotional_content

# Real-time analysis
penta = PentaCore(sample_rate=48000.0)
state = penta.get_state()

# Feed to intent system
emotion = analyze_emotional_content(state['chord'])
```

### 2. Plugin Integration

**Goal:** Package DAiW's intent system as presets for penta-core's VST3/AU plugin.

**Location:** `plugins/` directory for JUCE plugin code

### 3. MCP Tools for C++ Engine

**Goal:** Expose penta-core's C++ analysis via MCP protocol.

**Implementation:** Create new MCP server in `mcp_penta_core/`

### 4. Unified Teaching System

**Goal:** Combine technical analysis (penta-core) with music theory teaching (DAiW).

**Location:** Enhance `music_brain/session/teaching.py` with penta-core analysis

## Troubleshooting

### Python Import Errors

```bash
# If music_brain import fails:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# If penta_core import fails:
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"
```

### CMake Build Errors

```bash
# Clear build directory
rm -rf build
mkdir build && cd build

# Verbose build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON
cmake --build . -j --verbose
```

### Missing Dependencies

```bash
# Install all Python dependencies
pip install -e ".[all,dev]"

# Install C++ build tools
# Ubuntu/Debian:
sudo apt-get install cmake g++ libjack-dev

# macOS:
brew install cmake
```

## Next Steps

1. **Review Documentation:**
   - Read `README_penta-core.md` for C++ engine details
   - Read `README_music-brain.md` for Python tools
   - Check `MERGE_SUMMARY.md` for merge details

2. **Choose Your Path:**
   - Python-only: Focus on `music_brain/` package
   - C++ development: Focus on `include/` and `src_penta-core/`
   - Integration: Work with both systems

3. **Run Examples:**
   - Penta Core: `examples_penta-core/`
   - Music Brain: `examples_music-brain/`

4. **Contribute:**
   - Follow existing code style in each component
   - Run tests before submitting changes
   - Update documentation for new features

## Resources

### Documentation
- Main README: [README.md](README.md)
- Penta Core: [README_penta-core.md](README_penta-core.md)
- Music Brain: [README_music-brain.md](README_music-brain.md)
- Merge Details: [MERGE_SUMMARY.md](MERGE_SUMMARY.md)

### Roadmaps
- Penta Core: [ROADMAP_penta-core.md](ROADMAP_penta-core.md)
- Music Brain: [DEVELOPMENT_ROADMAP_music-brain.md](DEVELOPMENT_ROADMAP_music-brain.md)

### Technical Guides
- Penta Core Docs: `docs_penta-core/`
- Music Brain Docs: `docs_music-brain/`
- Knowledge Vault: `vault/`

### Support
- GitHub Issues: https://github.com/sburdges-eng/iDAW/issues
- Original Repos:
  - penta-core: https://github.com/sburdges-eng/penta-core
  - DAiW-Music-Brain: https://github.com/sburdges-eng/DAiW-Music-Brain

---

*Last Updated: December 3, 2025*
