# Penta Core - Build Instructions

## Prerequisites

### All Platforms

- CMake 3.20 or higher
- C++20 compatible compiler
- Python 3.8+ with development headers
- Git (for fetching dependencies)

### Platform-Specific

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake and Python (via Homebrew)
brew install cmake python@3.11
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install build-essential cmake python3-dev python3-pip git
```

#### Windows
```bash
# Install Visual Studio 2019 or later with C++ tools
# Install CMake from cmake.org
# Install Python from python.org
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/penta-core.git
cd penta-core
```

### 2. Build C++ Library and Python Bindings

```bash
# Create build directory
mkdir build && cd build

# Configure (Release build with all features)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DPENTA_BUILD_PYTHON_BINDINGS=ON \
         -DPENTA_BUILD_JUCE_PLUGIN=ON \
         -DPENTA_ENABLE_SIMD=ON

# Build (use -j for parallel compilation)
cmake --build . --config Release -j8

# Run tests
ctest --output-on-failure
```

### 3. Install Python Module

```bash
# Install to user site-packages
cmake --install . --prefix ~/.local

# Or install in development mode
pip install -e .
```

### 4. Run Examples

```bash
# Test harmony analysis
python examples/harmony_example.py

# Test groove analysis
python examples/groove_example.py

# Test full integration
python examples/integration_example.py
```

## Build Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Debug | Build type: Debug, Release, RelWithDebInfo |
| `PENTA_BUILD_PYTHON_BINDINGS` | ON | Build pybind11 Python module |
| `PENTA_BUILD_JUCE_PLUGIN` | ON | Build JUCE VST3/AU plugins |
| `PENTA_BUILD_TESTS` | ON | Build unit tests |
| `PENTA_ENABLE_SIMD` | ON | Enable SIMD optimizations (AVX2) |
| `PENTA_ENABLE_LTO` | OFF | Enable link-time optimization |

### Example Configurations

#### Development Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
      -DPENTA_BUILD_TESTS=ON \
      -DPENTA_ENABLE_SIMD=OFF
```

#### Release Build (Maximum Performance)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPENTA_ENABLE_SIMD=ON \
      -DPENTA_ENABLE_LTO=ON \
      -DCMAKE_CXX_FLAGS="-march=native"
```

#### Python-Only Build
```bash
cmake -B build -DPENTA_BUILD_PYTHON_BINDINGS=ON \
      -DPENTA_BUILD_JUCE_PLUGIN=OFF \
      -DPENTA_BUILD_TESTS=OFF
```

#### Plugin-Only Build
```bash
cmake -B build -DPENTA_BUILD_PYTHON_BINDINGS=OFF \
      -DPENTA_BUILD_JUCE_PLUGIN=ON \
      -DPENTA_BUILD_TESTS=OFF
```

## Advanced Configuration

### Custom Install Prefix

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --install build
```

### Cross-Compilation

#### macOS Universal Binary (x86_64 + ARM64)
```bash
cmake -B build -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
cmake --build build
```

#### Linux ARM64
```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=arm64-toolchain.cmake
cmake --build build
```

### Using System Libraries

By default, dependencies are fetched via CMake FetchContent. To use system libraries:

```bash
# Install dependencies first
# macOS
brew install juce pybind11

# Linux
sudo apt install libjuce-dev pybind11-dev

# Configure with system libraries
cmake -B build -DFETCHCONTENT_FULLY_DISCONNECTED=ON
```

## Building Specific Targets

### C++ Library Only
```bash
cmake --build build --target penta_core
```

### Python Bindings Only
```bash
cmake --build build --target penta_core_native
```

### JUCE Plugin Only
```bash
cmake --build build --target PentaCorePlugin_VST3
cmake --build build --target PentaCorePlugin_AU
cmake --build build --target PentaCorePlugin_Standalone
```

### Tests Only
```bash
cmake --build build --target penta_tests
./build/tests/penta_tests
```

## Plugin Installation

### macOS

```bash
# VST3
cp -r build/plugins/PentaCorePlugin_artefacts/Release/VST3/PentaCorePlugin.vst3 \
      ~/Library/Audio/Plug-Ins/VST3/

# AU
cp -r build/plugins/PentaCorePlugin_artefacts/Release/AU/PentaCorePlugin.component \
      ~/Library/Audio/Plug-Ins/Components/

# Standalone
open build/plugins/PentaCorePlugin_artefacts/Release/Standalone/PentaCorePlugin.app
```

### Linux

```bash
# VST3
cp -r build/plugins/PentaCorePlugin_artefacts/Release/VST3/PentaCorePlugin.vst3 \
      ~/.vst3/

# Standalone
./build/plugins/PentaCorePlugin_artefacts/Release/Standalone/PentaCorePlugin
```

### Windows

```powershell
# VST3
copy build\plugins\PentaCorePlugin_artefacts\Release\VST3\PentaCorePlugin.vst3 ^
     %CommonProgramFiles%\VST3\

# Standalone
start build\plugins\PentaCorePlugin_artefacts\Release\Standalone\PentaCorePlugin.exe
```

## Troubleshooting

### CMake Can't Find Python

```bash
# Specify Python explicitly
cmake -B build -DPython3_EXECUTABLE=/usr/bin/python3.11
```

### Missing pybind11

```bash
# Install via pip
pip install pybind11

# Or let CMake fetch it (default)
cmake -B build  # FetchContent will download pybind11
```

### SIMD Compilation Errors

If you get AVX2-related errors:

```bash
# Disable SIMD
cmake -B build -DPENTA_ENABLE_SIMD=OFF
```

### JUCE Build Errors

```bash
# Fetch latest JUCE
rm -rf build/_deps/juce-*
cmake -B build  # Will re-fetch JUCE
```

### Link Errors on Linux

```bash
# Install missing libraries
sudo apt install libasound2-dev libfreetype6-dev libx11-dev \
                 libxrandr-dev libxinerama-dev libxcursor-dev
```

## Development Workflow

### Incremental Builds

```bash
# Make changes to source files
# Rebuild only changed files
cmake --build build

# Rebuild specific target
cmake --build build --target penta_core
```

### Clean Build

```bash
# Clean build artifacts
cmake --build build --target clean

# Or remove build directory entirely
rm -rf build
cmake -B build
cmake --build build
```

### Rebuilding After CMakeLists.txt Changes

```bash
# Reconfigure
cmake -B build

# Rebuild
cmake --build build
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Install dependencies (Ubuntu)
      if: runner.os == 'Linux'
      run: |
        sudo apt update
        sudo apt install build-essential cmake python3-dev
    
    - name: Configure
      run: cmake -B build -DCMAKE_BUILD_TYPE=Release
    
    - name: Build
      run: cmake --build build --config Release
    
    - name: Test
      run: cd build && ctest --output-on-failure
```

## Performance Validation

### Benchmark Builds

```bash
# Build with optimizations and benchmarks
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPENTA_BUILD_BENCHMARKS=ON \
      -DPENTA_ENABLE_SIMD=ON \
      -DPENTA_ENABLE_LTO=ON

cmake --build build --target benchmarks

# Run benchmarks
./build/benchmarks/harmony_benchmark
./build/benchmarks/groove_benchmark
```

### Profile-Guided Optimization (Advanced)

```bash
# Step 1: Build with instrumentation
cmake -B build-pgo -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-fprofile-generate"
cmake --build build-pgo

# Step 2: Run representative workload
./build-pgo/examples/integration_example

# Step 3: Rebuild with profile data
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-fprofile-use"
cmake --build build
```

## Documentation Generation

### Doxygen (C++ API)

```bash
# Install Doxygen
brew install doxygen  # macOS
sudo apt install doxygen  # Linux

# Generate docs
doxygen Doxyfile

# Open docs
open docs/html/index.html
```

### Python API Docs

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Generate docs
cd docs/python
make html

# Open docs
open _build/html/index.html
```

## Support

For build issues, please:
1. Check this document first
2. Search existing GitHub issues
3. Create a new issue with:
   - OS and version
   - CMake output
   - Compiler version
   - Full error messages

## Next Steps

After successful build:
1. Run examples in `examples/` directory
2. Read `docs/PHASE3_DESIGN.md` for architecture overview
3. Check `docs/API.md` for API reference
4. Explore unit tests for usage examples
