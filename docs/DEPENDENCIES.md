# External Dependencies Setup

This guide helps you set up the required external dependencies for Kelly.

## JUCE Framework

Kelly uses JUCE 7 for audio processing and plugin development.

```bash
cd external
git clone --depth 1 --branch 7.0.x https://github.com/juce-framework/JUCE.git
```

## Catch2 Testing Framework

For C++ unit testing:

```bash
cd external
git clone --depth 1 --branch v3.x https://github.com/catchorg/Catch2.git
```

## Qt 6

### Ubuntu/Debian
```bash
sudo apt-get install qt6-base-dev qt6-multimedia-dev
```

### macOS
```bash
brew install qt@6
```

### Windows
Download and install from: https://www.qt.io/download-qt-installer

## Audio Drivers

### ASIO (Windows)
Download ASIO SDK from Steinberg: https://www.steinberg.net/developers/

### JACK (Linux)
```bash
sudo apt-get install libjack-jackd2-dev
```

### CoreAudio (macOS)
Included with Xcode.

## CMake

Requires CMake 3.27 or higher.

### Ubuntu/Debian
```bash
sudo apt-get install cmake ninja-build
```

### macOS
```bash
brew install cmake ninja
```

### Windows
Download from: https://cmake.org/download/

## Python Dependencies

All Python dependencies are managed via pip:

```bash
pip install -e ".[dev]"
```

This installs:
- music21 (music theory)
- librosa (audio analysis)
- mido (MIDI processing)
- typer (CLI framework)
- pytest (testing)
- black, ruff, mypy (code quality)

## Optional: Tracy Profiler

For performance profiling:

```bash
cd external
git clone https://github.com/wolfpld/tracy.git
```

Enable with CMake option:
```bash
cmake -B build -DENABLE_TRACY=ON
```

## Directory Structure

After setup, your external directory should look like:

```
external/
├── JUCE/
├── Catch2/
└── tracy/ (optional)
```

## Troubleshooting

### JUCE not found
Ensure JUCE is in `external/JUCE` or update CMakeLists.txt to point to your JUCE installation.

### Qt not found
Set `CMAKE_PREFIX_PATH` to your Qt installation:
```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/qt6
```

### ALSA errors (Linux)
```bash
sudo apt-get install libasound2-dev
```
