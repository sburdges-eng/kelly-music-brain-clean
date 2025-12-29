# Running All Components

This guide shows you how to run all components of the Kelly project.

## 🎯 Quick Start - Run Everything

### Option 1: Use the Run Script
```bash
cd "/Volumes/Extreme SSD/kelly-project"
./scripts/run_all.sh
```

### Option 2: Manual Setup (Multiple Terminals)

You'll need **4 terminal windows** to run all components simultaneously:

---

## 📋 Component Setup & Run Instructions

### 1. 🧠 Brain (Python) - Music Brain API

**Purpose**: Python ML & Music Brain API server

**Setup**:
```bash
cd "/Volumes/Extreme SSD/kelly-project"

# Activate virtual environment (if exists)
source .venv/bin/activate  # or: python3 -m venv .venv && source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -e ".[dev]"
```

**Run**:
```bash
# Option 1: Use the startup script (recommended)
./scripts/start_api_server.sh

# Option 2: Direct command
python -m music_brain.api

# Option 3: With custom port
PORT=8080 ./scripts/start_api_server.sh
```

**Access**:
- API: http://127.0.0.1:8000
- Interactive Docs: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

**Status Check**:
```bash
curl http://127.0.0.1:8000/health
```

---

### 2. 🎵 Audio Engine (C++) - C++ Audio/DSP

**Purpose**: C++ audio processing and DSP engine

**Setup**:
```bash
cd "/Volumes/Extreme SSD/kelly-project/audio-engine-cpp"

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build
make -j8  # Use -j8 for parallel build (adjust based on CPU cores)
```

**Run**:
```bash
# Run tests (if available)
cd build
ctest

# Or run specific executable (if built)
./audio_engine_test  # Example - actual name depends on CMake config
```

**Build Options**:
```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build
cmake -DCMAKE_BUILD_TYPE=Release ..

# With specific compiler
cmake -DCMAKE_CXX_COMPILER=clang++ ..
```

---

### 3. 🔌 Plugin (JUCE) - JUCE Audio Plugin

**Purpose**: JUCE audio plugin for DAW integration

**Setup**:
```bash
cd "/Volumes/Extreme SSD/kelly-project/plugin-juce"

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build
make -j8
```

**Run**:
```bash
# Build the plugin
cd build
make

# The plugin will be built in the build directory
# Load it in your DAW (Ableton, Logic, etc.)
```

**Testing**:
```bash
# Run plugin tests (if configured)
cd build
ctest
```

---

### 4. 🖥️ Desktop App - React + Tauri

**Purpose**: Desktop UI application

**Setup**:
```bash
cd "/Volumes/Extreme SSD/kelly-project"

# Install Node.js dependencies (if package.json exists at root)
npm install

# Or if desktop-app has its own package.json
cd desktop-app
npm install
```

**Run**:
```bash
# Option 1: Tauri dev mode (full desktop app)
npm run tauri dev

# Option 2: Web dev mode (UI only, no Tauri)
npm run dev

# Option 3: Build for production
npm run tauri build
```

**Note**: The desktop app connects to the Music Brain API at `http://127.0.0.1:8000`

---

### 5. 📦 Shared Data

**Purpose**: Shared configuration and data files

**No runtime required** - This is a data directory containing:
- Emotion thesaurus data
- Chord progression data
- Groove templates
- Scale/mode data
- Model definitions

**Access**: Components read from this directory as needed.

---

### 6. 📚 Docs

**Purpose**: Documentation

**No runtime required** - Contains project documentation.

**View**:
```bash
cd "/Volumes/Extreme SSD/kelly-project/docs"
# Open any .md file in your editor or markdown viewer
```

---

### 7. 🔗 Integration

**Purpose**: Integration bridges between components

**Components**:
- `bridges/api_bridge/` - HTTP API bridge
- `bridges/python_cpp_bridge/` - Python ↔ C++ bridge
- `bridges/osc_bridge/` - OSC communication
- `schemas/` - Shared schemas
- `protocols/` - Communication protocols

**Setup** (if needed):
```bash
cd "/Volumes/Extreme SSD/kelly-project/integration"

# Python bridges
cd bridges/python_cpp_bridge
pip install -e .  # if setup.py exists

# API bridge (usually runs as part of brain-python)
# OSC bridge (may need separate setup)
```

---

## 🚀 Full Stack Development Workflow

### Terminal 1: Python API Server
```bash
cd "/Volumes/Extreme SSD/kelly-project"
source .venv/bin/activate
./scripts/start_api_server.sh
```

### Terminal 2: Desktop App (Tauri)
```bash
cd "/Volumes/Extreme SSD/kelly-project"
npm run tauri dev
```

### Terminal 3: C++ Audio Engine (if developing)
```bash
cd "/Volumes/Extreme SSD/kelly-project/audio-engine-cpp/build"
make -j8
# Run tests or development tools
```

### Terminal 4: Plugin Development (if needed)
```bash
cd "/Volumes/Extreme SSD/kelly-project/plugin-juce/build"
make -j8
```

---

## 🧪 Testing All Components

### Python Tests
```bash
cd "/Volumes/Extreme SSD/kelly-project"
source .venv/bin/activate
pytest tests_music-brain/ -v
```

### C++ Tests
```bash
cd "/Volumes/Extreme SSD/kelly-project/audio-engine-cpp/build"
ctest -V
```

### Integration Tests
```bash
# Test API endpoints
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/emotions

# Test desktop app connection
# (Open desktop app and verify it connects to API)
```

---

## 🔧 Troubleshooting

### Python API Not Starting
```bash
# Check if port is in use
lsof -i :8000

# Kill existing process
kill -9 $(lsof -t -i:8000)

# Reinstall dependencies
pip install -e ".[dev]"
```

### C++ Build Fails
```bash
# Clean and rebuild
cd audio-engine-cpp
rm -rf build
mkdir build && cd build
cmake ..
make -j8
```

### Desktop App Not Connecting
- Ensure Python API is running on port 8000
- Check API status: `curl http://127.0.0.1:8000/health`
- Verify Tauri config points to correct API URL

### Missing Dependencies
```bash
# Python
pip install -e ".[dev]"

# Node.js
npm install

# C++ (macOS)
brew install cmake
```

---

## 📊 Component Dependencies

```
Desktop App ──HTTP──> Brain (Python) ──pybind11──> Audio Engine (C++)
     │                    │                              │
     │                    └──────────> Shared Data <─────┘
     │
     └──OSC──> Plugin (JUCE) <──C++──> Audio Engine (C++)
```

---

## 🎯 Recommended Development Order

1. **Start Python API** (required for desktop app)
2. **Start Desktop App** (main UI)
3. **Build C++ Engine** (if developing audio features)
4. **Build Plugin** (if developing DAW integration)

---

**Last Updated**: December 2024

