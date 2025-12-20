#!/bin/bash
# macOS Development Environment Setup Script
# For Kelly Music Brain / miDiKompanion

set -e

echo "🎵 Setting up miDiKompanion Development Environment for macOS..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check for required tools
echo "🔍 Checking prerequisites..."

# Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python ${PYTHON_VERSION} found${NC}"

# CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${YELLOW}⚠️  CMake not found. Installing via Homebrew...${NC}"
    if command -v brew &> /dev/null; then
        brew install cmake
    else
        echo -e "${RED}❌ Homebrew not found. Please install CMake manually${NC}"
        exit 1
    fi
fi
CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
echo -e "${GREEN}✅ CMake ${CMAKE_VERSION} found${NC}"

# Node.js
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠️  Node.js not found. Installing via Homebrew...${NC}"
    if command -v brew &> /dev/null; then
        brew install node
    else
        echo -e "${RED}❌ Homebrew not found. Please install Node.js manually${NC}"
        exit 1
    fi
fi
NODE_VERSION=$(node --version)
echo -e "${GREEN}✅ Node.js ${NODE_VERSION} found${NC}"

# Rust (for Tauri)
if ! command -v rustc &> /dev/null; then
    echo -e "${YELLOW}⚠️  Rust not found. Installing...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Rust ${RUST_VERSION} found${NC}"

# Homebrew (for macOS-specific dependencies)
if command -v brew &> /dev/null; then
    echo -e "${GREEN}✅ Homebrew found${NC}"
    
    # Install audio libraries if needed
    echo ""
    echo "📦 Installing audio libraries (if needed)..."
    brew install portaudio || echo "PortAudio may already be installed"
    brew install libsndfile || echo "libsndfile may already be installed"
else
    echo -e "${YELLOW}⚠️  Homebrew not found. Some audio libraries may need manual installation${NC}"
fi

echo ""
echo "🐍 Setting up Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install project dependencies
echo ""
echo "📦 Installing project dependencies..."
pip install -e .

# Install development dependencies
echo ""
echo "📦 Installing development dependencies..."
pip install -e ".[dev]"

# Verify installation
echo ""
echo "🧪 Verifying installation..."
python -c "import numpy, librosa, music21, fastapi; print('✅ Core dependencies imported successfully')" || {
    echo -e "${RED}❌ Failed to import core dependencies${NC}"
    exit 1
}

# Create necessary directories
echo ""
echo "📁 Creating workspace directories..."
mkdir -p data samples models outputs logs

# Check C++ build setup
echo ""
echo "🔨 Checking C++ build setup..."
if [ -f "src/CMakeLists.txt" ]; then
    echo -e "${GREEN}✅ CMakeLists.txt found${NC}"
    echo "   To build C++ components, run:"
    echo "   mkdir -p build && cd build && cmake .. && make"
else
    echo -e "${YELLOW}⚠️  CMakeLists.txt not found in src/${NC}"
fi

# Check frontend setup
echo ""
echo "🌐 Checking frontend setup..."
if [ -f "src/App.tsx" ]; then
    echo -e "${GREEN}✅ React frontend found${NC}"
    if [ -f "package.json" ]; then
        echo "   To set up frontend, run: npm install"
    else
        echo -e "${YELLOW}⚠️  package.json not found. Frontend may need separate setup${NC}"
    fi
fi

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Development environment setup complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run tests:"
echo "   pytest tests_music-brain/"
echo ""
echo "3. Start the API server:"
echo "   python -m music_brain.api"
echo "   # or use the script:"
echo "   ./scripts/start_api_server.sh"
echo ""
echo "4. Build C++ components (optional):"
echo "   mkdir -p build && cd build && cmake .. && make"
echo ""
echo "5. Set up frontend (if needed):"
echo "   npm install && npm run dev"
echo ""
echo "📚 For more information, see:"
echo "   - docs/COMPREHENSIVE_DEVELOPMENT_WORKFLOW.md"
echo "   - docs/BUILD.md"
echo ""
