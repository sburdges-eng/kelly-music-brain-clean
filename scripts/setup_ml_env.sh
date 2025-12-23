#!/bin/bash
# =============================================================================
# Kelly ML Environment Setup Script
# =============================================================================
# Sets up a Mac-optimized Python environment for ML training
#
# Usage:
#   chmod +x scripts/setup_ml_env.sh
#   ./scripts/setup_ml_env.sh
#
# Options:
#   --full     Install all optional dependencies
#   --minimal  Install only core dependencies
#   --clean    Remove existing venv before creating new one
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"

# Parse arguments
INSTALL_MODE="standard"
CLEAN_INSTALL=false

for arg in "$@"; do
    case $arg in
        --full)
            INSTALL_MODE="full"
            shift
            ;;
        --minimal)
            INSTALL_MODE="minimal"
            shift
            ;;
        --clean)
            CLEAN_INSTALL=true
            shift
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Kelly ML Environment Setup                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Detect system
echo -e "${YELLOW}→ Detecting system...${NC}"
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" != "Darwin" ]; then
    echo -e "${RED}✗ This script is designed for macOS${NC}"
    exit 1
fi

if [ "$ARCH" = "arm64" ]; then
    echo -e "${GREEN}✓ Apple Silicon detected (M1/M2/M3)${NC}"
    echo -e "  Metal Performance Shaders (MPS) backend will be available"
    CHIP="apple_silicon"
else
    echo -e "${YELLOW}⚠ Intel Mac detected${NC}"
    echo -e "  Training will use CPU only (slower)"
    CHIP="intel"
fi

# Check Python
echo ""
echo -e "${YELLOW}→ Checking Python installation...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${RED}✗ Python 3.10+ required (found $PYTHON_VERSION)${NC}"
        echo -e "  Install with: brew install python@3.11"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo -e "  Install with: brew install python@3.11"
    exit 1
fi

# Clean existing venv if requested
if [ "$CLEAN_INSTALL" = true ] && [ -d "$VENV_DIR" ]; then
    echo ""
    echo -e "${YELLOW}→ Removing existing virtual environment...${NC}"
    rm -rf "$VENV_DIR"
    echo -e "${GREEN}✓ Removed $VENV_DIR${NC}"
fi

# Create virtual environment
echo ""
echo -e "${YELLOW}→ Creating virtual environment...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists at $VENV_DIR${NC}"
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Created virtual environment at $VENV_DIR${NC}"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Activated virtual environment${NC}"

# Upgrade pip
echo ""
echo -e "${YELLOW}→ Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Install dependencies based on mode
echo ""
echo -e "${YELLOW}→ Installing dependencies (mode: $INSTALL_MODE)...${NC}"

case $INSTALL_MODE in
    minimal)
        echo "Installing minimal dependencies..."
        pip install numpy scipy librosa soundfile torch torchaudio tqdm pyyaml
        ;;
    full)
        echo "Installing all dependencies..."
        pip install -r "$PROJECT_ROOT/requirements.txt"
        # Optional extras
        pip install openai-whisper pedalboard
        ;;
    *)
        echo "Installing standard dependencies..."
        pip install -r "$PROJECT_ROOT/requirements.txt"
        ;;
esac

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Verify PyTorch installation
echo ""
echo -e "${YELLOW}→ Verifying PyTorch installation...${NC}"

python3 << 'EOF'
import torch
import sys

print(f"  PyTorch version: {torch.__version__}")
print(f"  Python version: {sys.version.split()[0]}")

# Check backends
backends = []
if torch.backends.mps.is_available():
    backends.append("MPS (Metal)")
if torch.cuda.is_available():
    backends.append("CUDA")
backends.append("CPU")

print(f"  Available backends: {', '.join(backends)}")

# Test MPS if available
if torch.backends.mps.is_available():
    try:
        x = torch.randn(10, device="mps")
        y = x * 2
        print("  MPS test: ✓ Working")
    except Exception as e:
        print(f"  MPS test: ✗ Failed ({e})")
EOF

# Create necessary directories
echo ""
echo -e "${YELLOW}→ Creating directory structure...${NC}"

# Audio data goes to external SSD for space/performance
AUDIO_DATA_ROOT="/Volumes/Extreme SSD"
if [ -d "$AUDIO_DATA_ROOT" ]; then
    echo -e "${GREEN}✓ External SSD detected: $AUDIO_DATA_ROOT${NC}"
    mkdir -p "$AUDIO_DATA_ROOT/kelly-audio-data/raw"
    mkdir -p "$AUDIO_DATA_ROOT/kelly-audio-data/processed"
    mkdir -p "$AUDIO_DATA_ROOT/kelly-audio-data/downloads"
    mkdir -p "$AUDIO_DATA_ROOT/kelly-audio-data/cache"
    
    # Create symlink in project if not exists
    if [ ! -L "$PROJECT_ROOT/data/audio" ]; then
        ln -sf "$AUDIO_DATA_ROOT/kelly-audio-data" "$PROJECT_ROOT/data/audio"
        echo -e "${GREEN}✓ Symlinked data/audio → $AUDIO_DATA_ROOT/kelly-audio-data${NC}"
    fi
else
    echo -e "${YELLOW}⚠ External SSD not found at $AUDIO_DATA_ROOT${NC}"
    echo -e "  Audio data will be stored locally in data/audio"
    mkdir -p "$PROJECT_ROOT/data/audio/raw"
    mkdir -p "$PROJECT_ROOT/data/audio/processed"
    mkdir -p "$PROJECT_ROOT/data/audio/downloads"
fi

mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/data/splits"
mkdir -p "$PROJECT_ROOT/logs/training"
mkdir -p "$PROJECT_ROOT/checkpoints"
mkdir -p "$PROJECT_ROOT/configs"
mkdir -p "$PROJECT_ROOT/docs/model_cards"

echo -e "${GREEN}✓ Directory structure created${NC}"

# Create activation script
echo ""
echo -e "${YELLOW}→ Creating activation helper...${NC}"

cat > "$PROJECT_ROOT/activate_ml.sh" << 'ACTIVATE_SCRIPT'
#!/bin/bash
# Quick activation script for Kelly ML environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
echo "Kelly ML environment activated"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
ACTIVATE_SCRIPT

chmod +x "$PROJECT_ROOT/activate_ml.sh"
echo -e "${GREEN}✓ Created activate_ml.sh${NC}"

# Summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Setup Complete!                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "To activate the environment:"
echo -e "  ${GREEN}source venv/bin/activate${NC}"
echo -e "  or"
echo -e "  ${GREEN}source activate_ml.sh${NC}"
echo ""
echo -e "To start training:"
echo -e "  ${GREEN}python scripts/train.py --list${NC}"
echo -e "  ${GREEN}python scripts/train.py --model emotion_recognizer --epochs 50${NC}"
echo ""
echo -e "Hardware: ${YELLOW}$CHIP${NC}"
if [ "$CHIP" = "apple_silicon" ]; then
    echo -e "  Use device='mps' in PyTorch for GPU acceleration"
else
    echo -e "  Training will use CPU (consider smaller batch sizes)"
fi
echo ""

