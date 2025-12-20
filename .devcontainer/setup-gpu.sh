#!/bin/bash
set -e

echo "🎮 Setting up GPU-Accelerated Audio Workspace..."

# Verify GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No GPU detected (will run on CPU)"
fi

# Update system
sudo apt-get update -qq

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libasound2-dev \
    libjack-jackd2-dev \
    libfreetype6-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libwebkit2gtk-4.0-dev \
    libssl-dev \
    curl \
    git \
    wget

# Install Rust (for Tauri)
if ! command -v rustc &> /dev/null; then
    echo "🦀 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Install Python dependencies
echo "🐍 Installing Python packages..."

# PyTorch with CUDA (if GPU available)
if command -v nvidia-smi &> /dev/null; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision torchaudio
fi

# Audio processing libraries
pip install \
    librosa \
    soundfile \
    audioread \
    resampy \
    numba \
    scipy \
    numpy

# GPU acceleration (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 Installing GPU acceleration libraries..."
    pip install cupy-cuda12x || echo "⚠️  CuPy installation failed, continuing without GPU acceleration"
fi

# Music AI libraries
pip install \
    music21 \
    pretty_midi \
    mido \
    python-rtmidi \
    essentia-tensorflow || echo "⚠️  Essentia optional, continuing..."

# ML for audio
pip install \
    madmom \
    openl3 \
    crepe

# Web framework
pip install \
    fastapi \
    uvicorn \
    pydantic \
    python-dotenv

# Development tools
pip install \
    black \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    ipython \
    jupyter

# Install project in development mode
if [ -f "/workspace/pyproject.toml" ]; then
    echo "📚 Installing project in development mode..."
    pip install -e /workspace
fi

# Create workspace directories
mkdir -p /workspace/{data,samples,models,outputs}

echo ""
echo "✅ GPU workspace ready!"
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
echo ""
echo "📝 Next steps:"
echo "  1. Test GPU: python -c 'import torch; print(torch.cuda.is_available())'"
echo "  2. Run audio analysis: python scripts/gpu_audio_batch.py"
echo "  3. Start API server: python -m music_brain.api"

