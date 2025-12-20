# GPU-Accelerated Development Environment

This devcontainer configuration sets up a GPU-enabled workspace for audio/music AI development.

## Features

- **NVIDIA CUDA 12.2** support
- **Python 3.11** with audio processing libraries
- **Rust** toolchain for Tauri development
- **C++ build tools** for JUCE/audio engine
- **GPU acceleration** for ML and audio processing

## Quick Start

### Option 1: GitHub Codespaces (Recommended)

```bash
# Create GPU-enabled codespace
gh codespace create \
  --repo sburdges-eng/miDiKompanion \
  --machine gpuEnabled \
  --devcontainer-path .devcontainer/devcontainer.json

# Connect to codespace
gh codespace ssh

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Option 2: Local Docker (with NVIDIA GPU)

```bash
# Requires: Docker Desktop with GPU support
# Or: nvidia-docker2 installed

docker build -t kelly-music-brain-gpu -f .devcontainer/Dockerfile .

docker run --gpus all -it \
  -v $(pwd):/workspace \
  kelly-music-brain-gpu
```

## Usage

### GPU-Accelerated Audio Processing

```bash
# Batch process audio samples
python scripts/gpu_audio_batch.py \
  --input-dir /workspace/samples \
  --output /workspace/outputs/analysis.json

# Expected speedup: 10-50x vs CPU
```

### Development Workflow

1. **Local Cursor (K2 Agents)** - Code generation, architecture, docs
2. **GPU Codespace** - Heavy compute, ML training, batch processing
3. **Hybrid** - Use both as needed

## Cost Optimization

- **GPU Codespace**: ~$1.18/hour (T4) or $3.60/hour (V100)
- **Use only when needed**: Shut down when not processing
- **Estimated monthly**: $20-50 for GPU compute

## Next Steps

1. Create GPU codespace: `gh codespace create --machine gpuEnabled`
2. Test GPU: Run `scripts/gpu_audio_batch.py`
3. Use K2 agents: See `.devcontainer/AGENT_PROMPTS.md`

