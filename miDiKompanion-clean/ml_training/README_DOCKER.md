# Docker Usage Guide for miDiKompanion ML Training

## Prerequisites

### 1. Install Docker

```bash
# macOS (via Homebrew)
brew install --cask docker

# Or download from: https://www.docker.com/products/docker-desktop
```

### 2. Install NVIDIA Docker (for GPU support)

```bash
# Linux only - macOS doesn't support GPU passthrough
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Note**: If you're on macOS, you'll need to use Google Colab or AWS for GPU training. Docker Desktop on macOS doesn't support GPU passthrough.

---

## Building Docker Images

### Initial Build (First Time)

The first build will download base images (~3-4GB) and may take 10-30 minutes depending on your internet connection.

#### Option 1: Comprehensive Build Script (Recommended)

```bash
cd ml_training

# Build and verify all images
./build-all.sh

# Build without cache (if you have issues)
./build-all.sh --no-cache

# Build and skip tests
./build-all.sh --skip-tests
```

This script will:
1. Build the data processing image (smaller, faster)
2. Build the training image (larger, includes PyTorch)
3. Verify both images exist
4. Test basic container functionality
5. Show image sizes and summary

#### Option 2: Using Helper Script

```bash
# Build using docker-compose
./docker-run.sh build

# Rebuild without cache
./docker-run.sh rebuild

# Comprehensive build and verification
./docker-run.sh build-all
```

#### Option 3: Direct Docker Compose

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build training

# Build without cache
docker-compose build --no-cache
```

### Build Optimization Tips

1. **Layer Caching**: Docker caches layers, so rebuilds are faster if you only change scripts
2. **Build Order**: Data processing image builds first (smaller), then training image
3. **Cache Management**: Use `--no-cache` only when dependencies change
4. **Image Sizes**:
   - Training image: ~4-5GB (includes PyTorch + CUDA)
   - Data processing image: ~500MB-1GB

### Verifying the Build

After building, verify everything works:

```bash
# Check images exist
docker images | grep midikompanion

# Test training container
docker run --rm midikompanion-training:latest python -c "import torch; print('OK')"

# Test data processing container
docker run --rm midikompanion-data:latest python -c "import mido; print('OK')"

# Check container status
./docker-run.sh status
```

### Troubleshooting Build Issues

**Out of disk space:**
```bash
# Clean up unused Docker resources
./docker-run.sh prune

# Or manually
docker system prune -a --volumes
```

**Build fails with dependency errors:**
```bash
# Rebuild without cache
./docker-run.sh rebuild
```

**Permission errors:**
- Ensure Docker Desktop is running
- Check file permissions on scripts
- On Linux, ensure user is in docker group

---

## Quick Start: Training Models

### Option 1: Docker Compose Helper Script (Easiest)

```bash
cd ml_training

# Make script executable (first time only)
chmod +x docker-run.sh

# Start training service
./docker-run.sh start

# Start in background
./docker-run.sh start --detach

# View logs
./docker-run.sh logs -f

# Open interactive shell
./docker-run.sh shell

# Start TensorBoard monitoring
./docker-run.sh monitor

# Run evaluation
./docker-run.sh evaluate

# Run benchmarks
./docker-run.sh benchmark

# Run tests
./docker-run.sh test

# Check status
./docker-run.sh status

# Stop all services
./docker-run.sh stop
```

### Option 2: Docker Compose Direct (Recommended)

```bash
cd ml_training

# Build and start training container
docker-compose up training

# Run in background
docker-compose up -d training

# View logs
docker-compose logs -f training

# Start with TensorBoard monitoring
docker-compose --profile monitoring up

# Start data processing
docker-compose --profile data up data-processing

# Run evaluation
docker-compose --profile evaluation up evaluation

# Run benchmarks
docker-compose --profile benchmark up benchmark

# Run tests
docker-compose --profile testing up testing

# Stop all services
docker-compose down
```

### Option 2: Docker CLI

```bash
cd ml_training

# Build image
docker build -t midikompanion-training:latest .

# Run training (with GPU)
docker run --gpus all \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  midikompanion-training:latest

# Run training (CPU only - for testing)
docker run \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  midikompanion-training:latest
```

### Option 3: Interactive Development

```bash
# Start interactive shell in container
docker run -it --gpus all \
  -v $(pwd):/workspace \
  midikompanion-training:latest \
  /bin/bash

# Inside container:
python prepare_datasets.py
python train_all_models.py --model emotion_recognizer --epochs 50
python export_to_onnx.py --model emotion_recognizer
exit
```

---

## Data Processing Pipeline

### Step 1: Prepare Datasets

```bash
# Process MIDI files from ~/Downloads/midi_dataset
docker run -v ~/Downloads/midi_dataset:/data/raw \
           -v $(pwd)/datasets:/data/processed \
           midikompanion-data:latest

# Or use docker-compose
docker-compose --profile data up data-processing
```

### Step 2: Train Models

```bash
# Train all 5 models (8-14 hours on GPU)
docker-compose up training

# Or train individual models
docker run --gpus all -v $(pwd):/workspace midikompanion-training:latest \
  python train_all_models.py --model emotion_recognizer --epochs 100

docker run --gpus all -v $(pwd):/workspace midikompanion-training:latest \
  python train_all_models.py --model melody_transformer --epochs 100

# ... repeat for harmony_predictor, dynamics_engine, groove_predictor
```

### Step 3: Export to ONNX

```bash
docker run -v $(pwd):/workspace midikompanion-training:latest \
  python export_to_onnx.py --all

# Output: models/onnx/*.onnx files
```

### Step 4: Validate Models

```bash
docker run -v $(pwd):/workspace midikompanion-training:latest \
  python validate_models.py

# Output: Validation report with accuracy/inference time
```

---

## Available Services

### 1. Training Service (Main)

The primary service for training ML models with GPU support.

```bash
docker-compose up training
# or
./docker-run.sh start
```

### 2. TensorBoard Monitoring

Monitor training progress with real-time visualizations.

```bash
docker-compose --profile monitoring up tensorboard
# or
./docker-run.sh monitor

# Access at: http://localhost:6006
```

### 3. Data Processing Service

Process raw MIDI files into training datasets.

```bash
docker-compose --profile data up data-processing
# or
./docker-run.sh data
```

### 4. Evaluation Service

Evaluate trained models on test datasets.

```bash
docker-compose --profile evaluation up evaluation
# or
./docker-run.sh evaluate
```

### 5. Benchmark Service

Benchmark model inference performance.

```bash
docker-compose --profile benchmark up benchmark
# or
./docker-run.sh benchmark
```

### 6. Testing Service

Run the full test suite with coverage reports.

```bash
docker-compose --profile testing up testing
# or
./docker-run.sh test
```

## Monitoring Training with TensorBoard

```bash
# Start TensorBoard (accessible at http://localhost:6006)
docker-compose --profile monitoring up tensorboard
# or
./docker-run.sh monitor

# Open browser to: http://localhost:6006
```

---

## Google Colab Integration

Since macOS doesn't support GPU passthrough, use Google Colab for free GPU training:

### 1. Upload Docker Image to Colab

```python
# In Colab notebook cell:
!pip install docker

# Build from Dockerfile in Colab
!git clone https://github.com/your-username/midikompanion.git
%cd midikompanion/ml_training

# Install dependencies directly (no Docker needed in Colab)
!pip install torch torchvision torchaudio \
             numpy pandas librosa mido music21 \
             onnx onnxruntime tensorboard

# Run training
!python train_all_models.py --epochs 100 --batch-size 64
```

### 2. Or Use Pre-built Image

```python
# In Colab notebook:
!docker pull your-dockerhub-username/midikompanion-training:latest
!docker run --gpus all midikompanion-training:latest
```

---

## Cloud Deployment

### AWS SageMaker

```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

docker tag midikompanion-training:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/midikompanion-training:latest

docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/midikompanion-training:latest

# Create SageMaker training job (use AWS Console or CLI)
```

### Google Cloud AI Platform

```bash
# Push to GCR
gcloud auth configure-docker
docker tag midikompanion-training:latest gcr.io/your-project-id/midikompanion-training:latest
docker push gcr.io/your-project-id/midikompanion-training:latest

# Create Cloud AI training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=midikompanion-training \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,replica-count=1,container-image-uri=gcr.io/your-project-id/midikompanion-training:latest
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU availability
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If error, check nvidia-docker installation
sudo systemctl status docker
nvidia-smi  # Should show GPU on host
```

### Out of Memory Errors

```bash
# Reduce batch size
docker run --gpus all -v $(pwd):/workspace midikompanion-training:latest \
  python train_all_models.py --batch-size 32  # Instead of 64

# Or set memory limit
docker run --gpus all --memory=16g -v $(pwd):/workspace midikompanion-training:latest
```

### Permission Errors on macOS

```bash
# Make sure directories are writable
chmod -R 755 datasets models logs

# Run with user ID mapping
docker run --user $(id -u):$(id -g) \
  -v $(pwd):/workspace midikompanion-training:latest
```

---

## When NOT to Use Docker

### ❌ Don't Use for Plugin Development

- JUCE plugins need native audio drivers (CoreAudio/ALSA)
- GUI frameworks don't work well in containers
- Can't test in real DAWs (Logic Pro, Ableton)

**Instead**: Build natively

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Test in real DAW
```

### ❌ Don't Use for Initial Setup

- Cloning JUCE, RTNeural is faster natively
- IDE integration works better natively

**Instead**: Use Docker only for training/testing, not development

---

## File Structure After Training

```
ml_training/
├── Dockerfile                    # GPU training image
├── Dockerfile.data               # Data processing image
├── docker-compose.yml            # Orchestration config
├── README_DOCKER.md             # This file
├── prepare_datasets.py           # Dataset preparation
├── train_all_models.py           # Training script
├── export_to_onnx.py            # ONNX export
├── validate_models.py           # Model validation
├── datasets/                     # Generated datasets (volume mount)
│   ├── emotion_train.npz
│   ├── melody_train.npz
│   └── ...
├── models/                       # Trained models (volume mount)
│   ├── onnx/
│   │   ├── emotion_recognizer.onnx
│   │   ├── melody_transformer.onnx
│   │   └── ...
│   └── pytorch/
│       ├── emotion_recognizer.pth
│       └── ...
└── logs/                         # TensorBoard logs (volume mount)
    ├── emotion_recognizer/
    └── ...
```

---

## Cost Comparison

| Platform | GPU | Cost | Training Time (5 models) |
|----------|-----|------|--------------------------|
| **Google Colab Free** | T4 | $0 | 12-15 hours |
| **Google Colab Pro** | T4/V100 | $10/month | 8-10 hours |
| **AWS SageMaker** | ml.p3.2xlarge (V100) | $3.06/hour | ~$25 for 8 hours |
| **Local GPU** (your own) | RTX 3080 | $0 | 10-12 hours |
| **Local CPU** (Docker) | N/A | $0 | 48-72 hours (NOT recommended) |

**Recommendation**: Use Google Colab Free for Agent 2's training (Weeks 3-6).
