# Docker Compose Quick Start Guide

## 🚀 Quick Commands

### Using Helper Script (Recommended)

```bash
./docker-run.sh [command]
```

### Direct Docker Compose

```bash
docker-compose [options] [service]
# or (Docker Compose v2)
docker compose [options] [service]
```

---

## 📋 Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `start` | Start training service | `./docker-run.sh start` |
| `stop` | Stop all services | `./docker-run.sh stop` |
| `restart` | Restart services | `./docker-run.sh restart` |
| `logs` | View logs | `./docker-run.sh logs -f` |
| `shell` | Interactive shell | `./docker-run.sh shell` |
| `build` | Build images | `./docker-run.sh build` |
| `monitor` | Start TensorBoard | `./docker-run.sh monitor` |
| `data` | Data processing | `./docker-run.sh data` |
| `evaluate` | Run evaluation | `./docker-run.sh evaluate` |
| `benchmark` | Run benchmarks | `./docker-run.sh benchmark` |
| `test` | Run tests | `./docker-run.sh test` |
| `status` | Show status | `./docker-run.sh status` |
| `clean` | Clean up | `./docker-run.sh clean` |

---

## 🎯 Common Workflows

### 1. Start Training

```bash
# Start training in foreground
./docker-run.sh start

# Start in background
./docker-run.sh start --detach

# View logs
./docker-run.sh logs -f
```

### 2. Training with Monitoring

```bash
# Terminal 1: Start training
./docker-run.sh start --detach

# Terminal 2: Start TensorBoard
./docker-run.sh monitor

# Open browser: http://localhost:6006
```

### 3. Interactive Development

```bash
# Open shell in container
./docker-run.sh shell

# Inside container:
python prepare_datasets.py
python train_all_models.py --model emotion_recognizer --epochs 50
python export_to_onnx.py --all
exit
```

### 4. Process Data First

```bash
# Step 1: Process raw data
./docker-run.sh data

# Step 2: Train models
./docker-run.sh start
```

### 5. Evaluate Trained Models

```bash
# After training completes
./docker-run.sh evaluate
```

### 6. Benchmark Performance

```bash
# Test inference speed
./docker-run.sh benchmark
```

### 7. Run Tests

```bash
# Full test suite with coverage
./docker-run.sh test
```

---

## 🏗️ Services Overview

### Training Service (Main)

- **Purpose**: Train ML models with GPU support
- **Volumes**: datasets, models, logs, trained_models, all scripts
- **GPU**: Yes (requires nvidia-docker)
- **Health Check**: Yes
- **Restart**: unless-stopped

### TensorBoard Service

- **Purpose**: Monitor training progress
- **Port**: 6006
- **URL**: <http://localhost:6006>
- **Profile**: `monitoring`
- **Health Check**: Yes

### Data Processing Service

- **Purpose**: Process raw MIDI files
- **Input**: `~/Downloads/midi_dataset`
- **Output**: `./datasets`
- **Profile**: `data`

### Evaluation Service

- **Purpose**: Evaluate model performance
- **Profile**: `evaluation`
- **Runs once**: Yes (restart: no)

### Benchmark Service

- **Purpose**: Benchmark inference speed
- **Profile**: `benchmark`
- **Runs once**: Yes (restart: no)

### Testing Service

- **Purpose**: Run pytest test suite
- **Profile**: `testing`
- **Coverage**: HTML report generated
- **Runs once**: Yes (restart: no)

---

## 🔧 Configuration

### Environment Variables

- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- `CUDA_VISIBLE_DEVICES=0`
- `PYTHONUNBUFFERED=1`
- `TORCH_USE_CUDA_DSA=1`

### Volume Mounts

- `./datasets` → `/workspace/datasets`
- `./models` → `/workspace/models`
- `./logs` → `/workspace/logs`
- `./trained_models` → `/workspace/trained_models`
- All Python scripts mounted individually
- `config.json` and `requirements.txt` mounted

### Network

- Bridge network: `ml-training`
- All services can communicate

---

## 🐛 Troubleshooting

### Check Docker Installation

```bash
docker --version
docker compose version  # or docker-compose --version
```

### Check GPU Support

```bash
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### View Container Logs

```bash
./docker-run.sh logs
# or
docker-compose logs training
```

### Check Container Status

```bash
./docker-run.sh status
# or
docker-compose ps
```

### Clean Up Everything

```bash
./docker-run.sh clean
# or
docker-compose down -v --remove-orphans
```

### Rebuild Images

```bash
./docker-run.sh build --no-cache
```

---

## 📝 Examples

### Train Single Model

```bash
./docker-run.sh shell
# Inside container:
python train_all_models.py --model emotion_recognizer --epochs 100
```

### Export Models

```bash
./docker-run.sh shell
# Inside container:
python export_to_onnx.py --all
```

### Validate Models

```bash
./docker-run.sh shell
# Inside container:
python validate_models.py
```

### Full Pipeline

```bash
# 1. Process data
./docker-run.sh data

# 2. Train models
./docker-run.sh start --detach

# 3. Monitor progress
./docker-run.sh monitor

# 4. Evaluate
./docker-run.sh evaluate

# 5. Benchmark
./docker-run.sh benchmark
```

---

## 🔗 Related Files

- `docker-compose.yml` - Main configuration
- `Dockerfile` - Training image
- `Dockerfile.data` - Data processing image
- `docker-run.sh` - Helper script
- `README_DOCKER.md` - Full documentation

---

## 💡 Tips

1. **Use profiles** to start only needed services
2. **Run in background** with `--detach` for long training
3. **Monitor logs** with `logs -f` to see real-time progress
4. **Use shell** for interactive development and debugging
5. **Clean up** regularly to free disk space
6. **Rebuild images** when dependencies change

---

## ⚠️ Notes

- **macOS**: GPU passthrough not supported, use CPU or cloud
- **Linux**: Requires nvidia-docker for GPU support
- **Windows**: Use WSL2 with Docker Desktop
- **Cloud**: Consider Google Colab or AWS for GPU training
