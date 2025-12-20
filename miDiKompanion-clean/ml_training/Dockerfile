# Dockerfile for ML Training Environment
# Agent 2: ML Training Specialist (Week 3-6)
# GPU training environment with PyTorch + CUDA

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for ML training
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    librosa==0.10.1 \
    mido==1.3.0 \
    onnx==1.15.0 \
    onnxruntime==1.16.3 \
    tensorboard==2.15.1 \
    scikit-learn==1.3.2 \
    tqdm==4.66.1 \
    matplotlib==3.7.1 \
    seaborn==0.12.2

# Copy training scripts
COPY prepare_datasets.py train_all_models.py export_to_onnx.py node_aware_training.py ./
COPY training_utils.py dataset_loaders.py ./

# Create models output directory
RUN mkdir -p /workspace/models/onnx

# Default command: train all models
CMD ["python", "train_all_models.py", "--epochs", "100", "--output-dir", "/workspace/models"]
