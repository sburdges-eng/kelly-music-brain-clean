# MidiKompanion ML Training Docker Image
# 
# GPU-enabled training environment with PyTorch and ONNX Runtime
#
# Build:
#   docker build -t midikompanion-ml-training .
#
# Run:
#   docker run --gpus all -v $(pwd)/models:/workspace/models midikompanion-ml-training

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY train_all_models.py .
COPY prepare_datasets.py .
COPY export_to_onnx.py .
COPY validate_models.py .
COPY node_aware_training.py .

# Create directories
RUN mkdir -p /workspace/models /workspace/data /workspace/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command: train all models
CMD ["python", "train_all_models.py", "--epochs", "100", "--output-dir", "/workspace/models"]
