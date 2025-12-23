#!/bin/bash

# Kelly ML Training Hardware Test Runner
# Automatically detects hardware and runs appropriate benchmarks

CONFIG_DIR="$(dirname "$0")"
SCRIPT_PATH="$CONFIG_DIR/hardware_benchmarks.py"

echo "===================================================="
echo "    Kelly ML Training: Hardware Test Suite"
echo "===================================================="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found."
    exit 1
fi

# Detect Hardware
IS_MAC=$(uname -a | grep -i "Darwin")
HAS_CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
HAS_MPS=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null)

if [ "$HAS_MPS" == "True" ]; then
    echo "[Detected] Apple Silicon / MPS Support"
    echo "Running 16GB Mac Optimized Test..."
    python3 "$SCRIPT_PATH" --config "$CONFIG_DIR/mac_mps_16gb.yaml"
fi

if [ "$HAS_CUDA" == "True" ]; then
    echo "[Detected] NVIDIA CUDA Support"
    echo "Running NVIDIA CUDA Optimized Test..."
    python3 "$SCRIPT_PATH" --config "$CONFIG_DIR/nvidia_cuda_gpu.yaml"
fi

if [ "$HAS_MPS" != "True" ] && [ "$HAS_CUDA" != "True" ]; then
    echo "[Warning] No hardware acceleration detected. Running Mac config on CPU for fallback test."
    python3 "$SCRIPT_PATH" --config "$CONFIG_DIR/mac_mps_16gb.yaml"
fi

echo "===================================================="
echo "           All Hardware Tests Completed"
echo "===================================================="
