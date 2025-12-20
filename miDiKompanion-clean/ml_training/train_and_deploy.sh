#!/bin/bash
# train_and_deploy.sh - Complete Training and Deployment Workflow
# ===============================================================
#
# Agent 2: ML Training Specialist (Week 3-6)
# Purpose: Complete workflow for training models and deploying to plugin
#
# Usage:
#   ./train_and_deploy.sh [--use-synthetic] [--epochs N] [--device cuda|cpu|mps]
#
# This script:
# 1. Prepares datasets (if needed)
# 2. Trains all 5 models
# 3. Exports to ONNX format
# 4. Validates models
# 5. Packages for deployment

set -e  # Exit on error

# Default values
USE_SYNTHETIC=false
EPOCHS=50
DEVICE="auto"
DATASETS_DIR=""
OUTPUT_DIR="./trained_models"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-synthetic)
            USE_SYNTHETIC=true
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --datasets-dir)
            DATASETS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--use-synthetic] [--epochs N] [--device cuda|cpu|mps] [--datasets-dir DIR] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ML Model Training and Deployment"
echo "=========================================="
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "Use Synthetic: $USE_SYNTHETIC"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Prepare datasets (optional)
if [ -n "$DATASETS_DIR" ] && [ "$USE_SYNTHETIC" = false ]; then
    echo "Step 1: Preparing datasets..."
    echo "----------------------------------------"

    # Check if prepare_datasets.py exists and datasets need preparation
    if [ -f "prepare_datasets.py" ]; then
        echo "Note: Use prepare_datasets.py to prepare node-aware datasets if needed"
        echo "  python prepare_datasets.py --audio-dir <dir> --midi-dir <dir> --output-dir datasets/prepared"
    fi
    echo ""
fi

# Step 2: Train all models
echo "Step 2: Training all models..."
echo "----------------------------------------"

TRAIN_ARGS="--output $OUTPUT_DIR --epochs $EPOCHS --device $DEVICE"
if [ "$USE_SYNTHETIC" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --use-synthetic"
fi
if [ -n "$DATASETS_DIR" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --datasets-dir $DATASETS_DIR"
fi

python train_all_models.py $TRAIN_ARGS

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "✓ Training complete"
echo ""

# Step 3: Export to ONNX (if not already done)
echo "Step 3: Exporting to ONNX format..."
echo "----------------------------------------"

ONNX_DIR="$OUTPUT_DIR/onnx"
if [ ! -d "$ONNX_DIR" ] || [ -z "$(ls -A $ONNX_DIR 2>/dev/null)" ]; then
    python export_to_onnx.py \
        --models-dir "$OUTPUT_DIR/checkpoints" \
        --output-dir "$ONNX_DIR"

    if [ $? -ne 0 ]; then
        echo "Error: ONNX export failed"
        exit 1
    fi
else
    echo "ONNX models already exist, skipping export"
fi

echo ""
echo "✓ ONNX export complete"
echo ""

# Step 4: Validate models
echo "Step 4: Validating models..."
echo "----------------------------------------"

if [ -f "validate_models.py" ]; then
    python validate_models.py "$ONNX_DIR" --verbose

    if [ $? -ne 0 ]; then
        echo "Warning: Some model validations failed"
    fi
else
    echo "Note: validate_models.py not found, skipping validation"
fi

echo ""
echo "✓ Validation complete"
echo ""

# Step 5: Package for deployment
echo "Step 5: Packaging for deployment..."
echo "----------------------------------------"

python deploy_models.py \
    --models-dir "$ONNX_DIR" \
    --output-dir "./deployment"

if [ $? -ne 0 ]; then
    echo "Error: Deployment packaging failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training and Deployment Complete!"
echo "=========================================="
echo ""
echo "Deployment package: ./deployment"
echo ""
echo "Next steps:"
echo "1. Copy models to plugin Resources:"
echo "   cp deployment/models/*.onnx /path/to/plugin/Resources/models/"
echo ""
echo "2. Enable ONNX Runtime in CMake:"
echo "   cmake -DENABLE_ONNX_RUNTIME=ON .."
echo ""
echo "3. Rebuild plugin"
echo ""
