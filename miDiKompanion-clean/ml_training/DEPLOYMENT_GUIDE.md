# ML Model Training and Deployment Guide

## Overview

This guide covers the complete workflow for training ML models and deploying them to the miDiKompanion plugin.

## Quick Start

### Option 1: Quick Training (Synthetic Data)

For testing the pipeline with synthetic data:

```bash
cd ml_training
./train_and_deploy.sh --use-synthetic --epochs 10
```

This will:
1. Train all 5 models with synthetic data (10 epochs)
2. Export to ONNX format
3. Validate models
4. Package for deployment

**Time**: ~5-10 minutes on CPU

### Option 2: Full Training (Real Datasets)

For production models with real datasets:

```bash
cd ml_training
./train_and_deploy.sh --epochs 50 --device cuda --datasets-dir ../datasets
```

**Time**: 8-14 hours on GPU (Google Colab free tier)

## Step-by-Step Workflow

### Step 1: Prepare Datasets (Optional)

If you have audio/MIDI datasets with emotion labels:

```bash
python prepare_datasets.py \
    --audio-dir datasets/audio \
    --midi-dir datasets/midi \
    --output-dir datasets/prepared \
    --vad-labels datasets/vad_labels.csv
```

This creates node-aware datasets with 216-node emotion thesaurus labels.

### Step 2: Train Models

Train all 5 models:

```bash
python train_all_models.py \
    --output ./trained_models \
    --epochs 50 \
    --batch-size 64 \
    --device cuda \
    --datasets-dir ./datasets
```

**Models trained:**
1. **EmotionRecognizer**: Audio → Emotion (128→64)
2. **MelodyTransformer**: Emotion → MIDI (64→128)
3. **HarmonyPredictor**: Context → Chords (128→64)
4. **DynamicsEngine**: Context → Expression (32→16)
5. **GroovePredictor**: Emotion → Groove (64→32)

### Step 3: Export to ONNX

Export trained models to ONNX format:

```bash
python export_to_onnx.py \
    --models-dir trained_models/checkpoints \
    --output-dir trained_models/onnx
```

### Step 4: Validate Models

Validate ONNX models:

```bash
python validate_models.py trained_models/onnx --verbose
```

### Step 5: Package for Deployment

Create deployment package:

```bash
python deploy_models.py \
    --models-dir trained_models/onnx \
    --output-dir deployment
```

### Step 6: Deploy to Plugin

1. **Copy models to plugin Resources**:
   ```bash
   cp deployment/models/*.onnx /path/to/plugin/Resources/models/
   ```

2. **Enable ONNX Runtime in CMake**:
   ```bash
   cmake -DENABLE_ONNX_RUNTIME=ON -B build
   ```

3. **Rebuild plugin**:
   ```bash
   cmake --build build
   ```

## Docker Training (GPU)

For GPU training with Docker:

```bash
# Build Docker image
docker build -t midikompanion-training:latest -f Dockerfile .

# Run training (with GPU)
docker run --gpus all \
    -v $(pwd):/workspace \
    -v $(pwd)/datasets:/workspace/datasets \
    midikompanion-training:latest \
    python train_all_models.py --epochs 50 --device cuda
```

## Google Colab Training

1. Upload `Train_MidiKompanion_Models.ipynb` to Google Colab
2. Upload datasets to Colab (or mount Google Drive)
3. Run all cells (8-14 hours on free GPU tier)
4. Download trained models
5. Export to ONNX locally

## Model Specifications

All models must match these specifications for C++ integration:

| Model | Input Size | Output Size | Description |
|-------|-----------|-------------|-------------|
| EmotionRecognizer | 128 | 64 | Audio features → Emotion embedding |
| MelodyTransformer | 64 | 128 | Emotion → MIDI note probabilities |
| HarmonyPredictor | 128 | 64 | Context → Chord probabilities |
| DynamicsEngine | 32 | 16 | Context → Expression parameters |
| GroovePredictor | 64 | 32 | Emotion → Groove parameters |

**ONNX Opset**: 14
**Format**: ONNX
**Total Size**: ~5 MB (all models)

## Validation Checklist

Before deployment, verify:

- [ ] All 5 models trained successfully
- [ ] ONNX export completed without errors
- [ ] Model input/output shapes match specifications
- [ ] Test inference works (validate_models.py)
- [ ] Models load in C++ (ONNXInference)
- [ ] Inference latency < 10ms per model
- [ ] Memory usage < 50MB for all models

## Troubleshooting

### Training Fails

- **Out of Memory**: Reduce batch size (`--batch-size 32`)
- **CUDA Errors**: Use CPU (`--device cpu`)
- **Dataset Not Found**: Use synthetic data (`--use-synthetic`)

### ONNX Export Fails

- **Model Not Found**: Check checkpoint directory
- **Shape Mismatch**: Verify model architecture matches specs
- **ONNX Runtime Error**: Update onnxruntime package

### C++ Integration Issues

- **Models Not Loading**: Check file paths and ONNX Runtime installation
- **Inference Errors**: Verify input shapes match model expectations
- **Performance Issues**: Enable ONNX Runtime optimizations

## Next Steps

After deployment:

1. Test models in plugin (A/B testing vs rule-based)
2. Monitor inference latency
3. Collect user feedback
4. Iterate on model training based on results

## Resources

- **Training Scripts**: `ml_training/train_all_models.py`
- **ONNX Export**: `ml_training/export_to_onnx.py`
- **Deployment**: `ml_training/deploy_models.py`
- **C++ Integration**: `src/ml/ONNXInference.h`
