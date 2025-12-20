# Quick Deployment Guide

## Fastest Path to Deployed Models

### 1. Train with Synthetic Data (5 minutes)

```bash
cd ml_training
./train_and_deploy.sh --use-synthetic --epochs 10
```

This creates a complete deployment package in `./deployment/` with all 5 models.

### 2. Deploy to Plugin

```bash
# Copy models to plugin Resources
mkdir -p /path/to/plugin/Resources/models
cp deployment/models/*.onnx /path/to/plugin/Resources/models/

# Rebuild with ONNX Runtime
cd /path/to/plugin
cmake -DENABLE_ONNX_RUNTIME=ON -B build
cmake --build build
```

### 3. Validate Deployment

```bash
cd ml_training
python validate_deployment.py deployment
```

## Production Training (8-14 hours)

For production-quality models:

```bash
# Option A: Google Colab (Free GPU)
# 1. Upload Train_MidiKompanion_Models.ipynb to Colab
# 2. Run all cells
# 3. Download trained models
# 4. Export locally: python export_to_onnx.py

# Option B: Docker (Local GPU)
docker run --gpus all \
    -v $(pwd):/workspace \
    midikompanion-training:latest \
    python train_all_models.py --epochs 50 --device cuda

# Then package:
python deploy_models.py --models-dir trained_models/onnx --output-dir deployment
```

## Model Files

After deployment, you should have:

```
deployment/
├── models/
│   ├── emotionrecognizer.onnx
│   ├── melodytransformer.onnx
│   ├── harmonypredictor.onnx
│   ├── dynamicsengine.onnx
│   └── groovepredictor.onnx
├── deployment_manifest.json
├── README.md
└── code_snippets/
    └── model_loading_example.cpp
```

## Integration Checklist

- [ ] Models copied to `plugin/Resources/models/`
- [ ] CMake configured with `-DENABLE_ONNX_RUNTIME=ON`
- [ ] Plugin rebuilt successfully
- [ ] Models load in `ONNXInference`
- [ ] Inference latency < 10ms per model
- [ ] A/B testing shows ML vs rule-based results

## Troubleshooting

**Models not loading?**
- Check file paths in `ONNXInference::loadModel()`
- Verify ONNX Runtime is installed and linked
- Check model file permissions

**Inference errors?**
- Verify input shapes match model specs (see deployment_manifest.json)
- Check ONNX opset version (should be 14)

**Performance issues?**
- Enable ONNX Runtime optimizations
- Consider model quantization for smaller/faster models
