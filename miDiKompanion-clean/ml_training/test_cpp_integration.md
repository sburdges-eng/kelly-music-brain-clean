# C++ Integration Testing Guide

## Overview

This guide helps verify that exported RTNeural JSON models can be loaded and used in the C++ plugin.

## Prerequisites

1. Models exported using `train_all_models.py`
2. Models validated using `validate_models.py`
3. C++ plugin built with `ENABLE_RTNEURAL=ON`

## Model File Locations

The plugin looks for models in these locations (in order):

1. `models/` directory (relative to plugin executable)
2. `Resources/models/` (inside plugin bundle)

Model files should be named:

- `emotionrecognizer.json`
- `melodytransformer.json`
- `harmonypredictor.json`
- `dynamicsengine.json`
- `groovepredictor.json`

## Testing Steps

### 1. Export Models

```bash
cd ml_training
python train_all_models.py --output ./trained_models --epochs 10 --use-synthetic
```

### 2. Validate Exported Models

```bash
python validate_models.py trained_models/*.json
```

### 3. Copy Models to Plugin Resources

```bash
# For macOS plugin bundle
cp trained_models/*.json "path/to/plugin.app/Contents/Resources/models/"

# Or create models directory next to plugin
mkdir -p models
cp trained_models/*.json models/
```

### 4. Build Plugin with RTNeural

```bash
mkdir build && cd build
cmake .. -DENABLE_RTNEURAL=ON
cmake --build . --config Release
```

### 5. Check Plugin Logs

When the plugin loads, check the logs for:

- `"Loaded model: EmotionRecognizer (497664 params)"`
- `"MultiModelProcessor initialized"`
- Any error messages about model loading

### 6. Test Inference

The plugin should:

- Load models on initialization
- Use fallback heuristics if models fail to load
- Log any exceptions during inference

## Common Issues

### Models Not Found

**Symptom**: Log shows "Model not found" or "Models directory not found"

**Solution**:

- Verify model files are in correct location
- Check file names match expected format (lowercase with .json extension)
- Ensure plugin has read permissions

### JSON Parse Errors

**Symptom**: "Failed to parse RTNeural model JSON"

**Solution**:

- Run `validate_models.py` to check JSON structure
- Verify LSTM weights are properly split into 4 gates
- Check JSON is valid (no syntax errors)

### Input/Output Size Mismatch

**Symptom**: "Input size mismatch" or "ModelWrapper: Input size mismatch"

**Solution**:

- Verify model input/output sizes match C++ `ModelSpec` definitions
- Run `verify_model_architectures.py` to check Python models
- Ensure exported model metadata has correct sizes

### RTNeural API Issues

**Symptom**: Compilation errors or runtime exceptions

**Solution**:

- Verify RTNeural version compatibility
- Check `forward()` API signature matches RTNeural version
- Ensure `ENABLE_RTNEURAL` is defined during compilation

## Verification Checklist

- [ ] Models exported successfully
- [ ] Models pass validation (`validate_models.py`)
- [ ] Model architectures match C++ specs (`verify_model_architectures.py`)
- [ ] Models copied to plugin resources
- [ ] Plugin built with `ENABLE_RTNEURAL=ON`
- [ ] Plugin logs show successful model loading
- [ ] No exceptions during inference
- [ ] Fallback heuristics work if models unavailable

## Next Steps

Once models load successfully:

1. Test full pipeline inference
2. Measure inference latency (<10ms target)
3. Measure memory usage (<4MB target)
4. Test with real audio features
5. Optimize if needed
