# RTNeural Export Verification Guide

This document describes the RTNeural export format verification and testing procedures implemented for the Kelly MIDI Companion ML models.

## Overview

The RTNeural export system has been updated to properly export PyTorch models to RTNeural-compatible JSON format. This includes:

1. **Fixed Export Function**: Properly handles LSTM layers and activation detection
2. **C++ Loading Fix**: MultiModelProcessor now correctly loads JSON models
3. **Validation Tools**: Scripts to verify JSON format compatibility
4. **Benchmarking**: Latency testing to ensure <10ms inference target
5. **Versioning System**: Track model compatibility and updates

## Export Format

### RTNeural JSON Structure

```json
{
  "layers": [
    {
      "type": "dense",
      "in_size": 128,
      "out_size": 512,
      "activation": "tanh",
      "weights": [[...], [...]],  // out_size x in_size
      "bias": [...]                // out_size
    },
    {
      "type": "lstm",
      "in_size": 256,
      "out_size": 128,
      "weights_ih": [[...], [...], [...], [...]],  // 4 gates
      "weights_hh": [[...], [...], [...], [...]],  // 4 gates
      "bias_ih": [[...], [...], [...], [...]],      // 4 gates
      "bias_hh": [[...], [...], [...], [...]]      // 4 gates
    }
  ],
  "metadata": {
    "model_name": "EmotionRecognizer",
    "framework": "PyTorch",
    "export_version": "2.0",
    "parameter_count": 497664,
    "memory_bytes": 1990656,
    "input_size": 128,
    "output_size": 64
  }
}
```

### Key Changes from v1.0 to v2.0

- **Structure**: Top-level `layers` array instead of mixed structure
- **LSTM Format**: Properly split LSTM weights into 4 gates (input, forget, cell, output)
- **Activation Detection**: Automatically detects activations from model structure
- **Metadata**: Enhanced metadata with version tracking

## Usage

### 1. Export Models

```bash
# Train and export all models
python ml_training/train_all_models.py --output ./trained_models

# Or use training_pipe version
python training_pipe/scripts/train_all_models.py --output ./trained_models
```

### 2. Validate Exported Models

```bash
# Validate a single model
python ml_training/validate_rtneural_export.py trained_models/emotionrecognizer.json

# With verbose output
python ml_training/validate_rtneural_export.py trained_models/emotionrecognizer.json --verbose
```

### 3. Benchmark Inference Latency

```bash
# Benchmark a model (Python simulation)
python ml_training/benchmark_inference.py trained_models/emotionrecognizer.json

# With custom iterations
python ml_training/benchmark_inference.py trained_models/emotionrecognizer.json --iterations 5000
```

### 4. Check Model Version

```bash
# Check model version and compatibility
python ml_training/model_versioning.py check trained_models/emotionrecognizer.json

# Get version info
python ml_training/model_versioning.py version trained_models/emotionrecognizer.json

# Migrate v1.0 to v2.0
python ml_training/model_versioning.py migrate trained_models/emotionrecognizer.json --output migrated.json
```

### 5. Test C++ Loading

```bash
# Compile test program (requires RTNeural)
g++ -std=c++17 -DENABLE_RTNEURAL -I/path/to/RTNeural/include \
    ml_training/test_model_loading.cpp -o test_model_loading \
    -L/path/to/RTNeural/lib -lRTNeural

# Run test
./test_model_loading trained_models/emotionrecognizer.json
```

## Validation Checklist

Before using exported models in production:

- [ ] **JSON Structure**: Run `validate_rtneural_export.py` - all checks pass
- [ ] **Layer Dimensions**: Input/output sizes match between layers
- [ ] **LSTM Format**: LSTM layers have 4 gates with correct dimensions
- [ ] **Activation Functions**: Valid activations (tanh, relu, sigmoid, softmax)
- [ ] **C++ Loading**: Model loads successfully in C++ using RTNeural
- [ ] **Inference Latency**: Benchmark shows <10ms average latency
- [ ] **Version Compatibility**: Model version is compatible with current system

## Common Issues and Solutions

### Issue: "Failed to parse RTNeural model JSON"

**Cause**: JSON format doesn't match RTNeural expectations.

**Solution**:

1. Run validation script to identify specific errors
2. Check that LSTM layers have 4 gates properly formatted
3. Verify activation functions are valid
4. Ensure layer dimensions are consistent

### Issue: "Input size mismatch" in C++

**Cause**: Model input size doesn't match what C++ code expects.

**Solution**:

1. Check `MODEL_SPECS` in `MultiModelProcessor.h` matches exported model
2. Verify metadata `input_size` matches first layer `in_size`
3. Update C++ code if model architecture changed

### Issue: High inference latency (>10ms)

**Cause**: Model too large or inefficient structure.

**Solution**:

1. Check parameter count - should be <1M total
2. Consider model quantization
3. Verify RTNeural is using optimized code paths
4. Profile to identify bottlenecks

### Issue: LSTM weights not loading

**Cause**: LSTM weight format incorrect.

**Solution**:

1. Verify LSTM weights are split into 4 gates correctly
2. Check `weights_ih` and `weights_hh` have correct dimensions
3. Ensure biases are also split into 4 gates

## Model Versioning

### Version Format

Models use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (incompatible JSON format)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Current Version: 2.0.0

- Proper LSTM weight splitting
- Automatic activation detection
- Enhanced metadata
- RTNeural-compatible JSON structure

### Compatibility

Models with the same major version are compatible. For example:

- v2.0.0, v2.1.0, v2.2.3 are all compatible
- v1.0.0 is NOT compatible with v2.x.x

Use `model_versioning.py migrate` to convert v1.0 models to v2.0.

## Testing in C++ Plugin

After exporting models, test loading in the plugin:

1. Place models in `Resources/models/` directory
2. Models should be named: `emotionrecognizer.json`, `melodytransformer.json`, etc.
3. Plugin will auto-load on initialization
4. Check plugin logs for loading success/failure

Example log output:

```
Loaded model: EmotionRecognizer (497,664 params)
Loaded model: MelodyTransformer (412,672 params)
...
MultiModelProcessor initialized:
  Total params: 1,016,880
  Total memory: 3,971 KB
  Estimated inference: <10ms
```

## Performance Targets

- **Inference Latency**: <10ms per model
- **Total Pipeline**: <50ms for all 5 models
- **Memory Usage**: <4MB for all models
- **CPU Usage**: <5% on modern processors

## Next Steps

1. **Real Dataset Training**: Replace synthetic data with actual datasets
2. **Model Optimization**: Quantization, pruning for better performance
3. **A/B Testing**: Framework for comparing model versions
4. **Automated Testing**: CI/CD integration for model validation

## References

- [RTNeural GitHub](https://github.com/jatinchowdhury18/RTNeural)
- [RTNeural Documentation](https://github.com/jatinchowdhury18/RTNeural#documentation)
- [Model Training Guide](../TRAINING_PIPE_QUICKSTART.md)
