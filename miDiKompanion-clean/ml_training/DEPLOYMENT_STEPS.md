# Deployment Steps 2-4: Package, Deploy, Rebuild

## Current Status

✅ **Step 1 Complete**: All 5 models trained
- Checkpoints: `trained_models/checkpoints/{ModelName}_best.pt`
- RTNeural JSON: `trained_models/{modelname}.json`
- TorchScript: `trained_models/onnx/{modelname}.pt` (fallback, ONNX pending)

⚠️ **ONNX Export**: Pending due to Python 3.14 compatibility
- Use Docker or Python 3.11/3.12 for ONNX export
- TorchScript models available as interim solution

## Step 2: Package for Deployment

### Option A: Package RTNeural JSON Models (Available Now)

```bash
cd ml_training

# Create deployment package with RTNeural models
mkdir -p deployment/models
cp trained_models/*.json deployment/models/
cp trained_models/checkpoints/*_best.pt deployment/models/

# Create manifest
python3 -c "
import json
from pathlib import Path

manifest = {
    'version': '1.0',
    'format': 'RTNeural_JSON',
    'models': {
        'EmotionRecognizer': {'file': 'emotionrecognizer.json', 'input': 128, 'output': 64},
        'MelodyTransformer': {'file': 'melodytransformer.json', 'input': 64, 'output': 128},
        'HarmonyPredictor': {'file': 'harmonypredictor.json', 'input': 128, 'output': 64},
        'DynamicsEngine': {'file': 'dynamicsengine.json', 'input': 32, 'output': 16},
        'GroovePredictor': {'file': 'groovepredictor.json', 'input': 64, 'output': 32}
    }
}

with open('deployment/deployment_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('✓ Created deployment manifest')
"
```

### Option B: Package ONNX Models (After Export)

Once ONNX models are available (via Docker or Python 3.11/3.12):

```bash
python3 deploy_models.py --models-dir trained_models/onnx --output-dir deployment
```

## Step 3: Deploy to Plugin

### Copy Models to Plugin Resources

```bash
# Find plugin Resources directory
# For macOS: Plugin.app/Contents/Resources/models/
# For Windows: Plugin.dll location/Resources/models/
# For Linux: Plugin.so location/Resources/models/

PLUGIN_RESOURCES="/path/to/plugin/Resources"
mkdir -p "$PLUGIN_RESOURCES/models"

# Copy RTNeural JSON models
cp deployment/models/*.json "$PLUGIN_RESOURCES/models/"

# Or copy ONNX models (when available)
# cp deployment/models/*.onnx "$PLUGIN_RESOURCES/models/"
```

### Plugin Integration Points

Models should be loaded in:

1. **`src/ml/ONNXInference.h`** - For ONNX models
2. **`src/ml/RTNeuralProcessor.h`** - For RTNeural JSON models (if implemented)
3. **`src/plugin/PluginProcessor.cpp`** - Main plugin processor

Example loading code (in `PluginProcessor::prepareToPlay`):

```cpp
// Load models from Resources
juce::File modelsDir = juce::File::getSpecialLocation(
    juce::File::currentExecutableFile
).getParentDirectory()
.getChildFile("Resources")
.getChildFile("models");

// Load EmotionRecognizer
if (ENABLE_ONNX_RUNTIME) {
    juce::File emotionModel = modelsDir.getChildFile("emotionrecognizer.onnx");
    if (emotionModel.existsAsFile()) {
        emotionRecognizer_->loadModel(emotionModel);
    }
}
```

## Step 4: Rebuild Plugin with ONNX Runtime

### CMake Configuration

```bash
cd /path/to/plugin/build
cmake .. -DENABLE_ONNX_RUNTIME=ON
```

Or if using RTNeural:

```bash
cmake .. -DENABLE_RTNEURAL=ON
```

### Build Plugin

```bash
cmake --build . --config Release
```

### Verify Models Load

Check plugin logs for:
```
EmotionRecognizer loaded successfully
MelodyTransformer loaded successfully
...
```

## Alternative: Use RTNeural JSON Models

If ONNX export continues to have issues, you can use RTNeural JSON models directly:

1. **RTNeural Integration**: Models are already in RTNeural format
2. **C++ Loading**: Use `RTNeuralProcessor` class (if implemented)
3. **No ONNX Runtime Required**: Lighter dependency

## Docker ONNX Export (When Docker Available)

```bash
# Build Docker image
docker build -t midikompanion-training:latest -f ml_training/Dockerfile ml_training/

# Export to ONNX
docker run --rm \
    -v "$(pwd)/ml_training/trained_models:/workspace/models" \
    midikompanion-training:latest \
    python export_to_onnx.py \
        --models-dir /workspace/models/checkpoints \
        --output-dir /workspace/models/onnx

# Models will be in ml_training/trained_models/onnx/
```

## Summary Checklist

- [x] Step 1: Train models ✅
- [ ] Step 2: Package models (RTNeural ✅, ONNX ⚠️ pending)
- [ ] Step 3: Copy to plugin Resources
- [ ] Step 4: Rebuild plugin with ONNX/RTNeural support
- [ ] Step 5: Test model loading and inference
- [ ] Step 6: A/B test ML vs rule-based generation

## Next Actions

1. **Immediate**: Use RTNeural JSON models (already available)
2. **Short-term**: Export ONNX via Docker or Python 3.11/3.12
3. **Integration**: Update plugin code to load models
4. **Testing**: Verify inference works in plugin
