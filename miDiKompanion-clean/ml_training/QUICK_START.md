# ML Training Quick Start Guide

## Prerequisites

```bash
pip install torch torchaudio librosa numpy matplotlib mido
```

## Quick Training (5 minutes)

### 1. Test Export Function

```bash
cd ml_training
python test_export_roundtrip.py
```

This verifies that models can be exported correctly.

### 2. Verify Architecture Alignment

```bash
python verify_architecture_alignment.py
```

This ensures Python models match C++ specifications.

### 3. Train with Synthetic Data (Testing)

```bash
python train_all_models.py \
    --output ./trained_models \
    --use-synthetic \
    --epochs 10 \
    --batch-size 32
```

### 4. Validate Exported Models

```bash
python validate_models.py trained_models/models/ --verbose
```

## Full Training Workflow

### Step 1: Prepare Datasets (Optional)

```bash
# Download datasets to datasets/ directory
# See docs/ML_TRAINING_GUIDE.md for details
```

### Step 2: Configure Training

Edit `config.json` if needed (defaults should work).

### Step 3: Train All Models

```bash
python train_all_models.py \
    --output ./trained_models \
    --datasets-dir ./datasets \
    --epochs 50 \
    --batch-size 64 \
    --device auto
```

### Step 4: Validate Models

```bash
python validate_models.py trained_models/models/ --verbose
```

### Step 5: Test C++ Loading

```bash
cd ../src/ml
# Compile test program (requires RTNeural)
g++ -std=c++17 -DENABLE_RTNEURAL \
    -I/path/to/RTNeural/include \
    test_model_loading.cpp \
    -o test_model_loading

# Run test
./test_model_loading ../ml_training/trained_models/models/emotionrecognizer.json
```

### Step 6: Integrate into Plugin

```bash
# Copy models to plugin resources
cp trained_models/models/*.json /path/to/plugin/Resources/models/
```

## Common Commands

```bash
# Train single model quickly (synthetic data)
python train_emotion_model.py --use-synthetic --epochs 5

# Validate specific model
python validate_models.py trained_models/models/emotionrecognizer.json

# Check architecture matches C++
python verify_architecture_alignment.py

# Test export function
python test_export_roundtrip.py
```

## Troubleshooting

### "CUDA out of memory"

- Use smaller batch size: `--batch-size 32`
- Use CPU: `--device cpu`

### "No datasets found"

- Use synthetic data: `--use-synthetic`
- Or download datasets (see ML_TRAINING_GUIDE.md)

### "Export failed"

- Run `python test_export_roundtrip.py` to diagnose
- Check `validate_models.py` output for JSON errors

## Next Steps

After training:

1. ✅ Validate models: `python validate_models.py models/*.json`
2. ✅ Test C++ loading: Use `test_model_loading.cpp`
3. ✅ Copy to plugin: `cp models/*.json plugin/Resources/models/`
4. ✅ Rebuild plugin and test in DAW

## File Reference

- **Training**: `train_all_models.py`
- **Validation**: `validate_models.py`
- **Architecture Check**: `verify_architecture_alignment.py`
- **Export Test**: `test_export_roundtrip.py`
- **Config**: `config.json`
- **Full Guide**: `../docs/ML_TRAINING_GUIDE.md`
- **Architecture**: `../docs/ML_ARCHITECTURE.md`
