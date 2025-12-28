# ML Training Data Status

## Current State

### What EXISTS (Code & Infrastructure)

| Component | Location | Status |
|-----------|----------|--------|
| Dataset utilities | `python/penta_core/ml/datasets/` | ✅ Complete |
| Training configs | `ML Kelly Training/backup/configs/` | ✅ 22 configs |
| Model registry | `models/registry.json` | ✅ 7 models defined |
| Prepare datasets script | `scripts/prepare_datasets.py` | ✅ Ready |
| Emotion sampler | `music_brain/emotion_sampler.py` | ✅ Cherry-picked |
| Extended sampler scripts | `scripts/sampler/` | ✅ 9 scripts |
| ML OSC learning | `music_brain/ml_osc/` | ✅ Ready |

### What's MISSING (Data & Samples)

| Item | Expected Location | Status | Action Required |
|------|-------------------|--------|-----------------|
| Audio samples | `datasets/emotion_dataset_v1/` | ❌ Empty | Run emotion sampler |
| Freesound API key | `configs/freesound_config.json` | ❌ Missing | Get API key |
| Downloaded staging | `~/.kelly/emotion_samples/` | ❌ Empty | Run downloads |
| MIDI training files | `datasets/*/raw/midi/` | ❌ Empty | Import or generate |
| Extracted features | `datasets/*/processed/features/` | ❌ Empty | Run extraction |
| Model checkpoints | `checkpoints/` | ⚠️ Minimal | Train models |

## Quick Start: Download Training Data

### 1. Get Freesound API Key

```bash
# Visit https://freesound.org/apiv2/apply/
# Copy your API key

# Create config
cp configs/freesound_config.json.template configs/freesound_config.json
# Edit and add your API key
```

### 2. Run Emotion Sampler

```bash
# Download samples for all 6 base emotions × 4 instruments
python scripts/auto_emotion_sampler.py start

# Or use the module directly
python -c "
from music_brain import EmotionSampler
sampler = EmotionSampler()
results = sampler.search_samples('happy', instrument='piano', max_results=10)
print(f'Found {len(results)} samples')
"
```

### 3. Create Dataset Structure

```bash
python scripts/prepare_datasets.py --create \
    --dataset emotion_dataset_v1 \
    --target-model emotionrecognizer
```

### 4. Import Downloaded Samples

```bash
python scripts/prepare_datasets.py --import-dir ~/.kelly/emotion_samples \
    --dataset emotion_dataset_v1
```

### 5. Extract Features & Augment

```bash
python scripts/prepare_datasets.py --extract-features --dataset emotion_dataset_v1
python scripts/prepare_datasets.py --augment --multiplier 10 --dataset emotion_dataset_v1
```

## Target Dataset Sizes

| Model | Target Samples | Current | Gap |
|-------|---------------|---------|-----|
| EmotionRecognizer | 1,000/emotion (6,000 total) | 0 | 6,000 |
| MelodyTransformer | 5,000+ | 0 | 5,000 |
| HarmonyPredictor | 2,000+ | 0 | 2,000 |
| DynamicsEngine | 1,000+ | 0 | 1,000 |
| GroovePredictor | 1,000/groove type | 0 | 5,000 |

## Extended Sampler Scripts

The `scripts/sampler/` directory contains additional tools:

| Script | Purpose | Command |
|--------|---------|---------|
| `batch_processing.py` | Bulk download | `python scripts/sampler/batch_processing.py` |
| `genre_based_sampler.py` | Sample by genre | `python scripts/sampler/genre_based_sampler.py start` |
| `intensity_sampler.py` | Intensity levels | `python scripts/sampler/intensity_sampler.py start` |
| `dataset_augmentation.py` | Augment existing | `python scripts/sampler/dataset_augmentation.py` |
| `quality_filtering.py` | Filter by quality | `python scripts/sampler/quality_filtering.py` |

## Storage Estimates

| Sample Count | MIDI Size | Audio Size | Features | Total |
|-------------|-----------|------------|----------|-------|
| 1,000 | ~50MB | ~2GB | ~100MB | ~2.2GB |
| 10,000 | ~500MB | ~20GB | ~1GB | ~21.5GB |
| 20,000 | ~1GB | ~40GB | ~2GB | ~43GB |

Your 3TB SSD budget allows for ~150,000 audio samples at 30s each.

## Next Steps

1. **Get Freesound API key** - https://freesound.org/apiv2/apply/
2. **Run auto_emotion_sampler.py** - Downloads base emotion samples
3. **Run extended samplers** - Genre, intensity, instrument families
4. **Prepare datasets** - Structure and extract features
5. **Train models** - Use `scripts/train.py`
