# Extended Sampler Scripts

A collection of utility scripts for the Emotion Sampler ecosystem.

## Scripts

| Script | Purpose |
|--------|---------|
| `batch_processing.py` | Batch download and process samples |
| `blend_emotions.py` | Blend multiple emotion samples |
| `dataset_augmentation.py` | Augment sample datasets for ML training |
| `extended_instruments.py` | Extended instrument family support |
| `genre_based_sampler.py` | Genre-aware sample selection |
| `instrument_family_sampler.py` | Sample by instrument family |
| `intensity_sampler.py` | Intensity-based sample filtering |
| `quality_filtering.py` | Quality assessment and filtering |
| `realtime_emotion_detection.py` | Real-time emotion detection from audio |

## Usage

These scripts work with the `EmotionSampler` module in `music_brain`:

```python
from music_brain import EmotionSampler

sampler = EmotionSampler()
results = sampler.search_samples("happy", instrument="piano")
```

See `docs/EXTENDED_SAMPLER_ECOSYSTEM.md` for full documentation.
