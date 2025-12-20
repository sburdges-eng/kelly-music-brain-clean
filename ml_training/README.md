# miDiKompanion ML Training

Train and deploy ML models for emotion-to-music generation.

## Quick Start

### Docker (Recommended)
```bash
docker-compose up training
docker-compose up export
```

### Local
```bash
pip install -r requirements.txt
python train_all_models.py --epochs 100
python export_to_onnx.py
```

### Google Colab
See `Train_MidiKompanion_Models.ipynb`

## Files

| File | Description |
|------|-------------|
| `train_all_models.py` | Main training script for all 5 models |
| `prepare_datasets.py` | Dataset preparation and synthetic data generation |
| `export_to_onnx.py` | Export PyTorch models to ONNX format |
| `validate_models.py` | Validate ONNX models |
| `Dockerfile` | GPU training environment |
| `docker-compose.yml` | Container orchestration |
| `requirements.txt` | Python dependencies |

## Models

| Model | Input | Output | Description |
|-------|-------|--------|-------------|
| EmotionRecognizer | 128 | 64 | Audio → Emotion |
| MelodyTransformer | 64 | 128 | Emotion → Melody |
| HarmonyPredictor | 128 | 64 | Context → Chords |
| DynamicsEngine | 32 | 16 | Intensity → Expression |
| GroovePredictor | 64 | 32 | Arousal → Groove |

## Output

Trained models are exported to `onnx/` directory:
- `emotion_recognizer.onnx`
- `melody_transformer.onnx`
- `harmony_predictor.onnx`
- `dynamics_engine.onnx`
- `groove_predictor.onnx`

Total size: < 5 MB
