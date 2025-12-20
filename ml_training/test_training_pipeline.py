#!/usr/bin/env python3
"""
Test suite for miDiKompanion ML training pipeline.

Tests:
- Model architecture validation
- Dataset generation
- ONNX export
- Inference correctness
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add ml_training to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules under test
try:
    from train_all_models import (
        EmotionRecognizer, 
        MelodyTransformer, 
        HarmonyPredictor, 
        DynamicsEngine, 
        GroovePredictor,
        SyntheticEmotionDataset
    )
    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False

try:
    from prepare_datasets import VADCalculator, DatasetBuilder
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ==============================================================================
# Model Architecture Tests
# ==============================================================================

@pytest.mark.skipif(not HAS_TRAINING, reason="Training module not available")
class TestModelArchitectures:
    """Test neural network architectures."""
    
    def test_emotion_recognizer_shape(self):
        """Test EmotionRecognizer input/output shapes."""
        import torch
        
        model = EmotionRecognizer(input_dim=128, hidden_dim=256, embedding_dim=64)
        x = torch.randn(32, 128)  # Batch of 32, 128 features
        y = model(x)
        
        assert y.shape == (32, 64), f"Expected (32, 64), got {y.shape}"
    
    def test_melody_transformer_shape(self):
        """Test MelodyTransformer input/output shapes."""
        import torch
        
        model = MelodyTransformer(embedding_dim=64, hidden_dim=256, output_dim=128)
        x = torch.randn(32, 64)
        y = model(x)
        
        assert y.shape == (32, 128), f"Expected (32, 128), got {y.shape}"
        assert torch.all(y >= 0) and torch.all(y <= 1), "Output should be in [0, 1]"
    
    def test_harmony_predictor_shape(self):
        """Test HarmonyPredictor input/output shapes."""
        import torch
        
        model = HarmonyPredictor(context_dim=128, hidden_dim=256, num_chords=64)
        x = torch.randn(32, 128)
        y = model(x)
        
        assert y.shape == (32, 64), f"Expected (32, 64), got {y.shape}"
        # Should be probabilities (sum to 1)
        sums = y.sum(dim=1)
        assert torch.allclose(sums, torch.ones(32), atol=1e-5), "Output should sum to 1"
    
    def test_dynamics_engine_shape(self):
        """Test DynamicsEngine input/output shapes."""
        import torch
        
        model = DynamicsEngine(intensity_dim=32, hidden_dim=128, output_dim=16)
        x = torch.randn(32, 32)
        y = model(x)
        
        assert y.shape == (32, 16), f"Expected (32, 16), got {y.shape}"
    
    def test_groove_predictor_shape(self):
        """Test GroovePredictor input/output shapes."""
        import torch
        
        model = GroovePredictor(arousal_dim=64, hidden_dim=128, output_dim=32)
        x = torch.randn(32, 64)
        y = model(x)
        
        assert y.shape == (32, 32), f"Expected (32, 32), got {y.shape}"


# ==============================================================================
# Dataset Tests
# ==============================================================================

@pytest.mark.skipif(not HAS_DATASETS, reason="Dataset module not available")
class TestDatasets:
    """Test dataset generation and processing."""
    
    def test_vad_calculator_node_0(self):
        """Test VAD calculation for node 0 (Happy/Subtle)."""
        vad = VADCalculator.get_vad_for_node(0)
        
        assert len(vad) == 4, "VAD should be 4-tuple (V, A, D, I)"
        assert -1 <= vad[0] <= 1, "Valence should be in [-1, 1]"
        assert 0 <= vad[1] <= 1, "Arousal should be in [0, 1]"
        assert 0 <= vad[2] <= 1, "Dominance should be in [0, 1]"
        assert 0 <= vad[3] <= 1, "Intensity should be in [0, 1]"
    
    def test_vad_calculator_node_215(self):
        """Test VAD calculation for node 215 (Disgust/Extreme)."""
        vad = VADCalculator.get_vad_for_node(215)
        
        assert vad[3] > 0.8, "Extreme intensity should be > 0.8"
    
    def test_node_indices_roundtrip(self):
        """Test node ID to indices conversion."""
        for node_id in [0, 42, 100, 215]:
            base, sub, intensity = VADCalculator.node_id_to_indices(node_id)
            reconstructed = base * 36 + sub * 6 + intensity
            assert reconstructed == node_id, f"Node {node_id} roundtrip failed"
    
    def test_synthetic_dataset_generation(self):
        """Test synthetic dataset generation."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(Path(tmpdir))
            data = builder.generate_synthetic_dataset('emotion', num_samples=100)
            
            assert 'X' in data, "Dataset should have X"
            assert 'y_vad' in data, "Dataset should have y_vad"
            assert data['X'].shape == (100, 128), "X shape should be (100, 128)"
            assert data['y_vad'].shape == (100, 4), "y_vad shape should be (100, 4)"


# ==============================================================================
# ONNX Export Tests
# ==============================================================================

@pytest.mark.skipif(not HAS_TRAINING, reason="Training module not available")
class TestONNXExport:
    """Test ONNX model export."""
    
    def test_model_export(self):
        """Test that a model can be exported to ONNX."""
        import torch
        import tempfile
        
        model = EmotionRecognizer(input_dim=128, hidden_dim=256, embedding_dim=64)
        model.eval()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as f:
            dummy_input = torch.randn(1, 128)
            
            torch.onnx.export(
                model,
                dummy_input,
                f.name,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify file exists and has content
            assert Path(f.name).stat().st_size > 0, "ONNX file should not be empty"
    
    def test_onnx_inference(self):
        """Test ONNX inference produces valid output."""
        import torch
        import tempfile
        
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("ONNX Runtime not available")
        
        model = EmotionRecognizer(input_dim=128, hidden_dim=256, embedding_dim=64)
        model.eval()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            dummy_input = torch.randn(1, 128)
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['output']
            )
            
            # Load and run
            session = ort.InferenceSession(onnx_path)
            input_data = np.random.randn(1, 128).astype(np.float32)
            output = session.run(None, {'input': input_data})[0]
            
            assert output.shape == (1, 64), f"Expected (1, 64), got {output.shape}"
        finally:
            Path(onnx_path).unlink()


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_emotion_node_to_embedding(self):
        """Test emotion node to embedding conversion."""
        if not HAS_DATASETS:
            pytest.skip("Dataset module not available")
        
        # Generate embedding for each base emotion
        for base_idx in range(6):
            node_id = base_idx * 36  # First node of each base emotion
            vad = VADCalculator.get_vad_for_node(node_id)
            
            # Create 64-dim embedding
            embedding = np.zeros(64, dtype=np.float32)
            embedding[:4] = vad
            
            assert not np.isnan(embedding).any(), f"Node {node_id} produced NaN"
    
    def test_latency_estimation(self):
        """Test that inference is fast enough for real-time."""
        if not HAS_TRAINING:
            pytest.skip("Training module not available")
        
        import torch
        import time
        
        model = MelodyTransformer(embedding_dim=64, hidden_dim=256, output_dim=128)
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                model(torch.randn(1, 64))
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                model(torch.randn(1, 64))
                times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        assert avg_time < 50, f"Inference took {avg_time:.2f}ms, should be < 50ms"
    
    def test_216_nodes_coverage(self):
        """Test that all 216 nodes produce valid output."""
        if not HAS_DATASETS:
            pytest.skip("Dataset module not available")
        
        for node_id in range(216):
            vad = VADCalculator.get_vad_for_node(node_id)
            
            # All values should be finite
            assert all(np.isfinite(v) for v in vad), f"Node {node_id} produced non-finite VAD"
            
            # Valence in [-1, 1], others in [0, 1]
            assert -1 <= vad[0] <= 1, f"Node {node_id} valence out of range"
            assert 0 <= vad[1] <= 1, f"Node {node_id} arousal out of range"
            assert 0 <= vad[2] <= 1, f"Node {node_id} dominance out of range"
            assert 0 <= vad[3] <= 1, f"Node {node_id} intensity out of range"


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
