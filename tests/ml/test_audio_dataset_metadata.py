"""
Test for enhanced AudioDataset with emotional and intent metadata.
"""

import json
import pytest
import tempfile
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Test if PyTorch is available
try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    pytest.skip("PyTorch not available", allow_module_level=True)

# Import after PyTorch check
from scripts.train_model import AudioDataset


def create_test_metadata():
    """Create test metadata with emotional and intent attributes."""
    return {
        "dataset_id": "test_emotion_v1",
        "samples": [
            {
                "file": "test_audio_001.wav",
                "emotion": "happy",
                "valence": 0.8,
                "arousal": 0.6,
                "intensity_tier": 3,
                "node_id": 42,
                "key": "C major",
                "tempo_bpm": 120.0,
                "mode": "major",
                "chord_progression": ["C", "Am", "F", "G"],
                "groove_type": "straight",
                "tags": ["uplifting", "energetic"],
                "quality_score": 0.9,
                "rule_breaks": [],
            },
            {
                "file": "test_audio_002.wav",
                "emotion": "sad",
                "valence": -0.7,
                "arousal": 0.3,
                "intensity_tier": 4,
                "node_id": 78,
                "key": "A minor",
                "tempo_bpm": 72.0,
                "mode": "minor",
                "chord_progression": ["Am", "F", "C", "G"],
                "groove_type": "laid_back",
                "tags": ["melancholic"],
                "quality_score": 0.85,
                "rule_breaks": [],
            },
            {
                "file": "test_audio_003.wav",
                "emotion": "grief",
                "valence": -0.9,
                "arousal": 0.2,
                "intensity_tier": 5,
                "node_id": 156,
                "key": "F major",
                "tempo_bpm": 68.0,
                "mode": "major",
                "chord_progression": ["F", "C", "Dm", "Bbm"],
                "groove_type": "laid_back",
                "tags": ["grief", "bittersweet"],
                "quality_score": 0.95,
                "rule_breaks": ["HARMONY_ModalInterchange"],
            },
        ]
    }


@pytest.fixture
def test_dataset_dir():
    """Create a temporary dataset directory with metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create metadata.json
        metadata = create_test_metadata()
        metadata_path = tmpdir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create processed directory structure (even though no actual audio files)
        processed_dir = tmpdir / "processed"
        processed_dir.mkdir()
        
        yield tmpdir


def test_load_metadata_with_emotional_attributes(test_dataset_dir):
    """Test loading metadata with emotional attributes."""
    dataset = AudioDataset(
        data_dir=test_dataset_dir,
        sample_rate=16000,
        n_mels=64,
        max_duration=5.0,
    )
    
    assert len(dataset) == 3
    
    # Test first sample
    metadata = dataset.get_sample_metadata(0)
    assert metadata["emotion"] == "happy"
    assert metadata["valence"] == 0.8
    assert metadata["arousal"] == 0.6
    assert metadata["intensity_tier"] == 3
    assert metadata["node_id"] == 42


def test_get_emotion_labels(test_dataset_dir):
    """Test extraction of emotion labels."""
    dataset = AudioDataset(
        data_dir=test_dataset_dir,
        sample_rate=16000,
    )
    
    # Test happy sample
    labels = dataset.get_emotion_labels(0)
    assert labels["emotion"] == "happy"
    assert labels["valence"] == 0.8
    assert labels["arousal"] == 0.6
    assert labels["intensity_tier"] == 3
    assert labels["node_id"] == 42
    
    # Test sad sample
    labels = dataset.get_emotion_labels(1)
    assert labels["emotion"] == "sad"
    assert labels["valence"] == -0.7
    assert labels["arousal"] == 0.3
    assert labels["intensity_tier"] == 4
    assert labels["node_id"] == 78


def test_get_musical_metadata(test_dataset_dir):
    """Test extraction of musical intent metadata."""
    dataset = AudioDataset(
        data_dir=test_dataset_dir,
        sample_rate=16000,
    )
    
    # Test first sample
    musical = dataset.get_musical_metadata(0)
    assert musical["key"] == "C major"
    assert musical["tempo_bpm"] == 120.0
    assert musical["mode"] == "major"
    assert musical["chord_progression"] == ["C", "Am", "F", "G"]
    assert musical["groove_type"] == "straight"
    
    # Test grief sample with modal interchange
    musical = dataset.get_musical_metadata(2)
    assert musical["key"] == "F major"
    assert musical["chord_progression"] == ["F", "C", "Dm", "Bbm"]
    assert musical["groove_type"] == "laid_back"


def test_metadata_validation(test_dataset_dir):
    """Test metadata validation and default values."""
    # Create dataset with out-of-range values
    invalid_metadata = {
        "samples": [
            {
                "file": "test.wav",
                "emotion": "test",
                "valence": 2.0,  # Out of range
                "arousal": -0.5,  # Out of range
                "intensity_tier": 10,  # Out of range
            }
        ]
    }
    
    metadata_path = test_dataset_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(invalid_metadata, f)
    
    dataset = AudioDataset(
        data_dir=test_dataset_dir,
        sample_rate=16000,
    )
    
    # Values should be clamped
    metadata = dataset.get_sample_metadata(0)
    assert -1.0 <= metadata["valence"] <= 1.0
    assert 0.0 <= metadata["arousal"] <= 1.0
    assert 0 <= metadata["intensity_tier"] <= 5


def test_fallback_directory_scanning(test_dataset_dir):
    """Test fallback to directory scanning when no metadata.json exists."""
    # Remove metadata.json
    metadata_path = test_dataset_dir / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
    
    # Create directory structure
    processed_dir = test_dataset_dir / "processed"
    happy_dir = processed_dir / "happy"
    happy_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy wav files
    (happy_dir / "test1.wav").touch()
    (happy_dir / "test2.wav").touch()
    
    dataset = AudioDataset(
        data_dir=test_dataset_dir,
        sample_rate=16000,
    )
    
    assert len(dataset) == 2
    
    # Check default values are set
    metadata = dataset.get_sample_metadata(0)
    assert metadata["emotion"] == "happy"
    assert metadata["valence"] == 0.0  # Default
    assert metadata["arousal"] == 0.0  # Default


def test_sample_metadata_defaults(test_dataset_dir):
    """Test that missing optional fields have proper defaults."""
    minimal_metadata = {
        "samples": [
            {
                "file": "minimal.wav",
                "emotion": "neutral",
            }
        ]
    }
    
    metadata_path = test_dataset_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(minimal_metadata, f)
    
    dataset = AudioDataset(
        data_dir=test_dataset_dir,
        sample_rate=16000,
    )
    
    metadata = dataset.get_sample_metadata(0)
    
    # Check defaults are set
    assert metadata["valence"] == 0.0
    assert metadata["arousal"] == 0.0
    assert metadata["intensity_tier"] == 3
    assert metadata["key"] == ""
    assert metadata["tempo_bpm"] == 0.0
    assert metadata["mode"] == ""
    assert metadata["chord_progression"] == []
    assert metadata["groove_type"] == "straight"
    assert metadata["tags"] == []
    assert metadata["quality_score"] == 0.0
    assert metadata["rule_breaks"] == []


def test_dataset_with_dataloader(test_dataset_dir):
    """Test that dataset works with PyTorch DataLoader."""
    dataset = AudioDataset(
        data_dir=test_dataset_dir,
        sample_rate=16000,
        n_mels=64,
        max_duration=5.0,
    )
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    
    # Test iteration (will use dummy data since no real audio files)
    for batch_idx, (inputs, labels) in enumerate(loader):
        assert inputs.shape[0] <= 2  # Batch size
        assert labels.shape[0] <= 2
        break  # Just test one batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
