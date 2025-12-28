"""
Tests for Intent Dataset ML Training Utilities.

Tests the intent_dataset module for ML training with CompleteSongIntent schema.
"""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np

# Import intent schema
try:
    from music_brain.session.intent_schema import (
        CompleteSongIntent,
        SongRoot,
        SongIntent,
        TechnicalConstraints,
    )
    INTENT_AVAILABLE = True
except ImportError:
    INTENT_AVAILABLE = False

# Import ML dataset utilities
try:
    from python.penta_core.ml.datasets.intent_dataset import (
        IntentDataset,
        IntentEncoder,
        IntentEncodingConfig,
        validate_dataset_for_training,
    )
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False


# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (INTENT_AVAILABLE and DATASET_AVAILABLE),
    reason="Intent schema or dataset utilities not available"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_intent():
    """Create a sample intent for testing."""
    return CompleteSongIntent(
        title="Test Song",
        created="2024-12-28",
        song_root=SongRoot(
            core_event="Test event",
            core_longing="Test longing",
        ),
        song_intent=SongIntent(
            mood_primary="grief",
            mood_secondary_tension=0.7,
            vulnerability_scale="High",
            narrative_arc="Slow Reveal",
        ),
        technical_constraints=TechnicalConstraints(
            technical_genre="Lo-fi",
            technical_tempo_range=(80, 100),
            technical_key="F",
            technical_mode="major",
            technical_rule_to_break="HARMONY_ModalInterchange",
            rule_breaking_justification="Test justification",
        ),
    )


@pytest.fixture
def temp_intent_dir(sample_intent):
    """Create temporary directory with intent JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save sample intent
        intent_path = tmpdir / "test_intent.json"
        sample_intent.save(str(intent_path))
        
        # Create second intent for batch tests
        intent2 = CompleteSongIntent(
            title="Test Song 2",
            created="2024-12-28",
            song_root=SongRoot(
                core_event="Test event 2",
                core_longing="Test longing 2",
            ),
            song_intent=SongIntent(
                mood_primary="joy",
                mood_secondary_tension=0.3,
                vulnerability_scale="Low",
                narrative_arc="Climb-to-Climax",
            ),
            technical_constraints=TechnicalConstraints(
                technical_tempo_range=(120, 140),
            ),
        )
        intent2.save(str(tmpdir / "test_intent2.json"))
        
        yield tmpdir


# =============================================================================
# IntentEncoder Tests
# =============================================================================


class TestIntentEncoder:
    """Test IntentEncoder class."""
    
    def test_encoder_initialization(self):
        """Test encoder initializes with default config."""
        encoder = IntentEncoder()
        assert encoder.config is not None
        assert len(encoder.emotion_to_id) > 0
        assert len(encoder.all_rules) > 0
    
    def test_encoder_custom_config(self):
        """Test encoder with custom config."""
        config = IntentEncodingConfig(
            emotion_labels=["grief", "joy", "anger"],
        )
        encoder = IntentEncoder(config)
        assert len(encoder.emotion_to_id) == 3
        assert "grief" in encoder.emotion_to_id
    
    def test_encode_intent(self, sample_intent):
        """Test encoding intent to targets."""
        encoder = IntentEncoder()
        targets = encoder.encode_intent(sample_intent)
        
        # Check all expected keys present
        assert "emotion_label" in targets
        assert "emotion_onehot" in targets
        assert "tension" in targets
        assert "vulnerability" in targets
        assert "narrative_arc_label" in targets
        assert "rule_break_id" in targets
        assert "tempo_normalized" in targets
        assert "intent_valid" in targets
    
    def test_encode_emotion(self, sample_intent):
        """Test emotion encoding."""
        encoder = IntentEncoder()
        targets = encoder.encode_intent(sample_intent)
        
        assert targets["emotion_str"] == "grief"
        assert isinstance(targets["emotion_label"], (int, np.integer))
        assert isinstance(targets["emotion_onehot"], np.ndarray)
        assert targets["emotion_onehot"].sum() == 1.0
    
    def test_encode_tension(self, sample_intent):
        """Test tension encoding."""
        encoder = IntentEncoder()
        targets = encoder.encode_intent(sample_intent)
        
        assert 0.0 <= targets["tension"] <= 1.0
        assert abs(targets["tension"] - 0.7) < 0.01
    
    def test_encode_rule_break(self, sample_intent):
        """Test rule break encoding."""
        encoder = IntentEncoder()
        targets = encoder.encode_intent(sample_intent)
        
        assert targets["rule_break_str"] == "HARMONY_ModalInterchange"
        assert isinstance(targets["rule_break_id"], (int, np.integer))
        assert targets["has_justification"] is True
    
    def test_decode_emotion(self):
        """Test decoding emotion ID."""
        encoder = IntentEncoder()
        emotion_str = encoder.decode_emotion(0)
        assert isinstance(emotion_str, str)
        assert emotion_str in encoder.config.emotion_labels


# =============================================================================
# IntentDataset Tests
# =============================================================================


class TestIntentDataset:
    """Test IntentDataset class."""
    
    def test_dataset_initialization(self, temp_intent_dir):
        """Test dataset loads intents from directory."""
        dataset = IntentDataset(intent_dir=temp_intent_dir)
        assert len(dataset) == 2
    
    def test_dataset_getitem(self, temp_intent_dir):
        """Test getting single sample."""
        dataset = IntentDataset(intent_dir=temp_intent_dir)
        sample = dataset[0]
        
        assert "intent" in sample
        assert "targets" in sample
        assert "path" in sample
        assert isinstance(sample["intent"], CompleteSongIntent)
        assert isinstance(sample["targets"], dict)
    
    def test_dataset_length(self, temp_intent_dir):
        """Test dataset length."""
        dataset = IntentDataset(intent_dir=temp_intent_dir)
        assert len(dataset) == 2
    
    def test_get_batch(self, temp_intent_dir):
        """Test getting batch of samples."""
        dataset = IntentDataset(intent_dir=temp_intent_dir)
        batch = dataset.get_batch([0, 1])
        
        assert "emotion_labels" in batch
        assert "tensions" in batch
        assert "tempo_bpm" in batch
        assert batch["emotion_labels"].shape == (2,)
        assert batch["tensions"].shape == (2,)
    
    def test_get_statistics(self, temp_intent_dir):
        """Test getting dataset statistics."""
        dataset = IntentDataset(intent_dir=temp_intent_dir)
        stats = dataset.get_statistics()
        
        assert "total_samples" in stats
        assert stats["total_samples"] == 2
        assert "emotion_distribution" in stats
        assert "tension_stats" in stats
        assert "tempo_stats" in stats
    
    def test_dataset_validation(self, temp_intent_dir):
        """Test dataset validates intents on load."""
        dataset = IntentDataset(
            intent_dir=temp_intent_dir,
            validate_on_load=True
        )
        assert len(dataset) == 2


# =============================================================================
# Validation Tests
# =============================================================================


class TestDatasetValidation:
    """Test dataset validation utilities."""
    
    def test_validate_dataset_for_training(self, temp_intent_dir):
        """Test validating dataset for training."""
        dataset = IntentDataset(intent_dir=temp_intent_dir)
        is_valid, issues = validate_dataset_for_training(
            dataset,
            min_samples_per_emotion=1
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
    
    def test_validation_detects_imbalance(self, temp_intent_dir):
        """Test validation detects class imbalance."""
        dataset = IntentDataset(intent_dir=temp_intent_dir)
        is_valid, issues = validate_dataset_for_training(
            dataset,
            min_samples_per_emotion=5  # High requirement
        )
        
        assert not is_valid
        assert len(issues) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Test integration scenarios."""
    
    def test_full_pipeline(self, temp_intent_dir):
        """Test full pipeline: load -> encode -> batch."""
        # Load dataset
        dataset = IntentDataset(intent_dir=temp_intent_dir)
        assert len(dataset) == 2
        
        # Get sample
        sample = dataset[0]
        assert sample["targets"]["intent_valid"] is True
        
        # Get batch
        batch = dataset.get_batch([0, 1])
        assert batch["emotion_labels"].shape == (2,)
        
        # Get statistics
        stats = dataset.get_statistics()
        assert stats["total_samples"] == 2
    
    def test_custom_encoding_config(self, temp_intent_dir):
        """Test using custom encoding configuration."""
        config = IntentEncodingConfig(
            emotion_labels=["grief", "joy"],
            tempo_range=(60, 140),
        )
        
        dataset = IntentDataset(
            intent_dir=temp_intent_dir,
            encoding_config=config
        )
        
        assert len(dataset) == 2
        sample = dataset[0]
        # Grief should be encoded as 0
        assert sample["targets"]["emotion_label"] in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
