"""
Tests for emotional mapping ML labeling examples.

These tests validate the documentation and examples for using
EmotionalState in ML training pipelines.
"""

import pytest
import json
from pathlib import Path
import sys

# Add data directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.emotional_mapping import (
    EmotionalState,
    get_parameters_for_state,
    EMOTIONAL_STATE_PRESETS,
    EMOTIONAL_PRESETS
)


class TestEmotionalStateLabeling:
    """Test EmotionalState usage for ML labeling."""
    
    def test_basic_labeling(self):
        """Test basic emotional state creation for labeling."""
        # Create emotional state as shown in documentation
        emotion = EmotionalState(
            valence=0.8,
            arousal=0.75,
            primary_emotion="joy"
        )
        
        # Validate
        assert emotion.valence == 0.8
        assert emotion.arousal == 0.75
        assert emotion.primary_emotion == "joy"
        assert emotion.secondary_emotions == []
        assert emotion.has_intrusions is False
        
        # Create label dictionary
        label = {
            "class_name": emotion.primary_emotion,
            "valence": emotion.valence,
            "arousal": emotion.arousal
        }
        
        assert label["class_name"] == "joy"
        assert label["valence"] == 0.8
    
    def test_multi_label_encoding(self):
        """Test multi-label emotional state encoding."""
        # Create complex emotional state
        emotion = EmotionalState(
            valence=-0.6,
            arousal=0.4,
            primary_emotion="grief",
            secondary_emotions=["nostalgia", "loss"]
        )
        
        # Create multi-hot encoding
        all_emotions = ["grief", "nostalgia", "loss", "anger", "joy", "calm"]
        multi_hot = [
            1 if (e == emotion.primary_emotion or e in emotion.secondary_emotions) else 0
            for e in all_emotions
        ]
        
        # Validate
        assert multi_hot == [1, 1, 1, 0, 0, 0]
        assert emotion.secondary_emotions == ["nostalgia", "loss"]
    
    def test_musical_parameters_integration(self):
        """Test converting emotional state to musical parameters."""
        # Create emotional state
        emotion = EmotionalState(
            valence=-0.8,
            arousal=0.3,
            primary_emotion="grief"
        )
        
        # Get musical parameters
        params = get_parameters_for_state(emotion)
        
        # Validate parameters exist
        assert params.tempo_suggested > 0
        assert params.tempo_min > 0
        assert params.tempo_max > 0
        assert 0 <= params.dissonance <= 1
        assert len(params.mode_weights) > 0
        
        # Grief should have specific characteristics
        assert params.tempo_suggested < 90  # Slow tempo
        assert params.dissonance > 0.2  # Some dissonance
    
    def test_presets_usage(self):
        """Test using preset emotional states."""
        # Verify presets exist
        assert "profound_grief" in EMOTIONAL_STATE_PRESETS
        assert "ptsd_anxiety" in EMOTIONAL_STATE_PRESETS
        assert "bittersweet_nostalgia" in EMOTIONAL_STATE_PRESETS
        
        # Use a preset
        emotion = EMOTIONAL_STATE_PRESETS["profound_grief"]
        
        # Validate preset
        assert emotion.primary_emotion == "grief"
        assert emotion.valence < 0  # Negative valence
        assert 0 <= emotion.arousal <= 1
        assert isinstance(emotion.secondary_emotions, list)
    
    def test_manifest_creation(self):
        """Test creating training dataset manifest."""
        # Simulated dataset
        dataset = [
            ("audio/sad_01.wav", EmotionalState(-0.8, 0.3, "grief", ["loss"])),
            ("audio/happy_01.wav", EmotionalState(0.7, 0.8, "joy", ["excitement"]))
        ]
        
        manifest = []
        for audio_path, emotion in dataset:
            entry = {
                "audio": audio_path,
                "label": emotion.primary_emotion,
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "secondary_emotions": emotion.secondary_emotions,
                "has_intrusions": emotion.has_intrusions
            }
            manifest.append(entry)
        
        # Validate manifest
        assert len(manifest) == 2
        assert manifest[0]["label"] == "grief"
        assert manifest[1]["label"] == "joy"
        assert "valence" in manifest[0]
        assert "arousal" in manifest[0]
    
    def test_emotion_to_index_mapping(self):
        """Test creating emotion-to-index mappings."""
        # Create emotion vocabulary
        emotions = ["grief", "joy", "anxiety", "anger", "calm", "nostalgia"]
        emotion_to_idx = {e: i for i, e in enumerate(emotions)}
        idx_to_emotion = {i: e for e, i in emotion_to_idx.items()}
        
        # Validate mappings
        assert len(emotion_to_idx) == 6
        assert emotion_to_idx["grief"] == 0
        assert emotion_to_idx["joy"] == 1
        assert idx_to_emotion[0] == "grief"
        assert idx_to_emotion[1] == "joy"
        
        # Use in labeling
        emotion = EmotionalState(-0.7, 0.3, "grief")
        label_idx = emotion_to_idx[emotion.primary_emotion]
        assert label_idx == 0


class TestEmotionalStateValidation:
    """Test EmotionalState validation."""
    
    def test_valid_ranges(self):
        """Test valid valence and arousal ranges."""
        # Valid state
        emotion = EmotionalState(
            valence=0.5,
            arousal=0.5,
            primary_emotion="calm"
        )
        assert -1 <= emotion.valence <= 1
        assert 0 <= emotion.arousal <= 1
    
    def test_invalid_valence(self):
        """Test invalid valence raises error."""
        with pytest.raises(AssertionError, match="Valence"):
            EmotionalState(valence=1.5, arousal=0.5, primary_emotion="joy")
        
        with pytest.raises(AssertionError, match="Valence"):
            EmotionalState(valence=-1.5, arousal=0.5, primary_emotion="grief")
    
    def test_invalid_arousal(self):
        """Test invalid arousal raises error."""
        with pytest.raises(AssertionError, match="Arousal"):
            EmotionalState(valence=0.5, arousal=-0.1, primary_emotion="calm")
        
        with pytest.raises(AssertionError, match="Arousal"):
            EmotionalState(valence=0.5, arousal=1.5, primary_emotion="joy")
    
    def test_invalid_intrusion_probability(self):
        """Test invalid intrusion probability raises error."""
        with pytest.raises(AssertionError, match="Intrusion"):
            EmotionalState(
                valence=0.5,
                arousal=0.5,
                primary_emotion="calm",
                intrusion_probability=1.5
            )
    
    def test_preset_consistency(self):
        """Test all presets are valid."""
        for name, emotion in EMOTIONAL_STATE_PRESETS.items():
            assert -1 <= emotion.valence <= 1, f"Invalid valence in preset {name}"
            assert 0 <= emotion.arousal <= 1, f"Invalid arousal in preset {name}"
            assert 0 <= emotion.intrusion_probability <= 1, f"Invalid intrusion_probability in preset {name}"


class TestMusicalParametersMapping:
    """Test mapping emotional states to musical parameters."""
    
    def test_grief_parameters(self):
        """Test grief emotion maps to expected parameters."""
        emotion = EmotionalState(
            valence=-0.8,
            arousal=0.3,
            primary_emotion="grief"
        )
        params = get_parameters_for_state(emotion)
        
        # Grief should have slow, sparse, behind-the-beat characteristics
        assert params.tempo_suggested < 90
        assert params.timing_feel.value == "behind"
        assert params.density.value in ["sparse", "medium"]
    
    def test_joy_parameters(self):
        """Test joy emotion maps to expected parameters."""
        # Joy isn't in presets, but should infer from valence/arousal
        emotion = EmotionalState(
            valence=0.8,
            arousal=0.8,
            primary_emotion="joy"
        )
        params = get_parameters_for_state(emotion)
        
        # High arousal should lead to higher tempo
        assert params.tempo_suggested > 100
    
    def test_intrusion_modifiers(self):
        """Test PTSD intrusions affect parameters."""
        # Without intrusions
        emotion1 = EmotionalState(
            valence=-0.6,
            arousal=0.8,
            primary_emotion="anxiety",
            has_intrusions=False
        )
        params1 = get_parameters_for_state(emotion1)
        
        # With intrusions
        emotion2 = EmotionalState(
            valence=-0.6,
            arousal=0.8,
            primary_emotion="anxiety",
            has_intrusions=True,
            intrusion_probability=0.2
        )
        params2 = get_parameters_for_state(emotion2)
        
        # Intrusions should increase dissonance and space
        assert params2.dissonance >= params1.dissonance
        assert params2.space_probability >= params1.space_probability


class TestIntegrationExamples:
    """Test integration examples from documentation."""
    
    def test_jsonl_format(self):
        """Test JSONL format for training manifests."""
        # Create sample data
        emotion = EmotionalState(-0.7, 0.3, "grief", ["loss"])
        
        entry = {
            "audio": "test.wav",
            "label": emotion.primary_emotion,
            "valence": emotion.valence,
            "arousal": emotion.arousal,
            "secondary_emotions": emotion.secondary_emotions
        }
        
        # Serialize to JSON
        json_str = json.dumps(entry)
        
        # Deserialize
        loaded = json.loads(json_str)
        
        # Validate round-trip
        assert loaded["label"] == "grief"
        assert loaded["valence"] == -0.7
        assert loaded["secondary_emotions"] == ["loss"]
    
    def test_training_sample_structure(self):
        """Test creating training sample with both emotional and musical labels."""
        emotion = EmotionalState(
            valence=-0.8,
            arousal=0.3,
            primary_emotion="grief"
        )
        params = get_parameters_for_state(emotion)
        
        # Create training sample
        training_sample = {
            "emotion": {
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "primary": emotion.primary_emotion
            },
            "musical_targets": {
                "tempo": params.tempo_suggested,
                "mode": max(params.mode_weights, key=params.mode_weights.get),
                "dissonance": params.dissonance,
                "timing_feel": params.timing_feel.value
            }
        }
        
        # Validate structure
        assert "emotion" in training_sample
        assert "musical_targets" in training_sample
        assert training_sample["emotion"]["primary"] == "grief"
        assert isinstance(training_sample["musical_targets"]["tempo"], int)
        assert isinstance(training_sample["musical_targets"]["dissonance"], float)
