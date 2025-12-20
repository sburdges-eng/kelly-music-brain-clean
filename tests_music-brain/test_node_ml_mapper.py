"""
Tests for NodeMLMapper - the Python ML-to-Node bridge.

Tests the mapping of emotions to the 216-node thesaurus and
extraction of musical parameters.
"""

import pytest
from pathlib import Path


class TestNodeMLMapperBasics:
    """Test basic NodeMLMapper functionality."""

    def test_import_node_ml_mapper(self):
        """NodeMLMapper can be imported."""
        from music_brain.audio.node_ml_mapper import (
            NodeMLMapper,
            EmotionNode,
            VADCoordinates,
            MusicalMapping,
        )
        assert NodeMLMapper is not None
        assert EmotionNode is not None
        assert VADCoordinates is not None
        assert MusicalMapping is not None

    def test_create_mapper(self):
        """NodeMLMapper can be instantiated."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        assert mapper is not None

    def test_loads_thesaurus(self):
        """NodeMLMapper loads the 216-node thesaurus."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        # Should have 216 nodes
        assert mapper.node_count == 216
        assert len(mapper) == 216

    def test_get_categories(self):
        """Can retrieve all emotion categories."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        categories = mapper.get_categories()
        
        assert len(categories) >= 6
        assert "joy" in categories
        assert "sad" in categories
        assert "anger" in categories
        assert "fear" in categories
        assert "surprise" in categories
        assert "disgust" in categories


class TestVADCoordinates:
    """Test VADCoordinates functionality."""

    def test_create_vad(self):
        """Can create VAD coordinates."""
        from music_brain.audio.node_ml_mapper import VADCoordinates
        vad = VADCoordinates(valence=0.5, arousal=0.6, dominance=0.7)
        assert vad.valence == 0.5
        assert vad.arousal == 0.6
        assert vad.dominance == 0.7

    def test_vad_distance(self):
        """VAD distance calculation works."""
        from music_brain.audio.node_ml_mapper import VADCoordinates
        
        vad1 = VADCoordinates(0.0, 0.0, 0.0)
        vad2 = VADCoordinates(1.0, 0.0, 0.0)
        
        # Distance should be 1.0 in valence axis
        assert abs(vad1.distance_to(vad2) - 1.0) < 0.01
        
        # Same point should have zero distance
        assert vad1.distance_to(vad1) == 0.0

    def test_vad_to_dict(self):
        """VAD can convert to dictionary."""
        from music_brain.audio.node_ml_mapper import VADCoordinates
        vad = VADCoordinates(0.5, 0.6, 0.7, 0.8)
        d = vad.to_dict()
        
        assert d["valence"] == 0.5
        assert d["arousal"] == 0.6
        assert d["dominance"] == 0.7
        assert d["intensity"] == 0.8

    def test_vad_from_dict(self):
        """VAD can be created from dictionary."""
        from music_brain.audio.node_ml_mapper import VADCoordinates
        d = {"valence": 0.5, "arousal": 0.6, "dominance": 0.7, "intensity": 0.8}
        vad = VADCoordinates.from_dict(d)
        
        assert vad.valence == 0.5
        assert vad.arousal == 0.6
        assert vad.dominance == 0.7
        assert vad.intensity == 0.8


class TestNodeQuerying:
    """Test node querying functionality."""

    def test_find_nearest_node(self):
        """Can find nearest node to VAD coordinates."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        # Find a happy emotion (high valence, medium arousal)
        node = mapper.find_nearest_node(valence=0.8, arousal=0.5, dominance=0.6)
        assert node is not None
        assert node.category == "joy"

    def test_find_sad_emotion(self):
        """Can find sad emotion nodes."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        # Find a sad emotion (low valence, low arousal)
        node = mapper.find_nearest_node(valence=-0.6, arousal=0.2, dominance=0.3)
        assert node is not None
        assert node.category == "sad"

    def test_find_angry_emotion(self):
        """Can find angry emotion nodes."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        # Find an angry emotion (negative valence, high arousal, high dominance)
        node = mapper.find_nearest_node(valence=-0.5, arousal=0.7, dominance=0.8)
        assert node is not None
        assert node.category == "anger"

    def test_find_nodes_in_range(self):
        """Can find multiple nodes within a VAD radius."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        results = mapper.find_nodes_in_range(
            valence=0.7,
            arousal=0.5,
            dominance=0.5,
            radius=0.3,
            max_results=10
        )
        
        assert len(results) > 0
        # Results should be sorted by distance
        if len(results) > 1:
            assert results[0][1] <= results[1][1]

    def test_get_node_by_id(self):
        """Can retrieve a node by ID."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        node = mapper.get_node(0)
        assert node is not None
        assert node.id == 0

    def test_get_nodes_by_category(self):
        """Can get all nodes in a category."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        joy_nodes = mapper.get_nodes_by_category("joy")
        assert len(joy_nodes) > 0
        for node in joy_nodes:
            assert node.category == "joy"

    def test_get_nodes_by_subcategory(self):
        """Can get nodes by subcategory."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        grief_nodes = mapper.get_nodes_by_subcategory("sad", "grief")
        assert len(grief_nodes) > 0
        for node in grief_nodes:
            assert node.category == "sad"
            assert node.subcategory == "grief"


class TestRelatedEmotions:
    """Test related emotion discovery."""

    def test_get_related_nodes(self):
        """Can get related emotion nodes."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        # Get a node with related emotions
        node = mapper.get_node(0)
        if node and node.related_emotions:
            related = mapper.get_related_nodes(node.id)
            assert len(related) > 0

    def test_get_node_context(self):
        """Can get full node context."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        context = mapper.get_node_context(0)
        assert context is not None
        assert context.node is not None
        assert context.confidence == 1.0


class TestMusicalMapping:
    """Test musical parameter extraction."""

    def test_get_musical_mapping(self):
        """Can get musical parameters from a node."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        node = mapper.get_node(0)
        params = mapper.get_musical_mapping(node)
        
        assert params is not None
        assert params.mode in ("major", "minor")
        assert params.tempo_multiplier > 0
        assert params.suggested_tempo_bpm > 0
        assert 0 <= params.dynamics_scale <= 1.0
        assert len(params.velocity_range) == 2

    def test_joy_is_major_mode(self):
        """Joy emotions should map to major mode."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        joy_nodes = mapper.get_nodes_by_category("joy")
        for node in joy_nodes:
            assert node.mode == "major"

    def test_sad_is_minor_mode(self):
        """Sad emotions should map to minor mode."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        sad_nodes = mapper.get_nodes_by_category("sad")
        for node in sad_nodes:
            assert node.mode == "minor"


class TestTransitions:
    """Test emotional transition paths."""

    def test_get_transition_path(self):
        """Can get transition path between emotions."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        # Transition from joy to sad
        joy_node = mapper.get_nodes_by_category("joy")[0]
        sad_node = mapper.get_nodes_by_category("sad")[0]
        
        path = mapper.get_transition_path(joy_node.id, sad_node.id, steps=5)
        
        assert len(path) >= 2
        assert path[0].id == joy_node.id
        assert path[-1].id == sad_node.id


class TestBasicEmotionMapping:
    """Test mapping from basic emotion labels."""

    def test_map_basic_happy(self):
        """Can map 'happy' to a node."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        node = mapper.map_basic_emotion("happy")
        assert node is not None
        assert node.category == "joy"

    def test_map_basic_sad(self):
        """Can map 'sad' to a node."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        node = mapper.map_basic_emotion("sad")
        assert node is not None
        assert node.category == "sad"

    def test_map_basic_angry(self):
        """Can map 'angry' to a node."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        node = mapper.map_basic_emotion("angry")
        assert node is not None
        assert node.category == "anger"

    def test_map_basic_neutral(self):
        """Can map 'neutral' to a node."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        node = mapper.map_basic_emotion("neutral")
        assert node is not None

    @pytest.mark.parametrize("emotion", [
        "fear", "surprise", "disgust", "calm", "excited", "anxious"
    ])
    def test_map_various_emotions(self, emotion):
        """Can map various emotion labels."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        node = mapper.map_basic_emotion(emotion)
        assert node is not None


class TestEmotionNode:
    """Test EmotionNode dataclass."""

    def test_emotion_node_to_dict(self):
        """EmotionNode can convert to dictionary."""
        from music_brain.audio.node_ml_mapper import NodeMLMapper
        mapper = NodeMLMapper()
        
        node = mapper.get_node(0)
        d = node.to_dict()
        
        assert "id" in d
        assert "name" in d
        assert "category" in d
        assert "subcategory" in d
        assert "vad" in d
        assert "mode" in d

    def test_emotion_node_from_dict(self):
        """EmotionNode can be created from dictionary."""
        from music_brain.audio.node_ml_mapper import EmotionNode
        
        d = {
            "id": 999,
            "name": "test_emotion",
            "category": "test",
            "subcategory": "test_sub",
            "vad": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5, "intensity": 0.5},
            "relatedEmotions": [1, 2, 3],
            "mode": "major",
            "tempoMultiplier": 1.0,
            "dynamicsScale": 0.75,
        }
        
        node = EmotionNode.from_dict(d)
        assert node.id == 999
        assert node.name == "test_emotion"
        assert node.mode == "major"


class TestModuleExports:
    """Test module-level exports."""

    def test_audio_module_exports(self):
        """NodeMLMapper is exported from audio module."""
        from music_brain.audio import (
            NodeMLMapper,
            NODE_ML_MAPPER_AVAILABLE,
        )
        assert NODE_ML_MAPPER_AVAILABLE is True
        assert NodeMLMapper is not None

    def test_create_mapper_function(self):
        """create_mapper convenience function works."""
        from music_brain.audio import create_mapper
        mapper = create_mapper()
        assert mapper is not None
        assert mapper.node_count == 216
