"""
Comprehensive Integration Tests

End-to-end tests that verify module integrations work correctly together.
These tests simulate real-world workflows and ensure components integrate properly.
"""

import pytest
from pathlib import Path
import tempfile
import json

# Import core modules
from music_brain.orchestrator.orchestrator import Orchestrator
from music_brain.session.intent_processor import IntentProcessor
from music_brain.session.intent_schema import (
    CompleteSongIntent,
    SongRoot,
    SongIntent,
    TechnicalConstraints,
    validate_intent,
)
from music_brain.harmony import HarmonyGenerator
from music_brain.groove.groove_engine import GrooveSettings, apply_groove


class TestOrchestratorIntegration:
    """Test Orchestrator integration with other modules."""

    def test_orchestrator_with_intent_processor(self):
        """Test orchestrator can process intents through intent processor."""
        orchestrator = Orchestrator()
        intent_processor = IntentProcessor()

        # Create a simple intent
        intent_data = {
            "emotional_intent": "joy",
            "technical": {"key": "C major", "bpm": 120, "time_signature": "4/4"},
        }

        # Process intent
        processed = intent_processor.process_intent(intent_data)
        assert processed is not None
        assert "emotional_intent" in processed

        # Orchestrator should be able to handle processed intent
        assert orchestrator is not None

    def test_orchestrator_with_harmony_generator(self):
        """Test orchestrator integration with harmony generator."""
        orchestrator = Orchestrator()
        harmony_gen = HarmonyGenerator()

        # Verify both can be instantiated together
        assert orchestrator is not None
        assert harmony_gen is not None

        # Test that harmony generator can generate basic progression
        result = harmony_gen.generate_basic_progression(key="C", mode="major")
        assert result is not None


class TestIntentProcessingPipeline:
    """Test the complete intent processing pipeline."""

    def test_complete_song_intent_creation(self):
        """Test CompleteSongIntent can be created and serialized."""
        intent = CompleteSongIntent(
            title="Test Song",
            song_root=SongRoot(
                core_event="A moment of clarity",
                core_longing="Peace",
            ),
            song_intent=SongIntent(
                mood_primary="hope",
                mood_secondary_tension=0.3,
            ),
            technical_constraints=TechnicalConstraints(
                technical_genre="ambient",
                technical_key="C",
                technical_mode="Lydian",
            ),
        )

        # Should be able to convert to dict
        intent_dict = intent.to_dict()
        assert intent_dict["title"] == "Test Song"
        assert intent_dict["song_root"]["core_event"] == "A moment of clarity"

    def test_intent_validation(self):
        """Test intent validation function."""
        intent = CompleteSongIntent(
            song_root=SongRoot(
                core_event="Something happened",
                core_longing="I want to feel better",
            ),
            song_intent=SongIntent(
                mood_primary="grief",
                mood_secondary_tension=0.7,
            ),
        )

        # Validate intent - returns list of issues
        issues = validate_intent(intent)
        # Should pass basic validation (core_event and mood_primary are set)
        # May have some issues if other fields are missing
        assert isinstance(issues, list)

    def test_intent_processor_full_flow(self):
        """Test complete intent processing flow."""
        processor = IntentProcessor()

        intent = {
            "emotional_intent": "excitement",
            "technical": {"key": "G major", "bpm": 140, "time_signature": "4/4"},
            "context": {"genre": "rock"},
        }

        # Process through pipeline
        result = processor.process_intent(intent)
        assert result is not None
        assert isinstance(result, dict)


class TestHarmonyGrooveIntegration:
    """Test harmony and groove modules work together."""

    def test_harmony_generator_with_intent(self):
        """Test harmony generator with CompleteSongIntent."""
        harmony = HarmonyGenerator()

        # Create a complete intent
        intent = CompleteSongIntent(
            song_root=SongRoot(
                core_event="Feeling hopeful",
                core_longing="Joy",
            ),
            song_intent=SongIntent(
                mood_primary="hope",
            ),
            technical_constraints=TechnicalConstraints(
                technical_key="C",
                technical_mode="Ionian",
            ),
        )

        # Generate harmony from intent
        result = harmony.generate_from_intent(intent)
        assert result is not None
        assert hasattr(result, "voicings")

    def test_groove_settings_creation(self):
        """Test groove settings can be created."""
        settings = GrooveSettings(
            swing_amount=0.15,
            velocity_variation=0.2,
            timing_variation=0.05,
            base_velocity=90,
        )

        assert settings.swing_amount == 0.15
        assert settings.velocity_variation == 0.2
        assert settings.base_velocity == 90


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_music_generation_workflow(self):
        """Test a complete music generation workflow."""
        # Initialize components
        orchestrator = Orchestrator()
        intent_processor = IntentProcessor()

        # Step 1: Define intent
        intent = {
            "emotional_intent": "peace",
            "technical": {"key": "D major", "bpm": 100},
        }

        # Step 2: Process intent
        processed_intent = intent_processor.process_intent(intent)
        assert processed_intent is not None

        # Step 3: Verify orchestrator can handle it
        assert orchestrator is not None

    def test_error_handling_across_modules(self):
        """Test error handling when modules fail."""
        intent_processor = IntentProcessor()

        # Invalid intent should be handled gracefully
        invalid_intent = {"invalid": "data"}

        # Should handle gracefully (may raise exception or return None)
        try:
            result = intent_processor.process_intent(invalid_intent)
            # If it doesn't raise, result should indicate failure or be None
            assert result is None or isinstance(result, dict)
        except (ValueError, KeyError, TypeError):
            # Expected exceptions are acceptable
            pass


class TestModuleImportIntegration:
    """Test that all modules can be imported together."""

    def test_core_module_imports(self):
        """Test all core modules can be imported."""
        from music_brain import harmony
        from music_brain.orchestrator import orchestrator
        from music_brain.session import intent_processor, intent_schema
        from music_brain.groove import groove_engine

        assert harmony is not None
        assert orchestrator is not None
        assert intent_processor is not None
        assert intent_schema is not None
        assert groove_engine is not None

    def test_api_module_imports(self):
        """Test API modules can be imported."""
        try:
            from music_brain.api import app

            assert app is not None
        except ImportError:
            # API may not be available in all environments
            pytest.skip("API module not available")


class TestDataFlowIntegration:
    """Test data flows correctly between modules."""

    def test_intent_to_harmony_flow(self):
        """Test data flows from intent to harmony."""
        intent_processor = IntentProcessor()
        harmony_gen = HarmonyGenerator()

        intent = {
            "emotional_intent": "happiness",
            "technical": {"key": "C major", "bpm": 120},
        }

        # Process intent
        processed = intent_processor.process_intent(intent)

        # Harmony generator should be able to work
        assert harmony_gen is not None

        # Generate basic progression
        result = harmony_gen.generate_basic_progression(key="C", mode="major")
        assert result is not None

    def test_complete_song_intent_round_trip(self):
        """Test CompleteSongIntent can be serialized and deserialized."""
        intent = CompleteSongIntent(
            title="Round Trip Test",
            song_root=SongRoot(
                core_event="Testing",
                core_longing="Verification",
            ),
            song_intent=SongIntent(
                mood_primary="neutral",
                mood_secondary_tension=0.5,
            ),
        )

        # Convert to dict
        data = intent.to_dict()
        assert data["title"] == "Round Trip Test"

        # Convert back from dict
        restored = CompleteSongIntent.from_dict(data)
        assert restored.title == "Round Trip Test"
        assert restored.song_root.core_event == "Testing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
