"""
Tests for music_brain.api module.

Tests cover:
- DAiWAPI class initialization
- Harmony generation methods
- Groove operations
- Chord analysis
- Therapy session
- Intent processing
- Preset management

Note: These tests require numpy to be installed. Tests are skipped if numpy
is unavailable, as the API module depends on numpy.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

# Check if numpy is available (required by music_brain.api)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Skip all tests in this module if numpy is not available
pytestmark = pytest.mark.skipif(
    not NUMPY_AVAILABLE,
    reason="numpy required for API module tests"
)


class TestDAiWAPIImport:
    """Tests for API module import and initialization."""

    def test_import_daiw_api(self):
        """Verify DAiWAPI can be imported."""
        from music_brain.api import DAiWAPI

        assert DAiWAPI is not None

    def test_import_api_instance(self):
        """Verify convenience api instance can be imported."""
        from music_brain.api import api

        assert api is not None

    def test_api_initialization(self):
        """Test DAiWAPI can be instantiated."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()

        assert api is not None
        assert hasattr(api, "harmony_generator")


class TestHarmonyGeneration:
    """Tests for harmony generation methods."""

    def test_generate_basic_progression(self):
        """Test basic progression generation."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()
        result = api.generate_basic_progression(
            key="C",
            mode="major",
            pattern="I-V-vi-IV",
        )

        assert "harmony" in result
        assert result["harmony"]["key"] == "C"
        assert result["harmony"]["mode"] == "major"

    def test_generate_basic_progression_minor(self):
        """Test minor key progression."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()
        result = api.generate_basic_progression(
            key="A",
            mode="minor",
            pattern="i-iv-VII-III",
        )

        assert result["harmony"]["key"] == "A"
        assert result["harmony"]["mode"] == "minor"

    def test_generate_harmony_from_intent(self):
        """Test intent-based harmony generation."""
        from music_brain.api import DAiWAPI
        from music_brain.session.intent_schema import CompleteSongIntent

        api = DAiWAPI()

        # Create minimal intent
        intent = CompleteSongIntent(
            core_event="loss of a friend",
            core_resistance="denial",
            core_longing="acceptance",
            mood_primary="grief",
            vulnerability_scale=7,
            narrative_arc="slow-reveal",
            technical_genre="indie",
            technical_key="Am",
        )

        result = api.generate_harmony_from_intent(intent)

        assert "harmony" in result
        assert "voicings" in result
        assert isinstance(result["voicings"], list)

    def test_generate_harmony_with_midi_output(self):
        """Test harmony generation with MIDI output."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "test_harmony.mid")

            result = api.generate_basic_progression(
                key="G",
                mode="major",
                pattern="I-IV-V-I",
                output_midi=midi_path,
            )

            assert "midi_path" in result
            assert result["midi_path"] == midi_path
            assert os.path.exists(midi_path)


class TestGrooveOperations:
    """Tests for groove-related methods."""

    def test_humanize_drums(self):
        """Test drum humanization."""
        from music_brain.api import DAiWAPI
        from music_brain.groove.templates import create_basic_drum_pattern
        import tempfile
        import os

        api = DAiWAPI()

        # Create a basic MIDI file for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "drums.mid")
            output_path = os.path.join(tmpdir, "drums_humanized.mid")

            # Create basic drum pattern
            create_basic_drum_pattern(input_path, tempo_bpm=120, bars=2)

            result = api.humanize_drums(
                midi_path=input_path,
                complexity=0.5,
                vulnerability=0.5,
                output_path=output_path,
            )

            assert "output_path" in result
            assert result["complexity"] == 0.5
            assert result["vulnerability"] == 0.5

    def test_humanize_drums_with_preset(self):
        """Test drum humanization with preset."""
        from music_brain.api import DAiWAPI
        from music_brain.groove.templates import create_basic_drum_pattern
        import tempfile
        import os

        api = DAiWAPI()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "drums.mid")
            output_path = os.path.join(tmpdir, "drums_humanized.mid")

            create_basic_drum_pattern(input_path, tempo_bpm=120, bars=2)

            result = api.humanize_drums(
                midi_path=input_path,
                preset="tight_mechanical",
                output_path=output_path,
            )

            assert result["preset_used"] == "tight_mechanical"


class TestChordAnalysis:
    """Tests for chord analysis methods."""

    def test_diagnose_progression(self):
        """Test progression diagnosis."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()
        result = api.diagnose_progression("F-C-Am-Dm")

        assert isinstance(result, dict)

    def test_suggest_reharmonizations(self):
        """Test reharmonization suggestions."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()
        suggestions = api.suggest_reharmonizations(
            progression="C-G-Am-F",
            style="jazz",
            count=3,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3


class TestIntentProcessing:
    """Tests for intent processing methods."""

    def test_suggest_rule_breaks(self):
        """Test rule break suggestions."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()
        suggestions = api.suggest_rule_breaks("grief")

        assert isinstance(suggestions, list)

    def test_list_available_rules(self):
        """Test listing available rules."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()
        rules = api.list_available_rules()

        assert isinstance(rules, dict)
        # Should have categorized rules
        assert len(rules) > 0

    def test_validate_song_intent(self):
        """Test intent validation."""
        from music_brain.api import DAiWAPI
        from music_brain.session.intent_schema import CompleteSongIntent

        api = DAiWAPI()

        intent = CompleteSongIntent(
            core_event="heartbreak",
            core_resistance="anger",
            core_longing="peace",
            mood_primary="sadness",
            vulnerability_scale=8,
            narrative_arc="climb-to-climax",
            technical_genre="pop",
            technical_key="C",
        )

        issues = api.validate_song_intent(intent)

        assert isinstance(issues, list)

    def test_process_song_intent(self):
        """Test full intent processing."""
        from music_brain.api import DAiWAPI
        from music_brain.session.intent_schema import CompleteSongIntent

        api = DAiWAPI()

        intent = CompleteSongIntent(
            core_event="finding hope",
            core_resistance="doubt",
            core_longing="confidence",
            mood_primary="hope",
            vulnerability_scale=5,
            narrative_arc="climb-to-climax",
            technical_genre="pop",
            technical_key="G",
        )

        result = api.process_song_intent(intent)

        assert "intent_summary" in result
        assert "harmony" in result
        assert "groove" in result
        assert "arrangement" in result
        assert "production" in result


class TestTherapySession:
    """Tests for therapy session method."""

    def test_therapy_session_basic(self):
        """Test basic therapy session."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()
        result = api.therapy_session(
            text="I feel overwhelmed by change",
            motivation=7,
            chaos_tolerance=0.5,
        )

        assert "affect" in result
        assert "plan" in result
        assert "plan" in result
        assert result["plan"]["root_note"] is not None

    def test_therapy_session_with_midi(self):
        """Test therapy session with MIDI output."""
        from music_brain.api import DAiWAPI
        import tempfile
        import os

        api = DAiWAPI()

        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "therapy.mid")

            result = api.therapy_session(
                text="I miss my childhood home",
                motivation=5,
                chaos_tolerance=0.3,
                output_midi=midi_path,
            )

            assert "midi_path" in result
            assert os.path.exists(midi_path)


class TestPresetManagement:
    """Tests for preset management methods."""

    def test_list_humanization_presets(self):
        """Test listing humanization presets."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()
        presets = api.list_humanization_presets()

        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_get_humanization_preset_info(self):
        """Test getting preset information."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()
        presets = api.list_humanization_presets()

        if presets:
            info = api.get_humanization_preset_info(presets[0])
            assert isinstance(info, dict)


class TestAPIModuleExports:
    """Tests for module-level exports."""

    def test_all_exports(self):
        """Test __all__ exports."""
        from music_brain import api

        assert hasattr(api, "DAiWAPI")
        assert hasattr(api, "api")

    def test_convenience_api_type(self):
        """Convenience api should be a DAiWAPI instance."""
        from music_brain.api import DAiWAPI, api

        assert isinstance(api, DAiWAPI)


class TestAPIMethodSignatures:
    """Tests verifying API method signatures match documentation."""

    def test_generate_harmony_from_intent_signature(self):
        """Test generate_harmony_from_intent accepts documented params."""
        from music_brain.api import DAiWAPI
        from music_brain.session.intent_schema import CompleteSongIntent
        import inspect

        api = DAiWAPI()
        sig = inspect.signature(api.generate_harmony_from_intent)

        params = list(sig.parameters.keys())
        assert "intent" in params
        assert "output_midi" in params
        assert "tempo_bpm" in params

    def test_generate_basic_progression_signature(self):
        """Test generate_basic_progression accepts documented params."""
        from music_brain.api import DAiWAPI
        import inspect

        api = DAiWAPI()
        sig = inspect.signature(api.generate_basic_progression)

        params = list(sig.parameters.keys())
        assert "key" in params
        assert "mode" in params
        assert "pattern" in params
        assert "output_midi" in params
        assert "tempo_bpm" in params

    def test_humanize_drums_signature(self):
        """Test humanize_drums accepts documented params."""
        from music_brain.api import DAiWAPI
        import inspect

        api = DAiWAPI()
        sig = inspect.signature(api.humanize_drums)

        params = list(sig.parameters.keys())
        assert "midi_path" in params
        assert "complexity" in params
        assert "vulnerability" in params
        assert "preset" in params
        assert "output_path" in params

    def test_therapy_session_signature(self):
        """Test therapy_session accepts documented params."""
        from music_brain.api import DAiWAPI
        import inspect

        api = DAiWAPI()
        sig = inspect.signature(api.therapy_session)

        params = list(sig.parameters.keys())
        assert "text" in params
        assert "motivation" in params
        assert "chaos_tolerance" in params
        assert "output_midi" in params


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_invalid_key_handling(self):
        """Test handling of invalid musical key."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()

        # This should handle gracefully (not crash)
        try:
            result = api.generate_basic_progression(
                key="X",  # Invalid key
                mode="major",
                pattern="I-V-vi-IV",
            )
            # If it doesn't raise, it should still return a result
            assert "harmony" in result
        except (ValueError, KeyError):
            # These exceptions are acceptable for invalid input
            pass

    def test_empty_progression_handling(self):
        """Test handling of empty progression."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()

        # Should handle empty progression gracefully
        result = api.diagnose_progression("")

        assert isinstance(result, dict)


class TestAudioAnalysis:
    """Tests for audio analysis methods."""

    def test_api_has_audio_analyzer(self):
        """Test DAiWAPI has audio_analyzer attribute."""
        from music_brain.api import DAiWAPI

        api = DAiWAPI()

        assert hasattr(api, "audio_analyzer")
        assert api.audio_analyzer is not None

    def test_audio_analyzer_import(self):
        """Test AudioAnalyzer can be imported from audio module."""
        from music_brain.audio import AudioAnalyzer, AudioAnalysis

        assert AudioAnalyzer is not None
        assert AudioAnalysis is not None

    def test_audio_analysis_to_dict(self):
        """Test AudioAnalysis.to_dict() method."""
        from music_brain.audio import AudioAnalysis

        analysis = AudioAnalysis(
            filename="test.wav",
            duration_seconds=120.0,
            tempo_bpm=120.0,
            detected_key="C",
            key_mode="major",
        )

        result = analysis.to_dict()

        assert isinstance(result, dict)
        assert result["file_info"]["filename"] == "test.wav"
        assert result["tempo"]["bpm"] == 120.0
        assert result["key"]["detected"] == "C"
        assert result["key"]["mode"] == "major"

    def test_analyze_audio_file_method_exists(self):
        """Test analyze_audio_file method exists on DAiWAPI."""
        from music_brain.api import DAiWAPI
        import inspect

        api = DAiWAPI()

        assert hasattr(api, "analyze_audio_file")
        assert callable(api.analyze_audio_file)

        sig = inspect.signature(api.analyze_audio_file)
        params = list(sig.parameters.keys())
        assert "audio_path" in params

    def test_analyze_audio_waveform_method_exists(self):
        """Test analyze_audio_waveform method exists on DAiWAPI."""
        from music_brain.api import DAiWAPI
        import inspect

        api = DAiWAPI()

        assert hasattr(api, "analyze_audio_waveform")
        assert callable(api.analyze_audio_waveform)

        sig = inspect.signature(api.analyze_audio_waveform)
        params = list(sig.parameters.keys())
        assert "samples" in params
        assert "sample_rate" in params

    def test_detect_audio_bpm_method_exists(self):
        """Test detect_audio_bpm method exists on DAiWAPI."""
        from music_brain.api import DAiWAPI
        import inspect

        api = DAiWAPI()

        assert hasattr(api, "detect_audio_bpm")
        assert callable(api.detect_audio_bpm)

        sig = inspect.signature(api.detect_audio_bpm)
        params = list(sig.parameters.keys())
        assert "samples" in params
        assert "sample_rate" in params

    def test_detect_audio_key_method_exists(self):
        """Test detect_audio_key method exists on DAiWAPI."""
        from music_brain.api import DAiWAPI
        import inspect

        api = DAiWAPI()

        assert hasattr(api, "detect_audio_key")
        assert callable(api.detect_audio_key)

        sig = inspect.signature(api.detect_audio_key)
        params = list(sig.parameters.keys())
        assert "samples" in params
        assert "sample_rate" in params

    def test_audio_analyzer_initialization(self):
        """Test AudioAnalyzer can be initialized with custom parameters."""
        from music_brain.audio import AudioAnalyzer

        # Default initialization
        analyzer1 = AudioAnalyzer()
        assert analyzer1.sample_rate == 44100
        assert analyzer1.hop_length == 512

        # Custom initialization
        analyzer2 = AudioAnalyzer(sample_rate=48000, hop_length=1024)
        assert analyzer2.sample_rate == 48000
        assert analyzer2.hop_length == 1024


class TestAudioAnalyzerWithMockedLibrosa:
    """Tests for audio analyzer with mocked librosa for unit testing."""

    def test_detect_bpm_with_mock(self):
        """Test BPM detection with mocked audio data."""
        from music_brain.api import DAiWAPI
        import numpy as np

        api = DAiWAPI()

        # Create a simple sine wave as mock audio
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # 120 BPM = 2 Hz beat frequency
        samples = np.sin(2 * np.pi * 2 * t)

        try:
            bpm = api.detect_audio_bpm(samples, sample_rate)
            # BPM should be a positive float
            assert isinstance(bpm, float)
            assert bpm > 0
        except ImportError:
            # Skip if librosa not installed
            pytest.skip("librosa not installed")

    def test_detect_key_with_mock(self):
        """Test key detection with mocked audio data."""
        from music_brain.api import DAiWAPI
        import numpy as np

        api = DAiWAPI()

        # Create mock audio data
        sample_rate = 44100
        duration = 2.0
        samples = np.random.randn(int(sample_rate * duration)) * 0.1

        try:
            key, mode = api.detect_audio_key(samples, sample_rate)
            # Should return valid key and mode
            assert isinstance(key, str)
            assert isinstance(mode, str)
            assert mode in ["major", "minor"]
        except ImportError:
            # Skip if librosa not installed
            pytest.skip("librosa not installed")
