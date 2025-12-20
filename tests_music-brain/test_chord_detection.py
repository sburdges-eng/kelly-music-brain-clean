"""
Tests for music_brain.audio.chord_detection module.

Tests cover:
- ChordDetection dataclass functionality
- ChordProgressionDetection dataclass functionality
- Chord template matching
- Chord name formatting
- Edge cases and error handling
"""

import pytest
from unittest.mock import patch, MagicMock
import math


class TestChordDetection:
    """Tests for ChordDetection dataclass."""

    def test_import_chord_detection(self):
        """Verify ChordDetection can be imported."""
        from music_brain.audio.chord_detection import ChordDetection
        assert ChordDetection is not None

    def test_chord_detection_creation(self):
        """Test creating a ChordDetection instance."""
        from music_brain.audio.chord_detection import ChordDetection

        detection = ChordDetection(
            chord_name="Cmaj",
            root="C",
            quality="maj",
            confidence=0.85,
            start_time=0.0,
            end_time=0.5,
            chroma_vector=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )

        assert detection.chord_name == "Cmaj"
        assert detection.root == "C"
        assert detection.quality == "maj"
        assert detection.confidence == 0.85
        assert detection.start_time == 0.0
        assert detection.end_time == 0.5

    def test_chord_detection_duration(self):
        """Test duration property calculation."""
        from music_brain.audio.chord_detection import ChordDetection

        detection = ChordDetection(
            chord_name="Am",
            root="A",
            quality="min",
            confidence=0.9,
            start_time=1.0,
            end_time=3.5,
        )

        assert detection.duration == 2.5

    def test_chord_detection_to_dict(self):
        """Test serialization to dictionary."""
        from music_brain.audio.chord_detection import ChordDetection

        detection = ChordDetection(
            chord_name="G7",
            root="G",
            quality="7",
            confidence=0.75,
            start_time=0.5,
            end_time=1.0,
        )

        result = detection.to_dict()

        assert result["chord"] == "G7"
        assert result["root"] == "G"
        assert result["quality"] == "7"
        assert result["confidence"] == 0.75
        assert result["start_time"] == 0.5
        assert result["end_time"] == 1.0

    def test_chord_detection_default_chroma_vector(self):
        """Test that chroma_vector defaults to empty list."""
        from music_brain.audio.chord_detection import ChordDetection

        detection = ChordDetection(
            chord_name="C",
            root="C",
            quality="maj",
            confidence=0.8,
            start_time=0.0,
            end_time=1.0,
        )

        assert detection.chroma_vector == []


class TestChordProgressionDetection:
    """Tests for ChordProgressionDetection dataclass."""

    def test_chord_progression_creation(self):
        """Test creating a ChordProgressionDetection instance."""
        from music_brain.audio.chord_detection import (
            ChordDetection,
            ChordProgressionDetection,
        )

        chords = [
            ChordDetection("C", "C", "maj", 0.9, 0.0, 1.0),
            ChordDetection("G", "G", "maj", 0.85, 1.0, 2.0),
            ChordDetection("Am", "A", "min", 0.88, 2.0, 3.0),
            ChordDetection("F", "F", "maj", 0.82, 3.0, 4.0),
        ]

        progression = ChordProgressionDetection(
            chords=chords,
            estimated_key="C major",
            confidence=0.86,
        )

        assert len(progression.chords) == 4
        assert progression.estimated_key == "C major"
        assert progression.confidence == 0.86

    def test_chord_sequence_property(self):
        """Test chord_sequence property returns chord names."""
        from music_brain.audio.chord_detection import (
            ChordDetection,
            ChordProgressionDetection,
        )

        chords = [
            ChordDetection("C", "C", "maj", 0.9, 0.0, 1.0),
            ChordDetection("Am", "A", "min", 0.85, 1.0, 2.0),
            ChordDetection("F", "F", "maj", 0.88, 2.0, 3.0),
            ChordDetection("G", "G", "maj", 0.82, 3.0, 4.0),
        ]

        progression = ChordProgressionDetection(chords=chords)

        assert progression.chord_sequence == ["C", "Am", "F", "G"]

    def test_unique_chords_property(self):
        """Test unique_chords property removes duplicates."""
        from music_brain.audio.chord_detection import (
            ChordDetection,
            ChordProgressionDetection,
        )

        chords = [
            ChordDetection("C", "C", "maj", 0.9, 0.0, 1.0),
            ChordDetection("G", "G", "maj", 0.85, 1.0, 2.0),
            ChordDetection("C", "C", "maj", 0.88, 2.0, 3.0),  # Repeat
            ChordDetection("F", "F", "maj", 0.82, 3.0, 4.0),
            ChordDetection("G", "G", "maj", 0.80, 4.0, 5.0),  # Repeat
        ]

        progression = ChordProgressionDetection(chords=chords)

        # Unique chords in order of first appearance
        assert progression.unique_chords == ["C", "G", "F"]

    def test_progression_to_dict(self):
        """Test serialization to dictionary."""
        from music_brain.audio.chord_detection import (
            ChordDetection,
            ChordProgressionDetection,
        )

        chords = [
            ChordDetection("C", "C", "maj", 0.9, 0.0, 1.0),
            ChordDetection("G", "G", "maj", 0.85, 1.0, 2.0),
        ]

        progression = ChordProgressionDetection(
            chords=chords,
            estimated_key="C major",
            confidence=0.87,
        )

        result = progression.to_dict()

        assert len(result["chords"]) == 2
        assert result["sequence"] == ["C", "G"]
        assert result["unique_chords"] == ["C", "G"]
        assert result["estimated_key"] == "C major"
        assert result["confidence"] == 0.87

    def test_empty_progression(self):
        """Test empty progression handling."""
        from music_brain.audio.chord_detection import ChordProgressionDetection

        progression = ChordProgressionDetection(chords=[])

        assert progression.chord_sequence == []
        assert progression.unique_chords == []


class TestChordTemplates:
    """Tests for chord template definitions."""

    def test_chord_templates_defined(self):
        """Verify all expected chord templates exist."""
        from music_brain.audio.chord_detection import CHORD_TEMPLATES

        expected_templates = [
            "maj", "min", "dim", "aug",  # Triads
            "maj7", "min7", "7", "dim7", "m7b5",  # Sevenths
            "sus2", "sus4",  # Suspended
        ]

        for template in expected_templates:
            assert template in CHORD_TEMPLATES, f"Missing template: {template}"

    def test_major_triad_intervals(self):
        """Test major triad has correct intervals."""
        from music_brain.audio.chord_detection import CHORD_TEMPLATES

        assert CHORD_TEMPLATES["maj"] == [0, 4, 7]

    def test_minor_triad_intervals(self):
        """Test minor triad has correct intervals."""
        from music_brain.audio.chord_detection import CHORD_TEMPLATES

        assert CHORD_TEMPLATES["min"] == [0, 3, 7]

    def test_dominant_seventh_intervals(self):
        """Test dominant 7th has correct intervals."""
        from music_brain.audio.chord_detection import CHORD_TEMPLATES

        assert CHORD_TEMPLATES["7"] == [0, 4, 7, 10]


class TestNoteNames:
    """Tests for note name constants."""

    def test_note_names_defined(self):
        """Verify all 12 note names are defined."""
        from music_brain.audio.chord_detection import NOTE_NAMES

        assert len(NOTE_NAMES) == 12
        assert NOTE_NAMES[0] == "C"
        assert NOTE_NAMES[9] == "A"


class TestChordNameFormatting:
    """Tests for chord name formatting function."""

    def test_format_major_chord(self):
        """Major chords should format without suffix."""
        from music_brain.audio.chord_detection import _format_chord_name

        assert _format_chord_name("C", "maj") == "C"
        assert _format_chord_name("G", "maj") == "G"

    def test_format_minor_chord(self):
        """Minor chords should add 'm' suffix."""
        from music_brain.audio.chord_detection import _format_chord_name

        assert _format_chord_name("A", "min") == "Am"
        assert _format_chord_name("E", "min") == "Em"

    def test_format_seventh_chords(self):
        """Test seventh chord formatting."""
        from music_brain.audio.chord_detection import _format_chord_name

        assert _format_chord_name("C", "maj7") == "Cmaj7"
        assert _format_chord_name("A", "min7") == "Am7"
        assert _format_chord_name("G", "7") == "G7"

    def test_format_diminished_chords(self):
        """Test diminished chord formatting."""
        from music_brain.audio.chord_detection import _format_chord_name

        assert _format_chord_name("B", "dim") == "Bdim"
        assert _format_chord_name("B", "dim7") == "Bdim7"
        assert _format_chord_name("B", "m7b5") == "Bm7b5"

    def test_format_augmented_chord(self):
        """Test augmented chord formatting."""
        from music_brain.audio.chord_detection import _format_chord_name

        assert _format_chord_name("C", "aug") == "Caug"

    def test_format_suspended_chords(self):
        """Test suspended chord formatting."""
        from music_brain.audio.chord_detection import _format_chord_name

        assert _format_chord_name("D", "sus2") == "Dsus2"
        assert _format_chord_name("D", "sus4") == "Dsus4"

    def test_format_unknown_quality(self):
        """Unknown qualities should append directly."""
        from music_brain.audio.chord_detection import _format_chord_name

        assert _format_chord_name("C", "add9") == "Cadd9"


class TestChordDetectorCreation:
    """Tests for ChordDetector initialization."""

    def test_detector_requires_librosa(self):
        """ChordDetector should raise ImportError if librosa unavailable."""
        from music_brain.audio.chord_detection import LIBROSA_AVAILABLE

        if not LIBROSA_AVAILABLE:
            from music_brain.audio.chord_detection import ChordDetector

            with pytest.raises(ImportError, match="librosa required"):
                ChordDetector()

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_detector_creation_with_defaults(self):
        """Test ChordDetector creation with default parameters."""
        from music_brain.audio.chord_detection import ChordDetector

        detector = ChordDetector()

        assert detector.hop_length == 512
        assert detector.window_size == 0.5
        assert detector.min_confidence == 0.3

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_detector_creation_with_custom_params(self):
        """Test ChordDetector creation with custom parameters."""
        from music_brain.audio.chord_detection import ChordDetector

        detector = ChordDetector(
            hop_length=256,
            window_size=1.0,
            min_confidence=0.5,
        )

        assert detector.hop_length == 256
        assert detector.window_size == 1.0
        assert detector.min_confidence == 0.5


class TestChordMatching:
    """Tests for chord template matching logic."""

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_create_chord_template(self):
        """Test chord template creation."""
        from music_brain.audio.chord_detection import _create_chord_template
        import numpy as np

        # C major template (root=0, intervals=[0, 4, 7])
        template = _create_chord_template(0, [0, 4, 7])

        # Should have energy at C, E, G (indices 0, 4, 7)
        assert template[0] > 0  # C
        assert template[4] > 0  # E
        assert template[7] > 0  # G

        # Other notes should be zero
        assert template[1] == 0
        assert template[2] == 0
        assert template[3] == 0

        # Should be normalized
        assert abs(np.sum(template) - 1.0) < 0.001

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_match_chord_major(self):
        """Test matching a clear major chord chroma."""
        from music_brain.audio.chord_detection import _match_chord
        import numpy as np

        # Create a C major chroma (energy at C, E, G)
        chroma = np.zeros(12)
        chroma[0] = 1.0  # C
        chroma[4] = 0.8  # E
        chroma[7] = 0.9  # G

        root, quality, confidence = _match_chord(chroma)

        assert root == "C"
        assert quality == "maj"
        assert confidence > 0.5

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_match_chord_minor(self):
        """Test matching a clear minor chord chroma."""
        from music_brain.audio.chord_detection import _match_chord
        import numpy as np

        # Create an A minor chroma (A=9, C=0, E=4)
        chroma = np.zeros(12)
        chroma[9] = 1.0  # A
        chroma[0] = 0.8  # C
        chroma[4] = 0.9  # E

        root, quality, confidence = _match_chord(chroma)

        # Should match A minor
        assert root == "A"
        assert quality == "min"
        assert confidence > 0.5


class TestChordDetectorMethods:
    """Tests for ChordDetector methods with mocked librosa."""

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_merge_consecutive_chords(self):
        """Test merging consecutive identical chords."""
        from music_brain.audio.chord_detection import ChordDetector, ChordDetection

        detector = ChordDetector()

        chords = [
            ChordDetection("C", "C", "maj", 0.8, 0.0, 0.5),
            ChordDetection("C", "C", "maj", 0.85, 0.5, 1.0),  # Same chord
            ChordDetection("G", "G", "maj", 0.9, 1.0, 1.5),
            ChordDetection("G", "G", "maj", 0.87, 1.5, 2.0),  # Same chord
            ChordDetection("Am", "A", "min", 0.82, 2.0, 2.5),
        ]

        merged = detector._merge_consecutive_chords(chords)

        assert len(merged) == 3
        assert merged[0].chord_name == "C"
        assert merged[0].end_time == 1.0  # Extended
        assert merged[1].chord_name == "G"
        assert merged[2].chord_name == "Am"

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_merge_empty_list(self):
        """Test merging empty chord list."""
        from music_brain.audio.chord_detection import ChordDetector

        detector = ChordDetector()

        merged = detector._merge_consecutive_chords([])

        assert merged == []

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_estimate_key_from_chords(self):
        """Test key estimation from chord sequence."""
        from music_brain.audio.chord_detection import ChordDetector, ChordDetection

        detector = ChordDetector()

        # C major progression: C-G-Am-F
        chords = [
            ChordDetection("C", "C", "maj", 0.9, 0.0, 1.0),
            ChordDetection("G", "G", "maj", 0.85, 1.0, 2.0),
            ChordDetection("Am", "A", "min", 0.88, 2.0, 3.0),
            ChordDetection("F", "F", "maj", 0.82, 3.0, 4.0),
            ChordDetection("C", "C", "maj", 0.9, 4.0, 5.0),
        ]

        key = detector._estimate_key_from_chords(chords)

        # C is the most common root
        assert "C" in key

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_estimate_key_empty_list(self):
        """Test key estimation with empty chord list."""
        from music_brain.audio.chord_detection import ChordDetector

        detector = ChordDetector()

        key = detector._estimate_key_from_chords([])

        assert key is None

    @pytest.mark.skipif(
        "not __import__('music_brain.audio.chord_detection', fromlist=['LIBROSA_AVAILABLE']).LIBROSA_AVAILABLE"
    )
    def test_confidence_score(self):
        """Test confidence score retrieval."""
        from music_brain.audio.chord_detection import ChordDetector, ChordDetection

        detector = ChordDetector()
        detection = ChordDetection("C", "C", "maj", 0.85, 0.0, 1.0)

        score = detector.confidence_score(detection)

        assert score == 0.85


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_detect_chords_from_audio_exists(self):
        """Verify convenience function can be imported."""
        from music_brain.audio.chord_detection import detect_chords_from_audio

        assert callable(detect_chords_from_audio)
