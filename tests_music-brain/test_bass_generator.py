"""
Tests for music_brain.arrangement.bass_generator module.

Tests cover:
- BassPattern enum
- BassNote and BassLine dataclasses
- Chord root parsing
- Chord tone extraction
- Pattern generation (root, root-fifth, walking, pedal, funk)
- Bass line generation
- Genre-based pattern suggestion
"""

import pytest


class TestBassPatternEnum:
    """Tests for BassPattern enum."""

    def test_all_patterns_defined(self):
        """Verify all expected bass patterns exist."""
        from music_brain.arrangement.bass_generator import BassPattern

        expected_patterns = [
            "ROOT_ONLY",
            "ROOT_FIFTH",
            "WALKING",
            "PEDAL",
            "OCTAVE_JUMP",
            "ARPEGGIO",
            "SYNCOPATED",
            "FUNK",
        ]

        for pattern in expected_patterns:
            assert hasattr(BassPattern, pattern), f"Missing pattern: {pattern}"

    def test_pattern_values(self):
        """Test pattern enum values."""
        from music_brain.arrangement.bass_generator import BassPattern

        assert BassPattern.ROOT_ONLY.value == "root_only"
        assert BassPattern.WALKING.value == "walking"
        assert BassPattern.FUNK.value == "funk"


class TestBassNote:
    """Tests for BassNote dataclass."""

    def test_bass_note_creation(self):
        """Test creating a BassNote instance."""
        from music_brain.arrangement.bass_generator import BassNote

        note = BassNote(
            pitch=36,  # C2
            start_tick=0,
            duration_ticks=480,
            velocity=80,
        )

        assert note.pitch == 36
        assert note.start_tick == 0
        assert note.duration_ticks == 480
        assert note.velocity == 80

    def test_bass_note_default_velocity(self):
        """Test default velocity."""
        from music_brain.arrangement.bass_generator import BassNote

        note = BassNote(pitch=36, start_tick=0, duration_ticks=480)
        assert note.velocity == 80

    def test_bass_note_to_dict(self):
        """Test serialization to dictionary."""
        from music_brain.arrangement.bass_generator import BassNote

        note = BassNote(pitch=40, start_tick=480, duration_ticks=240, velocity=100)
        result = note.to_dict()

        assert result["pitch"] == 40
        assert result["start_tick"] == 480
        assert result["duration_ticks"] == 240
        assert result["velocity"] == 100


class TestBassLine:
    """Tests for BassLine dataclass."""

    def test_bass_line_creation(self):
        """Test creating a BassLine instance."""
        from music_brain.arrangement.bass_generator import BassLine, BassNote, BassPattern

        notes = [
            BassNote(36, 0, 480),
            BassNote(43, 480, 480),
        ]

        line = BassLine(
            notes=notes,
            pattern=BassPattern.ROOT_FIFTH,
            octave=2,
        )

        assert len(line.notes) == 2
        assert line.pattern == BassPattern.ROOT_FIFTH
        assert line.octave == 2

    def test_bass_line_default_octave(self):
        """Test default octave is 2."""
        from music_brain.arrangement.bass_generator import BassLine, BassPattern

        line = BassLine(notes=[], pattern=BassPattern.ROOT_ONLY)
        assert line.octave == 2

    def test_bass_line_to_dict(self):
        """Test serialization to dictionary."""
        from music_brain.arrangement.bass_generator import BassLine, BassNote, BassPattern

        notes = [BassNote(36, 0, 480)]
        line = BassLine(notes=notes, pattern=BassPattern.WALKING, octave=3)

        result = line.to_dict()

        assert len(result["notes"]) == 1
        assert result["pattern"] == "walking"
        assert result["octave"] == 3


class TestParseChordRoot:
    """Tests for parse_chord_root function."""

    def test_parse_c_major(self):
        """Parse C chord root."""
        from music_brain.arrangement.bass_generator import parse_chord_root

        # C2 = MIDI note 36
        root = parse_chord_root("C")
        assert root == 24  # C2 in their octave scheme

    def test_parse_c_minor(self):
        """Parse Cm chord root (quality shouldn't affect root)."""
        from music_brain.arrangement.bass_generator import parse_chord_root

        root = parse_chord_root("Cm")
        assert root == 24

    def test_parse_sharp_notes(self):
        """Parse sharp notes."""
        from music_brain.arrangement.bass_generator import parse_chord_root

        c_sharp = parse_chord_root("C#m")
        assert c_sharp == 25  # C#2

        f_sharp = parse_chord_root("F#")
        assert f_sharp == 30  # F#2

    def test_parse_flat_notes(self):
        """Parse flat notes (converted to sharps)."""
        from music_brain.arrangement.bass_generator import parse_chord_root

        # Bb becomes A#
        bb = parse_chord_root("Bb")
        assert bb == 34  # A#2

        # Eb becomes D#
        eb = parse_chord_root("Eb")
        assert eb == 27  # D#2

    def test_parse_seventh_chords(self):
        """Parse seventh chord roots."""
        from music_brain.arrangement.bass_generator import parse_chord_root

        g7 = parse_chord_root("G7")
        assert g7 == 31  # G2

        am7 = parse_chord_root("Am7")
        assert am7 == 33  # A2

    def test_parse_invalid_returns_c(self):
        """Invalid chord names should return C."""
        from music_brain.arrangement.bass_generator import parse_chord_root

        # Invalid note letter
        result = parse_chord_root("X")
        assert result == 36  # Defaults to C2


class TestGetChordTones:
    """Tests for get_chord_tones function."""

    def test_major_triad(self):
        """Major triads have 0, 4, 7 intervals."""
        from music_brain.arrangement.bass_generator import get_chord_tones

        tones = get_chord_tones("C")
        root = tones[0]

        # Should contain root, M3, P5
        assert tones[1] == root + 4  # Major third
        assert tones[2] == root + 7  # Perfect fifth

    def test_minor_triad(self):
        """Minor triads have 0, 3, 7 intervals."""
        from music_brain.arrangement.bass_generator import get_chord_tones

        tones = get_chord_tones("Am")
        root = tones[0]

        assert tones[1] == root + 3  # Minor third
        assert tones[2] == root + 7  # Perfect fifth

    def test_dominant_seventh(self):
        """Dominant 7th has 0, 4, 7, 10 intervals."""
        from music_brain.arrangement.bass_generator import get_chord_tones

        tones = get_chord_tones("G7")
        root = tones[0]

        assert len(tones) == 4
        assert tones[1] == root + 4   # Major third
        assert tones[2] == root + 7   # Perfect fifth
        assert tones[3] == root + 10  # Minor seventh

    def test_minor_seventh(self):
        """Minor 7th has 0, 3, 7, 10 intervals."""
        from music_brain.arrangement.bass_generator import get_chord_tones

        tones = get_chord_tones("Am7")
        root = tones[0]

        assert len(tones) == 4
        assert tones[1] == root + 3   # Minor third
        assert tones[3] == root + 10  # Minor seventh

    def test_major_seventh(self):
        """Major 7th has 0, 4, 7, 11 intervals."""
        from music_brain.arrangement.bass_generator import get_chord_tones

        tones = get_chord_tones("Cmaj7")
        root = tones[0]

        assert len(tones) == 4
        assert tones[3] == root + 11  # Major seventh

    def test_diminished_triad(self):
        """Diminished has 0, 3, 6 intervals."""
        from music_brain.arrangement.bass_generator import get_chord_tones

        tones = get_chord_tones("Bdim")
        root = tones[0]

        assert tones[1] == root + 3  # Minor third
        assert tones[2] == root + 6  # Diminished fifth

    def test_half_diminished(self):
        """Half-diminished has 0, 3, 6, 10 intervals."""
        from music_brain.arrangement.bass_generator import get_chord_tones

        tones = get_chord_tones("Bm7b5")
        root = tones[0]

        assert len(tones) == 4
        assert tones[2] == root + 6   # Diminished fifth
        assert tones[3] == root + 10  # Minor seventh


class TestGenerateRootOnly:
    """Tests for generate_root_only function."""

    def test_generates_notes_on_beats(self):
        """Should generate one note per beat."""
        from music_brain.arrangement.bass_generator import generate_root_only

        notes = generate_root_only("C", bars=1, ppq=480)

        # 4/4 time = 4 beats per bar
        assert len(notes) == 4

    def test_all_notes_are_root(self):
        """All notes should be the root."""
        from music_brain.arrangement.bass_generator import generate_root_only

        notes = generate_root_only("C", bars=1, ppq=480)

        for note in notes:
            assert note.pitch == 24  # C2

    def test_multiple_bars(self):
        """Multiple bars should multiply notes."""
        from music_brain.arrangement.bass_generator import generate_root_only

        notes = generate_root_only("G", bars=2, ppq=480)

        # 2 bars * 4 beats = 8 notes
        assert len(notes) == 8

    def test_note_timing(self):
        """Notes should be on beat boundaries."""
        from music_brain.arrangement.bass_generator import generate_root_only

        notes = generate_root_only("C", bars=1, ppq=480)

        expected_starts = [0, 480, 960, 1440]
        actual_starts = [n.start_tick for n in notes]

        assert actual_starts == expected_starts


class TestGenerateRootFifth:
    """Tests for generate_root_fifth function."""

    def test_alternates_root_and_fifth(self):
        """Should alternate between root and fifth."""
        from music_brain.arrangement.bass_generator import generate_root_fifth

        notes = generate_root_fifth("C", bars=1, ppq=480)

        # Beat 1, 3 = root; Beat 2, 4 = fifth
        c2 = 24  # C2
        g2 = c2 + 7  # G2 (fifth)

        assert notes[0].pitch == c2  # Beat 1
        assert notes[1].pitch == g2  # Beat 2
        assert notes[2].pitch == c2  # Beat 3
        assert notes[3].pitch == g2  # Beat 4


class TestGenerateWalkingBass:
    """Tests for generate_walking_bass function."""

    def test_walking_bass_progression(self):
        """Walking bass should create chromatic approaches."""
        from music_brain.arrangement.bass_generator import generate_walking_bass

        chords = ["C", "F", "G", "C"]
        notes = generate_walking_bass(chords, ppq=480)

        # Should have 4 beats per chord * 4 chords = 16 notes
        assert len(notes) == 16

    def test_starts_on_root(self):
        """Each bar should start on root."""
        from music_brain.arrangement.bass_generator import generate_walking_bass

        chords = ["C", "Am"]
        notes = generate_walking_bass(chords, ppq=480)

        # First note of each bar
        assert notes[0].pitch == 24  # C2
        assert notes[4].pitch == 33  # A2

    def test_chromatic_approach(self):
        """Last beat should approach next chord chromatically."""
        from music_brain.arrangement.bass_generator import generate_walking_bass

        chords = ["C", "F"]
        notes = generate_walking_bass(chords, ppq=480)

        # Beat 4 of first bar should be half-step below F
        f2 = 29  # F2
        approach_note = notes[3].pitch

        assert approach_note == f2 - 1  # E approaching F


class TestGeneratePedalTone:
    """Tests for generate_pedal_tone function."""

    def test_single_sustained_note(self):
        """Pedal should generate one long note."""
        from music_brain.arrangement.bass_generator import generate_pedal_tone

        chords = ["C", "G", "Am", "F"]
        notes = generate_pedal_tone(chords, ppq=480)

        assert len(notes) == 1

    def test_pedal_uses_first_chord_root(self):
        """Pedal note should be root of first chord."""
        from music_brain.arrangement.bass_generator import generate_pedal_tone

        chords = ["G", "C", "D"]
        notes = generate_pedal_tone(chords, ppq=480)

        assert notes[0].pitch == 31  # G2

    def test_pedal_duration_spans_all_bars(self):
        """Pedal should span entire progression."""
        from music_brain.arrangement.bass_generator import generate_pedal_tone

        chords = ["C", "C", "C", "C"]  # 4 bars
        notes = generate_pedal_tone(chords, ppq=480)

        # 4 bars * 4 beats * 480 ppq = 7680 ticks
        expected_duration = 4 * 4 * 480
        assert notes[0].duration_ticks == expected_duration


class TestGenerateFunkBass:
    """Tests for generate_funk_bass function."""

    def test_funk_syncopation(self):
        """Funk pattern should be syncopated."""
        from music_brain.arrangement.bass_generator import generate_funk_bass

        notes = generate_funk_bass("E", bars=1, ppq=480)

        # Funk pattern has specific placements
        # Should have notes off the beat (syncopated)
        starts = [n.start_tick for n in notes]

        # Verify there are off-beat notes (not all on quarter-note boundaries)
        on_beat = [s % 480 == 0 for s in starts]
        assert not all(on_beat), "Funk should have off-beat notes"

    def test_funk_uses_root_and_fifth(self):
        """Funk pattern uses root and fifth."""
        from music_brain.arrangement.bass_generator import generate_funk_bass

        notes = generate_funk_bass("A", bars=1, ppq=480)

        a2 = 33
        e3 = a2 + 7  # Fifth

        pitches = [n.pitch for n in notes]
        assert a2 in pitches
        assert e3 in pitches


class TestGenerateBassLine:
    """Tests for main generate_bass_line function."""

    def test_generates_root_only_pattern(self):
        """Generate ROOT_ONLY pattern."""
        from music_brain.arrangement.bass_generator import generate_bass_line, BassPattern

        line = generate_bass_line(
            ["C", "G", "Am", "F"],
            pattern=BassPattern.ROOT_ONLY,
            ppq=480,
        )

        assert line.pattern == BassPattern.ROOT_ONLY
        assert len(line.notes) == 16  # 4 chords * 4 beats

    def test_generates_walking_pattern(self):
        """Generate WALKING pattern."""
        from music_brain.arrangement.bass_generator import generate_bass_line, BassPattern

        line = generate_bass_line(
            ["C", "F", "G", "C"],
            pattern=BassPattern.WALKING,
            ppq=480,
        )

        assert line.pattern == BassPattern.WALKING

    def test_empty_progression(self):
        """Empty progression returns empty line."""
        from music_brain.arrangement.bass_generator import generate_bass_line, BassPattern

        line = generate_bass_line([], pattern=BassPattern.ROOT_ONLY)

        assert line.notes == []

    def test_octave_shift(self):
        """Test octave parameter shifts pitches."""
        from music_brain.arrangement.bass_generator import generate_bass_line, BassPattern

        # Octave 2 (default)
        line2 = generate_bass_line(["C"], pattern=BassPattern.ROOT_ONLY, octave=2)

        # Octave 3
        line3 = generate_bass_line(["C"], pattern=BassPattern.ROOT_ONLY, octave=3)

        # Octave 3 should be 12 semitones higher
        pitch_diff = line3.notes[0].pitch - line2.notes[0].pitch
        assert pitch_diff == 12

    def test_custom_ppq(self):
        """Test custom PPQ affects timing."""
        from music_brain.arrangement.bass_generator import generate_bass_line, BassPattern

        line = generate_bass_line(
            ["C"],
            pattern=BassPattern.ROOT_ONLY,
            ppq=960,  # Double resolution
        )

        # Notes should be at 0, 960, 1920, 2880
        starts = [n.start_tick for n in line.notes]
        assert starts[1] == 960


class TestSuggestBassPattern:
    """Tests for suggest_bass_pattern function."""

    def test_funk_genre(self):
        """Funk genre suggests FUNK pattern."""
        from music_brain.arrangement.bass_generator import suggest_bass_pattern, BassPattern

        pattern = suggest_bass_pattern("funk")
        assert pattern == BassPattern.FUNK

    def test_disco_genre(self):
        """Disco genre suggests FUNK pattern."""
        from music_brain.arrangement.bass_generator import suggest_bass_pattern, BassPattern

        pattern = suggest_bass_pattern("disco")
        assert pattern == BassPattern.FUNK

    def test_jazz_genre(self):
        """Jazz genre suggests WALKING pattern."""
        from music_brain.arrangement.bass_generator import suggest_bass_pattern, BassPattern

        pattern = suggest_bass_pattern("jazz")
        assert pattern == BassPattern.WALKING

    def test_ambient_genre(self):
        """Ambient genre suggests PEDAL pattern."""
        from music_brain.arrangement.bass_generator import suggest_bass_pattern, BassPattern

        pattern = suggest_bass_pattern("ambient")
        assert pattern == BassPattern.PEDAL

    def test_rock_high_energy(self):
        """High-energy rock suggests ROOT_ONLY."""
        from music_brain.arrangement.bass_generator import suggest_bass_pattern, BassPattern

        pattern = suggest_bass_pattern("rock", energy_level=0.9)
        assert pattern == BassPattern.ROOT_ONLY

    def test_rock_medium_energy(self):
        """Medium-energy rock suggests ROOT_FIFTH."""
        from music_brain.arrangement.bass_generator import suggest_bass_pattern, BassPattern

        pattern = suggest_bass_pattern("rock", energy_level=0.5)
        assert pattern == BassPattern.ROOT_FIFTH

    def test_case_insensitivity(self):
        """Genre matching should be case-insensitive."""
        from music_brain.arrangement.bass_generator import suggest_bass_pattern, BassPattern

        assert suggest_bass_pattern("FUNK") == BassPattern.FUNK
        assert suggest_bass_pattern("Jazz") == BassPattern.WALKING

    def test_unknown_genre_uses_energy(self):
        """Unknown genres default based on energy level."""
        from music_brain.arrangement.bass_generator import suggest_bass_pattern, BassPattern

        assert suggest_bass_pattern("unknown", energy_level=0.9) == BassPattern.ROOT_ONLY
        assert suggest_bass_pattern("unknown", energy_level=0.1) == BassPattern.PEDAL
        assert suggest_bass_pattern("unknown", energy_level=0.5) == BassPattern.ROOT_FIFTH
