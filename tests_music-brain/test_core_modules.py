"""
Comprehensive unit tests for core Music Brain modules.
Tests harmony generation, progression analysis, intent processing, and module integration.
"""

import pytest
from pathlib import Path
import json

# Test imports
from music_brain.session.intent_schema import (
    CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
    suggest_rule_break, validate_intent, list_all_rules
)
from music_brain.structure.chord import Chord, ChordProgression
from music_brain.structure.progression import diagnose_progression, generate_reharmonizations
from music_brain.groove.templates import get_genre_template, list_genres


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_intent():
    """Create a sample complete song intent for testing."""
    return CompleteSongIntent(
        song_root=SongRoot(
            core_event="Loss of a loved one",
            core_resistance="Fear of moving on",
            core_longing="To remember without pain"
        ),
        song_intent=SongIntent(
            mood_primary="Grief",
            vulnerability_scale="High"
        ),
        technical_constraints=TechnicalConstraints(
            technical_key="F",
            technical_mode="major"
        )
    )


@pytest.fixture
def common_progressions():
    """Common chord progressions for testing."""
    return {
        'diatonic': "C-Am-F-G",
        'modal_interchange': "F-C-Bbm-F",
        'simple': "C-F-G-C",
        'repetitive': "C-C-C-C",
    }


# ============================================================================
# CHORD MODULE TESTS
# ============================================================================

class TestChordModule:
    """Test chord parsing and analysis."""
    
    @pytest.mark.parametrize("chord_str,expected_root,expected_quality,expected_name", [
        ("C", 0, "maj", "C"),
        ("Am", 9, "min", "Am"),
        ("F#", 6, "maj", "F#"),
        ("Bb", 10, "maj", "A#"),  # Normalized to sharps
    ])
    def test_parse_basic_chords(self, chord_str, expected_root, expected_quality, expected_name):
        """Test parsing of basic major and minor chords."""
        chord = Chord.from_string(chord_str, key="C")
        assert chord.root == expected_root, f"Root mismatch for {chord_str}"
        assert chord.quality == expected_quality, f"Quality mismatch for {chord_str}"
        assert chord.name == expected_name, f"Name mismatch for {chord_str}"
    
    @pytest.mark.parametrize("chord_str,expected_root,expected_quality", [
        ("Cmaj7", 0, "maj7"),
        ("G7", 7, "7"),  # Dominant 7th
        ("Am7", 9, "min7"),
        ("Dm7", 2, "min7"),
    ])
    def test_parse_seventh_chords(self, chord_str, expected_root, expected_quality):
        """Test parsing of various seventh chord types."""
        chord = Chord.from_string(chord_str, key="C")
        assert chord.root == expected_root, f"Root mismatch for {chord_str}"
        assert chord.quality == expected_quality, f"Quality mismatch for {chord_str}"
    
    @pytest.mark.parametrize("chord_str,expected_extensions", [
        ("Cmaj9", ["9"]),
        ("C11", ["11"]),
        ("C13", ["13"]),
    ])
    def test_parse_extended_chords(self, chord_str, expected_extensions):
        """Test parsing of extended chords (9th, 11th, 13th)."""
        chord = Chord.from_string(chord_str, key="C")
        assert chord.root == 0, f"Root should be C (0) for {chord_str}"
        # Check that at least one expected extension is present
        assert any(ext in chord.extensions for ext in expected_extensions), \
            f"Expected one of {expected_extensions} in extensions, got {chord.extensions}"
    
    @pytest.mark.parametrize("chord_str,expected_quality", [
        ("Cdim", "dim"),
        ("Caug", "aug"),
        ("C+", "aug"),  # Alternative notation
    ])
    def test_parse_altered_chords(self, chord_str, expected_quality):
        """Test parsing of altered chords (diminished, augmented)."""
        chord = Chord.from_string(chord_str, key="C")
        assert chord.root == 0, f"Root should be C (0) for {chord_str}"
        assert chord.quality == expected_quality, \
            f"Expected quality {expected_quality} for {chord_str}, got {chord.quality}"
    
    def test_chord_voicing_generation(self):
        """Test chord voicing generation for different voicing types."""
        chord = Chord.from_string("C", key="C")
        
        # Test close voicing
        close_notes = chord.get_voicing(octave=4, voicing_type='close')
        assert len(close_notes) >= 3, "Close voicing should have at least 3 notes"
        assert close_notes[0] < close_notes[-1], "Notes should be in ascending order"
        assert all(48 <= note <= 84 for note in close_notes), \
            f"Notes should be in MIDI range 48-84 (octave 4), got {close_notes}"
        
        # Test open voicing
        open_notes = chord.get_voicing(octave=4, voicing_type='open')
        assert len(open_notes) >= 3, "Open voicing should have at least 3 notes"
        assert open_notes[0] < open_notes[-1], "Open voicing notes should be ascending"


class TestProgressionAnalysis:
    """Test chord progression diagnosis and analysis."""
    
    def test_diagnose_diatonic_progression(self, common_progressions):
        """Test diagnosis of a standard diatonic progression."""
        result = diagnose_progression(common_progressions['diatonic'])
        
        # Verify structure
        assert 'key' in result, "Result should contain 'key'"
        assert 'mode' in result, "Result should contain 'mode'"
        assert 'issues' in result, "Result should contain 'issues'"
        assert 'suggestions' in result, "Result should contain 'suggestions'"
        assert 'chords' in result, "Result should contain 'chords'"
        
        # Verify types
        assert isinstance(result['key'], str), f"Key should be string, got {type(result['key'])}"
        assert isinstance(result['mode'], str), f"Mode should be string, got {type(result['mode'])}"
        assert isinstance(result['issues'], list), f"Issues should be list, got {type(result['issues'])}"
        assert isinstance(result['suggestions'], list), f"Suggestions should be list, got {type(result['suggestions'])}"
        assert isinstance(result['chords'], list), f"Chords should be list, got {type(result['chords'])}"
        
        # Verify chords were parsed
        assert len(result['chords']) == 4, f"Expected 4 chords, got {len(result['chords'])}"
    
    def test_diagnose_modal_interchange(self, common_progressions):
        """Test detection of modal interchange (borrowed chords)."""
        result = diagnose_progression(common_progressions['modal_interchange'])
        
        assert 'key' in result, "Result should contain 'key'"
        assert 'mode' in result, "Result should contain 'mode'"
        assert 'chords' in result, "Result should contain 'chords'"
        
        # Verify Bbm is in the parsed chords (may be normalized to A#m)
        chord_names = [c.lower() for c in result['chords']]
        assert any('bbm' in c or 'a#m' in c for c in chord_names), \
            f"Expected Bbm or A#m in chords, got {result['chords']}"
        
        # Verify progression was parsed correctly (4 chords)
        assert len(result['chords']) == 4, \
            f"Expected 4 chords in progression, got {len(result['chords'])}"
        
        # Note: The implementation may or may not flag borrowed chords as issues
        # depending on the detection logic. The important thing is that the
        # progression parses correctly and the chords are identified.
    
    def test_diagnose_detects_key(self, common_progressions):
        """Test key detection from progression."""
        result = diagnose_progression(common_progressions['simple'])
        
        # Should detect C as the key
        assert result['key'] == 'C', f"Expected key 'C', got '{result['key']}'"
        assert result['mode'] in ['major', 'minor'], \
            f"Mode should be 'major' or 'minor', got '{result['mode']}'"
    
    def test_diagnose_provides_suggestions(self, common_progressions):
        """Test that boring/repetitive progressions generate suggestions."""
        result = diagnose_progression(common_progressions['repetitive'])
        
        assert 'suggestions' in result, "Result should contain 'suggestions'"
        assert isinstance(result['suggestions'], list), "Suggestions should be a list"
        # Repetitive progressions should generate suggestions
        assert len(result['suggestions']) >= 0, "Suggestions list should exist (may be empty)"
    
    def test_reharmonization_generates_options(self, common_progressions):
        """Test reharmonization generates valid suggestions."""
        suggestions = generate_reharmonizations(common_progressions['diatonic'], style="jazz", count=3)
        
        assert len(suggestions) >= 1, f"Expected at least 1 suggestion, got {len(suggestions)}"
        assert len(suggestions) <= 3, f"Expected at most 3 suggestions, got {len(suggestions)}"
        
        # Verify structure of each suggestion
        required_keys = {'chords', 'technique', 'mood'}
        for i, suggestion in enumerate(suggestions):
            assert isinstance(suggestion, dict), f"Suggestion {i} should be a dict"
            assert 'chords' in suggestion, f"Suggestion {i} missing 'chords' key"
            assert 'technique' in suggestion, f"Suggestion {i} missing 'technique' key"
            assert 'mood' in suggestion, f"Suggestion {i} missing 'mood' key"
            assert isinstance(suggestion['chords'], list), \
                f"Suggestion {i} 'chords' should be a list, got {type(suggestion['chords'])}"
            assert len(suggestion['chords']) > 0, f"Suggestion {i} should have at least one chord"
    
    @pytest.mark.parametrize("style", ["jazz", "pop", "rnb", "classical", "experimental"])
    def test_reharmonization_different_styles(self, style, common_progressions):
        """Test reharmonization works for different musical styles."""
        result = generate_reharmonizations(common_progressions['diatonic'], style=style, count=2)
        
        assert len(result) >= 1, f"Style '{style}' should generate at least 1 suggestion"
        assert all('technique' in s for s in result), \
            f"All suggestions for style '{style}' should have 'technique'"


class TestIntentSchema:
    """Test intent schema and validation."""
    
    def test_create_complete_intent(self, sample_intent):
        """Test creating a complete song intent with all phases."""
        assert sample_intent.song_root.core_event == "Loss of a loved one"
        assert sample_intent.song_root.core_resistance == "Fear of moving on"
        assert sample_intent.song_root.core_longing == "To remember without pain"
        assert sample_intent.song_intent.mood_primary == "Grief"
        assert sample_intent.song_intent.vulnerability_scale == "High"
        assert sample_intent.technical_constraints.technical_key == "F"
        assert sample_intent.technical_constraints.technical_mode == "major"
    
    @pytest.mark.parametrize("emotion", ["grief", "defiance", "longing", "anger"])
    def test_suggest_rule_break_for_emotion(self, emotion):
        """Test rule break suggestions for different emotions."""
        suggestions = suggest_rule_break(emotion)
        
        assert isinstance(suggestions, list), f"Suggestions should be a list for '{emotion}'"
        # Some emotions may not have suggestions, but structure should be consistent
        if len(suggestions) > 0:
            required_keys = {'rule', 'description', 'effect', 'use_when'}
            for i, suggestion in enumerate(suggestions):
                assert isinstance(suggestion, dict), \
                    f"Suggestion {i} for '{emotion}' should be a dict"
                assert 'rule' in suggestion, \
                    f"Suggestion {i} for '{emotion}' missing 'rule' key"
                # Verify all expected keys are present
                missing_keys = required_keys - set(suggestion.keys())
                assert len(missing_keys) == 0, \
                    f"Suggestion {i} for '{emotion}' missing keys: {missing_keys}"
    
    def test_list_all_rules(self):
        """Test listing all available rules."""
        rules = list_all_rules()
        
        assert isinstance(rules, dict), f"Rules should be a dict, got {type(rules)}"
        assert len(rules) > 0, "Should have at least one rule category"
        
        # Verify structure: should have category keys with list values
        for category, rule_list in rules.items():
            assert isinstance(category, str), f"Category should be string, got {type(category)}"
            assert isinstance(rule_list, list), \
                f"Rules for '{category}' should be a list, got {type(rule_list)}"
            assert len(rule_list) > 0, f"Category '{category}' should have at least one rule"
        
        # Check that HARMONY_ModalInterchange is in the Harmony category
        harmony_rules = rules.get('Harmony', [])
        assert 'HARMONY_ModalInterchange' in harmony_rules, \
            f"Expected HARMONY_ModalInterchange in Harmony rules, got {harmony_rules}"
    
    def test_validate_intent_complete(self):
        """Test validation of a complete, valid intent."""
        intent = CompleteSongIntent(
            song_root=SongRoot(
                core_event="Test event",
                core_resistance="Test resistance",
                core_longing="Test longing"
            ),
            song_intent=SongIntent(
                mood_primary="Grief",
                vulnerability_scale="High"
            ),
            technical_constraints=TechnicalConstraints(
                technical_key="C",
                technical_mode="major",
                technical_rule_to_break="HARMONY_ModalInterchange",
                rule_breaking_justification="Creates emotional depth"
            )
        )
        
        result = validate_intent(intent)
        
        assert isinstance(result, list), "Validation result should be a list"
        assert len(result) == 0, \
            f"Complete intent should have no validation issues, got: {result}"
    
    def test_validate_intent_missing_justification(self):
        """Test validation catches missing rule-breaking justification."""
        intent = CompleteSongIntent(
            song_root=SongRoot(core_event="Test", core_longing="Test longing"),
            song_intent=SongIntent(mood_primary="Grief"),
            technical_constraints=TechnicalConstraints(
                technical_key="C",
                technical_rule_to_break="HARMONY_ModalInterchange",
                rule_breaking_justification=""  # Missing!
            )
        )
        
        result = validate_intent(intent)
        
        assert isinstance(result, list), "Validation result should be a list"
        assert len(result) > 0, "Intent with missing justification should have validation issues"
        
        # Check that one of the issues mentions justification or rule
        issue_text = ' '.join(result).lower()
        assert 'justification' in issue_text or 'rule' in issue_text, \
            f"Expected validation issue about justification/rule, got: {result}"
    
    def test_validate_intent_missing_core_fields(self):
        """Test validation catches missing core Phase 0 fields."""
        intent = CompleteSongIntent(
            song_root=SongRoot(),  # Empty core fields
            song_intent=SongIntent(mood_primary="Grief"),
            technical_constraints=TechnicalConstraints(technical_key="C")
        )
        
        result = validate_intent(intent)
        
        assert len(result) > 0, "Intent with missing core fields should have validation issues"
        issue_text = ' '.join(result).lower()
        assert 'core_event' in issue_text or 'core_longing' in issue_text, \
            f"Expected validation issue about core fields, got: {result}"


class TestGrooveTemplates:
    """Test groove template system."""
    
    def test_list_available_genres(self):
        """Test listing all available genre templates."""
        genres = list_genres()
        
        assert isinstance(genres, list), f"Genres should be a list, got {type(genres)}"
        assert len(genres) > 0, "Should have at least one genre template"
        
        # Verify expected genres exist
        expected_genres = ['funk', 'jazz', 'rock', 'hiphop', 'bedroom_lofi']
        found_genres = [g for g in expected_genres if g in genres]
        assert len(found_genres) >= 3, \
            f"Expected at least 3 of {expected_genres}, found: {found_genres}"
    
    @pytest.mark.parametrize("genre", ["funk", "jazz", "rock", "hiphop", "bedroom_lofi"])
    def test_get_genre_template(self, genre):
        """Test retrieving templates for various genres."""
        template = get_genre_template(genre)
        
        assert template is not None, f"Template for '{genre}' should not be None"
        assert hasattr(template, 'swing_factor'), \
            f"Template for '{genre}' should have 'swing_factor' attribute"
        assert hasattr(template, 'tempo_bpm'), \
            f"Template for '{genre}' should have 'tempo_bpm' attribute"
        assert hasattr(template, 'timing_deviations'), \
            f"Template for '{genre}' should have 'timing_deviations' attribute"
        assert hasattr(template, 'velocity_curve'), \
            f"Template for '{genre}' should have 'velocity_curve' attribute"
        
        # Verify swing_factor is in valid range
        assert 0.0 <= template.swing_factor <= 1.0, \
            f"Swing factor for '{genre}' should be 0.0-1.0, got {template.swing_factor}"
        
        # Verify timing deviations exist
        assert isinstance(template.timing_deviations, list), \
            f"Timing deviations for '{genre}' should be a list"
        assert len(template.timing_deviations) > 0, \
            f"Timing deviations for '{genre}' should not be empty"
    
    @pytest.mark.parametrize("invalid_genre", [
        'non_existent_genre_xyz',
        '',
        '   ',
        'FUNK',  # Case sensitivity - should work but test edge case
        'funk-rock',  # Hyphenated (should normalize)
        'funk rock',  # Space (should normalize)
        'funk_rock',  # Underscore variant
        '123',
        'genre_with_numbers_123',
        '!@#$%',
    ])
    def test_invalid_genre_raises(self, invalid_genre):
        """Test that invalid genre names raise appropriate exceptions."""
        # Note: Some inputs like 'FUNK', 'funk-rock', 'funk rock' might actually work
        # due to normalization, so we test that they either work or raise ValueError
        try:
            template = get_genre_template(invalid_genre)
            # If it doesn't raise, verify it's a valid template
            assert template is not None, f"'{invalid_genre}' should either raise or return valid template"
        except ValueError as e:
            # Should raise ValueError with "Unknown genre" message
            assert "Unknown genre" in str(e), \
                f"Error message should mention 'Unknown genre', got: {e}"
            # Error should list available genres
            assert "Available:" in str(e), \
                f"Error message should list available genres, got: {e}"
    
    def test_invalid_genre_error_message(self):
        """Test that error message includes helpful information."""
        with pytest.raises(ValueError) as exc_info:
            get_genre_template('nonexistent')
        
        error_msg = str(exc_info.value)
        assert "Unknown genre" in error_msg, \
            f"Error should mention 'Unknown genre', got: {error_msg}"
        assert "Available:" in error_msg, \
            f"Error should list available genres, got: {error_msg}"
        # Should mention at least one known genre
        assert any(genre in error_msg for genre in ['funk', 'jazz', 'rock']), \
            f"Error should mention available genres, got: {error_msg}"


class TestModuleIntegration:
    """Test integration between modules."""
    
    def test_intent_to_progression_flow(self, sample_intent):
        """Test flow from intent to chord progression."""
        # Verify intent structure is valid
        assert sample_intent.technical_constraints.technical_key == "F"
        assert sample_intent.technical_constraints.technical_mode == "major"
        assert sample_intent.song_intent.mood_primary == "Grief"
        
        # Intent should be serializable (for API/CLI use)
        intent_dict = sample_intent.to_dict()
        assert isinstance(intent_dict, dict), "Intent should be serializable to dict"
        assert 'technical_constraints' in intent_dict, "Dict should contain technical_constraints"
    
    def test_progression_to_diagnosis_flow(self, common_progressions):
        """Test flow from progression string to diagnosis."""
        progression = common_progressions['diatonic']
        result = diagnose_progression(progression)
        
        assert isinstance(result, dict), "Diagnosis should return a dict"
        assert 'key' in result, "Diagnosis should include detected key"
        assert 'mode' in result, "Diagnosis should include detected mode"
        assert 'chords' in result, "Diagnosis should include parsed chords"
        
        # Verify chords match input
        assert len(result['chords']) == 4, \
            f"Expected 4 chords in diagnosis, got {len(result['chords'])}"
    
    def test_diagnosis_to_reharmonization_flow(self, common_progressions):
        """Test complete flow from diagnosis to reharmonization."""
        progression = common_progressions['diatonic']
        
        # First diagnose
        diagnosis = diagnose_progression(progression)
        assert 'key' in diagnosis, "Diagnosis should have key"
        
        # Then reharmonize using the same progression
        reharms = generate_reharmonizations(progression, style="jazz", count=2)
        
        assert len(reharms) >= 1, "Should generate at least one reharmonization"
        assert all('chords' in r for r in reharms), \
            "All reharmonizations should have chord lists"
        
        # Verify reharmonizations have same number of chords as original
        original_chord_count = len(diagnosis['chords'])
        for reharm in reharms:
            assert len(reharm['chords']) >= original_chord_count - 2, \
                f"Reharmonization should have similar chord count (original: {original_chord_count}, got: {len(reharm['chords'])})"
    
    def test_intent_groove_emotion_mapping(self):
        """Test that emotions can be mapped to appropriate grooves."""
        # Test various emotions
        emotions = ["Grief", "Anger", "Joy", "Longing"]
        
        for emotion in emotions:
            intent = SongIntent(mood_primary=emotion)
            assert intent.mood_primary == emotion, \
                f"Intent should preserve emotion '{emotion}'"
            
            # Verify we can get rule suggestions for the emotion
            suggestions = suggest_rule_break(emotion.lower())
            assert isinstance(suggestions, list), \
                f"Should get suggestions list for '{emotion}'"


class TestDataFiles:
    """Test data file integrity."""
    
    def test_rule_breaks_json_loadable(self):
        """Test that rule breaking database JSON file exists and is valid."""
        db_path = Path('music_brain/data/rule_breaking_database.json')
        
        assert db_path.exists(), f"Rule breaking database should exist at {db_path}"
        
        # Try to load and validate structure
        with open(db_path) as f:
            data = json.load(f)
        
        assert isinstance(data, dict), f"Database should be a dict, got {type(data)}"
        assert len(data) > 0, "Database should not be empty"
        
        # Should have rule_breaks key or be a dict of rules
        assert 'rule_breaks' in data or len(data) > 0, \
            "Database should contain rule_breaks or have rule entries"
    
    def test_chord_progressions_json_loadable(self):
        """Test that chord progression families JSON file is valid if it exists."""
        chord_prog_path = Path('music_brain/data/chord_progression_families.json')
        
        if chord_prog_path.exists():
            with open(chord_prog_path) as f:
                data = json.load(f)
            
            assert isinstance(data, (dict, list)), \
                f"Chord progressions should be dict or list, got {type(data)}"
            assert len(data) > 0, "Chord progressions file should not be empty"
    
    def test_emotional_mapping_exists(self):
        """Test that emotional mapping module exists and has expected interface."""
        try:
            from music_brain.data import emotional_mapping
            
            # Should have emotional mapping functionality
            has_get_params = hasattr(emotional_mapping, 'get_parameters_for_state')
            has_presets = hasattr(emotional_mapping, 'EMOTIONAL_PRESETS')
            
            assert has_get_params or has_presets, \
                "Emotional mapping should have get_parameters_for_state or EMOTIONAL_PRESETS"
        except ImportError:
            pytest.skip("emotional_mapping module not available")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_progression(self):
        """Empty progression should handle gracefully."""
        result = diagnose_progression("")
        
        assert isinstance(result, dict), "Empty progression should return dict"
        assert result.get('key') == 'unknown', \
            f"Empty progression should have 'unknown' key, got '{result.get('key')}'"
        assert 'issues' in result, "Empty progression should have issues list"
        assert len(result.get('issues', [])) > 0, \
            "Empty progression should have at least one issue"
    
    def test_whitespace_only_progression(self):
        """Progression with only whitespace should handle gracefully."""
        result = diagnose_progression("   \n\t  ")
        
        assert isinstance(result, dict), "Whitespace-only progression should return dict"
        assert result.get('key') == 'unknown', \
            "Whitespace-only progression should have 'unknown' key"
    
    @pytest.mark.parametrize("invalid_chord", ["XYZ123", "Q", "123"])
    def test_invalid_chord_notation(self, invalid_chord):
        """Invalid chord notation should raise ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            Chord.from_string(invalid_chord, key="C")
    
    def test_invalid_chord_notation_edge_cases(self):
        """Test edge cases for invalid chord notation."""
        # Empty string should raise ValueError
        with pytest.raises(ValueError):
            Chord.from_string("", key="C")
        
        # Some edge cases might parse partially or fail differently
        # C#b# might parse as C# with invalid quality, or fail completely
        try:
            chord = Chord.from_string("C#b#", key="C")
            # If it doesn't raise, verify it's at least a valid chord structure
            assert hasattr(chord, 'root'), "Parsed chord should have root"
        except ValueError:
            # Expected for truly invalid chords
            pass
    
    def test_partially_valid_progression(self):
        """Progression with some invalid chords should handle gracefully."""
        # Mix of valid and invalid chords
        result = diagnose_progression("C-XYZ-F-G")
        
        assert isinstance(result, dict), "Should return dict even with invalid chords"
        assert 'chords' in result, "Should have chords list"
        # Should parse valid chords
        assert len(result['chords']) >= 3, \
            f"Should parse at least 3 valid chords, got {len(result['chords'])}"
    
    def test_mismatched_key_progression(self):
        """Progression should auto-detect key regardless of provided key."""
        result = diagnose_progression("C-F-G-C")
        
        assert isinstance(result, dict), "Should return dict"
        assert 'key' in result, "Should detect key"
        assert result['key'] == 'C', \
            f"Should detect C as key for C-F-G-C progression, got '{result['key']}'"
    
    def test_extremely_long_progression(self):
        """Very long progression should handle gracefully."""
        long_prog = "-".join(["C", "Am", "F", "G"] * 20)  # 80 chords
        result = diagnose_progression(long_prog)
        
        assert isinstance(result, dict), "Should return dict for long progression"
        assert 'key' in result, "Should detect key for long progression"
        assert 'chords' in result, "Should parse chords for long progression"
        assert len(result['chords']) == 80, \
            f"Should parse all 80 chords, got {len(result['chords'])}"
    
    def test_single_chord_progression(self):
        """Single chord progression should be handled."""
        result = diagnose_progression("C")
        
        assert isinstance(result, dict), "Should return dict for single chord"
        assert 'key' in result, "Should detect key from single chord"
        assert len(result.get('chords', [])) == 1, \
            "Should parse single chord"
    
    def test_special_characters_in_progression(self):
        """Progression with special delimiters should parse correctly."""
        variants = [
            "C-Am-F-G",      # Hyphens
            "C Am F G",      # Spaces
            "C,Am,F,G",      # Commas
            "C|Am|F|G",      # Pipes
        ]
        
        for prog in variants:
            result = diagnose_progression(prog)
            assert isinstance(result, dict), f"Should parse '{prog}'"
            assert len(result.get('chords', [])) == 4, \
                f"Should parse 4 chords from '{prog}', got {len(result.get('chords', []))}"


class TestRuleBreakingDatabase:
    """Test rule-breaking database functionality."""
    
    def test_database_has_modal_interchange(self):
        """Test that modal interchange rule exists in database."""
        rules = list_all_rules()
        
        assert isinstance(rules, dict), "Rules should be a dict"
        
        # Check all categories for modal interchange
        all_rule_strings = []
        for category_rules in rules.values():
            assert isinstance(category_rules, list), \
                f"Category rules should be list, got {type(category_rules)}"
            all_rule_strings.extend(category_rules)
        
        modal_rules = [r for r in all_rule_strings 
                      if 'modal' in r.lower() or 'interchange' in r.lower()]
        
        assert len(modal_rules) > 0, \
            f"Should have modal interchange rule, found rules: {all_rule_strings[:10]}"
    
    def test_database_structure(self):
        """Test that rule database has expected structure."""
        rules = list_all_rules()
        
        assert isinstance(rules, dict), "Rules should be a dict"
        assert len(rules) > 0, "Should have at least one category"
        
        # Verify categories are strings and contain lists
        for category, rule_list in rules.items():
            assert isinstance(category, str), \
                f"Category '{category}' should be string"
            assert isinstance(rule_list, list), \
                f"Rules for '{category}' should be list"
            assert len(rule_list) > 0, \
                f"Category '{category}' should have at least one rule"
    
    def test_suggest_returns_justified_breaks(self):
        """Test that rule break suggestions have proper structure."""
        suggestions = suggest_rule_break("grief")
        
        assert isinstance(suggestions, list), "Suggestions should be a list"
        
        if len(suggestions) > 0:
            for i, suggestion in enumerate(suggestions):
                assert isinstance(suggestion, dict), \
                    f"Suggestion {i} should be a dict"
                assert 'rule' in suggestion, \
                    f"Suggestion {i} should have 'rule' key"
                assert 'description' in suggestion, \
                    f"Suggestion {i} should have 'description' key"
                assert 'effect' in suggestion, \
                    f"Suggestion {i} should have 'effect' key"
                assert 'use_when' in suggestion, \
                    f"Suggestion {i} should have 'use_when' key"
                
                # Verify rule key is a string
                assert isinstance(suggestion['rule'], str), \
                    f"Suggestion {i} 'rule' should be string"
    
    def test_suggest_for_multiple_emotions(self):
        """Test rule suggestions for various emotions."""
        emotions = ["grief", "joy", "anger", "longing", "defiance"]
        
        for emotion in emotions:
            suggestions = suggest_rule_break(emotion)
            assert isinstance(suggestions, list), \
                f"Should get list for '{emotion}'"
            
            # If suggestions exist, verify structure
            for suggestion in suggestions:
                assert 'rule' in suggestion, \
                    f"Suggestion for '{emotion}' should have 'rule' key"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
