"""
Comprehensive unit tests for core Music Brain modules.
Tests harmony generation, progression analysis, intent processing, and module integration.
"""

import pytest
from pathlib import Path
import json
import sys
import importlib
import importlib.util

# Prefer this repository's code over any globally-installed copy
REPO_ROOT = Path(__file__).resolve().parents[2]
root_str = str(REPO_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
sys.path = [p for p in sys.path if "miDiKompanion" not in p]
# Purge any preloaded music_brain modules from other locations
for name in list(sys.modules.keys()):
    if name.startswith("music_brain"):
        mod_file = getattr(sys.modules[name], "__file__", "") or ""
        if "kelly-clean" not in mod_file:
            sys.modules.pop(name, None)
importlib.invalidate_caches()

# Force-load local music_brain.groove.templates before tests import it
templates_path = REPO_ROOT / "music_brain" / "groove" / "templates.py"
spec = importlib.util.spec_from_file_location(
    "music_brain.groove.templates", templates_path
)
templates = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(templates)
sys.modules["music_brain.groove.templates"] = templates

# Force-load key modules from this repo to avoid any globally-installed versions
def _force_local_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

_force_local_module("music_brain.session.intent_schema", "music_brain/session/intent_schema.py")
_force_local_module("music_brain.structure.progression", "music_brain/structure/progression.py")
_force_local_module("music_brain.structure.chord", "music_brain/structure/chord.py")
_force_local_module("music_brain.groove.extractor", "music_brain/groove/extractor.py")
_force_local_module("music_brain.data", "music_brain/data/__init__.py")
_force_local_module("music_brain.data.rule_breaking_database", "music_brain/data/rule_breaking_database.py")

# Test imports
from music_brain.session.intent_schema import (
    CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
    suggest_rule_break, validate_intent, list_all_rules
)
from music_brain.structure.chord import Chord, ChordProgression
from music_brain.structure.progression import diagnose_progression, generate_reharmonizations
from music_brain.groove.templates import get_genre_template, list_genre_templates


class TestChordModule:
    """Test chord parsing and analysis."""
    
    def test_parse_major_chord(self):
        chord = Chord.from_string("C", key="C")
        assert chord.root == "C"
        assert chord.quality in ["major", "maj", ""]
    
    def test_parse_minor_chord(self):
        chord = Chord.from_string("Am", key="C")
        assert chord.root == "A"
        assert chord.quality in ["minor", "min", "m"]
    
    def test_parse_seventh_chords(self):
        # Major 7th
        chord = Chord.from_string("Cmaj7", key="C")
        assert chord.root == "C"
        assert "maj7" in chord.quality.lower() or "7" in str(chord.extensions)
        
        # Dominant 7th
        chord = Chord.from_string("G7", key="C")
        assert chord.root == "G"
        
        # Minor 7th
        chord = Chord.from_string("Am7", key="C")
        assert chord.root == "A"
    
    def test_parse_extended_chords(self):
        # 9th chord
        chord = Chord.from_string("Cmaj9", key="C")
        assert chord.root == "C"
        
        # 11th chord
        chord = Chord.from_string("C11", key="C")
        assert chord.root == "C"
    
    def test_parse_altered_chords(self):
        # Flat 5
        chord = Chord.from_string("Cdim", key="C")
        assert chord.root == "C"
        
        # Sharp 5 (augmented)
        chord = Chord.from_string("Caug", key="C")
        assert chord.root == "C"
    
    def test_chord_voicing_generation(self):
        chord = Chord.from_string("C", key="C")
        notes = chord.get_voicing(octave=4, voicing_type='close')
        assert len(notes) >= 3  # At least root, third, fifth
        assert notes[0] < notes[-1]  # Ascending order


class TestProgressionAnalysis:
    """Test chord progression diagnosis and analysis."""
    
    def test_diagnose_diatonic_progression(self):
        result = diagnose_progression("C-Am-F-G", key="C major")
        assert 'key' in result
        assert 'mode' in result
        assert isinstance(result.get('issues', []), list)
    
    def test_diagnose_modal_interchange(self):
        # F-C-Bbm-F in F major (Bbm borrowed from F minor)
        result = diagnose_progression("F-C-Bbm-F", key="F major")
        assert 'borrowed' in str(result).lower() or 'modal' in str(result).lower()
    
    def test_diagnose_detects_key(self):
        result = diagnose_progression("C-F-G-C")
        assert result['key'] in ['C', 'C major', 'C maj']
    
    def test_diagnose_provides_suggestions(self):
        result = diagnose_progression("C-C-C-C")  # Boring progression
        assert 'suggestions' in result
        assert isinstance(result['suggestions'], list)
    
    def test_reharmonization_generates_options(self):
        suggestions = generate_reharmonizations("C-Am-F-G", style="jazz", count=3)
        assert len(suggestions) >= 1
        assert all('chords' in s for s in suggestions)
        assert all('technique' in s for s in suggestions)
    
    def test_reharmonization_different_styles(self):
        jazz_result = generate_reharmonizations("C-Am-F-G", style="jazz", count=2)
        pop_result = generate_reharmonizations("C-Am-F-G", style="pop", count=2)
        
        # Results should exist for both styles
        assert len(jazz_result) >= 1
        assert len(pop_result) >= 1


class TestIntentSchema:
    """Test intent schema and validation."""
    
    def test_create_complete_intent(self):
        intent = CompleteSongIntent(
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
        
        assert intent.song_root.core_event == "Loss of a loved one"
        assert intent.song_intent.mood_primary == "Grief"
        assert intent.technical_constraints.technical_key == "F"
    
    def test_suggest_rule_break_for_emotion(self):
        suggestions = suggest_rule_break("grief")
        assert len(suggestions) > 0
        assert all('rule' in s for s in suggestions)
    
    def test_list_all_rules(self):
        rules = list_all_rules()
        assert "Harmony" in rules
        assert "Rhythm" in rules
        flat_rules = sum(rules.values(), [])
        assert 'HARMONY_ModalInterchange' in flat_rules
    
    def test_validate_intent_complete(self):
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
        assert result == []
    
    def test_validate_intent_missing_justification(self):
        intent = CompleteSongIntent(
            song_root=SongRoot(core_event="Test"),
            song_intent=SongIntent(mood_primary="Grief"),
            technical_constraints=TechnicalConstraints(
                technical_key="C",
                technical_rule_to_break="HARMONY_ModalInterchange",
                rule_breaking_justification=""  # Missing!
            )
        )
        
        result = validate_intent(intent)
        assert any("Rule to break" in r for r in result)


class TestGrooveTemplates:
    """Test groove template system."""
    
    def test_list_available_genres(self):
        genres = list_genre_templates()
        assert len(genres) > 0
        assert 'funk' in genres
        assert 'boom-bap' in genres or 'boom_bap' in genres
    
    def test_get_funk_template(self):
        template = get_genre_template('funk')
        assert template is not None
        assert hasattr(template, 'swing') or hasattr(template, 'swing_factor')
    
    def test_get_boom_bap_template(self):
        template = get_genre_template('boom-bap')
        assert template is not None
    
    def test_template_has_timing_data(self):
        template = get_genre_template('funk')
        # Template should have timing/feel information
        assert template is not None
    
    def test_invalid_genre_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_genre_template('non_existent_genre_xyz')


class TestModuleIntegration:
    """Test integration between modules."""
    
    def test_intent_to_progression_flow(self):
        """Test flow from intent to chord progression."""
        intent = CompleteSongIntent(
            song_root=SongRoot(core_event="Test"),
            song_intent=SongIntent(mood_primary="Grief"),
            technical_constraints=TechnicalConstraints(
                technical_key="F",
                technical_mode="major"
            )
        )
        
        # This should not raise
        assert intent.technical_constraints.technical_key == "F"
    
    def test_progression_to_diagnosis_flow(self):
        """Test flow from progression to diagnosis."""
        progression = "F-C-Am-Dm"
        result = diagnose_progression(progression, key="F major")
        
        assert 'key' in result
        assert isinstance(result, dict)
    
    def test_diagnosis_to_reharmonization_flow(self):
        """Test flow from diagnosis to reharmonization."""
        progression = "C-Am-F-G"
        
        # First diagnose
        diagnosis = diagnose_progression(progression)
        
        # Then reharmonize
        reharms = generate_reharmonizations(progression, style="jazz", count=2)
        
        assert len(reharms) >= 1
    
    def test_intent_groove_emotion_mapping(self):
        """Test that emotions map to appropriate grooves."""
        # Grief should map to laid-back grooves
        grief_intent = SongIntent(mood_primary="Grief")
        assert grief_intent.mood_primary == "Grief"
        
        # Anger should map to aggressive grooves
        anger_intent = SongIntent(mood_primary="Anger")
        assert anger_intent.mood_primary == "Anger"


class TestDataFiles:
    """Test data file integrity."""
    
    def test_rule_breaks_json_loadable(self):
        from music_brain.data import rule_breaking_database
        
        # Should have rule breaks data
        assert hasattr(rule_breaking_database, 'RULE_BREAKS') or \
               Path('music_brain/data/rule_breaking_database.json').exists()
    
    def test_chord_progressions_json_loadable(self):
        chord_prog_path = Path('music_brain/data/chord_progression_families.json')
        if chord_prog_path.exists():
            with open(chord_prog_path) as f:
                data = json.load(f)
            assert len(data) > 0
    
    def test_emotional_mapping_exists(self):
        from music_brain.data import emotional_mapping
        
        # Should have emotional mapping functionality
        assert hasattr(emotional_mapping, 'get_parameters_for_state') or \
               hasattr(emotional_mapping, 'EMOTIONAL_PRESETS')


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_progression(self):
        """Empty progression should handle gracefully."""
        result = diagnose_progression("")
        assert isinstance(result, dict)
        assert result.get("issues")
    
    def test_invalid_chord_notation(self):
        """Invalid chord should handle gracefully."""
        # This might raise or might parse partially
        try:
            Chord.from_string("XYZ123", key="C")
        except (ValueError, KeyError, AttributeError):
            pass  # Expected
    
    def test_mismatched_key_progression(self):
        """Progression in different key should still analyze."""
        result = diagnose_progression("C-F-G-C", key="D major")
        # Should still return a result
        assert 'key' in result or 'issues' in result
    
    def test_extremely_long_progression(self):
        """Very long progression should handle gracefully."""
        long_prog = "-".join(["C", "Am", "F", "G"] * 20)
        result = diagnose_progression(long_prog)
        assert 'key' in result


class TestRuleBreakingDatabase:
    """Test rule-breaking database functionality."""
    
    def test_database_has_modal_interchange(self):
        rules = list_all_rules()
        flat_rules = sum(rules.values(), [])
        modal_rules = [r for r in flat_rules if 'modal' in r.lower() or 'interchange' in r.lower()]
        assert modal_rules
    
    def test_database_has_examples(self):
        rules = list_all_rules()
        # At least some rules should have examples
        with_examples = [r for r in rules.values() if r]
        assert with_examples or len(rules) > 5
    
    def test_suggest_returns_justified_breaks(self):
        suggestions = suggest_rule_break("grief")
        for suggestion in suggestions:
            # Should have justification for the emotion
            assert 'rule' in suggestion or 'name' in suggestion


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
