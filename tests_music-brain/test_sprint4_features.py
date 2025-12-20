"""
Tests for Sprint 4 features: Audio Analysis and Arrangement Generation.

Run with: pytest tests_music-brain/test_sprint4_features.py -v
"""

import pytest
from pathlib import Path


class TestAudioAnalysisImports:
    """Test that audio analysis modules can be imported."""
    
    def test_import_audio_module(self):
        from music_brain.audio import (
            analyze_feel, AudioFeatures,
            ChordDetector, ChordDetection,
            analyze_frequency_bands, FrequencyProfile,
            analyze_reference, ReferenceProfile
        )
        assert callable(analyze_feel)
        assert AudioFeatures is not None
        assert ChordDetector is not None
        assert callable(analyze_frequency_bands)
        assert callable(analyze_reference)
    
    def test_import_chord_detection(self):
        from music_brain.audio.chord_detection import ChordDetector, ChordDetection
        assert ChordDetector is not None
        assert ChordDetection is not None
    
    def test_import_frequency_analysis(self):
        from music_brain.audio.frequency_analysis import (
            analyze_frequency_bands,
            FrequencyProfile,
            compare_frequency_profiles,
            suggest_eq_adjustments
        )
        assert callable(analyze_frequency_bands)
        assert FrequencyProfile is not None
        assert callable(compare_frequency_profiles)
        assert callable(suggest_eq_adjustments)


class TestArrangementImports:
    """Test that arrangement modules can be imported."""
    
    def test_import_arrangement_module(self):
        from music_brain.arrangement import (
            generate_arrangement,
            ArrangementGenerator,
            SectionTemplate,
            EnergyArc,
            NarrativeArc
        )
        assert callable(generate_arrangement)
        assert ArrangementGenerator is not None
        assert SectionTemplate is not None
        assert EnergyArc is not None
        assert NarrativeArc is not None
    
    def test_import_templates(self):
        from music_brain.arrangement.templates import (
            SectionTemplate,
            ArrangementTemplate,
            get_genre_template,
            list_available_genres
        )
        assert SectionTemplate is not None
        assert ArrangementTemplate is not None
        assert callable(get_genre_template)
        assert callable(list_available_genres)
    
    def test_import_energy_arc(self):
        from music_brain.arrangement.energy_arc import (
            EnergyArc,
            NarrativeArc,
            calculate_energy_curve,
            map_emotion_to_arc
        )
        assert EnergyArc is not None
        assert NarrativeArc is not None
        assert callable(calculate_energy_curve)
        assert callable(map_emotion_to_arc)
    
    def test_import_bass_generator(self):
        from music_brain.arrangement.bass_generator import (
            BassPattern,
            generate_bass_line,
            suggest_bass_pattern
        )
        assert BassPattern is not None
        assert callable(generate_bass_line)
        assert callable(suggest_bass_pattern)


class TestFrequencyAnalysis:
    """Test frequency analysis functionality."""
    
    def test_frequency_profile_creation(self):
        from music_brain.audio.frequency_analysis import FrequencyProfile
        
        profile = FrequencyProfile(
            bass=0.8,
            mids=0.5,
            brilliance=0.3,
            brightness=0.4,
            warmth=0.7,
            clarity=0.5
        )
        
        assert profile.bass == 0.8
        assert profile.mids == 0.5
        assert profile.brilliance == 0.3
    
    def test_frequency_profile_to_dict(self):
        from music_brain.audio.frequency_analysis import FrequencyProfile
        
        profile = FrequencyProfile(bass=0.8, mids=0.5)
        data = profile.to_dict()
        
        assert 'bands' in data
        assert 'characteristics' in data
        assert data['bands']['bass'] == 0.8
        assert data['bands']['mids'] == 0.5
    
    def test_frequency_profile_production_notes(self):
        from music_brain.audio.frequency_analysis import FrequencyProfile
        
        # Create profile with strong bass
        profile = FrequencyProfile(bass=0.9, mids=0.3)
        notes = profile.get_production_notes()
        
        assert isinstance(notes, list)
        assert len(notes) > 0
        # Should mention bass-heavy mix
        assert any('bass' in note.lower() for note in notes)


class TestArrangementGeneration:
    """Test arrangement generation functionality."""
    
    def test_section_template_creation(self):
        from music_brain.arrangement.templates import create_verse, SectionType
        
        verse = create_verse(length_bars=8, energy_level=0.5)
        
        assert verse.section_type == SectionType.VERSE
        assert verse.length_bars == 8
        assert verse.energy_level == 0.5
        assert isinstance(verse.instruments, list)
    
    def test_genre_templates(self):
        from music_brain.arrangement.templates import (
            get_genre_template,
            list_available_genres
        )
        
        genres = list_available_genres()
        assert 'pop' in genres
        assert 'rock' in genres
        assert 'edm' in genres
        
        # Test pop template
        pop_template = get_genre_template('pop')
        assert pop_template.genre == 'pop'
        assert len(pop_template.sections) > 0
        assert pop_template.total_bars > 0
    
    def test_genre_template_invalid(self):
        from music_brain.arrangement.templates import get_genre_template
        
        with pytest.raises(ValueError):
            get_genre_template('invalid_genre_xyz')
    
    def test_energy_arc_calculation(self):
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc
        )
        
        arc = calculate_energy_curve(
            NarrativeArc.CLIMB_TO_CLIMAX,
            num_sections=8,
            base_intensity=0.6
        )
        
        assert arc.narrative_arc == NarrativeArc.CLIMB_TO_CLIMAX
        assert len(arc.energy_curve) == 8
        assert all(0.0 <= e <= 1.0 for e in arc.energy_curve)
    
    def test_emotion_to_arc_mapping(self):
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc
        )
        
        # Test grief maps to slow reveal
        arc = map_emotion_to_arc("grief")
        assert arc == NarrativeArc.SLOW_REVEAL
        
        # Test anger maps to explosive start
        arc = map_emotion_to_arc("anger")
        assert arc == NarrativeArc.EXPLOSIVE_START
    
    def test_bass_line_generation(self):
        from music_brain.arrangement.bass_generator import (
            generate_bass_line,
            BassPattern
        )
        
        chords = ["C", "G", "Am", "F"]
        bass_line = generate_bass_line(
            chord_progression=chords,
            pattern=BassPattern.ROOT_FIFTH,
            ppq=480
        )
        
        assert bass_line.pattern == BassPattern.ROOT_FIFTH
        assert len(bass_line.notes) > 0
        # Should have notes for each chord
        assert len(bass_line.notes) >= len(chords)
    
    def test_bass_pattern_suggestion(self):
        from music_brain.arrangement.bass_generator import (
            suggest_bass_pattern,
            BassPattern
        )
        
        # Funk should suggest funk pattern
        pattern = suggest_bass_pattern("funk", energy_level=0.7)
        assert pattern == BassPattern.FUNK
        
        # Jazz should suggest walking bass
        pattern = suggest_bass_pattern("jazz", energy_level=0.5)
        assert pattern == BassPattern.WALKING
    
    def test_arrangement_generation(self):
        from music_brain.arrangement import generate_arrangement
        
        arrangement = generate_arrangement(
            genre="pop",
            emotion="grief",
            intensity=0.6
        )
        
        assert arrangement is not None
        assert arrangement.template.genre == "pop"
        assert len(arrangement.template.sections) > 0
        assert len(arrangement.instruments) > 0
        assert arrangement.energy_arc is not None
    
    def test_arrangement_production_notes(self):
        from music_brain.arrangement import generate_arrangement
        
        arrangement = generate_arrangement(genre="rock", emotion="anger")
        notes = arrangement.get_production_notes()
        
        assert isinstance(notes, list)
        assert len(notes) > 0
        # Should mention genre and tempo
        assert any('rock' in note.lower() for note in notes)


class TestBassGenerator:
    """Test bass line generator functionality."""
    
    def test_chord_root_parsing(self):
        from music_brain.arrangement.bass_generator import parse_chord_root
        
        # Test basic parsing (bass is in octave 1, not 2)
        assert parse_chord_root("C") == 24  # C1
        assert parse_chord_root("G") == 31  # G1
        
        # Test with quality
        assert parse_chord_root("Cm") == 24
        assert parse_chord_root("Gmaj7") == 31
    
    def test_chord_tones_extraction(self):
        from music_brain.arrangement.bass_generator import get_chord_tones
        
        # C major should have C, E, G
        tones = get_chord_tones("C")
        assert len(tones) == 3
        assert 24 in tones  # C in bass range
        
        # Cmaj7 should have 4 notes
        tones = get_chord_tones("Cmaj7")
        assert len(tones) == 4
    
    def test_bass_patterns_generate_notes(self):
        from music_brain.arrangement.bass_generator import (
            generate_root_only,
            generate_root_fifth,
            generate_walking_bass
        )
        
        # Root only
        notes = generate_root_only("C", bars=1, ppq=480)
        assert len(notes) > 0
        
        # Root and fifth
        notes = generate_root_fifth("C", bars=1, ppq=480)
        assert len(notes) > 0
        
        # Walking bass
        notes = generate_walking_bass(["C", "F", "G", "C"], ppq=480)
        assert len(notes) > 0


class TestCLICommands:
    """Test that CLI commands are registered correctly."""
    
    def test_audio_command_exists(self):
        from music_brain.cli import get_audio_module
        
        # Should be able to get audio module
        modules = get_audio_module()
        assert len(modules) == 6  # Should return 6 functions/classes
    
    def test_arrangement_command_exists(self):
        from music_brain.cli import get_arrangement_module
        
        # Should be able to get arrangement module
        modules = get_arrangement_module()
        assert len(modules) == 2  # Should return 2 items


class TestDataSerialization:
    """Test data serialization for audio and arrangement features."""
    
    def test_frequency_profile_serialization(self):
        from music_brain.audio.frequency_analysis import FrequencyProfile
        
        profile = FrequencyProfile(bass=0.8, mids=0.5, brilliance=0.3)
        data = profile.to_dict()
        
        # Should be JSON-serializable
        import json
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        
        # Should be able to deserialize
        loaded_data = json.loads(json_str)
        assert loaded_data['bands']['bass'] == 0.8
    
    def test_arrangement_serialization(self):
        from music_brain.arrangement import generate_arrangement
        import json
        
        arrangement = generate_arrangement(genre="pop", emotion="neutral")
        data = arrangement.to_dict()
        
        # Should be JSON-serializable
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        
        # Should contain expected fields
        loaded_data = json.loads(json_str)
        assert 'template' in loaded_data
        assert 'energy_arc' in loaded_data
        assert 'instruments' in loaded_data


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_chord_progression(self):
        from music_brain.arrangement.bass_generator import generate_bass_line, BassPattern
        
        # Empty progression should return empty bass line
        bass_line = generate_bass_line([], pattern=BassPattern.ROOT_ONLY)
        assert len(bass_line.notes) == 0
    
    def test_single_section_energy_arc(self):
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc
        )
        
        # Single section should work
        arc = calculate_energy_curve(NarrativeArc.CLIMB_TO_CLIMAX, num_sections=1)
        assert len(arc.energy_curve) == 1
        assert 0.0 <= arc.energy_curve[0] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
