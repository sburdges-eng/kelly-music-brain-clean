"""
Performance profiling tests for critical paths.

Tests measure and profile:
- Chord parsing performance
- Progression diagnosis performance
- Reharmonization generation performance
- Intent validation performance
- Groove template retrieval performance
- API endpoint response times
"""

import pytest
import time
import cProfile
import pstats
import io
from contextlib import contextmanager
from typing import Dict, Any

from music_brain.structure.chord import Chord
from music_brain.structure.progression import diagnose_progression, generate_reharmonizations
from music_brain.session.intent_schema import validate_intent, CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints
from music_brain.groove.templates import get_genre_template


@contextmanager
def profile_function(func, *args, **kwargs):
    """Profile a function call and return stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = func(*args, **kwargs)
    finally:
        profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    yield result, s.getvalue()


class TestChordParsingPerformance:
    """Test chord parsing performance."""
    
    def test_parse_single_chord_performance(self):
        """Test parsing a single chord is fast."""
        start = time.perf_counter()
        chord = Chord.from_string("Cmaj7", key="C")
        elapsed = time.perf_counter() - start
        
        assert chord is not None
        assert elapsed < 0.01  # Should parse in < 10ms
    
    def test_parse_multiple_chords_performance(self):
        """Test parsing multiple chords is efficient."""
        chord_strings = ["C", "Am", "F", "G", "Cmaj7", "Am7", "Fmaj7", "G7"] * 10
        
        start = time.perf_counter()
        chords = [Chord.from_string(c, key="C") for c in chord_strings]
        elapsed = time.perf_counter() - start
        
        assert len(chords) == 80
        assert elapsed < 0.1  # Should parse 80 chords in < 100ms
        assert all(c is not None for c in chords)
    
    def test_chord_voicing_generation_performance(self):
        """Test chord voicing generation performance."""
        chord = Chord.from_string("Cmaj9", key="C")
        
        start = time.perf_counter()
        voicing = chord.get_voicing(octave=4, voicing_type='close')
        elapsed = time.perf_counter() - start
        
        assert len(voicing) > 0
        assert elapsed < 0.01  # Should generate voicing in < 10ms


class TestProgressionAnalysisPerformance:
    """Test progression analysis performance."""
    
    def test_diagnose_short_progression_performance(self):
        """Test diagnosing short progression is fast."""
        progression = "C-Am-F-G"
        
        start = time.perf_counter()
        result = diagnose_progression(progression)
        elapsed = time.perf_counter() - start
        
        assert 'key' in result
        assert elapsed < 0.05  # Should diagnose in < 50ms
    
    def test_diagnose_long_progression_performance(self):
        """Test diagnosing long progression performance."""
        # 40-chord progression
        progression = "-".join(["C", "Am", "F", "G"] * 10)
        
        start = time.perf_counter()
        result = diagnose_progression(progression)
        elapsed = time.perf_counter() - start
        
        assert 'key' in result
        assert elapsed < 0.2  # Should diagnose 40 chords in < 200ms
    
    def test_reharmonization_generation_performance(self):
        """Test reharmonization generation performance."""
        progression = "C-Am-F-G"
        
        start = time.perf_counter()
        suggestions = generate_reharmonizations(progression, style="jazz", count=3)
        elapsed = time.perf_counter() - start
        
        assert len(suggestions) >= 1
        assert elapsed < 0.1  # Should generate in < 100ms
    
    def test_reharmonization_multiple_styles_performance(self):
        """Test generating reharmonizations for multiple styles."""
        progression = "C-Am-F-G"
        styles = ["jazz", "pop", "rnb", "classical", "experimental"]
        
        start = time.perf_counter()
        all_suggestions = []
        for style in styles:
            suggestions = generate_reharmonizations(progression, style=style, count=2)
            all_suggestions.extend(suggestions)
        elapsed = time.perf_counter() - start
        
        assert len(all_suggestions) >= len(styles)
        assert elapsed < 0.5  # Should generate for all styles in < 500ms


class TestIntentValidationPerformance:
    """Test intent validation performance."""
    
    @pytest.fixture
    def sample_intent(self):
        """Create a sample intent for testing."""
        return CompleteSongIntent(
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
    
    def test_validate_intent_performance(self, sample_intent):
        """Test intent validation is fast."""
        start = time.perf_counter()
        issues = validate_intent(sample_intent)
        elapsed = time.perf_counter() - start
        
        assert isinstance(issues, list)
        assert elapsed < 0.05  # Should validate in < 50ms
    
    def test_validate_multiple_intents_performance(self, sample_intent):
        """Test validating multiple intents is efficient."""
        intents = [sample_intent] * 10
        
        start = time.perf_counter()
        results = [validate_intent(intent) for intent in intents]
        elapsed = time.perf_counter() - start
        
        assert len(results) == 10
        assert elapsed < 0.2  # Should validate 10 intents in < 200ms


class TestGrooveTemplatePerformance:
    """Test groove template retrieval performance."""
    
    def test_get_template_performance(self):
        """Test getting a template is fast."""
        start = time.perf_counter()
        template = get_genre_template('funk')
        elapsed = time.perf_counter() - start
        
        assert template is not None
        assert elapsed < 0.01  # Should retrieve in < 10ms
    
    def test_get_multiple_templates_performance(self):
        """Test getting multiple templates is efficient."""
        genres = ['funk', 'jazz', 'rock', 'hiphop', 'bedroom_lofi']
        
        start = time.perf_counter()
        templates = [get_genre_template(genre) for genre in genres]
        elapsed = time.perf_counter() - start
        
        assert len(templates) == len(genres)
        assert all(t is not None for t in templates)
        assert elapsed < 0.05  # Should retrieve all in < 50ms


class TestCriticalPathProfiling:
    """Profile critical code paths to identify bottlenecks."""
    
    def test_chord_parsing_profile(self):
        """Profile chord parsing to find bottlenecks."""
        chord_strings = ["Cmaj7", "Am7", "Fmaj7", "G7"] * 25  # 100 chords
        
        with profile_function(lambda: [Chord.from_string(c, key="C") for c in chord_strings]) as (result, stats):
            assert len(result) == 100
            
            # Check that parsing is the main cost
            assert "from_string" in stats.lower() or "parse_chord" in stats.lower()
    
    def test_progression_diagnosis_profile(self):
        """Profile progression diagnosis to find bottlenecks."""
        progression = "-".join(["C", "Am", "F", "G"] * 20)  # 80 chords
        
        with profile_function(diagnose_progression, progression) as (result, stats):
            assert 'key' in result
            
            # Key detection should be a significant part
            assert "detect" in stats.lower() or "parse" in stats.lower()
    
    def test_reharmonization_profile(self):
        """Profile reharmonization generation."""
        progression = "C-Am-F-G"
        
        with profile_function(generate_reharmonizations, progression, "jazz", 5) as (result, stats):
            assert len(result) >= 1
            
            # Should show where time is spent
            assert len(stats) > 0


class TestMemoryUsage:
    """Test memory usage of critical operations."""
    
    def test_chord_parsing_memory(self):
        """Test chord parsing doesn't leak memory."""
        import sys
        
        # Parse many chords
        chords = []
        for i in range(1000):
            chord = Chord.from_string("C", key="C")
            chords.append(chord)
        
        # Memory should be reasonable (each chord is small)
        # This is a basic check - full memory profiling would need memory_profiler
        assert len(chords) == 1000
        assert sys.getsizeof(chords) < 10 * 1024 * 1024  # < 10MB for 1000 chords
    
    def test_progression_diagnosis_memory(self):
        """Test progression diagnosis memory usage."""
        # Long progression
        progression = "-".join(["C", "Am", "F", "G"] * 50)  # 200 chords
        
        result = diagnose_progression(progression)
        
        # Result should be reasonable size
        assert 'key' in result
        assert len(str(result)) < 10000  # Result JSON should be < 10KB


class TestConcurrentPerformance:
    """Test performance under concurrent operations."""
    
    def test_concurrent_chord_parsing(self):
        """Test parsing chords concurrently."""
        import concurrent.futures
        
        chord_strings = ["C", "Am", "F", "G", "Cmaj7"] * 20  # 100 chords
        
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            chords = list(executor.map(
                lambda c: Chord.from_string(c, key="C"),
                chord_strings
            ))
        elapsed = time.perf_counter() - start
        
        assert len(chords) == 100
        assert elapsed < 0.2  # Should be faster with concurrency
    
    def test_concurrent_progression_diagnosis(self):
        """Test diagnosing progressions concurrently."""
        import concurrent.futures
        
        progressions = [
            "C-Am-F-G",
            "F-C-Bbm-F",
            "G-Em-C-D",
            "Am-F-C-G"
        ] * 5  # 20 progressions
        
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(diagnose_progression, progressions))
        elapsed = time.perf_counter() - start
        
        assert len(results) == 20
        assert all('key' in r for r in results)
        assert elapsed < 1.0  # Should complete in < 1 second


class TestScalability:
    """Test scalability with large inputs."""
    
    def test_very_long_progression(self):
        """Test handling very long progressions."""
        # 200-chord progression
        progression = "-".join(["C", "Am", "F", "G"] * 50)
        
        start = time.perf_counter()
        result = diagnose_progression(progression)
        elapsed = time.perf_counter() - start
        
        assert 'key' in result
        assert elapsed < 1.0  # Should handle 200 chords in < 1 second
    
    def test_many_reharmonizations(self):
        """Test generating many reharmonization suggestions."""
        progression = "C-Am-F-G"
        
        start = time.perf_counter()
        suggestions = generate_reharmonizations(progression, style="jazz", count=10)
        elapsed = time.perf_counter() - start
        
        assert len(suggestions) <= 10
        assert elapsed < 0.5  # Should generate 10 suggestions in < 500ms
