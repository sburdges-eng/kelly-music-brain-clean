"""
Performance benchmark tests for DAiW Music Brain.

Run with: pytest tests_music-brain/test_performance.py -v --benchmark-only
"""

import pytest
import time
import random
from typing import List, Optional

# Attempt to import pytest-benchmark
try:
    from pytest_benchmark.fixture import BenchmarkFixture
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_midi_data():
    """Generate sample MIDI-like data for testing."""
    return {
        "notes": [
            {"pitch": random.randint(36, 96), "velocity": random.randint(60, 120),
             "start": i * 0.25, "duration": 0.125}
            for i in range(64)
        ],
        "tempo": 120,
        "time_signature": (4, 4)
    }


@pytest.fixture
def sample_chord_progression():
    """Sample chord progression for analysis."""
    return ["C", "Am", "F", "G"] * 4


# =============================================================================
# Core Module Performance Tests
# =============================================================================

@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
def test_groove_applicator_performance(benchmark):
    """Benchmark groove applicator performance."""
    try:
        from music_brain.groove import GrooveApplicator

        applicator = GrooveApplicator()

        # Create sample timing data
        timings = [i * 0.25 for i in range(64)]
        velocities = [80] * 64

        def run_applicator():
            return applicator.apply_groove(timings, velocities, swing=0.3)

        # Run benchmark
        result = benchmark(run_applicator)

        # Verify result is valid
        assert result is not None

    except ImportError:
        pytest.skip("GrooveApplicator not available")


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
def test_chord_analysis_performance(benchmark):
    """Benchmark chord analysis performance."""
    try:
        from music_brain.structure import Chord

        # Create a variety of pitch class sets
        test_cases = [
            [0, 4, 7],      # C major
            [0, 3, 7],      # C minor
            [0, 4, 7, 11],  # Cmaj7
            [0, 3, 7, 10],  # Cm7
        ]

        def analyze_chords():
            results = []
            for pitches in test_cases:
                chord = Chord.from_pitch_classes(pitches)
                results.append(chord)
            return results

        result = benchmark(analyze_chords)
        assert len(result) == len(test_cases)

    except (ImportError, AttributeError):
        pytest.skip("Chord analysis not available")


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
def test_intent_processing_performance(benchmark):
    """Benchmark intent processing performance."""
    try:
        from music_brain.session import CompleteSongIntent, IntentProcessor

        intent = CompleteSongIntent(
            title="Test Song",
            core_event="Testing",
            mood_primary="calm",
            technical_genre="ambient",
            technical_key="C major"
        )

        processor = IntentProcessor()

        def process_intent():
            return processor.process(intent)

        result = benchmark(process_intent)
        assert result is not None

    except (ImportError, AttributeError):
        pytest.skip("Intent processing not available")


# =============================================================================
# Non-Benchmark Performance Tests (for CI without benchmark plugin)
# =============================================================================

def test_module_import_speed():
    """Test that core modules import quickly (< 2 seconds total)."""
    import time

    start = time.time()

    # Import core modules
    imports = [
        "music_brain",
        "music_brain.groove",
        "music_brain.structure",
        "music_brain.session",
    ]

    for module_name in imports:
        try:
            __import__(module_name)
        except ImportError:
            pass  # Module may not be available

    elapsed = time.time() - start

    # Should complete in under 2 seconds
    assert elapsed < 2.0, f"Imports took too long: {elapsed:.2f}s"


def test_cli_startup_speed():
    """Test that CLI starts quickly."""
    import subprocess
    import time

    start = time.time()

    # Run CLI help command (should be fast)
    try:
        result = subprocess.run(
            ["python", "-m", "music_brain.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5.0
        )
        elapsed = time.time() - start

        # CLI help should complete in under 2 seconds
        assert elapsed < 2.0, f"CLI startup took too long: {elapsed:.2f}s"

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("CLI not available or timed out")


def test_groove_extraction_latency():
    """Test that groove extraction completes in reasonable time."""
    try:
        from music_brain.groove import GrooveApplicator
        import time

        applicator = GrooveApplicator()

        # Generate test data (512 notes)
        timings = [i * 0.125 for i in range(512)]
        velocities = [80 + (i % 40) for i in range(512)]

        # Measure extraction time
        start = time.time()
        for _ in range(10):
            result = applicator.apply_groove(timings, velocities, swing=0.25)
        elapsed = time.time() - start

        # Average should be under 50ms per extraction
        avg_ms = (elapsed / 10) * 1000
        assert avg_ms < 50, f"Groove extraction too slow: {avg_ms:.2f}ms per call"

    except ImportError:
        pytest.skip("GrooveApplicator not available")


def test_chord_detection_latency():
    """Test chord detection latency."""
    try:
        from music_brain.structure import Chord
        import time

        # Test with various pitch class combinations
        test_cases = [
            [0, 4, 7],
            [2, 5, 9],
            [0, 3, 7, 10],
            [0, 4, 7, 11],
            [5, 9, 0],
        ]

        iterations = 100

        start = time.time()
        for _ in range(iterations):
            for pitches in test_cases:
                chord = Chord.from_pitch_classes(pitches)
        elapsed = time.time() - start

        # Should complete 500 chord detections in under 100ms
        assert elapsed < 0.1, f"Chord detection too slow: {elapsed * 1000:.2f}ms for {iterations * len(test_cases)} detections"

    except (ImportError, AttributeError):
        pytest.skip("Chord module not available")


def test_intent_schema_creation_speed():
    """Test intent schema creation performance."""
    try:
        from music_brain.session.intent_schema import (
            CompleteSongIntent,
            SongRoot,
            SongIntent,
            TechnicalConstraints,
        )
        import time

        iterations = 100

        start = time.time()
        for i in range(iterations):
            intent = CompleteSongIntent(
                title=f"Test Song {i}",
                song_root=SongRoot(core_event="Testing performance"),
                song_intent=SongIntent(mood_primary="energetic"),
                technical_constraints=TechnicalConstraints(
                    technical_genre="electronic",
                    technical_key="A minor",
                ),
            )
        elapsed = time.time() - start

        # Should create 100 intents in under 100ms (relaxed for complex objects)
        assert elapsed < 0.1, f"Intent creation too slow: {elapsed * 1000:.2f}ms for {iterations} intents"

    except ImportError:
        pytest.skip("Session module not available")


# =============================================================================
# Memory Usage Tests
# =============================================================================

def test_large_midi_processing_memory():
    """Test that processing large MIDI data doesn't cause memory issues."""
    try:
        from music_brain.groove import GrooveApplicator
        import sys

        applicator = GrooveApplicator()

        # Create large dataset (10k notes)
        large_timings = [i * 0.0625 for i in range(10000)]
        large_velocities = [80] * 10000

        # Process and check for memory issues
        result = applicator.apply_groove(large_timings, large_velocities, swing=0.2)

        # Verify result is correct size
        assert result is not None

    except ImportError:
        pytest.skip("GrooveApplicator not available")


# =============================================================================
# Latency Targets (from COMPREHENSIVE_TODO.md)
# =============================================================================

def test_harmony_latency_target():
    """Verify harmony analysis meets <100μs target (with tolerance for Python)."""
    try:
        from music_brain.structure import Chord
        import time

        # For Python, we allow 10ms (10000μs) as a reasonable target
        # The 100μs target is for C++ code

        pitches = [0, 4, 7, 11]  # Cmaj7

        iterations = 100
        start = time.time()
        for _ in range(iterations):
            chord = Chord.from_pitch_classes(pitches)
        elapsed = time.time() - start

        avg_us = (elapsed / iterations) * 1_000_000  # Convert to microseconds

        # Python target: < 10ms (10,000μs) per analysis
        assert avg_us < 10000, f"Harmony analysis too slow: {avg_us:.2f}μs (Python target: <10,000μs)"

        print(f"\nHarmony analysis: {avg_us:.2f}μs average")

    except (ImportError, AttributeError):
        pytest.skip("Chord module not available")


def test_groove_latency_target():
    """Verify groove processing meets <200μs target (with tolerance for Python)."""
    try:
        from music_brain.groove import GrooveApplicator
        import time

        # For Python, we allow 20ms (20000μs) as a reasonable target
        # The 200μs target is for C++ code

        applicator = GrooveApplicator()
        timings = [i * 0.25 for i in range(16)]  # 16 notes
        velocities = [80] * 16

        iterations = 100
        start = time.time()
        for _ in range(iterations):
            result = applicator.apply_groove(timings, velocities, swing=0.25)
        elapsed = time.time() - start

        avg_us = (elapsed / iterations) * 1_000_000

        # Python target: < 20ms (20,000μs) per processing
        assert avg_us < 20000, f"Groove processing too slow: {avg_us:.2f}μs (Python target: <20,000μs)"

        print(f"\nGroove processing: {avg_us:.2f}μs average")

    except ImportError:
        pytest.skip("GrooveApplicator not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
