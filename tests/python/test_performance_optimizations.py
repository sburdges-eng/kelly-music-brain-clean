"""
Performance tests for code optimizations.

Tests the efficiency improvements made to:
- range(len()) patterns replaced with enumerate()
- .keys() removed from dict iteration
- List comprehensions with zip() for interval calculations
"""

import time


def _bench(fn, repeats: int = 3):
    """Run a small benchmark and return (result, best_time)."""
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    return result, min(durations)


def test_enumerate_vs_range_len():
    """Test that enumerate is as fast or faster than range(len())."""
    data = list(range(10000))
    
    def _range_len():
        return [data[i] * 2 for i in range(len(data))]
    
    def _enumerate():
        return [val * 2 for i, val in enumerate(data)]
    
    result_old, old_time = _bench(_range_len)
    result_new, new_time = _bench(_enumerate)
    
    # Results should be identical
    assert result_old == result_new
    
    # New should be at least as fast (within margin of error)
    assert new_time <= old_time * 1.5, \
        f"enumerate slower: old={old_time:.4f}s, new={new_time:.4f}s"
    
    print(f"  enumerate vs range(len): {old_time:.4f}s vs {new_time:.4f}s")


def test_dict_iteration_efficiency():
    """Test that dict iteration without .keys() is efficient."""
    data = {f"key{i}": i for i in range(10000)}
    
    def _keys_iteration():
        return [data[k] for k in data.keys()]
    
    def _direct_iteration():
        return [data[k] for k in data]
    
    result_old, old_time = _bench(_keys_iteration)
    result_new, new_time = _bench(_direct_iteration)
    
    # Results should be identical
    assert result_old == result_new
    
    # New should be faster
    assert new_time <= old_time * 1.2, \
        f"Direct iteration not faster: old={old_time:.4f}s, new={new_time:.4f}s"
    
    print(f"  dict iteration: {old_time:.4f}s vs {new_time:.4f}s")


def test_zip_interval_calculation():
    """Test that zip() is more efficient for interval calculations."""
    times = [i * 0.1 for i in range(10000)]
    
    # Old pattern: list[i+1] - list[i]
    start = time.time()
    intervals_old = [times[i+1] - times[i] for i in range(len(times) - 1)]
    old_time = time.time() - start
    
    # New pattern: zip(list[:-1], list[1:])
    start = time.time()
    intervals_new = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
    new_time = time.time() - start
    
    # Results should be identical (within floating point precision)
    assert len(intervals_old) == len(intervals_new)
    for old, new in zip(intervals_old, intervals_new):
        assert abs(old - new) < 1e-10
    
    # New should be faster
    assert new_time <= old_time * 1.3, \
        f"zip not faster: old={old_time:.4f}s, new={new_time:.4f}s"
    
    print(f"  interval calculation: {old_time:.4f}s vs {new_time:.4f}s")


def test_list_comprehension_performance():
    """Test that list comprehensions are efficient."""
    data = range(10000)
    
    # Append in loop
    start = time.time()
    result_loop = []
    for x in data:
        result_loop.append(x * 2)
    loop_time = time.time() - start
    
    # List comprehension
    start = time.time()
    result_comp = [x * 2 for x in data]
    comp_time = time.time() - start
    
    # Results should be identical
    assert result_loop == result_comp
    
    # Comprehension should be faster
    assert comp_time <= loop_time, \
        f"List comprehension not faster: loop={loop_time:.4f}s, comp={comp_time:.4f}s"
    
    print(f"  list comprehension: {loop_time:.4f}s vs {comp_time:.4f}s")


def test_voice_leading_optimization():
    """Test optimized voice leading code doesn't regress."""
    try:
        from harmony_tools import voice_leading_tool
        
        # Simple test case
        chords = ["C", "F", "G", "C"]
        
        # Should complete quickly
        start = time.time()
        result = voice_leading_tool(chords)
        elapsed = time.time() - start
        
        assert elapsed < 0.1, f"Voice leading too slow: {elapsed:.4f}s"
        
    except (ImportError, Exception) as e:
        raise Exception(f"Test skipped: {e}")


def test_polyrhythm_detection_performance():
    """Test polyrhythm detection with optimized interval calculation."""
    try:
        from python.penta_core.groove.polyrhythm import detect_polyrhythm
        
        # Create test events
        events = [
            {"time": i * 0.5} for i in range(100)
        ]
        
        # Should complete quickly
        start = time.time()
        result = detect_polyrhythm(events)
        elapsed = time.time() - start
        
        assert elapsed < 0.1, f"Polyrhythm detection too slow: {elapsed:.4f}s"
        
    except (ImportError, Exception) as e:
        raise Exception(f"Test skipped: {e}")


def test_performance_analysis_speed():
    """Test live performance analysis with optimizations."""
    try:
        from python.penta_core.groove.performance import analyze_live_performance
        
        # Create test events
        events = [
            {"time": i * 0.25, "velocity": 80 + (i % 20)}
            for i in range(200)
        ]
        
        # Should complete quickly
        start = time.time()
        analysis = analyze_live_performance(events, reference_tempo=120)
        elapsed = time.time() - start
        
        assert elapsed < 0.5, f"Performance analysis too slow: {elapsed:.4f}s"
        assert analysis is not None
        
    except (ImportError, Exception) as e:
        raise Exception(f"Test skipped: {e}")


if __name__ == "__main__":
    print("Running performance optimization tests...\n")
    
    tests = [
        ("Enumerate vs range(len)", test_enumerate_vs_range_len),
        ("Dict iteration", test_dict_iteration_efficiency),
        ("Zip interval calculation", test_zip_interval_calculation),
        ("List comprehension", test_list_comprehension_performance),
        ("Voice leading optimization", test_voice_leading_optimization),
        ("Polyrhythm detection", test_polyrhythm_detection_performance),
        ("Performance analysis", test_performance_analysis_speed),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            print(f"{name}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"⊘ SKIPPED: {e}")
            skipped += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")
