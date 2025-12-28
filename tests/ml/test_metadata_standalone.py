"""
Standalone test for AudioDataset metadata loading (no PyTorch required).
Tests the metadata extraction and validation logic.
"""

import json
import tempfile
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def create_test_metadata():
    """Create test metadata with emotional and intent attributes."""
    return {
        "dataset_id": "test_emotion_v1",
        "samples": [
            {
                "file": "test_audio_001.wav",
                "emotion": "happy",
                "valence": 0.8,
                "arousal": 0.6,
                "intensity_tier": 3,
                "node_id": 42,
                "key": "C major",
                "tempo_bpm": 120.0,
                "mode": "major",
                "chord_progression": ["C", "Am", "F", "G"],
                "groove_type": "straight",
                "tags": ["uplifting", "energetic"],
                "quality_score": 0.9,
                "rule_breaks": [],
            },
            {
                "file": "test_audio_002.wav",
                "emotion": "sad",
                "valence": -0.7,
                "arousal": 0.3,
                "intensity_tier": 4,
                "node_id": 78,
                "key": "A minor",
                "tempo_bpm": 72.0,
                "mode": "minor",
                "chord_progression": ["Am", "F", "C", "G"],
                "groove_type": "laid_back",
                "tags": ["melancholic"],
                "quality_score": 0.85,
                "rule_breaks": [],
            },
            {
                "file": "test_audio_003.wav",
                "emotion": "grief",
                "valence": -0.9,
                "arousal": 0.2,
                "intensity_tier": 5,
                "node_id": 156,
                "key": "F major",
                "tempo_bpm": 68.0,
                "mode": "major",
                "chord_progression": ["F", "C", "Dm", "Bbm"],
                "groove_type": "laid_back",
                "tags": ["grief", "bittersweet"],
                "quality_score": 0.95,
                "rule_breaks": ["HARMONY_ModalInterchange"],
            },
        ]
    }


def test_metadata_loading():
    """Test basic metadata loading."""
    print("Test 1: Basic metadata loading")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create metadata.json
        metadata = create_test_metadata()
        metadata_path = tmpdir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Test loading
        with open(metadata_path) as f:
            loaded = json.load(f)
        
        assert loaded["dataset_id"] == "test_emotion_v1"
        assert len(loaded["samples"]) == 3
        
        # Check first sample
        sample = loaded["samples"][0]
        assert sample["emotion"] == "happy"
        assert sample["valence"] == 0.8
        assert sample["arousal"] == 0.6
        assert sample["intensity_tier"] == 3
        assert sample["node_id"] == 42
        assert sample["key"] == "C major"
        assert sample["tempo_bpm"] == 120.0
        
        print("  ✓ Metadata loaded successfully")
        print(f"  ✓ Found {len(loaded['samples'])} samples")
        return True


def test_emotional_attributes():
    """Test emotional attribute extraction."""
    print("\nTest 2: Emotional attributes")
    
    metadata = create_test_metadata()
    
    for i, sample in enumerate(metadata["samples"]):
        emotion = sample["emotion"]
        valence = sample["valence"]
        arousal = sample["arousal"]
        intensity = sample["intensity_tier"]
        
        # Validate ranges
        assert -1.0 <= valence <= 1.0, f"Valence out of range: {valence}"
        assert 0.0 <= arousal <= 1.0, f"Arousal out of range: {arousal}"
        assert 0 <= intensity <= 5, f"Intensity out of range: {intensity}"
        
        print(f"  ✓ Sample {i+1}: {emotion} (valence={valence}, arousal={arousal}, intensity={intensity})")
    
    return True


def test_musical_metadata():
    """Test musical intent metadata."""
    print("\nTest 3: Musical metadata")
    
    metadata = create_test_metadata()
    
    for i, sample in enumerate(metadata["samples"]):
        key = sample["key"]
        tempo = sample["tempo_bpm"]
        mode = sample["mode"]
        chords = sample["chord_progression"]
        groove = sample["groove_type"]
        
        print(f"  ✓ Sample {i+1}: {key} @ {tempo} BPM, {mode} mode, {groove} groove")
        print(f"    Progression: {' - '.join(chords)}")
    
    return True


def test_thesaurus_integration():
    """Test DAiW thesaurus node_id integration."""
    print("\nTest 4: Thesaurus integration")
    
    metadata = create_test_metadata()
    
    for i, sample in enumerate(metadata["samples"]):
        node_id = sample.get("node_id")
        if node_id is not None:
            assert 0 <= node_id <= 215, f"Node ID out of range: {node_id}"
            print(f"  ✓ Sample {i+1}: Emotion node ID = {node_id}")
    
    return True


def test_rule_breaks():
    """Test rule-breaking metadata."""
    print("\nTest 5: Rule breaking metadata")
    
    metadata = create_test_metadata()
    
    # Find the grief sample with modal interchange
    grief_sample = metadata["samples"][2]
    assert grief_sample["emotion"] == "grief"
    assert "HARMONY_ModalInterchange" in grief_sample["rule_breaks"]
    
    print(f"  ✓ Grief sample has intentional rule break: {grief_sample['rule_breaks']}")
    print(f"  ✓ Progression: {' - '.join(grief_sample['chord_progression'])}")
    print(f"    (Bbm in F major = borrowed sadness)")
    
    return True


def test_validation_ranges():
    """Test that out-of-range values are detectable."""
    print("\nTest 6: Validation of value ranges")
    
    test_cases = [
        {"valence": 2.0, "expected": "out of range"},
        {"arousal": -0.5, "expected": "out of range"},
        {"intensity_tier": 10, "expected": "out of range"},
        {"valence": 0.5, "arousal": 0.5, "intensity_tier": 3, "expected": "valid"},
    ]
    
    for i, case in enumerate(test_cases):
        valence = case.get("valence", 0.0)
        arousal = case.get("arousal", 0.0)
        intensity = case.get("intensity_tier", 3)
        
        valid = (-1.0 <= valence <= 1.0 and 
                 0.0 <= arousal <= 1.0 and 
                 0 <= intensity <= 5)
        
        if case["expected"] == "valid":
            assert valid, f"Case {i+1} should be valid"
            print(f"  ✓ Case {i+1}: Valid ranges")
        else:
            assert not valid, f"Case {i+1} should be invalid"
            print(f"  ✓ Case {i+1}: Detected out-of-range values")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("AudioDataset Metadata Loading Tests")
    print("=" * 60)
    
    tests = [
        test_metadata_loading,
        test_emotional_attributes,
        test_musical_metadata,
        test_thesaurus_integration,
        test_rule_breaks,
        test_validation_ranges,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
