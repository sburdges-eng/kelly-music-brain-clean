#!/usr/bin/env python3
"""
Demo script showing AudioDataset metadata loading capabilities.
"""

import json
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_metadata_loading():
    """Demonstrate metadata loading from example file."""
    print_header("Demo: AudioDataset Metadata Loading")
    
    # Load the example metadata
    example_path = ROOT / "examples" / "metadata_example.json"
    
    if not example_path.exists():
        print(f"Error: Example metadata not found at {example_path}")
        return
    
    with open(example_path) as f:
        metadata = json.load(f)
    
    print(f"\nDataset: {metadata['dataset_id']}")
    print(f"Version: {metadata['version']}")
    print(f"Samples: {len(metadata['samples'])}")
    
    return metadata


def demo_emotional_attributes(metadata):
    """Demonstrate emotional attribute extraction."""
    print_header("Emotional Attributes")
    
    for i, sample in enumerate(metadata['samples']):
        emotion = sample.get('emotion', 'unknown')
        valence = sample.get('valence', 0.0)
        arousal = sample.get('arousal', 0.0)
        intensity = sample.get('intensity_tier', 0)
        node_id = sample.get('node_id', 'N/A')
        
        print(f"\nSample {i+1}: {emotion.upper()}")
        print(f"  Valence:    {valence:+.1f} (negative ← 0 → positive)")
        print(f"  Arousal:    {arousal:+.1f} (calm ← 0 → energetic)")
        print(f"  Intensity:  {intensity}/5")
        print(f"  Node ID:    {node_id} (DAiW thesaurus)")
        
        # Interpret the emotional state
        if valence > 0.5 and arousal > 0.5:
            state = "High-energy positive (excited, joyful)"
        elif valence > 0.5 and arousal < 0.5:
            state = "Low-energy positive (peaceful, content)"
        elif valence < -0.5 and arousal > 0.5:
            state = "High-energy negative (angry, anxious)"
        elif valence < -0.5 and arousal < 0.5:
            state = "Low-energy negative (sad, melancholic)"
        else:
            state = "Neutral or mixed"
        
        print(f"  State:      {state}")


def demo_musical_metadata(metadata):
    """Demonstrate musical metadata extraction."""
    print_header("Musical Intent Metadata")
    
    for i, sample in enumerate(metadata['samples']):
        emotion = sample.get('emotion', 'unknown')
        key = sample.get('key', 'N/A')
        tempo = sample.get('tempo_bpm', 0.0)
        mode = sample.get('mode', 'N/A')
        chords = sample.get('chord_progression', [])
        groove = sample.get('groove_type', 'N/A')
        
        print(f"\nSample {i+1}: {emotion.upper()}")
        print(f"  Key:        {key}")
        print(f"  Tempo:      {tempo} BPM")
        print(f"  Mode:       {mode}")
        print(f"  Groove:     {groove}")
        
        if chords:
            chord_str = " → ".join(chords)
            print(f"  Progression: {chord_str}")


def demo_rule_breaking(metadata):
    """Demonstrate rule-breaking metadata."""
    print_header("Intentional Rule Breaking")
    
    for i, sample in enumerate(metadata['samples']):
        rule_breaks = sample.get('rule_breaks', [])
        
        if rule_breaks:
            emotion = sample.get('emotion', 'unknown')
            chords = sample.get('chord_progression', [])
            notes = sample.get('notes', '')
            
            print(f"\nSample {i+1}: {emotion.upper()}")
            print(f"  Rule breaks: {', '.join(rule_breaks)}")
            print(f"  Progression: {' - '.join(chords)}")
            print(f"  Why: {notes}")


def demo_thesaurus_mapping():
    """Demonstrate thesaurus node mapping."""
    print_header("DAiW Thesaurus Node Mapping")
    
    # Example node ID calculations
    # node_id = base_idx * 36 + sub_idx * 6 + subsub_idx
    
    examples = [
        {"node_id": 0, "hierarchy": "HAPPY → JOY → elated"},
        {"node_id": 42, "hierarchy": "HAPPY → CONTENTMENT → satisfied"},
        {"node_id": 78, "hierarchy": "SAD → GRIEF → sorrowful"},
        {"node_id": 156, "hierarchy": "FEAR → ANXIETY → worried"},
    ]
    
    print("\nExample node ID → emotion hierarchy mappings:")
    print("(6 base emotions × 6 sub-emotions × 6 sub-sub-emotions = 216 nodes)")
    
    for ex in examples:
        node_id = ex['node_id']
        base_idx = node_id // 36
        sub_idx = (node_id % 36) // 6
        subsub_idx = node_id % 6
        
        print(f"\n  Node {node_id:3d}: {ex['hierarchy']}")
        print(f"           Indices: [{base_idx}, {sub_idx}, {subsub_idx}]")


def demo_validation():
    """Demonstrate metadata validation."""
    print_header("Metadata Validation")
    
    print("\nValidation rules:")
    print("  • valence:        -1.0 to +1.0 (clamped if outside)")
    print("  • arousal:         0.0 to  1.0 (clamped if outside)")
    print("  • intensity_tier:  0 to 5 (clamped if outside)")
    print("  • file:            Required (raises error if missing)")
    print("  • All other fields: Optional with sensible defaults")
    
    print("\nExample validation:")
    
    test_cases = [
        {"valence": 2.0, "result": "Clamped to 1.0"},
        {"arousal": -0.5, "result": "Clamped to 0.0"},
        {"intensity_tier": 10, "result": "Clamped to 5"},
    ]
    
    for case in test_cases:
        field = list(case.keys())[0]
        value = case[field]
        result = case["result"]
        print(f"  • {field} = {value} → {result}")


def main():
    """Run all demos."""
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "   AudioDataset Metadata Loading Demo".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Load metadata
    metadata = demo_metadata_loading()
    
    if metadata:
        # Run demos
        demo_emotional_attributes(metadata)
        demo_musical_metadata(metadata)
        demo_rule_breaking(metadata)
        demo_thesaurus_mapping()
        demo_validation()
        
        print("\n" + "=" * 60)
        print("Demo complete! See docs/DATASET_METADATA_SCHEMA.md for details.")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
