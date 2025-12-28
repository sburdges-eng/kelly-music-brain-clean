#!/usr/bin/env python3
"""
Emotional Labeling Example for ML Training

Demonstrates how to use EmotionalState for creating ML training labels.
"""

import json
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.emotional_mapping import (
    EmotionalState,
    get_parameters_for_state,
    EMOTIONAL_STATE_PRESETS
)

def main():
    print("Emotional Mapping ML Labeling Examples")
    print("="*60)
    
    # Example 1: Basic usage
    emotion = EmotionalState(valence=0.8, arousal=0.75, primary_emotion="joy")
    print(f"\nExample 1: Created EmotionalState: {emotion.primary_emotion}")
    print(f"  Valence: {emotion.valence}, Arousal: {emotion.arousal}")
    
    # Example 2: Using presets
    grief = EMOTIONAL_STATE_PRESETS["profound_grief"]
    print(f"\nExample 2: Using preset: {grief.primary_emotion}")
    params = get_parameters_for_state(grief)
    print(f"  Suggested tempo: {params.tempo_suggested} BPM")
    
    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()