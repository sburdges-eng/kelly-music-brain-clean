#!/usr/bin/env python3
"""
Test script for emotion detection integration.

Usage:
    python -m music_brain.audio.test_emotion_detection <audio_file>
    
Example:
    python -m music_brain.audio.test_emotion_detection path/to/song.wav
"""

import sys
from pathlib import Path

def test_emotion_detection(audio_path: str):
    """Test emotion detection on an audio file."""
    try:
        from music_brain.audio.emotion_detection import (
            EmotionDetector,
            detect_emotion_from_audio,
            get_emotional_state_from_audio,
        )
        from data.emotional_mapping import get_parameters_for_state
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Install SpeechBrain with:")
        print("   pip install speechbrain")
        print("   pip install torch torchaudio")
        return False
    
    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    print("=" * 60)
    print("Emotion Detection Test")
    print("=" * 60)
    print(f"\n📁 Audio file: {audio_file}")
    
    try:
        # Test 1: Quick function
        print("\n1️⃣ Testing quick detection function...")
        result = detect_emotion_from_audio(str(audio_file))
        print(f"   ✓ Emotion: {result['emotion']}")
        print(f"   ✓ Confidence: {result['confidence']:.2%}")
        print(f"   ✓ Valence: {result['valence']:.2f}")
        print(f"   ✓ Arousal: {result['arousal']:.2f}")
        print(f"   ✓ Primary emotion: {result['primary_emotion']}")
        
        # Test 2: Get EmotionalState
        print("\n2️⃣ Testing EmotionalState integration...")
        state = get_emotional_state_from_audio(str(audio_file))
        print(f"   ✓ Valence: {state.valence:.2f}")
        print(f"   ✓ Arousal: {state.arousal:.2f}")
        print(f"   ✓ Primary emotion: {state.primary_emotion}")
        
        # Test 3: Get musical parameters
        print("\n3️⃣ Testing musical parameter mapping...")
        params = get_parameters_for_state(state)
        print(f"   ✓ Suggested tempo: {params.tempo_suggested} BPM")
        print(f"   ✓ Tempo range: {params.tempo_min}-{params.tempo_max} BPM")
        print(f"   ✓ Dominant mode: {max(params.mode_weights, key=params.mode_weights.get)}")
        print(f"   ✓ Dissonance: {params.dissonance:.2f}")
        print(f"   ✓ Timing feel: {params.timing_feel.value}")
        print(f"   ✓ Density: {params.density.value}")
        print(f"   ✓ Dynamics: {params.dynamics}")
        
        # Test 4: Class-based API
        print("\n4️⃣ Testing class-based API...")
        detector = EmotionDetector()
        result2 = detector.detect_emotion(str(audio_file))
        print(f"   ✓ Emotion: {result2['emotion']}")
        print(f"   ✓ Confidence: {result2['confidence']:.2%}")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m music_brain.audio.test_emotion_detection <audio_file>")
        print("\nExample:")
        print("  python -m music_brain.audio.test_emotion_detection path/to/song.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    success = test_emotion_detection(audio_file)
    sys.exit(0 if success else 1)
