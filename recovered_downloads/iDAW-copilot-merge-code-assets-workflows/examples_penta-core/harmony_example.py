"""
Example: Using Penta Core for real-time MIDI harmony analysis
"""

import time
from penta_core import HarmonyEngine

def main():
    # Initialize harmony engine
    harmony = HarmonyEngine(sample_rate=48000.0, confidence_threshold=0.6)
    
    # Simulate MIDI input - C major triad
    midi_notes = [
        (60, 80),  # C4
        (64, 75),  # E4
        (67, 70),  # G4
    ]
    
    print("Processing MIDI notes...")
    harmony.process_midi_notes(midi_notes)
    
    # Get detected chord
    chord = harmony.get_current_chord()
    print(f"\nDetected Chord: {chord['name']}")
    print(f"Root: {chord['root']}, Quality: {chord['quality']}")
    print(f"Confidence: {chord['confidence']:.2f}")
    print(f"Pitch Classes: {chord['pitch_classes']}")
    
    # Get detected scale
    scale = harmony.get_current_scale()
    print(f"\nDetected Scale: {scale['name']}")
    print(f"Tonic: {scale['tonic']}, Mode: {scale['mode']}")
    print(f"Confidence: {scale['confidence']:.2f}")
    print(f"Degrees: {scale['degrees']}")
    
    # Voice leading example
    print("\n--- Voice Leading Example ---")
    current_voices = [60, 64, 67]  # C major
    target_root = 5  # F major
    
    suggested_voices = harmony.suggest_voice_leading(target_root, current_voices)
    print(f"Current voices: {current_voices}")
    print(f"Suggested voices for F major: {suggested_voices}")


if __name__ == '__main__':
    main()
