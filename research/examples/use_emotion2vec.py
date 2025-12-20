#!/usr/bin/env python3
"""
Simple usage example for emotion2vec integration.

Usage:
    python use_emotion2vec.py path/to/audio.wav
    python use_emotion2vec.py path/to/audio.wav --model plus
    python use_emotion2vec.py path/to/audio.wav --embeddings
"""

import argparse
import sys
from pathlib import Path

from emotion2vec_integration import (
    Emotion2VecExtractor,
    create_emotional_state_from_emotion2vec,
    EMOTION2VEC_AVAILABLE,
)


def main():
    parser = argparse.ArgumentParser(
        description="Use emotion2vec for emotion detection"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file (must be 16kHz mono WAV)",
    )
    parser.add_argument(
        "--model",
        choices=["base", "plus"],
        default="plus",
        help="Model type: 'base' for embeddings, 'plus' for emotion recognition",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Extract embeddings instead of recognizing emotion",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert audio to 16kHz mono WAV if needed",
    )
    parser.add_argument(
        "--integrate",
        action="store_true",
        help="Create EmotionalState and get musical parameters",
    )

    args = parser.parse_args()

    if not EMOTION2VEC_AVAILABLE:
        print("ERROR: emotion2vec not installed.")
        print("Install with: pip install -U funasr modelscope")
        sys.exit(1)

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    # Initialize extractor
    try:
        extractor = Emotion2VecExtractor(model_type=args.model)
    except Exception as e:
        print(f"ERROR: Failed to initialize extractor: {e}")
        sys.exit(1)

    # Convert audio if needed
    if args.convert:
        if audio_path.suffix.lower() != ".wav":
            print(f"Converting {audio_path} to 16kHz mono WAV...")
            try:
                audio_path = extractor.convert_to_16khz_mono(audio_path)
                print(f"Converted to: {audio_path}")
            except Exception as e:
                print(f"ERROR: Failed to convert audio: {e}")
                sys.exit(1)

    # Extract embeddings or recognize emotion
    if args.embeddings:
        if args.model != "base":
            print("WARNING: Using 'base' model for embeddings is recommended")
        print(f"Extracting embeddings from {audio_path}...")
        try:
            result = extractor.extract_embeddings(audio_path)
            print(f"\n✓ Embeddings extracted successfully!")
            print(f"  Model: {result['model']}")
            if result["embeddings"] is not None:
                print(f"  Embedding shape: {result['embeddings'].shape}")
            else:
                print("  WARNING: Embeddings not found in output")
        except Exception as e:
            print(f"ERROR: Failed to extract embeddings: {e}")
            sys.exit(1)
    else:
        if args.model != "plus":
            print("WARNING: Using 'plus' model for emotion recognition is recommended")
        print(f"Recognizing emotion from {audio_path}...")
        try:
            result = extractor.recognize_emotion(audio_path)
            print(f"\n✓ Emotion recognized successfully!")
            print(f"  Emotion: {result['emotion']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Valence: {result['valence']:.2f}")
            print(f"  Arousal: {result['arousal']:.2f}")
            print(f"  Primary emotion: {result['primary_emotion']}")

            # Integration with existing system
            if args.integrate:
                print("\n--- Integration with miDiKompanion ---")
                try:
                    state = create_emotional_state_from_emotion2vec(
                        extractor, audio_path
                    )
                    print(f"✓ EmotionalState created:")
                    print(f"  Valence: {state.valence:.2f}")
                    print(f"  Arousal: {state.arousal:.2f}")
                    print(f"  Primary emotion: {state.primary_emotion}")

                    # Get musical parameters
                    try:
                        from data.emotional_mapping import get_parameters_for_state

                        params = get_parameters_for_state(state)
                        print(f"\n✓ Musical parameters:")
                        print(f"  Tempo: {params.tempo_suggested} BPM")
                        print(f"  Mode: {params.mode}")
                        print(f"  Key: {params.key}")
                    except ImportError:
                        print("  (Musical parameters not available)")
                except Exception as e:
                    print(f"ERROR: Failed to create EmotionalState: {e}")
        except Exception as e:
            print(f"ERROR: Failed to recognize emotion: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
