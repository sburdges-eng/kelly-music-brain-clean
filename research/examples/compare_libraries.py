"""
Library Comparison Script

Compare different emotion detection libraries on the same audio file
to help choose the best one for your use case.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# Try importing all libraries
try:
    from research.examples.speechbrain_integration import (
        SpeechBrainEmotionDetector,
    )
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

try:
    from research.examples.opensmile_integration import (
        OpenSMILEFeatureExtractor,
    )
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False

try:
    from research.examples.emotion2vec_integration import (
        Emotion2VecExtractor,
    )
    EMOTION2VEC_AVAILABLE = True
except ImportError:
    EMOTION2VEC_AVAILABLE = False


def compare_libraries(audio_path: str | Path) -> dict:
    """
    Compare all available libraries on the same audio file.

    Args:
        audio_path: Path to audio file to test (str or Path)

    Returns:
        Dictionary with results from each library
    """
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        msg = f"Audio file not found: {audio_path_obj}"
        raise FileNotFoundError(msg)

    results: dict[str, Any] = {
        "audio_file": str(audio_path_obj),
        "libraries": {},
    }

    # Test SpeechBrain
    if SPEECHBRAIN_AVAILABLE:
        try:
            print("Testing SpeechBrain...")
            start_time = time.time()
            detector = SpeechBrainEmotionDetector()
            result = detector.detect_emotion(str(audio_path_obj))
            elapsed_time = time.time() - start_time

            results["libraries"]["SpeechBrain"] = {
                "status": "success",
                "emotion": result["emotion"],
                "confidence": result["confidence"],
                "valence": result["valence"],
                "arousal": result["arousal"],
                "processing_time": elapsed_time,
                "primary_emotion": result["primary_emotion"],
            }
            emotion_msg = (
                f"  ✓ Emotion: {result['emotion']}, "
                f"Confidence: {result['confidence']:.2f}"
            )
            print(emotion_msg)
            print(f"  ✓ Processing time: {elapsed_time:.2f}s")
        except Exception as e:
            results["libraries"]["SpeechBrain"] = {
                "status": "error",
                "error": str(e),
            }
            print(f"  ✗ Error: {e}")
    else:
        results["libraries"]["SpeechBrain"] = {
            "status": "not_available",
            "message": "Install with: pip install speechbrain",
        }
        print("  - SpeechBrain not available")

    # Test OpenSMILE
    if OPENSMILE_AVAILABLE:
        try:
            print("\nTesting OpenSMILE...")
            start_time = time.time()
            opensmile_extractor = OpenSMILEFeatureExtractor(
                feature_set="eGeMAPSv02"
            )
            features = opensmile_extractor.extract_features(
                str(audio_path_obj)
            )
            elapsed_time = time.time() - start_time

            results["libraries"]["OpenSMILE"] = {
                "status": "success",
                "feature_count": len(features),
                "feature_names": opensmile_extractor.get_feature_names()[:10],
                "processing_time": elapsed_time,
                "note": (
                    "Feature extraction only - "
                    "requires classifier for emotion"
                ),
            }
            print(f"  ✓ Extracted {len(features)} features")
            print(f"  ✓ Processing time: {elapsed_time:.2f}s")
            print(
                "  ✓ Note: OpenSMILE extracts features, "
                "not emotions directly"
            )
        except Exception as e:
            results["libraries"]["OpenSMILE"] = {
                "status": "error",
                "error": str(e),
            }
            print(f"  ✗ Error: {e}")
    else:
        results["libraries"]["OpenSMILE"] = {
            "status": "not_available",
            "message": "Install with: pip install opensmile",
        }
        print("  - OpenSMILE not available")

    # Test emotion2vec
    if EMOTION2VEC_AVAILABLE:
        try:
            print("\nTesting emotion2vec...")
            start_time = time.time()
            emotion2vec_extractor = Emotion2VecExtractor(model_type="plus")
            result = emotion2vec_extractor.recognize_emotion(
                str(audio_path_obj)
            )
            elapsed_time = time.time() - start_time

            results["libraries"]["emotion2vec"] = {
                "status": "success",
                "emotion": result["emotion"],
                "confidence": result["confidence"],
                "valence": result["valence"],
                "arousal": result["arousal"],
                "processing_time": elapsed_time,
                "primary_emotion": result["primary_emotion"],
            }
            emotion_msg = (
                f"  ✓ Emotion: {result['emotion']}, "
                f"Confidence: {result['confidence']:.2f}"
            )
            print(emotion_msg)
            print(f"  ✓ Processing time: {elapsed_time:.2f}s")
        except Exception as e:
            results["libraries"]["emotion2vec"] = {
                "status": "error",
                "error": str(e),
            }
            print(f"  ✗ Error: {e}")
    else:
        results["libraries"]["emotion2vec"] = {
            "status": "not_available",
            "message": "Install with: pip install -U funasr modelscope",
        }
        print("  - emotion2vec not available")

    return results


def print_comparison_summary(results: dict) -> None:
    """
    Print a formatted comparison summary.

    Args:
        results: Results dictionary from compare_libraries()
    """
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nAudio file: {results['audio_file']}\n")

    for lib_name, lib_results in results["libraries"].items():
        print(f"{lib_name}:")
        if lib_results["status"] == "success":
            if "emotion" in lib_results:
                print(f"  Emotion: {lib_results['emotion']}")
                print(f"  Confidence: {lib_results['confidence']:.2f}")
                print(f"  Valence: {lib_results['valence']:.2f}")
                print(f"  Arousal: {lib_results['arousal']:.2f}")
            if "feature_count" in lib_results:
                print(f"  Features: {lib_results['feature_count']}")
            print(f"  Processing time: {lib_results['processing_time']:.2f}s")
        elif lib_results["status"] == "error":
            print(f"  Error: {lib_results['error']}")
        elif lib_results["status"] == "not_available":
            print(f"  {lib_results['message']}")
        print()


if __name__ == "__main__":
    MIN_ARGS = 2

    if len(sys.argv) < MIN_ARGS:
        print("Usage: python compare_libraries.py <audio_file>")
        print("\nExample:")
        print("  python compare_libraries.py path/to/audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]

    print("Comparing emotion detection libraries...")
    print("=" * 60)

    results = compare_libraries(audio_file)
    print_comparison_summary(results)

    # Save results to JSON
    output_file = Path("comparison_results.json")
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
