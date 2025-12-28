#!/usr/bin/env python3
"""
Real-Time Emotion Detection Workflow (Integration Example 1)
===========================================================
Demonstrates real-time emotion detection using downloaded samples.

Workflow:
1. Load audio input (microphone or file)
2. Extract features (MFCC, chroma, spectral)
3. Classify emotion using trained model
4. Match to downloaded samples
5. Output emotion label + confidence
"""

import sys
from pathlib import Path
import json

BRAIN_PYTHON_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BRAIN_PYTHON_DIR / "scripts"))


def extract_features_mock(audio_file):
    """Mock feature extraction (would use librosa in production)."""
    return {
        "mfcc_mean": [0.1, 0.2, 0.3],
        "chroma_mean": [0.4, 0.5],
        "spectral_centroid": 1500.0,
        "tempo": 120.0
    }


def classify_emotion_mock(features):
    """Mock emotion classifier (would use trained model)."""
    # Simulated classification
    emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust"]
    confidences = [0.7, 0.1, 0.05, 0.05, 0.05, 0.05]
    
    return {
        "primary_emotion": emotions[0],
        "confidence": confidences[0],
        "all_predictions": dict(zip(emotions, confidences))
    }


def find_matching_samples(emotion, min_confidence=0.5):
    """Find downloaded samples matching the detected emotion."""
    staging_dir = BRAIN_PYTHON_DIR / "scripts" / "emotion_instrument_staging"
    
    if not staging_dir.exists():
        return []
    
    samples = []
    emotion_dir = staging_dir / "base" / emotion.lower()
    
    if emotion_dir.exists():
        for instrument_dir in emotion_dir.iterdir():
            if instrument_dir.is_dir():
                for sample in instrument_dir.glob("*.mp3")[:3]:  # Top 3 per instrument
                    samples.append({
                        "path": str(sample),
                        "instrument": instrument_dir.name,
                        "emotion": emotion
                    })
    
    return samples


class RealtimeEmotionDetector:
    """Real-time emotion detection system."""
    
    def __init__(self):
        self.emotion_history = []
        self.sample_cache = {}
    
    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio and detect emotion."""
        # Extract features
        features = extract_features_mock(audio_chunk)
        
        # Classify
        result = classify_emotion_mock(features)
        
        # Record history
        self.emotion_history.append({
            "emotion": result["primary_emotion"],
            "confidence": result["confidence"],
            "timestamp": "now"
        })
        
        # Find matching samples
        if result["confidence"] >= 0.5:
            samples = find_matching_samples(result["primary_emotion"])
            result["matching_samples"] = samples
        
        return result
    
    def get_emotion_trend(self, window_size=10):
        """Get recent emotion trend."""
        recent = self.emotion_history[-window_size:]
        
        if not recent:
            return None
        
        # Most common emotion in window
        emotions = [r["emotion"] for r in recent]
        return max(set(emotions), key=emotions.count)


def realtime_demo():
    """Demo of real-time emotion detection."""
    print("="*70)
    print("REAL-TIME EMOTION DETECTION WORKFLOW")
    print("="*70)
    
    detector = RealtimeEmotionDetector()
    
    # Simulate processing audio chunks
    mock_audio_chunks = [
        "chunk_1.wav",
        "chunk_2.wav",
        "chunk_3.wav"
    ]
    
    for i, chunk in enumerate(mock_audio_chunks):
        print(f"\nProcessing chunk {i+1}...")
        result = detector.process_audio_chunk(chunk)
        
        print(f"  Detected: {result['primary_emotion'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
        if "matching_samples" in result:
            print(f"  Matching samples: {len(result['matching_samples'])}")
            for sample in result["matching_samples"][:2]:
                print(f"    - {sample['instrument']}: {Path(sample['path']).name[:40]}")
    
    # Show trend
    trend = detector.get_emotion_trend()
    if trend:
        print(f"\nEmotion trend: {trend.upper()}")


def main():
    if len(sys.argv) < 2:
        print("REAL-TIME EMOTION DETECTION (Integration Example 1)")
        print("\nUSAGE:")
        print("  python realtime_emotion_detection.py demo")
        print("  python realtime_emotion_detection.py process <audio_file>")
        return
    
    if sys.argv[1].lower() == "demo":
        realtime_demo()
    elif sys.argv[1].lower() == "process":
        detector = RealtimeEmotionDetector()
        audio_file = sys.argv[2] if len(sys.argv) > 2 else "audio.wav"
        result = detector.process_audio_chunk(audio_file)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
