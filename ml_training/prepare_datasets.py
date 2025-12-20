#!/usr/bin/env python3
"""
Dataset Preparation for MidiKompanion ML Training

Processes raw audio and MIDI files into training datasets with emotion labels.
Integrates with the 216-node emotion thesaurus.

Usage:
    python prepare_datasets.py --input ./raw_data --output ./data
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import librosa
    import mido
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    print("Warning: librosa or mido not available. Using synthetic data.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 216-Node Thesaurus Structure
# =============================================================================

BASE_EMOTIONS = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"]

SUB_EMOTIONS = {
    "Happy": ["Joyful", "Content", "Cheerful", "Playful", "Elated", "Blissful"],
    "Sad": ["Melancholic", "Sorrowful", "Gloomy", "Wistful", "Dejected", "Heartbroken"],
    "Angry": ["Irritated", "Frustrated", "Furious", "Resentful", "Enraged", "Hostile"],
    "Fear": ["Anxious", "Worried", "Terrified", "Nervous", "Panicked", "Dreadful"],
    "Surprise": ["Amazed", "Astonished", "Startled", "Shocked", "Stunned", "Bewildered"],
    "Disgust": ["Repulsed", "Revolted", "Nauseated", "Contemptuous", "Loathing", "Aversion"]
}

INTENSITY_LEVELS = ["Subtle", "Mild", "Moderate", "Strong", "Intense", "Extreme"]


def calculate_vad(base_idx: int, sub_idx: int, intensity_idx: int) -> Dict[str, float]:
    """Calculate VAD coordinates for a node."""
    
    # Base emotion determines primary quadrant
    base_vad = {
        0: {"v": 0.7, "a": 0.5, "d": 0.6},   # Happy
        1: {"v": -0.5, "a": 0.2, "d": 0.3},  # Sad
        2: {"v": -0.4, "a": 0.8, "d": 0.7},  # Angry
        3: {"v": -0.3, "a": 0.7, "d": 0.2},  # Fear
        4: {"v": 0.1, "a": 0.8, "d": 0.5},   # Surprise
        5: {"v": -0.5, "a": 0.5, "d": 0.6}   # Disgust
    }
    
    bvad = base_vad[base_idx]
    
    # Sub-emotion modifies slightly
    sub_mod = (sub_idx - 2.5) / 10
    
    # Intensity scales magnitude
    intensity = (intensity_idx + 1) / 6
    
    return {
        "valence": np.clip(bvad["v"] + sub_mod, -1, 1) * (0.5 + intensity * 0.5),
        "arousal": np.clip(bvad["a"] + sub_mod * 0.5, 0, 1) * (0.7 + intensity * 0.3),
        "dominance": np.clip(bvad["d"] + sub_mod * 0.3, 0, 1),
        "intensity": intensity
    }


def generate_thesaurus_nodes() -> List[Dict]:
    """Generate all 216 emotion nodes with VAD coordinates."""
    nodes = []
    node_id = 0
    
    for base_idx, base in enumerate(BASE_EMOTIONS):
        for sub_idx, sub in enumerate(SUB_EMOTIONS[base]):
            for intensity_idx, intensity in enumerate(INTENSITY_LEVELS):
                vad = calculate_vad(base_idx, sub_idx, intensity_idx)
                
                node = {
                    "id": node_id,
                    "name": f"{sub} ({intensity})",
                    "category": base,
                    "subcategory": sub,
                    "intensity_level": intensity,
                    "valence": vad["valence"],
                    "arousal": vad["arousal"],
                    "dominance": vad["dominance"],
                    "intensity": vad["intensity"]
                }
                nodes.append(node)
                node_id += 1
    
    return nodes


# =============================================================================
# Audio Feature Extraction
# =============================================================================

def extract_audio_features(audio_path: Path, sr: int = 22050) -> Optional[np.ndarray]:
    """Extract 128-dim audio features for emotion recognition."""
    
    if not HAS_AUDIO_LIBS:
        return None
    
    try:
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
        
        features = []
        
        # MFCCs (40 dims)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.extend(mfccs.mean(axis=1))
        features.extend(mfccs.std(axis=1))
        
        # Chroma (24 dims)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(chroma.mean(axis=1))
        features.extend(chroma.std(axis=1))
        
        # Spectral features (32 dims)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        features.append(spectral_centroid.mean())
        features.append(spectral_centroid.std())
        features.append(spectral_rolloff.mean())
        features.append(spectral_rolloff.std())
        features.extend(spectral_contrast.mean(axis=1))
        features.extend(spectral_contrast.std(axis=1))
        
        # RMS energy (8 dims)
        rms = librosa.feature.rms(y=y)
        features.append(rms.mean())
        features.append(rms.std())
        features.append(rms.max())
        features.append(rms.min())
        
        # Zero crossing rate (4 dims)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(zcr.mean())
        features.append(zcr.std())
        
        # Tempo and beat (4 dims)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo / 200.0)  # Normalize
        
        # Pad or truncate to 128 dims
        features = np.array(features, dtype=np.float32)
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))
        else:
            features = features[:128]
        
        return features
        
    except Exception as e:
        logger.warning(f"Error extracting features from {audio_path}: {e}")
        return None


# =============================================================================
# MIDI Feature Extraction
# =============================================================================

def extract_midi_features(midi_path: Path) -> Optional[Dict]:
    """Extract features from MIDI file for training."""
    
    if not HAS_AUDIO_LIBS:
        return None
    
    try:
        mid = mido.MidiFile(str(midi_path))
        
        notes = []
        tempo = 120
        
        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append({
                        "pitch": msg.note,
                        "velocity": msg.velocity,
                        "time": current_time,
                        "channel": msg.channel
                    })
                elif msg.type == 'set_tempo':
                    tempo = mido.tempo2bpm(msg.tempo)
        
        if not notes:
            return None
        
        pitches = [n["pitch"] for n in notes]
        velocities = [n["velocity"] for n in notes]
        
        return {
            "num_notes": len(notes),
            "pitch_mean": np.mean(pitches),
            "pitch_std": np.std(pitches),
            "pitch_range": max(pitches) - min(pitches),
            "velocity_mean": np.mean(velocities),
            "velocity_std": np.std(velocities),
            "tempo": tempo,
            "notes": notes
        }
        
    except Exception as e:
        logger.warning(f"Error processing MIDI {midi_path}: {e}")
        return None


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_dataset(
    num_samples: int,
    nodes: List[Dict],
    output_dir: Path
) -> pd.DataFrame:
    """Generate synthetic training data when real data is not available."""
    
    logger.info(f"Generating {num_samples} synthetic samples...")
    
    data = []
    
    for i in tqdm(range(num_samples)):
        # Random node selection
        node = nodes[np.random.randint(len(nodes))]
        
        # Generate audio features based on VAD
        audio_features = np.random.randn(128).astype(np.float32) * 0.1
        
        # Embed VAD in first dimensions
        audio_features[0] = node["valence"]
        audio_features[1] = node["arousal"]
        audio_features[2] = node["dominance"]
        audio_features[3] = node["intensity"]
        
        # Add some coherent structure
        audio_features[4:8] = np.array([
            node["valence"] * 0.5 + np.random.randn() * 0.1,
            node["arousal"] * 0.5 + np.random.randn() * 0.1,
            node["dominance"] * 0.5 + np.random.randn() * 0.1,
            node["intensity"] * 0.5 + np.random.randn() * 0.1
        ])
        
        data.append({
            "sample_id": i,
            "node_id": node["id"],
            "node_name": node["name"],
            "category": node["category"],
            "valence": node["valence"],
            "arousal": node["arousal"],
            "dominance": node["dominance"],
            "intensity": node["intensity"],
            "audio_features": audio_features.tolist()
        })
    
    df = pd.DataFrame(data)
    
    # Save to files
    train_df = df.iloc[:int(len(df) * 0.8)]
    val_df = df.iloc[int(len(df) * 0.8):int(len(df) * 0.9)]
    test_df = df.iloc[int(len(df) * 0.9):]
    
    train_df.to_json(output_dir / "train.json", orient="records", lines=True)
    val_df.to_json(output_dir / "val.json", orient="records", lines=True)
    test_df.to_json(output_dir / "test.json", orient="records", lines=True)
    
    logger.info(f"Saved: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for ML training")
    parser.add_argument("--input", type=str, default="./raw_data", help="Input directory")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--synthetic-samples", type=int, default=50000,
                        help="Number of synthetic samples if no real data")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate thesaurus nodes
    nodes = generate_thesaurus_nodes()
    
    # Save thesaurus
    with open(output_dir / "thesaurus.json", "w") as f:
        json.dump(nodes, f, indent=2)
    logger.info(f"Saved {len(nodes)} emotion nodes to thesaurus.json")
    
    # Check for real data
    audio_files = list(input_dir.glob("**/*.wav")) + list(input_dir.glob("**/*.mp3"))
    midi_files = list(input_dir.glob("**/*.mid")) + list(input_dir.glob("**/*.midi"))
    
    if audio_files or midi_files:
        logger.info(f"Found {len(audio_files)} audio files and {len(midi_files)} MIDI files")
        # TODO: Process real data with emotion labels
        # For now, fall through to synthetic data
    
    # Generate synthetic data
    logger.info("Generating synthetic training data...")
    generate_synthetic_dataset(args.synthetic_samples, nodes, output_dir)
    
    logger.info(f"Dataset preparation complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
