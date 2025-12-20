#!/usr/bin/env python3
"""
prepare_datasets.py - Dataset Preparation with Node Labels
==========================================================

Agent 2: ML Training Specialist (Week 3-6)
Purpose: Prepare training datasets with 216-node emotion thesaurus labels.

This script:
1. Loads audio and MIDI datasets
2. Labels data with node IDs (0-215) based on VAD coordinates
3. Maps MIDI sequences to node musical attributes
4. Creates node relationship graphs for context
5. Exports prepared datasets for training
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import librosa
import mido
from tqdm import tqdm

# Load 216-node emotion thesaurus
EMOTION_THESAURUS_PATH = Path(__file__).parent.parent / "data" / "emotion_thesaurus.json"


def load_emotion_thesaurus() -> Dict:
    """Load the 216-node emotion thesaurus."""
    if not EMOTION_THESAURUS_PATH.exists():
        raise FileNotFoundError(f"Emotion thesaurus not found: {EMOTION_THESAURUS_PATH}")

    with open(EMOTION_THESAURUS_PATH, 'r') as f:
        return json.load(f)


def vad_to_node_id(vad_coords: Tuple[float, float, float], thesaurus: Dict) -> int:
    """
    Map VAD coordinates to nearest node ID.

    Args:
        vad_coords: (valence, arousal, dominance) tuple
        thesaurus: Loaded emotion thesaurus

    Returns:
        Node ID (0-215)
    """
    valence, arousal, dominance = vad_coords

    min_distance = float('inf')
    best_node_id = 0

    for node in thesaurus.get('nodes', []):
        node_vad = (node.get('valence', 0.0),
                   node.get('arousal', 0.0),
                   node.get('dominance', 0.0))

        # Euclidean distance in VAD space
        distance = np.sqrt(
            (valence - node_vad[0])**2 +
            (arousal - node_vad[1])**2 +
            (dominance - node_vad[2])**2
        )

        if distance < min_distance:
            min_distance = distance
            best_node_id = node.get('id', 0)

    return best_node_id


def extract_audio_features(audio_path: str, sample_rate: int = 44100) -> np.ndarray:
    """
    Extract 128-dimensional audio features for EmotionRecognizer.

    Returns:
        128-dim feature vector
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sample_rate, duration=3.0)  # 3-second clips

        features = []

        # 1. MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(mfccs.mean(axis=1))

        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(spectral_centroids.mean())

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(spectral_rolloff.mean())

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        features.append(zero_crossing_rate.mean())

        # 3. Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma(y=y, sr=sr)
        features.extend(chroma.mean(axis=1))

        # 4. Tonnetz (6 features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features.extend(tonnetz.mean(axis=1))

        # 5. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo / 200.0)  # Normalize

        # 6. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features.append(rms.mean())

        # 7. Spectral contrast (7 bands)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(spectral_contrast.mean(axis=1))

        # 8. Harmonic/percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features.append(np.mean(np.abs(y_harmonic)))
        features.append(np.mean(np.abs(y_percussive)))

        # Pad or truncate to 128 dimensions
        feature_array = np.array(features)
        if len(feature_array) < 128:
            feature_array = np.pad(feature_array, (0, 128 - len(feature_array)))
        elif len(feature_array) > 128:
            feature_array = feature_array[:128]

        return feature_array

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return np.zeros(128)


def extract_midi_features(midi_path: str) -> Dict:
    """
    Extract MIDI features for MelodyTransformer and HarmonyPredictor.

    Returns:
        Dictionary with melody, harmony, and groove features
    """
    try:
        mid = mido.MidiFile(midi_path)

        # Extract notes
        notes = []
        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append({
                        'pitch': msg.note,
                        'time': current_time,
                        'velocity': msg.velocity
                    })

        if not notes:
            return {
                'melody': np.zeros(128),  # 128 MIDI pitches
                'harmony': np.zeros(64),  # Chord probabilities
                'groove': np.zeros(32)    # Groove parameters
            }

        # Melody features (128-dim: pitch probabilities)
        melody = np.zeros(128)
        for note in notes:
            if 0 <= note['pitch'] < 128:
                melody[note['pitch']] += 1.0
        melody = melody / (np.sum(melody) + 1e-10)  # Normalize

        # Harmony features (64-dim: simplified chord representation)
        # Group notes into chords (simplified)
        harmony = np.zeros(64)
        if len(notes) > 0:
            # Extract chord roots and qualities (simplified)
            pitches = [n['pitch'] % 12 for n in notes[:10]]  # First 10 notes
            for pitch in pitches:
                harmony[pitch] += 1.0
        harmony = harmony / (np.sum(harmony) + 1e-10)

        # Groove features (32-dim: rhythm patterns)
        if len(notes) > 1:
            intervals = np.diff([n['time'] for n in notes[:32]])
            groove = np.histogram(intervals, bins=32, range=(0, 2.0))[0]
            groove = groove / (np.sum(groove) + 1e-10)
        else:
            groove = np.zeros(32)

        return {
            'melody': melody,
            'harmony': harmony,
            'groove': groove
        }

    except Exception as e:
        print(f"Error extracting MIDI features from {midi_path}: {e}")
        return {
            'melody': np.zeros(128),
            'harmony': np.zeros(64),
            'groove': np.zeros(32)
        }


def prepare_dataset(
    audio_dir: str,
    midi_dir: Optional[str] = None,
    output_dir: str = "datasets/prepared",
    vad_labels_file: Optional[str] = None
) -> None:
    """
    Prepare dataset with node labels.

    Args:
        audio_dir: Directory containing audio files
        midi_dir: Optional directory containing MIDI files
        output_dir: Output directory for prepared datasets
        vad_labels_file: Optional CSV file with VAD labels (columns: file, valence, arousal, dominance)
    """
    print("Loading emotion thesaurus...")
    thesaurus = load_emotion_thesaurus()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load VAD labels if provided
    vad_labels = {}
    if vad_labels_file and os.path.exists(vad_labels_file):
        df = pd.read_csv(vad_labels_file)
        for _, row in df.iterrows():
            vad_labels[row['file']] = (row['valence'], row['arousal'], row['dominance'])

    # Prepare datasets
    emotion_recognizer_data = []
    melody_transformer_data = []
    harmony_predictor_data = []
    dynamics_engine_data = []
    groove_predictor_data = []

    audio_files = list(Path(audio_dir).glob("*.wav")) + list(Path(audio_dir).glob("*.mp3"))

    print(f"Processing {len(audio_files)} audio files...")

    for audio_file in tqdm(audio_files):
        # Get VAD coordinates (use file name or labels file)
        if audio_file.name in vad_labels:
            vad_coords = vad_labels[audio_file.name]
        else:
            # Default: neutral emotion
            vad_coords = (0.0, 0.5, 0.5)

        # Map to node ID
        node_id = vad_to_node_id(vad_coords, thesaurus)
        node = next((n for n in thesaurus.get('nodes', []) if n.get('id') == node_id), None)

        if not node:
            continue

        # Extract audio features (128-dim for EmotionRecognizer)
        audio_features = extract_audio_features(str(audio_file))

        # EmotionRecognizer data: audio → 64-dim embedding → node ID
        emotion_recognizer_data.append({
            'input': audio_features.tolist(),
            'node_id': node_id,
            'vad': vad_coords,
            'file': audio_file.name
        })

        # Extract MIDI features if available
        if midi_dir:
            midi_file = Path(midi_dir) / (audio_file.stem + ".mid")
            if midi_file.exists():
                midi_features = extract_midi_features(str(midi_file))

                # MelodyTransformer data: node embedding → MIDI notes
                melody_transformer_data.append({
                    'node_id': node_id,
                    'node_embedding': [node.get('valence', 0.0), node.get('arousal', 0.0),
                                      node.get('dominance', 0.0), node.get('intensity', 0.0)],
                    'output': midi_features['melody'].tolist(),
                    'file': audio_file.name
                })

                # HarmonyPredictor data: node context → chords
                harmony_predictor_data.append({
                    'node_id': node_id,
                    'related_nodes': node.get('relatedEmotions', []),
                    'output': midi_features['harmony'].tolist(),
                    'file': audio_file.name
                })

                # DynamicsEngine data: node intensity → expression
                dynamics_engine_data.append({
                    'node_id': node_id,
                    'intensity': node.get('intensity', 0.5),
                    'output': [node.get('intensity', 0.5)] * 16,  # 16 expression parameters
                    'file': audio_file.name
                })

                # GroovePredictor data: node arousal → groove
                groove_predictor_data.append({
                    'node_id': node_id,
                    'arousal': node.get('arousal', 0.5),
                    'output': midi_features['groove'].tolist(),
                    'file': audio_file.name
                })

    # Save prepared datasets
    print("Saving prepared datasets...")

    datasets = {
        'emotion_recognizer': emotion_recognizer_data,
        'melody_transformer': melody_transformer_data,
        'harmony_predictor': harmony_predictor_data,
        'dynamics_engine': dynamics_engine_data,
        'groove_predictor': groove_predictor_data
    }

    for name, data in datasets.items():
        output_file = output_path / f"{name}_dataset.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} samples to {output_file}")

    # Save metadata
    metadata = {
        'num_samples': {
            'emotion_recognizer': len(emotion_recognizer_data),
            'melody_transformer': len(melody_transformer_data),
            'harmony_predictor': len(harmony_predictor_data),
            'dynamics_engine': len(dynamics_engine_data),
            'groove_predictor': len(groove_predictor_data)
        },
        'feature_dims': {
            'emotion_recognizer_input': 128,
            'emotion_recognizer_output': 64,
            'melody_transformer_input': 64,
            'melody_transformer_output': 128,
            'harmony_predictor_input': 128,
            'harmony_predictor_output': 64,
            'dynamics_engine_input': 32,
            'dynamics_engine_output': 16,
            'groove_predictor_input': 64,
            'groove_predictor_output': 32
        }
    }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset preparation complete!")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare datasets with node labels")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--midi-dir", help="Optional directory containing MIDI files")
    parser.add_argument("--output-dir", default="datasets/prepared", help="Output directory")
    parser.add_argument("--vad-labels", help="Optional CSV file with VAD labels")

    args = parser.parse_args()

    prepare_dataset(
        args.audio_dir,
        args.midi_dir,
        args.output_dir,
        args.vad_labels
    )
