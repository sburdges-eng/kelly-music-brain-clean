#!/usr/bin/env python3
"""
Real Dataset Loaders for Kelly ML Training
==========================================
Loads actual datasets for training the 5 ML models:
1. DEAM - Emotion recognition (audio + valence/arousal)
2. Lakh MIDI - Melody generation
3. MAESTRO - Dynamics engine
4. Groove MIDI - Groove prediction
5. Harmony - Chord progressions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Optional imports with graceful fallback
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available. Audio processing will be limited.")

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    warnings.warn("mido not available. MIDI processing will be limited.")


# =============================================================================
# 1. DEAM Dataset Loader (Emotion Recognition)
# =============================================================================

class DEAMDataset(DEAMDatasetBase):
    """DEAM dataset with full structure."""
    pass


class EmotionDataset(DEAMDatasetBase):
    """
    Alias for DEAMDataset for backward compatibility.
    Supports both new DEAM structure and old simple structure.
    """

    def __init__(
        self,
        audio_dir: Path,
        labels_file: Optional[Path] = None,
        n_mels: int = 128,
        duration: float = 2.0,
        sr: int = 22050,
        cache_features: bool = False
    ):
        """
        Backward-compatible initializer for simple audio directory structure.

        Args:
            audio_dir: Directory containing audio files
            labels_file: Path to labels CSV/JSON file
            n_mels: Number of mel bands
            duration: Duration to analyze (seconds)
            sr: Sample rate
            cache_features: Cache extracted features
        """
        # Try DEAM structure first
        annotations_file = None
        if labels_file:
            annotations_file = labels_file
        else:
            # Try to find labels in common locations
            for path in [audio_dir / "labels.csv", audio_dir.parent / "labels.csv"]:
                if path.exists():
                    annotations_file = path
                    break

        # Check if this looks like DEAM structure
        deam_annotations = audio_dir.parent / "annotations" / "annotations.csv"
        if deam_annotations.exists():
            # Use DEAM structure
            super().__init__(
                deam_dir=audio_dir.parent,
                annotations_file=deam_annotations,
                sample_rate=sr,
                n_mels=n_mels
            )
            return

        # Otherwise use simple structure
        self.audio_dir = Path(audio_dir)
        self.n_mels = n_mels
        self.duration = duration
        self.sr = sr
        self.cache_features = cache_features

        # Find audio files
        self.audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            self.audio_files.extend(list(self.audio_dir.glob(ext)))
            self.audio_files.extend(list(self.audio_dir.glob(f'**/{ext}')))

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")

        # Load labels
        self.labels = self._load_labels_simple(annotations_file or labels_file)

        # Filter to files with labels
        if self.labels:
            self.audio_files = [
                f for f in self.audio_files
                if f.name in self.labels or f.stem in self.labels
            ]

        if len(self.audio_files) == 0:
            raise ValueError("No audio files with labels found")

        self.feature_cache: Optional[Dict[Path, np.ndarray]] = {} if cache_features else None
        print(f"Loaded {len(self.audio_files)} audio files with emotion labels")

    def _load_labels_simple(self, labels_file: Optional[Path]) -> Dict[str, np.ndarray]:
        """Load labels from simple CSV/JSON format."""
        if labels_file is None or not Path(labels_file).exists():
            print(f"Warning: No labels file found. Using placeholder labels.")
            return {}

        labels = {}
        labels_path = Path(labels_file)

        if labels_path.suffix == '.csv':
            with open(labels_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('filename', row.get('file', ''))
                    if not filename:
                        continue
                    try:
                        valence = float(row.get('valence', row.get('val', 0.0)))
                        arousal = float(row.get('arousal', row.get('ar', 0.0)))
                        emotion_vec = np.zeros(64, dtype=np.float32)
                        emotion_vec[:32] = valence
                        emotion_vec[32:] = arousal
                        labels[filename] = emotion_vec
                        labels[Path(filename).stem] = emotion_vec
                    except (ValueError, KeyError):
                        continue
        elif labels_path.suffix == '.json':
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for filename, label_data in data.items():
                    if isinstance(label_data, dict):
                        valence = float(label_data.get('valence', 0.0))
                        arousal = float(label_data.get('arousal', 0.0))
                    elif isinstance(label_data, (list, tuple)) and len(label_data) >= 2:
                        valence = float(label_data[0])
                        arousal = float(label_data[1])
                    else:
                        continue
                    emotion_vec = np.zeros(64, dtype=np.float32)
                    emotion_vec[:32] = valence
                    emotion_vec[32:] = arousal
                    labels[filename] = emotion_vec
                    labels[Path(filename).stem] = emotion_vec

        print(f"Loaded {len(labels)} emotion labels from {labels_file}")
        return labels

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with dict format for compatibility."""
        audio_path = self.audio_files[idx]
        features = self._extract_mel_features_simple(audio_path)
        filename = audio_path.name
        label_key = filename if filename in self.labels else audio_path.stem
        label = self.labels.get(label_key)

        if label is None:
            label = np.random.randn(64).astype(np.float32)
            label = np.tanh(label * 0.5)

        return {
            'mel_features': torch.tensor(features, dtype=torch.float32),
            'emotion': torch.tensor(label, dtype=torch.float32)
        }

    def _extract_mel_features_simple(self, audio_path: Path) -> np.ndarray:
        """Extract mel-spectrogram features."""
        if self.feature_cache is not None and audio_path in self.feature_cache:
            return self.feature_cache[audio_path]

        if not LIBROSA_AVAILABLE:
            return np.zeros(self.n_mels, dtype=np.float32)

        try:
            y, sr = librosa.load(
                str(audio_path),
                sr=self.sr,
                duration=self.duration,
                mono=True
            )
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.n_mels,
                hop_length=512,
                n_fft=2048
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            features = np.mean(log_mel, axis=1)
            features = np.clip(features / 80.0, -1.0, 1.0)

            if self.feature_cache is not None:
                self.feature_cache[audio_path] = features

            return features.astype(np.float32)
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return np.zeros(self.n_mels, dtype=np.float32)


class DEAMDatasetBase(Dataset):
    """
    DEAM (Dataset for Emotion Analysis in Music) loader.
    Expected structure:
        deam_dir/
        ├── audio/
        │   ├── audio_001.wav
        │   └── ...
        └── annotations/
            └── annotations.csv  (filename,valence,arousal)
    """

    def __init__(
        self,
        deam_dir: Path,
        annotations_file: Optional[Path] = None,
        sample_rate: int = 22050,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048
    ):
        self.deam_dir = Path(deam_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

        # Find annotations file
        if annotations_file is None:
            # Try common locations
            possible_paths = [
                self.deam_dir / "annotations" / "annotations.csv",
                self.deam_dir / "annotations.csv",
                self.deam_dir / "DEAM_annotations" / "annotations.csv",
                self.deam_dir / "labels.csv"
            ]
            for path in possible_paths:
                if path.exists():
                    annotations_file = path
                    break

        if annotations_file is None or not Path(annotations_file).exists():
            raise FileNotFoundError(
                f"Annotations file not found. Tried: {possible_paths}"
            )

        # Load annotations
        self.samples = []
        audio_dir = self.deam_dir / "audio"
        if not audio_dir.exists():
            audio_dir = self.deam_dir  # Try root directory

        with open(annotations_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', row.get('song_id', ''))
                if not filename:
                    continue

                # Try different filename formats
                audio_path = None
                for ext in ['.wav', '.mp3', '.flac']:
                    test_path = audio_dir / f"{filename}{ext}"
                    if test_path.exists():
                        audio_path = test_path
                        break
                    # Also try without extension
                    test_path = audio_dir / filename
                    if test_path.exists():
                        audio_path = test_path
                        break

                if audio_path and audio_path.exists():
                    try:
                        valence = float(row.get('valence', 0.0))
                        arousal = float(row.get('arousal', 0.0))
                        self.samples.append({
                            'audio_path': audio_path,
                            'valence': valence,
                            'arousal': arousal
                        })
                    except (ValueError, KeyError):
                        continue

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {annotations_file}")

        print(f"Loaded {len(self.samples)} DEAM samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample['audio_path']

        # Load audio and extract mel-spectrogram
        if LIBROSA_AVAILABLE:
            try:
                y, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=30.0)
                # Extract mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_mels=self.n_mels,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft
                )
                # Convert to log scale and take mean over time
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_features = np.mean(mel_spec_db, axis=1)  # Average over time
            except Exception as e:
                warnings.warn(f"Error loading {audio_path}: {e}")
                mel_features = np.zeros(self.n_mels, dtype=np.float32)
        else:
            mel_features = np.zeros(self.n_mels, dtype=np.float32)

        # Normalize to [-1, 1]
        if mel_features.max() > 0:
            mel_features = mel_features / (abs(mel_features).max() + 1e-8)

        # Create emotion embedding from valence/arousal
        # First 32 dims: valence-related, last 32 dims: arousal-related
        emotion = np.zeros(64, dtype=np.float32)
        emotion[:32] = sample['valence']
        emotion[32:] = sample['arousal']

        return {
            'mel_features': torch.tensor(mel_features, dtype=torch.float32),
            'emotion': torch.tensor(emotion, dtype=torch.float32)
        }


# =============================================================================
# 2. Lakh MIDI Dataset Loader (Melody Generation)
# =============================================================================

class LakhMIDIDataset(Dataset):
    """
    Lakh MIDI Dataset loader for melody generation.
    Expected structure:
        lakh_dir/
        ├── midi_files/
        │   ├── *.mid
        │   └── ...
        └── emotion_labels.json  (optional, {filename: {valence, arousal}})
    """

    def __init__(
        self,
        lakh_dir: Path,
        emotion_labels_file: Optional[Path] = None,
        max_files: Optional[int] = None
    ):
        self.lakh_dir = Path(lakh_dir)
        self.midi_files = []

        # Find MIDI files
        for ext in ['.mid', '.midi']:
            files = list(self.lakh_dir.rglob(f"*{ext}"))
            self.midi_files.extend(files)
            if max_files and len(self.midi_files) >= max_files:
                self.midi_files = self.midi_files[:max_files]
                break

        if len(self.midi_files) == 0:
            raise ValueError(f"No MIDI files found in {lakh_dir}")

        # Load emotion labels if provided
        self.emotion_labels = {}
        if emotion_labels_file and Path(emotion_labels_file).exists():
            with open(emotion_labels_file, 'r') as f:
                self.emotion_labels = json.load(f)

        print(f"Loaded {len(self.midi_files)} MIDI files")

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_path = self.midi_files[idx]
        filename = midi_path.name

        # Load MIDI and extract melody notes
        note_probs = np.zeros(128, dtype=np.float32)

        if MIDO_AVAILABLE:
            try:
                mid = mido.MidiFile(str(midi_path))
                notes = []

                for track in mid.tracks:
                    current_time = 0
                    active_notes = {}

                    for msg in track:
                        current_time += msg.time

                        if msg.type == 'note_on' and msg.velocity > 0:
                            active_notes[msg.note] = current_time
                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                            if msg.note in active_notes:
                                notes.append(msg.note)
                                del active_notes[msg.note]

                # Convert to probability distribution
                if len(notes) > 0:
                    unique_notes, counts = np.unique(notes, return_counts=True)
                    for note, count in zip(unique_notes, counts):
                        if 0 <= note < 128:
                            note_probs[note] = count / len(notes)
            except Exception as e:
                warnings.warn(f"Error loading MIDI {midi_path}: {e}")

        # Get emotion embedding (from labels or generate from filename)
        if filename in self.emotion_labels:
            label = self.emotion_labels[filename]
            valence = label.get('valence', 0.0)
            arousal = label.get('arousal', 0.0)
        else:
            # Generate synthetic emotion based on filename or random
            valence = np.random.uniform(-1.0, 1.0)
            arousal = np.random.uniform(0.0, 1.0)

        emotion = np.zeros(64, dtype=np.float32)
        emotion[:32] = valence
        emotion[32:] = arousal

        return {
            'emotion': torch.tensor(emotion, dtype=torch.float32),
            'notes': torch.tensor(note_probs, dtype=torch.float32)
        }


# =============================================================================
# 3. MAESTRO Dataset Loader (Dynamics Engine)
# =============================================================================

class MAESTRODataset(Dataset):
    """
    MAESTRO Dataset loader for dynamics/expression.
    Expected structure:
        maestro_dir/
        ├── *.midi
        └── ...
    """

    def __init__(self, maestro_dir: Path, max_files: Optional[int] = None):
        self.maestro_dir = Path(maestro_dir)
        self.midi_files = []

        for ext in ['.mid', '.midi']:
            files = list(self.maestro_dir.rglob(f"*{ext}"))
            self.midi_files.extend(files)
            if max_files and len(self.midi_files) >= max_files:
                self.midi_files = self.midi_files[:max_files]
                break

        if len(self.midi_files) == 0:
            raise ValueError(f"No MIDI files found in {maestro_dir}")

        print(f"Loaded {len(self.midi_files)} MAESTRO MIDI files")

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_path = self.midi_files[idx]

        # Extract velocity and timing data
        velocities = []
        timings = []

        if MIDO_AVAILABLE:
            try:
                mid = mido.MidiFile(str(midi_path))
                current_time = 0

                for track in mid.tracks:
                    for msg in track:
                        current_time += msg.time
                        if msg.type == 'note_on' and msg.velocity > 0:
                            velocities.append(msg.velocity / 127.0)  # Normalize
                            timings.append(current_time)

                # Extract features: mean velocity, std velocity, mean timing, etc.
                if len(velocities) > 0:
                    vel_mean = np.mean(velocities)
                    vel_std = np.std(velocities)
                    vel_max = np.max(velocities)
                    vel_min = np.min(velocities)
                else:
                    vel_mean = vel_std = vel_max = vel_min = 0.5

                if len(timings) > 0:
                    timing_mean = np.mean(timings)
                    timing_std = np.std(timings)
                else:
                    timing_mean = timing_std = 0.0

            except Exception as e:
                warnings.warn(f"Error loading MIDI {midi_path}: {e}")
                vel_mean = vel_std = vel_max = vel_min = 0.5
                timing_mean = timing_std = 0.0
        else:
            vel_mean = vel_std = vel_max = vel_min = 0.5
            timing_mean = timing_std = 0.0

        # Create compact context (32 dims) from emotion + dynamics
        # In practice, this would come from emotion model output
        compact_context = np.zeros(32, dtype=np.float32)
        compact_context[0] = vel_mean
        compact_context[1] = vel_std
        compact_context[2] = vel_max
        compact_context[3] = vel_min
        compact_context[4] = timing_mean / 1000.0  # Normalize timing
        compact_context[5] = timing_std / 1000.0
        # Rest filled with emotion-like features (would come from emotion model)
        compact_context[6:] = np.random.uniform(-1.0, 1.0, 26)

        # Expression parameters (16 dims): velocity, timing, dynamics
        expression = np.zeros(16, dtype=np.float32)
        expression[0:4] = [vel_mean, vel_std, vel_max, vel_min]
        expression[4:8] = [timing_mean / 1000.0, timing_std / 1000.0, 0.0, 0.0]
        expression[8:] = np.random.uniform(0.0, 1.0, 8)  # Additional expression params

        return {
            'context': torch.tensor(compact_context, dtype=torch.float32),
            'expression': torch.tensor(expression, dtype=torch.float32)
        }


# =============================================================================
# 4. Groove MIDI Dataset Loader (Groove Prediction)
# =============================================================================

class GrooveMIDIDataset(Dataset):
    """
    Groove MIDI Dataset loader for drum pattern/groove prediction.
    Expected structure:
        groove_dir/
        ├── *.midi
        └── ...
    """

    def __init__(self, groove_dir: Path, max_files: Optional[int] = None):
        self.groove_dir = Path(groove_dir)
        self.midi_files = []

        for ext in ['.mid', '.midi']:
            files = list(self.groove_dir.rglob(f"*{ext}"))
            self.midi_files.extend(files)
            if max_files and len(self.midi_files) >= max_files:
                self.midi_files = self.midi_files[:max_files]
                break

        if len(self.midi_files) == 0:
            raise ValueError(f"No MIDI files found in {groove_dir}")

        print(f"Loaded {len(self.midi_files)} Groove MIDI files")

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_path = self.midi_files[idx]

        # Extract groove parameters: timing deviations, velocity patterns
        timing_deviations = []
        velocity_pattern = []

        if MIDO_AVAILABLE:
            try:
                mid = mido.MidiFile(str(midi_path))
                ticks_per_beat = mid.ticks_per_beat
                current_time = 0
                last_note_time = 0

                for track in mid.tracks:
                    for msg in track:
                        current_time += msg.time
                        if msg.type == 'note_on' and msg.velocity > 0:
                            # Calculate timing deviation from expected grid
                            expected_time = round(current_time / ticks_per_beat) * ticks_per_beat
                            deviation = (current_time - expected_time) / ticks_per_beat
                            timing_deviations.append(deviation)
                            velocity_pattern.append(msg.velocity / 127.0)

            except Exception as e:
                warnings.warn(f"Error loading MIDI {midi_path}: {e}")

        # Create groove parameters (32 dims)
        groove_params = np.zeros(32, dtype=np.float32)

        if len(timing_deviations) > 0:
            groove_params[0] = np.mean(timing_deviations)
            groove_params[1] = np.std(timing_deviations)
            groove_params[2:16] = np.histogram(
                timing_deviations, bins=14, range=(-0.5, 0.5)
            )[0] / len(timing_deviations)

        if len(velocity_pattern) > 0:
            groove_params[16] = np.mean(velocity_pattern)
            groove_params[17] = np.std(velocity_pattern)
            groove_params[18:32] = np.histogram(
                velocity_pattern, bins=14, range=(0.0, 1.0)
            )[0] / len(velocity_pattern)

        # Generate emotion embedding (would come from emotion model)
        emotion = np.zeros(64, dtype=np.float32)
        emotion[:32] = np.random.uniform(-1.0, 1.0, 32)
        emotion[32:] = np.random.uniform(0.0, 1.0, 32)

        return {
            'emotion': torch.tensor(emotion, dtype=torch.float32),
            'groove': torch.tensor(groove_params, dtype=torch.float32)
        }


# =============================================================================
# 5. Harmony Dataset Loader (Harmony Prediction)
# =============================================================================

class HarmonyDataset(Dataset):
    """
    Harmony dataset loader for chord progression prediction.
    Expected structure:
        harmony_dir/
        └── chord_progressions.json
    """

    def __init__(self, harmony_file: Path):
        self.harmony_file = Path(harmony_file)

        if not self.harmony_file.exists():
            raise FileNotFoundError(f"Harmony file not found: {harmony_file}")

        with open(self.harmony_file, 'r') as f:
            data = json.load(f)

        self.progressions = []
        if isinstance(data, dict):
            if "progressions" in data:
                self.progressions = data["progressions"]
            elif "chord_progressions" in data:
                self.progressions = data["chord_progressions"]
        elif isinstance(data, list):
            self.progressions = data

        if len(self.progressions) == 0:
            raise ValueError(f"No progressions found in {harmony_file}")

        # Chord name to index mapping (64 chords)
        self.chord_to_idx = self._build_chord_mapping()

        print(f"Loaded {len(self.progressions)} chord progressions")

    def _build_chord_mapping(self) -> Dict[str, int]:
        """Build mapping from chord names to indices."""
        # Common chords: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        # With variations: maj, min, dim, aug, 7, maj7, min7, etc.
        base_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_types = ['', 'm', 'dim', 'aug', '7', 'maj7', 'min7', 'sus4', 'sus2']
        mapping = {}
        idx = 0

        for note in base_notes:
            for chord_type in chord_types:
                if idx < 64:
                    mapping[f"{note}{chord_type}"] = idx
                    idx += 1

        return mapping

    def chord_to_index(self, chord_name: str) -> int:
        """Convert chord name to index."""
        # Normalize chord name
        chord = chord_name.strip().upper()
        return self.chord_to_idx.get(chord, 0)

    def __len__(self):
        return len(self.progressions)

    def __getitem__(self, idx):
        progression = self.progressions[idx]

        # Get chords
        if isinstance(progression, dict):
            chords = progression.get('chords', [])
            emotion_data = progression.get('emotion', {})
        else:
            chords = progression if isinstance(progression, list) else []
            emotion_data = {}

        # Convert chords to indices
        chord_indices = [self.chord_to_index(chord) for chord in chords]
        chord_probs = np.zeros(64, dtype=np.float32)

        if len(chord_indices) > 0:
            unique_chords, counts = np.unique(chord_indices, return_counts=True)
            for chord_idx, count in zip(unique_chords, counts):
                if 0 <= chord_idx < 64:
                    chord_probs[chord_idx] = count / len(chord_indices)

        # Get emotion or generate
        valence = emotion_data.get('valence', np.random.uniform(-1.0, 1.0))
        arousal = emotion_data.get('arousal', np.random.uniform(0.0, 1.0))

        # Create context (128 dims): emotion + audio features
        context = np.zeros(128, dtype=np.float32)
        context[:32] = valence
        context[32:64] = arousal
        context[64:] = np.random.uniform(-1.0, 1.0, 64)  # Audio features

        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'chords': torch.tensor(chord_probs, dtype=torch.float32)
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Full dataset
        val_ratio: Ratio of validation samples (0.0 to 1.0)
        random_seed: Random seed for shuffling

    Returns:
        (train_dataset, val_dataset)
    """
    from torch.utils.data import random_split

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    return train_dataset, val_dataset


# =============================================================================
# Dataset Factory
# =============================================================================

def create_dataset(
    dataset_type: str,
    data_dir: Path,
    **kwargs
) -> Dataset:
    """
    Factory function to create appropriate dataset.

    Args:
        dataset_type: One of 'deam', 'lakh', 'maestro', 'groove', 'harmony'
        data_dir: Path to dataset directory
        **kwargs: Additional arguments for specific dataset

    Returns:
        Dataset instance
    """
    data_dir = Path(data_dir)

    if dataset_type.lower() == 'deam':
        return DEAMDataset(data_dir, **kwargs)
    elif dataset_type.lower() == 'lakh':
        return LakhMIDIDataset(data_dir, **kwargs)
    elif dataset_type.lower() == 'maestro':
        return MAESTRODataset(data_dir, **kwargs)
    elif dataset_type.lower() == 'groove':
        return GrooveMIDIDataset(data_dir, **kwargs)
    elif dataset_type.lower() == 'harmony':
        harmony_file = kwargs.get('harmony_file', data_dir / 'chord_progressions.json')
        return HarmonyDataset(harmony_file)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


if __name__ == "__main__":
    # Test dataset loaders
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset loaders")
    parser.add_argument("dataset_type", choices=['deam', 'lakh', 'maestro', 'groove', 'harmony'])
    parser.add_argument("data_dir", type=str, help="Path to dataset directory")
    args = parser.parse_args()

    try:
        dataset = create_dataset(args.dataset_type, args.data_dir)
        print(f"✓ Dataset created: {len(dataset)} samples")
        sample = dataset[0]
        print(f"✓ Sample keys: {sample.keys()}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
