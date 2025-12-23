#!/usr/bin/env python3
"""
Real Dataset Preparation Script for Kelly ML Training.

Downloads, preprocesses, and prepares real audio datasets for training:
- Emotion recognition datasets
- MIDI melody datasets
- Chord progression datasets
- Groove/timing datasets

All data is stored on: /Volumes/Extreme SSD/kelly-audio-data/

Usage:
    python scripts/prepare_datasets.py --dataset emotion --download
    python scripts/prepare_datasets.py --dataset all --preprocess
    python scripts/prepare_datasets.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Audio data root on external SSD
AUDIO_DATA_ROOT = Path("/Volumes/Extreme SSD/kelly-audio-data")


# =============================================================================
# Dataset Configurations
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    task: str
    description: str
    sources: List[Dict[str, Any]]
    output_dir: str
    sample_rate: int = 16000
    max_duration: float = 10.0
    min_duration: float = 0.5


DATASETS = {
    "emotion_ravdess": DatasetConfig(
        name="RAVDESS",
        task="emotion",
        description="Ryerson Audio-Visual Database of Emotional Speech and Song",
        sources=[
            {
                "type": "kaggle",
                "dataset": "uwrfkaggler/ravdess-emotional-speech-audio",
                "files": ["*.wav"],
            }
        ],
        output_dir="emotions/ravdess",
        sample_rate=16000,
        max_duration=5.0,
    ),
    "emotion_cremad": DatasetConfig(
        name="CREMA-D",
        task="emotion",
        description="Crowd-sourced Emotional Multimodal Actors Dataset",
        sources=[
            {
                "type": "url",
                "url": "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip",
            }
        ],
        output_dir="emotions/cremad",
        sample_rate=16000,
    ),
    "emotion_tess": DatasetConfig(
        name="TESS",
        task="emotion",
        description="Toronto Emotional Speech Set",
        sources=[
            {
                "type": "kaggle",
                "dataset": "ejlok1/toronto-emotional-speech-set-tess",
            }
        ],
        output_dir="emotions/tess",
        sample_rate=16000,
    ),
    "groove_midi": DatasetConfig(
        name="Groove MIDI Dataset",
        task="groove",
        description="Expressive drum performances from Magenta",
        sources=[
            {
                "type": "url",
                "url": "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip",
            }
        ],
        output_dir="grooves/groove_midi",
    ),
    "maestro": DatasetConfig(
        name="MAESTRO",
        task="melody",
        description="MIDI and Audio Edited for Synchronous TRacks and Organization",
        sources=[
            {
                "type": "url",
                "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
            }
        ],
        output_dir="melodies/maestro",
    ),
    "lakh_midi": DatasetConfig(
        name="Lakh MIDI Dataset (Clean)",
        task="harmony",
        description="Clean subset of Lakh MIDI Dataset",
        sources=[
            {
                "type": "url",
                "url": "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
            }
        ],
        output_dir="chord_progressions/lakh",
    ),
    "musicnet": DatasetConfig(
        name="MusicNet",
        task="melody",
        description="Classical music with note annotations",
        sources=[
            {
                "type": "huggingface",
                "dataset": "albertvillanova/musicnet",
            }
        ],
        output_dir="melodies/musicnet",
    ),
    "gtzan": DatasetConfig(
        name="GTZAN",
        task="emotion",
        description="Music genre classification dataset (10 genres)",
        sources=[
            {
                "type": "kaggle",
                "dataset": "andradaolteanu/gtzan-dataset-music-genre-classification",
            }
        ],
        output_dir="emotions/gtzan",
        sample_rate=22050,
        max_duration=30.0,
    ),
}


# =============================================================================
# Download Functions
# =============================================================================

def download_from_url(url: str, output_dir: Path) -> Optional[Path]:
    """Download file from URL."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        logger.error("requests and tqdm required: pip install requests tqdm")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get filename from URL
    filename = url.split("/")[-1]
    output_path = output_dir / filename
    
    if output_path.exists():
        logger.info(f"Already downloaded: {output_path}")
        return output_path
    
    logger.info(f"Downloading: {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(output_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    logger.info(f"Downloaded: {output_path}")
    return output_path


def download_from_kaggle(dataset: str, output_dir: Path) -> bool:
    """Download dataset from Kaggle."""
    try:
        import kaggle
    except ImportError:
        logger.error("kaggle package required: pip install kaggle")
        logger.info("Also set up ~/.kaggle/kaggle.json with your API key")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading from Kaggle: {dataset}")
    
    try:
        kaggle.api.dataset_download_files(
            dataset,
            path=str(output_dir),
            unzip=True,
        )
        logger.info(f"Downloaded to: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        return False


def download_from_huggingface(dataset_name: str, output_dir: Path, split: str = "train") -> bool:
    """Download dataset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package required: pip install datasets")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading from Hugging Face: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        # Save to disk
        dataset.save_to_disk(str(output_dir))
        logger.info(f"Saved to: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Hugging Face download failed: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract zip/tar archive."""
    import tarfile
    import zipfile
    
    logger.info(f"Extracting: {archive_path}")
    
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(output_dir)
        elif archive_path.suffix in [".tar", ".gz", ".tgz"] or ".tar" in archive_path.name:
            mode = "r:gz" if ".gz" in archive_path.name else "r"
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(output_dir)
        else:
            logger.error(f"Unknown archive format: {archive_path}")
            return False
        
        logger.info(f"Extracted to: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_dataset(config: DatasetConfig) -> bool:
    """Download a dataset based on its configuration."""
    downloads_dir = AUDIO_DATA_ROOT / "downloads"
    raw_dir = AUDIO_DATA_ROOT / "raw" / config.output_dir
    
    success = True
    
    for source in config.sources:
        source_type = source.get("type")
        
        if source_type == "url":
            archive_path = download_from_url(source["url"], downloads_dir)
            if archive_path:
                extract_archive(archive_path, raw_dir)
            else:
                success = False
                
        elif source_type == "kaggle":
            if not download_from_kaggle(source["dataset"], raw_dir):
                success = False
                
        elif source_type == "huggingface":
            if not download_from_huggingface(source["dataset"], raw_dir):
                success = False
    
    return success


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_audio_file(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000,
    max_duration: float = 10.0,
    normalize: bool = True,
) -> bool:
    """Preprocess a single audio file."""
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        logger.error("librosa and soundfile required")
        return False
    
    try:
        # Load audio
        y, sr = librosa.load(str(input_path), sr=target_sr, duration=max_duration)
        
        # Normalize
        if normalize:
            y = y / (np.max(np.abs(y)) + 1e-8)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), y, target_sr)
        
        return True
    except Exception as e:
        logger.debug(f"Failed to process {input_path}: {e}")
        return False


def extract_mel_spectrogram(
    audio_path: Path,
    output_path: Path,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> bool:
    """Extract and save mel spectrogram."""
    try:
        import librosa
    except ImportError:
        return False
    
    try:
        y, _ = librosa.load(str(audio_path), sr=sr)
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), mel_spec_db)
        
        return True
    except Exception as e:
        logger.debug(f"Failed to extract mel: {e}")
        return False


def parse_ravdess_filename(filename: str) -> Dict[str, Any]:
    """Parse RAVDESS filename to extract metadata."""
    # Format: 03-01-06-01-02-01-12.wav
    # Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor
    
    parts = filename.replace(".wav", "").split("-")
    if len(parts) != 7:
        return {}
    
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    }
    
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion": emotion_map.get(parts[2], "unknown"),
        "intensity": "normal" if parts[3] == "01" else "strong",
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6],
    }


def preprocess_emotion_dataset(
    input_dir: Path,
    output_dir: Path,
    config: DatasetConfig,
) -> Tuple[int, int]:
    """Preprocess emotion dataset and create metadata."""
    from tqdm import tqdm
    
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    success_count = 0
    fail_count = 0
    
    # Find all audio files
    audio_files = list(input_dir.rglob("*.wav")) + list(input_dir.rglob("*.mp3"))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    for audio_path in tqdm(audio_files, desc="Processing"):
        # Parse filename for metadata
        if "ravdess" in str(input_dir).lower():
            meta = parse_ravdess_filename(audio_path.name)
            emotion = meta.get("emotion", "unknown")
        else:
            # Try to infer emotion from directory structure
            emotion = audio_path.parent.name.lower()
        
        if emotion == "unknown":
            fail_count += 1
            continue
        
        # Create output path
        output_filename = f"{audio_path.stem}.wav"
        emotion_dir = processed_dir / emotion
        output_path = emotion_dir / output_filename
        
        # Process audio
        if preprocess_audio_file(
            audio_path,
            output_path,
            target_sr=config.sample_rate,
            max_duration=config.max_duration,
        ):
            metadata.append({
                "file": str(output_path.relative_to(output_dir)),
                "emotion": emotion,
                "original_file": audio_path.name,
            })
            success_count += 1
            else:
            fail_count += 1
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({"samples": metadata}, f, indent=2)
    
    # Also save as CSV
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "emotion", "original_file"])
        writer.writeheader()
        writer.writerows(metadata)
    
    logger.info(f"Saved metadata: {len(metadata)} samples")
    
    return success_count, fail_count


def preprocess_midi_dataset(
    input_dir: Path,
    output_dir: Path,
    config: DatasetConfig,
) -> Tuple[int, int]:
    """Preprocess MIDI dataset for melody/harmony training."""
    try:
        import mido
    except ImportError:
        logger.error("mido required for MIDI processing: pip install mido")
        return 0, 0
    
    from tqdm import tqdm
    
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    success_count = 0
    fail_count = 0
    
    # Find all MIDI files
    midi_files = list(input_dir.rglob("*.mid")) + list(input_dir.rglob("*.midi"))
    
    logger.info(f"Found {len(midi_files)} MIDI files")
    
    for midi_path in tqdm(midi_files, desc="Processing MIDI"):
        try:
            mid = mido.MidiFile(str(midi_path))
            
            # Extract note sequences
            notes = []
            current_time = 0
            
            for track in mid.tracks:
                for msg in track:
                    current_time += msg.time
                    if msg.type == "note_on" and msg.velocity > 0:
                        notes.append({
                            "time": current_time,
                            "pitch": msg.note,
                            "velocity": msg.velocity,
                            "channel": msg.channel,
                        })
            
            if len(notes) < 10:
                fail_count += 1
                continue
            
            # Save processed notes
            output_path = processed_dir / f"{midi_path.stem}.json"
            with open(output_path, "w") as f:
                json.dump({
                    "notes": notes,
                    "ticks_per_beat": mid.ticks_per_beat,
                    "length": mid.length,
                }, f)
            
            metadata.append({
                "file": str(output_path.relative_to(output_dir)),
                "original_file": midi_path.name,
                "num_notes": len(notes),
                "duration": mid.length,
            })
            success_count += 1
            
        except Exception as e:
            logger.debug(f"Failed to process {midi_path}: {e}")
            fail_count += 1
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({"samples": metadata}, f, indent=2)
    
    return success_count, fail_count


def preprocess_dataset(dataset_name: str) -> bool:
    """Preprocess a downloaded dataset."""
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False
    
    config = DATASETS[dataset_name]
    
    input_dir = AUDIO_DATA_ROOT / "raw" / config.output_dir
    output_dir = AUDIO_DATA_ROOT / "processed" / config.output_dir
    
    if not input_dir.exists():
        logger.error(f"Dataset not downloaded: {input_dir}")
        return False
    
    logger.info(f"Preprocessing: {config.name}")
    
    if config.task == "emotion":
        success, fail = preprocess_emotion_dataset(input_dir, output_dir, config)
    elif config.task in ["melody", "harmony", "groove"]:
        success, fail = preprocess_midi_dataset(input_dir, output_dir, config)
    else:
        logger.error(f"Unknown task type: {config.task}")
        return False
    
    logger.info(f"Preprocessing complete: {success} success, {fail} failed")
    return success > 0


# =============================================================================
# Dataset Statistics
# =============================================================================

def compute_dataset_stats(dataset_name: str) -> Dict[str, Any]:
    """Compute statistics for a processed dataset."""
    if dataset_name not in DATASETS:
        return {}
    
    config = DATASETS[dataset_name]
    processed_dir = AUDIO_DATA_ROOT / "processed" / config.output_dir
    
    if not processed_dir.exists():
        return {"error": "Dataset not processed"}
    
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        return {"error": "Metadata not found"}
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    samples = metadata.get("samples", [])
    
    stats = {
        "name": config.name,
        "task": config.task,
        "total_samples": len(samples),
    }
    
    if config.task == "emotion":
        # Count per emotion
        emotion_counts = {}
        for sample in samples:
            emotion = sample.get("emotion", "unknown")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        stats["emotion_distribution"] = emotion_counts
    
    return stats


# =============================================================================
# Main
# =============================================================================

def list_datasets():
    """List available datasets."""
    print("\n" + "=" * 70)
    print("Available Datasets")
    print("=" * 70)
    
    for name, config in DATASETS.items():
        raw_dir = AUDIO_DATA_ROOT / "raw" / config.output_dir
        processed_dir = AUDIO_DATA_ROOT / "processed" / config.output_dir
        
        status = "❌ Not downloaded"
        if processed_dir.exists():
            status = "✅ Processed"
        elif raw_dir.exists():
            status = "📦 Downloaded (not processed)"
        
        print(f"\n  {name}")
        print(f"    Name: {config.name}")
        print(f"    Task: {config.task}")
        print(f"    Status: {status}")
        print(f"    Description: {config.description}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Kelly ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/prepare_datasets.py --list
    python scripts/prepare_datasets.py --dataset emotion_ravdess --download
    python scripts/prepare_datasets.py --dataset emotion_ravdess --preprocess
    python scripts/prepare_datasets.py --dataset all --download --preprocess
        """,
    )
    
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name (or 'all')")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess dataset")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    
    args = parser.parse_args()
    
    # Check SSD is mounted
    if not AUDIO_DATA_ROOT.parent.exists():
        logger.error(f"External SSD not mounted: {AUDIO_DATA_ROOT.parent}")
        logger.info("Please connect the external SSD and try again")
        sys.exit(1)
    
    # Ensure directories exist
    AUDIO_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIO_DATA_ROOT / "raw").mkdir(exist_ok=True)
    (AUDIO_DATA_ROOT / "processed").mkdir(exist_ok=True)
    (AUDIO_DATA_ROOT / "downloads").mkdir(exist_ok=True)
    
    if args.list:
        list_datasets()
        return
    
    if not args.dataset:
        parser.print_help()
        return
    
    # Get datasets to process
    if args.dataset == "all":
        datasets = list(DATASETS.keys())
    else:
        if args.dataset not in DATASETS:
            logger.error(f"Unknown dataset: {args.dataset}")
            list_datasets()
            sys.exit(1)
        datasets = [args.dataset]
    
    # Process each dataset
    for dataset_name in datasets:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing: {dataset_name}")
        logger.info("=" * 50)
        
        if args.download:
            download_dataset(DATASETS[dataset_name])
        
        if args.preprocess:
            preprocess_dataset(dataset_name)
        
        if args.stats:
            stats = compute_dataset_stats(dataset_name)
            print(f"\nStatistics for {dataset_name}:")
            print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
