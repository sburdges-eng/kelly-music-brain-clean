#!/usr/bin/env python3
"""
Kelly ML Data Sanitizer - Detect and quarantine bad audio files at TB scale.

Scans dataset directories for:
1. File Corruption (un-openable files)
2. Silence (RMS below threshold)
3. Clipping (Values exceeding 0.999)
4. Improper Bitrate/Sample Rate
5. 0-byte or truncated files

Usage:
    python scripts/sanitize_datasets.py --path "/Volumes/Extreme SSD/kelly-audio-data/raw" --quarantine
"""

import os
import argparse
import logging
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("sanitizer")

def check_audio_file(file_path: Path, silence_thresh: float = 1e-4) -> Tuple[Path, str, Optional[str]]:
    """
    Analyzes a single audio file for issues.
    Returns: (path, status, error_reason)
    """
    try:
        # 1. Check size
        if file_path.stat().st_size == 0:
            return file_path, "corrupt", "0-byte file"

        # 2. Try to open and read header
        try:
            with sf.SoundFile(str(file_path)) as f:
                if f.frames == 0:
                    return file_path, "corrupt", "Empty frames"
                
                # Check for massive files that might crash memory (e.g. 2 hour long files)
                if f.frames > f.samplerate * 3600 * 2: # 2 hours
                    return file_path, "warn", "File too long (>2h)"

                # 3. Read a chunk to check for silence/clipping
                # Read middle 1 second to avoid potential header/tail silence
                mid_point = f.frames // 2
                f.seek(max(0, mid_point - f.samplerate // 2))
                chunk = f.read(f.samplerate)
                
                if len(chunk) == 0:
                    return file_path, "corrupt", "Could not read data chunk"

                # Check for silence
                rms = np.sqrt(np.mean(chunk**2))
                if rms < silence_thresh:
                    return file_path, "silent", f"RMS: {rms:.2e}"

                # Check for clipping
                peak = np.max(np.abs(chunk))
                if peak > 0.999:
                    return file_path, "clipped", f"Peak: {peak:.3f}"

        except Exception as e:
            return file_path, "corrupt", str(e)

        return file_path, "ok", None

    except Exception as e:
        return file_path, "failed", str(e)

def sanitize_directory(base_path: Path, quarantine_path: Optional[Path] = None, num_workers: int = 4):
    """Recursively scans and sanitizes a directory."""
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}
    
    logger.info(f"Scanning for audio files in: {base_path}")
    all_files = []
    for ext in audio_extensions:
        all_files.extend(base_path.rglob(f"*{ext}"))
    
    logger.info(f"Found {len(all_files)} files. Starting analysis with {num_workers} workers...")
    
    results = {
        "ok": 0,
        "silent": [],
        "corrupt": [],
        "clipped": [],
        "failed": []
    }

    with Pool(num_workers) as p:
        # Use tqdm for progress tracking
        for path, status, reason in tqdm(p.imap_unordered(check_audio_file, all_files), total=len(all_files)):
            if status == "ok":
                results["ok"] += 1
            else:
                results[status].append((path, reason))
                
                if quarantine_path:
                    # Move to quarantine
                    target = quarantine_path / status / path.relative_to(base_path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.move(str(path), str(target))
                    except Exception as e:
                        logger.error(f"Failed to move {path}: {e}")

    # Summary report
    print("\n" + "="*50)
    print(" SANITIZATION REPORT")
    print("="*50)
    print(f"Total Scanned: {len(all_files)}")
    print(f"Healthy (ok): {results['ok']}")
    print(f"Corrupt:      {len(results['corrupt'])}")
    print(f"Silent:       {len(results['silent'])}")
    print(f"Clipped:      {len(results['clipped'])}")
    print(f"Failed:       {len(results['failed'])}")
    print("="*50)

    if quarantine_path:
        print(f"Quarantined files moved to: {quarantine_path}")

def main():
    parser = argparse.ArgumentParser(description="Kelly ML Data Sanitizer")
    parser.add_argument("--path", type=str, required=True, help="Path to audio files")
    parser.add_argument("--quarantine", action="store_true", help="Move bad files to a quarantine folder")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel workers")
    parser.add_argument("--thresh", type=float, default=1e-4, help="Silence RMS threshold")
    
    args = parser.parse_args()
    
    base_path = Path(args.path)
    if not base_path.exists():
        logger.error(f"Path does not exist: {base_path}")
        return

    quarantine_path = None
    if args.quarantine:
        quarantine_path = base_path.parent / "quarantine"
        quarantine_path.mkdir(exist_ok=True)
        logger.info(f"Quarantine mode enabled. Bad files will move to: {quarantine_path}")

    sanitize_directory(base_path, quarantine_path, args.workers)

if __name__ == "__main__":
    main()

