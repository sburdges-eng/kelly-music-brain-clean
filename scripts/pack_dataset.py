#!/usr/bin/env python3
"""
Audio Dataset Packer - Pack 3 TB of small files into a high-performance LMDB database.

Why? 
Reading 1 million small .wav files from an SSD is 10x slower than reading 
one large indexed database due to filesystem overhead.
"""

import os
import lmdb
import argparse
import logging
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("packer")

def pack_dataset(manifest_path: str, output_db_path: str, map_size_tb: float = 3.0):
    """
    Reads a manifest and packs audio into LMDB.
    """
    manifest_path = Path(manifest_path)
    output_db_path = Path(output_db_path)
    
    # 1. Estimate required map size
    map_size = int(map_size_tb * 1024 * 1024 * 1024 * 1024)
    
    # 2. Create LMDB environment
    env = lmdb.open(str(output_db_path), map_size=map_size)
    
    # 3. Read manifest
    with open(manifest_path, "r") as f:
        lines = f.readlines()
    
    logger.info(f"Packing {len(lines)} files into LMDB at {output_db_path}...")

    # 4. Packing loop
    with env.begin(write=True) as txn:
        for i, line in enumerate(tqdm(lines)):
            try:
                item = json.loads(line)
                audio_path = Path(item["audio"])
                if not audio_path.is_absolute():
                    audio_path = manifest_path.parent / audio_path
                
                # Read raw bytes (packing the original compressed file is more efficient than raw samples)
                with open(audio_path, "rb") as af:
                    audio_bytes = af.read()
                
                # Key: original path or index
                key = str(i).encode('ascii')
                txn.put(key, audio_bytes)
                
                # Store metadata separately if needed
                meta_key = f"meta_{i}".encode('ascii')
                txn.put(meta_key, line.encode('utf-8'))

            except Exception as e:
                logger.error(f"Failed to pack {line}: {e}")

    env.close()
    logger.info("Packing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    pack_dataset(args.manifest, args.output)

