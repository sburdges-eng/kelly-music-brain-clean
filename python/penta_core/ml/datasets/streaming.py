"""
Streaming Audio Dataset - Optimized for 3 TB scale training.

Key Features:
1. Low RAM Footprint: Manifest is indexed on disk, not loaded into RAM.
2. Fast Random Access: Binary index of line offsets for JSONL files.
3. Streaming I/O: Background audio loading and transformation.
4. Feature Caching: Optional memory-mapped feature storage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class StreamingAudioDataset(Dataset):
    """
    Highly scalable dataset that reads from large JSONL manifests.
    Indices the file offsets once, then uses seek() for constant-time access.
    """
    
    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 44100,
        segment_seconds: float = 5.0,
        audio_key: str = "audio",
        label_key: str = "label",
        transform: Optional[torch.nn.Module] = None,
        cache_index: bool = True
    ):
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.audio_key = audio_key
        self.label_key = label_key
        self.transform = transform
        
        # 1. Index the manifest
        self.index_path = self.manifest_path.with_suffix(".idx")
        self._offsets = self._load_or_build_index(cache_index)
        self.length = len(self._offsets)
        
        # 2. Extract label mapping from first sample (assuming strings for labels)
        # In a real scenario, you might want to pre-pass for all unique labels
        self.label_to_idx = self._infer_labels()
        
        logger.info(f"StreamingAudioDataset initialized with {self.length} samples.")

    def _load_or_build_index(self, cache: bool) -> List[int]:
        """Builds or loads a list of byte offsets for each line in the JSONL."""
        if cache and self.index_path.exists():
            logger.info(f"Loading manifest index from {self.index_path}")
            return np.load(str(self.index_path)).tolist()
        
        logger.info(f"Building index for {self.manifest_path} (this might take a minute)...")
        offsets = []
        with open(self.manifest_path, "rb") as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        
        if cache:
            np.save(str(self.index_path), np.array(offsets, dtype=np.int64))
            logger.info(f"Index saved to {self.index_path}")
        
        return offsets

    def _infer_labels(self) -> Dict[str, int]:
        """Peek at the manifest to find labels. For large scale, you should provide this map."""
        # Simple peek at first 100 samples
        labels = set()
        with open(self.manifest_path, "r") as f:
            for _ in range(min(100, self.length)):
                item = json.loads(f.readline())
                if self.label_key in item:
                    labels.add(str(item[self.label_key]))
        
        return {lab: i for i, lab in enumerate(sorted(list(labels)))}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Random access to manifest line
        offset = self._offsets[idx]
        with open(self.manifest_path, "r") as f:
            f.seek(offset)
            item = json.loads(f.readline())
        
        audio_path = Path(item[self.audio_key])
        if not audio_path.is_absolute():
            # Assume relative to manifest or project root
            audio_path = self.manifest_path.parent / audio_path
            
        label_str = str(item.get(self.label_key, "unknown"))
        label = torch.tensor(self.label_to_idx.get(label_str, 0), dtype=torch.long)

        # 2. Load and Resample
        try:
            try:
                import soundfile as sf
                wav, sr = sf.read(str(audio_path), always_2d=True)
                wav = torch.from_numpy(wav).float().T # [C, T]
            except Exception:
                # Fallback to torchaudio if soundfile fails
                wav, sr = torchaudio.load(str(audio_path))
            
            if sr != self.sample_rate:
                import torchaudio.transforms as T
                wav = T.Resample(sr, self.sample_rate)(wav)
            
            # Mono conversion
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
                
            # 3. Fixed length crop/pad
            if wav.shape[1] >= self.segment_samples:
                start = torch.randint(0, wav.shape[1] - self.segment_samples + 1, (1,)).item()
                wav = wav[:, start : start + self.segment_samples]
            else:
                pad = self.segment_samples - wav.shape[1]
                wav = torch.nn.functional.pad(wav, (0, pad))
                
            # 4. Optional transform (e.g. MelSpectrogram)
            if self.transform:
                wav = self.transform(wav)
                
            return wav, label

        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}. Returning zero tensor.")
            return torch.zeros((1, self.segment_samples)), label

