#!/usr/bin/env python3
"""
Emotion-Driven Sound Sampling System
=====================================

Integrates Freesound.org hierarchical search with Music-Brain emotion taxonomy.
Supports fetching real-world audio samples that match emotional states.

Key Features:
- Music-Brain emotion hierarchy (HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST)
- Target instruments (Piano, Guitar, Drums, Vocals)
- Hierarchical emotion search (base, sub, sub-sub emotions)
- Freesound.org API integration for audio sample discovery
- Download tracking and size management

Usage:
    from music_brain.emotion_sampler import EmotionSampler
    
    sampler = EmotionSampler()
    
    # Search for samples
    results = sampler.search_samples("happy", instrument="piano", max_results=5)
    
    # Download samples
    sampler.download_sample(sound_id=12345, emotion="happy", instrument="piano")
"""

from __future__ import annotations

import json
import os
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass


# Default paths - can be overridden via constructor
DEFAULT_CONFIG_FILE = Path.home() / ".kelly" / "freesound_config.json"
DEFAULT_DOWNLOAD_LOG = Path.home() / ".kelly" / "emotion_sampler_downloads.json"
DEFAULT_STAGING_DIR = Path.home() / ".kelly" / "emotion_samples"

# Target instruments aligned with Music-Brain
INSTRUMENTS = ["piano", "guitar", "drums", "vocals"]

# Base emotions aligned with Music-Brain taxonomy
BASE_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "DISGUST"]

# Size limits per emotion-instrument combination
MAX_SIZE_PER_COMBO_MB = 25
MAX_SIZE_PER_COMBO_BYTES = MAX_SIZE_PER_COMBO_MB * 1024 * 1024


@dataclass
class SoundResult:
    """Represents a sound search result from Freesound API."""
    id: int
    name: str
    duration: float
    filesize: int
    tags: List[str]
    preview_url: str
    license: str
    rating: Optional[float] = None


@dataclass
class DownloadedSound:
    """Represents a downloaded sound with metadata."""
    id: int
    name: str
    filename: str
    size_bytes: int
    emotion: str
    instrument: str
    level: str
    downloaded: str


class FreesoundAPI:
    """
    Freesound.org API wrapper for sound search and download.
    
    Requires a Freesound API key. Get one at: https://freesound.org/apiv2/apply/
    """

    def __init__(self, api_key: Optional[str] = None, config_file: Optional[Path] = None):
        """
        Initialize Freesound API client.
        
        Args:
            api_key: Freesound API key. If None, loads from config file.
            config_file: Path to config JSON file with 'freesound_api_key' field.
        """
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self.api_key = api_key or self._load_api_key()
        self.base_url = "https://freesound.org/apiv2"
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({"Authorization": f"Token {self.api_key}"})

    def _load_api_key(self) -> Optional[str]:
        """Load API key from config file or environment."""
        # Try environment variable first
        api_key = os.environ.get('FREESOUND_API_KEY')
        if api_key:
            return api_key
        
        # Try config file
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('freesound_api_key')
            except (json.JSONDecodeError, IOError):
                pass
        
        return None

    def search(
        self,
        query: str,
        instrument: Optional[str] = None,
        page_size: int = 10,
        duration_min: float = 1.0,
        duration_max: float = 30.0,
        file_type: str = "wav",
        sort: str = "rating_desc"
    ) -> List[SoundResult]:
        """
        Search for sounds with optional instrument filter.
        
        Args:
            query: Search query (emotion or description)
            instrument: Optional instrument filter (piano, guitar, drums, vocals)
            page_size: Number of results to return (max 150)
            duration_min: Minimum duration in seconds
            duration_max: Maximum duration in seconds
            file_type: File type filter (wav, mp3, etc.)
            sort: Sort order (rating_desc, downloads_desc, etc.)
        
        Returns:
            List of SoundResult objects
        """
        if not self.api_key:
            raise ValueError(
                "Freesound API key required. Set FREESOUND_API_KEY environment variable "
                "or provide api_key parameter."
            )

        # Build search query
        full_query = f"{query} {instrument}" if instrument else query

        params = {
            'query': full_query,
            'page_size': min(page_size, 150),
            'fields': 'id,name,tags,duration,filesize,type,previews,download,license,avg_rating',
            'filter': f'type:{file_type} duration:[{duration_min} TO {duration_max}]',
            'sort': sort
        }

        try:
            response = self.session.get(f"{self.base_url}/search/text/", params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for sound in data.get('results', []):
                results.append(SoundResult(
                    id=sound['id'],
                    name=sound['name'],
                    duration=sound.get('duration', 0.0),
                    filesize=sound.get('filesize', 0),
                    tags=sound.get('tags', []),
                    preview_url=sound['previews'].get('preview-hq-mp3', ''),
                    license=sound.get('license', 'unknown'),
                    rating=sound.get('avg_rating')
                ))
            
            return results
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Freesound API error: {e}") from e

    def download_preview(self, sound_id: int, output_path: Path) -> int:
        """
        Download sound preview (HQ MP3 - no OAuth needed).
        
        Args:
            sound_id: Freesound sound ID
            output_path: Path to save the downloaded file
        
        Returns:
            Size of downloaded file in bytes
        """
        try:
            # Get sound details
            response = self.session.get(f"{self.base_url}/sounds/{sound_id}/")
            response.raise_for_status()
            sound_data = response.json()

            # Use HQ preview (no OAuth required)
            preview_url = sound_data['previews']['preview-hq-mp3']

            # Download
            download_response = requests.get(preview_url, stream=True)
            download_response.raise_for_status()

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in download_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return output_path.stat().st_size

        except (requests.exceptions.RequestException, IOError, KeyError) as e:
            raise RuntimeError(f"Download error for sound {sound_id}: {e}") from e


class EmotionHierarchy:
    """
    Music-Brain emotion hierarchy navigator.
    
    Provides access to the 6-base emotion taxonomy structure
    without requiring full EmotionThesaurus data files.
    """
    
    BASE_EMOTIONS = BASE_EMOTIONS
    
    def __init__(self):
        """Initialize with default Music-Brain emotion structure."""
        self.base_emotions = self.BASE_EMOTIONS
    
    def get_base_emotions(self) -> List[str]:
        """Get list of base emotions."""
        return self.base_emotions.copy()
    
    def normalize_emotion(self, emotion: str) -> str:
        """
        Normalize emotion name to uppercase base format.
        
        Args:
            emotion: Emotion name (any case)
        
        Returns:
            Normalized emotion name (e.g., "happy" -> "HAPPY")
        """
        emotion_upper = emotion.upper()
        if emotion_upper in self.base_emotions:
            return emotion_upper
        return emotion  # Return as-is if not a base emotion


class EmotionSampler:
    """
    Main interface for emotion-driven sound sampling.
    
    Integrates Freesound API with Music-Brain emotion taxonomy
    to fetch and organize audio samples by emotion and instrument.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_file: Optional[Path] = None,
        download_log: Optional[Path] = None,
        staging_dir: Optional[Path] = None
    ):
        """
        Initialize emotion sampler.
        
        Args:
            api_key: Freesound API key (optional, loads from config if not provided)
            config_file: Path to config file (default: ~/.kelly/freesound_config.json)
            download_log: Path to download log (default: ~/.kelly/emotion_sampler_downloads.json)
            staging_dir: Directory for downloaded samples (default: ~/.kelly/emotion_samples)
        """
        self.api = FreesoundAPI(api_key=api_key, config_file=config_file)
        self.hierarchy = EmotionHierarchy()
        
        self.download_log_path = download_log or DEFAULT_DOWNLOAD_LOG
        self.staging_dir = staging_dir or DEFAULT_STAGING_DIR
        
        self.download_log = self._load_download_log()
        
        # Ensure directories exist
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.download_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_download_log(self) -> Dict[str, Any]:
        """Load download history from JSON file."""
        if self.download_log_path.exists():
            try:
                with open(self.download_log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {
            "created": datetime.now().isoformat(),
            "combinations": {},
            "total_size_mb": 0.0,
            "total_files": 0,
            "last_emotion": None,
            "last_instrument": None
        }

    def _save_download_log(self) -> None:
        """Save download history to JSON file."""
        self.download_log["total_size_mb"] = sum(
            c.get('total_size_bytes', 0) for c in self.download_log['combinations'].values()
        ) / (1024 * 1024)
        self.download_log["total_files"] = sum(
            len(c.get('files', [])) for c in self.download_log['combinations'].values()
        )
        
        with open(self.download_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.download_log, f, indent=2)

    def _get_combo_key(self, emotion: str, instrument: str) -> str:
        """Get combination key for tracking."""
        emotion_norm = self.hierarchy.normalize_emotion(emotion)
        return f"{emotion_norm}_{instrument.lower()}"

    def _get_combo_size(self, emotion: str, instrument: str) -> int:
        """Get current download size for emotion-instrument combination."""
        key = self._get_combo_key(emotion, instrument)
        combo_data = self.download_log['combinations'].get(key, {})
        return combo_data.get('total_size_bytes', 0)

    def can_download_more(self, emotion: str, instrument: str, file_size: int) -> bool:
        """
        Check if more downloads are allowed for this combination.
        
        Args:
            emotion: Emotion name
            instrument: Instrument name
            file_size: Size of file to download in bytes
        
        Returns:
            True if download would not exceed size limit
        """
        current_size = self._get_combo_size(emotion, instrument)
        return (current_size + file_size) <= MAX_SIZE_PER_COMBO_BYTES

    def search_samples(
        self,
        emotion: str,
        instrument: Optional[str] = None,
        max_results: int = 10,
        level: str = "base"
    ) -> List[SoundResult]:
        """
        Search for samples matching emotion and optional instrument.
        
        Args:
            emotion: Emotion name (e.g., "happy", "sad")
            instrument: Optional instrument filter
            max_results: Maximum number of results
            level: Emotion level ("base", "sub", "sub_sub")
        
        Returns:
            List of SoundResult objects
        """
        query = emotion.lower()
        return self.api.search(
            query=query,
            instrument=instrument,
            page_size=max_results
        )

    def download_sample(
        self,
        sound_id: int,
        emotion: str,
        instrument: str,
        level: str = "base",
        sound_name: Optional[str] = None
    ) -> Optional[DownloadedSound]:
        """
        Download a specific sample and track it.
        
        Args:
            sound_id: Freesound sound ID
            emotion: Emotion name
            instrument: Instrument name
            level: Emotion hierarchy level ("base", "sub", "sub_sub")
            sound_name: Optional sound name for filename
        
        Returns:
            DownloadedSound object if successful, None otherwise
        """
        emotion_norm = self.hierarchy.normalize_emotion(emotion)
        key = self._get_combo_key(emotion_norm, instrument)
        
        # Initialize tracking for this combination if needed
        if key not in self.download_log['combinations']:
            self.download_log['combinations'][key] = {
                'emotion': emotion_norm,
                'instrument': instrument.lower(),
                'level': level,
                'total_size_bytes': 0,
                'files': [],
                'last_updated': datetime.now().isoformat()
            }
        
        combo_data = self.download_log['combinations'][key]
        
        # Check if already at size limit
        if combo_data['total_size_bytes'] >= MAX_SIZE_PER_COMBO_BYTES:
            return None
        
        # Create filename
        name_part = sound_name[:40] if sound_name else str(sound_id)
        # Sanitize filename
        name_part = "".join(c for c in name_part if c.isalnum() or c in '._- ')
        filename = f"{sound_id}_{name_part}.mp3"
        
        # Create output path
        output_dir = self.staging_dir / level / emotion_norm / instrument.lower()
        output_path = output_dir / filename
        
        # Skip if already downloaded
        if output_path.exists():
            return None
        
        try:
            # Download
            downloaded_size = self.api.download_preview(sound_id, output_path)
            
            # Update tracking
            downloaded = DownloadedSound(
                id=sound_id,
                name=sound_name or str(sound_id),
                filename=filename,
                size_bytes=downloaded_size,
                emotion=emotion_norm,
                instrument=instrument.lower(),
                level=level,
                downloaded=datetime.now().isoformat()
            )
            
            combo_data['files'].append({
                'id': downloaded.id,
                'name': downloaded.name,
                'filename': downloaded.filename,
                'size_bytes': downloaded.size_bytes,
                'downloaded': downloaded.downloaded
            })
            combo_data['total_size_bytes'] += downloaded_size
            combo_data['last_updated'] = downloaded.downloaded
            
            self.download_log['last_emotion'] = emotion_norm
            self.download_log['last_instrument'] = instrument.lower()
            
            self._save_download_log()
            
            return downloaded
            
        except (requests.exceptions.RequestException, IOError, FileNotFoundError) as e:
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Failed to download sound {sound_id}: {e}") from e

    def fetch_for_combination(
        self,
        emotion: str,
        instrument: str,
        max_files: int = 5,
        level: str = "base"
    ) -> List[DownloadedSound]:
        """
        Fetch multiple samples for an emotion-instrument combination.
        
        Args:
            emotion: Emotion name
            instrument: Instrument name
            max_files: Maximum number of files to download
            level: Emotion hierarchy level
        
        Returns:
            List of successfully downloaded DownloadedSound objects
        """
        # Check current state
        current_size = self._get_combo_size(emotion, instrument)
        if current_size >= MAX_SIZE_PER_COMBO_BYTES:
            return []
        
        # Search for samples
        results = self.search_samples(
            emotion=emotion,
            instrument=instrument,
            max_results=max_files * 2,  # Search for more to account for failures
            level=level
        )
        
        downloaded = []
        for result in results:
            if len(downloaded) >= max_files:
                break
            
            if not self.can_download_more(emotion, instrument, result.filesize):
                break
            
            try:
                sound = self.download_sample(
                    sound_id=result.id,
                    emotion=emotion,
                    instrument=instrument,
                    level=level,
                    sound_name=result.name
                )
                
                if sound:
                    downloaded.append(sound)
                    # Rate limiting
                    time.sleep(1)
                    
            except (requests.exceptions.RequestException, IOError, RuntimeError) as e:
                # Skip failed downloads and continue with next sample
                continue
        
        return downloaded

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get download statistics.
        
        Returns:
            Dictionary with statistics about downloads
        """
        stats = {
            "total_combinations": len(self.download_log['combinations']),
            "total_files": self.download_log['total_files'],
            "total_size_mb": self.download_log['total_size_mb'],
            "combinations": []
        }
        
        for key, combo in self.download_log['combinations'].items():
            stats['combinations'].append({
                "key": key,
                "emotion": combo['emotion'],
                "instrument": combo['instrument'],
                "level": combo['level'],
                "files": len(combo['files']),
                "size_mb": combo['total_size_bytes'] / (1024 * 1024),
                "last_updated": combo.get('last_updated', 'unknown')
            })
        
        return stats
