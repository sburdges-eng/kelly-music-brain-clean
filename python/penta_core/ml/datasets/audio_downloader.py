"""
Audio Downloader - Download and manage audio datasets.

Downloads audio to /Volumes/Extreme SSD/kelly-audio-data/

Supported sources:
- Freesound (requires API key)
- Direct URLs
- Hugging Face datasets
- Local file copying

Usage:
    from python.penta_core.ml.datasets import AudioDownloader
    
    downloader = AudioDownloader()
    
    # Download from URL
    downloader.download_url(
        "https://example.com/audio.zip",
        extract=True
    )
    
    # Download from Freesound (requires API key)
    downloader.download_freesound_pack("emotion_pack_id")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

from . import get_downloads_dir, get_raw_audio_dir

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    path: Optional[Path] = None
    files_count: int = 0
    total_size_mb: float = 0.0
    error: Optional[str] = None


class AudioDownloader:
    """
    Download and manage audio datasets.
    
    All downloads go to /Volumes/Extreme SSD/kelly-audio-data/downloads/
    """
    
    def __init__(
        self,
        download_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        freesound_api_key: Optional[str] = None,
    ):
        """
        Initialize downloader.
        
        Args:
            download_dir: Directory for downloaded archives (default: SSD/downloads)
            output_dir: Directory for extracted audio (default: SSD/raw)
            freesound_api_key: Freesound API key (or set FREESOUND_API_KEY env var)
        """
        self.download_dir = download_dir or get_downloads_dir()
        self.output_dir = output_dir or get_raw_audio_dir()
        self.freesound_api_key = freesound_api_key or os.environ.get("FREESOUND_API_KEY")
        
        # Ensure directories exist
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AudioDownloader initialized")
        logger.info(f"  Download dir: {self.download_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def download_url(
        self,
        url: str,
        filename: Optional[str] = None,
        extract: bool = True,
        extract_to: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DownloadResult:
        """
        Download file from URL.
        
        Args:
            url: URL to download
            filename: Output filename (default: from URL)
            extract: Extract if archive
            extract_to: Extraction directory (default: output_dir)
            progress_callback: Callback(downloaded_bytes, total_bytes)
        
        Returns:
            DownloadResult
        """
        try:
            import requests
        except ImportError:
            return DownloadResult(
                success=False,
                error="requests not installed: pip install requests"
            )
        
        try:
            # Determine filename
            if not filename:
                parsed = urlparse(url)
                filename = Path(parsed.path).name or "download"
            
            download_path = self.download_dir / filename
            
            logger.info(f"Downloading: {url}")
            logger.info(f"  To: {download_path}")
            
            # Download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)
            
            logger.info(f"Downloaded: {downloaded / 1024 / 1024:.1f} MB")
            
            # Extract if needed
            if extract and self._is_archive(download_path):
                extract_path = extract_to or self.output_dir
                files_count = self._extract_archive(download_path, extract_path)
                
                return DownloadResult(
                    success=True,
                    path=extract_path,
                    files_count=files_count,
                    total_size_mb=downloaded / 1024 / 1024,
                )
            
            return DownloadResult(
                success=True,
                path=download_path,
                files_count=1,
                total_size_mb=downloaded / 1024 / 1024,
            )
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return DownloadResult(success=False, error=str(e))
    
    def download_freesound_pack(
        self,
        pack_id: Union[str, int],
        output_subdir: Optional[str] = None,
        max_files: int = 100,
    ) -> DownloadResult:
        """
        Download a Freesound pack.
        
        Requires FREESOUND_API_KEY environment variable.
        
        Args:
            pack_id: Freesound pack ID
            output_subdir: Subdirectory in output_dir
            max_files: Maximum files to download
        
        Returns:
            DownloadResult
        """
        if not self.freesound_api_key:
            return DownloadResult(
                success=False,
                error="FREESOUND_API_KEY not set. Get one at https://freesound.org/apiv2/apply/"
            )
        
        try:
            import requests
        except ImportError:
            return DownloadResult(
                success=False,
                error="requests not installed: pip install requests"
            )
        
        try:
            # Get pack info
            pack_url = f"https://freesound.org/apiv2/packs/{pack_id}/"
            headers = {"Authorization": f"Token {self.freesound_api_key}"}
            
            response = requests.get(pack_url, headers=headers)
            response.raise_for_status()
            pack_info = response.json()
            
            pack_name = pack_info.get("name", f"pack_{pack_id}")
            logger.info(f"Downloading Freesound pack: {pack_name}")
            
            # Get sounds in pack
            sounds_url = f"https://freesound.org/apiv2/packs/{pack_id}/sounds/"
            response = requests.get(sounds_url, headers=headers, params={"page_size": max_files})
            response.raise_for_status()
            sounds = response.json().get("results", [])
            
            # Create output directory
            output_path = self.output_dir / (output_subdir or pack_name)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Download each sound
            downloaded_count = 0
            total_size = 0
            
            for sound in sounds[:max_files]:
                sound_id = sound["id"]
                sound_name = Path(sound["name"]).name.replace("..", "")
                
                # Get download URL
                sound_url = f"https://freesound.org/apiv2/sounds/{sound_id}/download/"
                response = requests.get(sound_url, headers=headers, allow_redirects=True)
                
                if response.status_code == 200:
                    # Save file
                    file_path = output_path / sound_name
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    
                    downloaded_count += 1
                    total_size += len(response.content)
                    logger.debug(f"  Downloaded: {sound_name}")
            
            logger.info(f"Downloaded {downloaded_count} files ({total_size / 1024 / 1024:.1f} MB)")
            
            return DownloadResult(
                success=True,
                path=output_path,
                files_count=downloaded_count,
                total_size_mb=total_size / 1024 / 1024,
            )
            
        except Exception as e:
            logger.error(f"Freesound download failed: {e}")
            return DownloadResult(success=False, error=str(e))
    
    def download_huggingface_dataset(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "train",
        output_subdir: Optional[str] = None,
    ) -> DownloadResult:
        """
        Download audio dataset from Hugging Face.
        
        Args:
            dataset_name: HF dataset name (e.g., "speech_commands")
            subset: Dataset subset/config
            split: Data split (train, test, validation)
            output_subdir: Subdirectory in output_dir
        
        Returns:
            DownloadResult
        """
        try:
            from datasets import load_dataset
        except ImportError:
            return DownloadResult(
                success=False,
                error="datasets not installed: pip install datasets"
            )
        
        try:
            logger.info(f"Loading HuggingFace dataset: {dataset_name}")
            
            # Load dataset
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # Create output directory
            output_path = self.output_dir / (output_subdir or dataset_name.replace("/", "_"))
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save audio files
            saved_count = 0
            total_size = 0
            
            for i, item in enumerate(dataset):
                if "audio" in item:
                    audio = item["audio"]
                    
                    # Get label if available
                    label = item.get("label", item.get("text", "unknown"))
                    if isinstance(label, int):
                        label = f"class_{label}"
                    label = Path(str(label)).name.replace("..", "")
                    
                    # Create label subdirectory
                    label_dir = output_path / str(label)
                    label_dir.mkdir(exist_ok=True)
                    
                    # Save audio
                    import soundfile as sf
                    
                    file_path = label_dir / f"{i:06d}.wav"
                    sf.write(file_path, audio["array"], audio["sampling_rate"])
                    
                    saved_count += 1
                    total_size += file_path.stat().st_size
            
            logger.info(f"Saved {saved_count} audio files ({total_size / 1024 / 1024:.1f} MB)")
            
            return DownloadResult(
                success=True,
                path=output_path,
                files_count=saved_count,
                total_size_mb=total_size / 1024 / 1024,
            )
            
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return DownloadResult(success=False, error=str(e))
    
    def copy_local_files(
        self,
        source_dir: Union[str, Path],
        output_subdir: Optional[str] = None,
        extensions: List[str] = [".wav", ".mp3", ".flac", ".ogg"],
        preserve_structure: bool = True,
    ) -> DownloadResult:
        """
        Copy local audio files to the data directory.
        
        Args:
            source_dir: Source directory
            output_subdir: Subdirectory in output_dir
            extensions: Audio file extensions to copy
            preserve_structure: Preserve directory structure
        
        Returns:
            DownloadResult
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            return DownloadResult(
                success=False,
                error=f"Source directory not found: {source_path}"
            )
        
        try:
            output_path = self.output_dir / (output_subdir or source_path.name)
            output_path.mkdir(parents=True, exist_ok=True)
            
            copied_count = 0
            total_size = 0
            
            for ext in extensions:
                for src_file in source_path.rglob(f"*{ext}"):
                    if preserve_structure:
                        rel_path = src_file.relative_to(source_path)
                        dst_file = output_path / rel_path
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        dst_file = output_path / src_file.name
                    
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
                    total_size += dst_file.stat().st_size
            
            logger.info(f"Copied {copied_count} files ({total_size / 1024 / 1024:.1f} MB)")
            
            return DownloadResult(
                success=True,
                path=output_path,
                files_count=copied_count,
                total_size_mb=total_size / 1024 / 1024,
            )
            
        except Exception as e:
            logger.error(f"Copy failed: {e}")
            return DownloadResult(success=False, error=str(e))
    
    def _is_archive(self, path: Path) -> bool:
        """Check if file is an archive."""
        return path.suffix.lower() in [".zip", ".tar", ".gz", ".tgz", ".tar.gz"]
    
    def _extract_archive(self, archive_path: Path, extract_to: Path) -> int:
        """Extract archive and return file count."""
        logger.info(f"Extracting: {archive_path}")
        
        files_count = 0
        
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_to)
                files_count = len(zf.namelist())
        
        elif archive_path.name.endswith((".tar.gz", ".tgz", ".tar")):
            mode = "r:gz" if archive_path.name.endswith((".tar.gz", ".tgz")) else "r"
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(extract_to)
                files_count = len(tf.getnames())
        
        elif archive_path.suffix == ".gz":
            # Plain gzip file (single file)
            import gzip
            output_file = extract_to / archive_path.stem
            with gzip.open(archive_path, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            files_count = 1
        
        logger.info(f"Extracted {files_count} files to {extract_to}")
        return files_count


# Convenience function
def download_audio(
    url: str,
    output_subdir: Optional[str] = None,
    extract: bool = True,
) -> DownloadResult:
    """
    Quick download helper.
    
    Downloads to /Volumes/Extreme SSD/kelly-audio-data/
    
    Args:
        url: URL to download
        output_subdir: Subdirectory name
        extract: Extract if archive
    
    Returns:
        DownloadResult
    """
    downloader = AudioDownloader()
    return downloader.download_url(
        url,
        extract=extract,
        extract_to=downloader.output_dir / output_subdir if output_subdir else None,
    )

