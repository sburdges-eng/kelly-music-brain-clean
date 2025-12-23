#!/usr/bin/env python3
"""
Automatic Emotion-Instrument Sampler
Systematically downloads samples following Music-Brain hierarchy:
1. Base Emotions (6): HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST
2. Sub-Emotions (36 total)
3. Sub-Sub-Emotions (216 total)

Instruments: Piano, Guitar, Drums, Vocals
"""

import json
import os
import sys
import requests
import time
from pathlib import Path
from datetime import datetime
import urllib.parse

# Paths
SCRIPT_DIR = Path(__file__).parent
MUSIC_BRAIN_DIR = SCRIPT_DIR / "music_brain"
METADATA_PATH = MUSIC_BRAIN_DIR / "metadata.json"
GDRIVE_ROOT = Path.home() / "sburdges@gmail.com - Google Drive" / "My Drive"
GDRIVE_SAMPLES = GDRIVE_ROOT / "iDAW_Samples" / "Emotion_Instrument_Library"

# Local staging
LOCAL_STAGING = SCRIPT_DIR / "emotion_instrument_staging"

# Config
CONFIG_FILE = SCRIPT_DIR / "freesound_config.json"
DOWNLOAD_LOG = SCRIPT_DIR / "emotion_instrument_downloads.json"

# Target instruments
INSTRUMENTS = ["piano", "guitar", "drums", "vocals"]

# Size limits
MAX_SIZE_PER_COMBO_MB = 25  # 25MB per emotion-instrument combo
MAX_SIZE_PER_COMBO_BYTES = MAX_SIZE_PER_COMBO_MB * 1024 * 1024

class FreesoundAPI:
    """Freesound.org API wrapper"""

    def __init__(self):
        self.api_key = self.load_api_key()
        self.base_url = "https://freesound.org/apiv2"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Token {self.api_key}"})

    def load_api_key(self):
        """Load API key from config"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('freesound_api_key')
        return None

    def search(self, query, instrument=None, page_size=10):
        """Search for sounds with optional instrument filter"""
        if not self.api_key:
            print("⚠ Freesound API key required. Set FREESOUND_API_KEY environment variable or run setup.")
            return None

        # Build search query
        if instrument:
            full_query = f"{query} {instrument}"
        else:
            full_query = query

        params = {
            'query': full_query,
            'page_size': page_size,
            'fields': 'id,name,tags,duration,filesize,type,previews,download,license',
            'filter': 'type:wav duration:[1 TO 30]',  # 1-30 second .wav files
            'sort': 'rating_desc'  # Highest rated first
        }

        try:
            response = self.session.get(f"{self.base_url}/search/text/", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  ✗ API Error: {e}")
            return None

    def download_preview(self, sound_id, output_path):
        """Download sound preview (HQ MP3 - no OAuth needed)"""
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

        except Exception as e:
            print(f"  ✗ Download error: {e}")
            return 0


class EmotionHierarchy:
    """Load and navigate Music-Brain emotion taxonomy"""

    def __init__(self):
        self.metadata = self.load_metadata()
        self.base_emotions = self.load_base_emotions()
        self.emotion_tree = self.build_emotion_tree()

    def load_metadata(self):
        """Load metadata.json"""
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                return json.load(f)
        return None

    def load_base_emotions(self):
        """Load base emotion data"""
        if not self.metadata:
            return []

        base_emotions = []
        for base in self.metadata.get('base_emotions', []):
            emotion_file = MUSIC_BRAIN_DIR / base['file']
            if emotion_file.exists():
                with open(emotion_file, 'r') as f:
                    data = json.load(f)
                    base_emotions.append({
                        'id': base['id'],
                        'name': data['name'],
                        'data': data
                    })

        return base_emotions

    def build_emotion_tree(self):
        """Build hierarchical emotion tree"""
        tree = {
            'base': [],
            'sub': [],
            'sub_sub': []
        }

        for base in self.base_emotions:
            base_name = base['name']
            tree['base'].append(base_name)

            # Extract sub-emotions
            sub_emotions = base['data'].get('sub_emotions', {})
            for sub_name, sub_data in sub_emotions.items():
                tree['sub'].append({
                    'base': base_name,
                    'name': sub_name,
                    'data': sub_data
                })

                # Extract sub-sub-emotions
                sub_sub_emotions = sub_data.get('sub_sub_emotions', {})
                for subsub_name, subsub_data in sub_sub_emotions.items():
                    tree['sub_sub'].append({
                        'base': base_name,
                        'sub': sub_name,
                        'name': subsub_name,
                        'data': subsub_data
                    })

        return tree


class AutoSampler:
    """Automatic emotion-instrument sampler"""

    def __init__(self):
        self.api = FreesoundAPI()
        self.hierarchy = EmotionHierarchy()
        self.download_log = self.load_download_log()

        # Create directories
        LOCAL_STAGING.mkdir(parents=True, exist_ok=True)
        GDRIVE_SAMPLES.mkdir(parents=True, exist_ok=True)

    def load_download_log(self):
        """Load download history"""
        if DOWNLOAD_LOG.exists():
            with open(DOWNLOAD_LOG, 'r') as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "combinations": {},
            "total_size_mb": 0,
            "total_files": 0,
            "last_emotion": None,
            "last_instrument": None
        }

    def save_download_log(self):
        """Save download history"""
        with open(DOWNLOAD_LOG, 'w') as f:
            json.dump(self.download_log, f, indent=2)

    def get_combo_key(self, emotion, instrument):
        """Get combination key"""
        return f"{emotion}_{instrument}"

    def get_combo_size(self, emotion, instrument):
        """Get current size for emotion-instrument combination"""
        key = self.get_combo_key(emotion, instrument)
        combo_data = self.download_log['combinations'].get(key, {})
        return combo_data.get('total_size_bytes', 0)

    def can_download_more(self, emotion, instrument, file_size):
        """Check if we can download more"""
        current_size = self.get_combo_size(emotion, instrument)
        return (current_size + file_size) <= MAX_SIZE_PER_COMBO_BYTES

    def download_for_combo(self, emotion, instrument, level="base", max_files=5):
        """Download samples for emotion-instrument combination"""
        key = self.get_combo_key(emotion, instrument)

        # Initialize tracking
        if key not in self.download_log['combinations']:
            self.download_log['combinations'][key] = {
                'emotion': emotion,
                'instrument': instrument,
                'level': level,
                'total_size_bytes': 0,
                'files': [],
                'last_updated': datetime.now().isoformat()
            }

        combo_data = self.download_log['combinations'][key]
        current_size = combo_data['total_size_bytes']

        if current_size >= MAX_SIZE_PER_COMBO_BYTES:
            print(f"  ✓ {emotion}/{instrument} already at {MAX_SIZE_PER_COMBO_MB}MB")
            return 0

        print(f"\n{'='*70}")
        print(f"[{level.upper()}] {emotion.upper()} + {instrument.upper()}")
        print(f"Progress: {current_size / 1024 / 1024:.2f}MB / {MAX_SIZE_PER_COMBO_MB}MB")
        print(f"{'='*70}")

        # Create output directory
        output_dir = LOCAL_STAGING / level / emotion / instrument
        output_dir.mkdir(parents=True, exist_ok=True)

        # Search
        search_query = emotion.lower()
        results = self.api.search(search_query, instrument=instrument, page_size=max_files * 2)

        if not results or 'results' not in results:
            print(f"  ⚠ No results found")
            return 0

        downloaded_count = 0

        for sound in results['results']:
            if current_size >= MAX_SIZE_PER_COMBO_BYTES:
                break
            if downloaded_count >= max_files:
                break

            sound_id = sound['id']
            sound_name = sound['name']
            file_size = sound.get('filesize', 5 * 1024 * 1024)  # Estimate 5MB if unknown

            # Check if we can download
            if not self.can_download_more(emotion, instrument, file_size):
                continue

            # Create filename
            filename = f"{sound_id}_{sound_name[:40]}.mp3"
            filename = "".join(c for c in filename if c.isalnum() or c in '._- ')
            output_path = output_dir / filename

            if output_path.exists():
                continue

            print(f"  ⬇ {sound_name[:50]}...")
            downloaded_size = self.api.download_preview(sound_id, output_path)

            if downloaded_size > 0:
                # Update tracking
                combo_data['files'].append({
                    'id': sound_id,
                    'name': sound_name,
                    'filename': filename,
                    'size_bytes': downloaded_size,
                    'downloaded': datetime.now().isoformat()
                })
                current_size += downloaded_size
                combo_data['total_size_bytes'] = current_size
                combo_data['last_updated'] = datetime.now().isoformat()

                downloaded_count += 1
                print(f"    ✓ {downloaded_size / 1024 / 1024:.2f}MB (Total: {current_size / 1024 / 1024:.2f}MB)")

                # Save progress
                self.download_log['last_emotion'] = emotion
                self.download_log['last_instrument'] = instrument
                self.save_download_log()

                # Rate limiting
                time.sleep(1)

        return downloaded_count

    def auto_fetch_all(self, files_per_combo=5):
        """Automatically fetch samples in hierarchical order"""
        print("="*70)
        print("AUTOMATIC EMOTION-INSTRUMENT SAMPLING")
        print("="*70)
        print(f"\nHierarchy:")
        print(f"  Base Emotions: {len(self.hierarchy.emotion_tree['base'])}")
        print(f"  Sub-Emotions: {len(self.hierarchy.emotion_tree['sub'])}")
        print(f"  Sub-Sub-Emotions: {len(self.hierarchy.emotion_tree['sub_sub'])}")
        print(f"\nInstruments: {', '.join(INSTRUMENTS)}")
        print(f"Target: {files_per_combo} files × {MAX_SIZE_PER_COMBO_MB}MB per combination")
        print("")

        total_downloaded = 0

        # 1. Base emotions first
        print("\n" + "="*70)
        print("PHASE 1: BASE EMOTIONS")
        print("="*70)

        for base_emotion in self.hierarchy.emotion_tree['base']:
            for instrument in INSTRUMENTS:
                count = self.download_for_combo(base_emotion, instrument, level="base", max_files=files_per_combo)
                total_downloaded += count

        # 2. Sub-emotions
        print("\n" + "="*70)
        print("PHASE 2: SUB-EMOTIONS")
        print("="*70)

        for sub in self.hierarchy.emotion_tree['sub'][:20]:  # First 20 sub-emotions
            sub_name = sub['name']
            for instrument in INSTRUMENTS:
                count = self.download_for_combo(sub_name, instrument, level="sub", max_files=files_per_combo)
                total_downloaded += count

        # 3. Sync to Google Drive
        self.sync_to_gdrive()

        # Final stats
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"Total files downloaded: {total_downloaded}")
        print(f"Combinations filled: {len(self.download_log['combinations'])}")
        self.show_stats()

    def sync_to_gdrive(self):
        """Sync to Google Drive"""
        print(f"\n{'='*70}")
        print("SYNCING TO GOOGLE DRIVE")
        print(f"{'='*70}")

        import shutil

        total_synced = 0
        
        # Use generator expression to flatten the nested directory structure
        # This reduces nesting from 4 levels to 1
        def iter_all_mp3_files():
            """Generator to iterate through all MP3 files in the directory tree"""
            for level_dir in LOCAL_STAGING.iterdir():
                if not level_dir.is_dir():
                    continue
                for emotion_dir in level_dir.iterdir():
                    if not emotion_dir.is_dir():
                        continue
                    for instrument_dir in emotion_dir.iterdir():
                        if not instrument_dir.is_dir():
                            continue
                        for file in instrument_dir.glob("*.mp3"):
                            yield level_dir, emotion_dir, instrument_dir, file
        
        # Process files with single loop
        for level_dir, emotion_dir, instrument_dir, file in iter_all_mp3_files():
            # Create target directory
            target_dir = GDRIVE_SAMPLES / level_dir.name / emotion_dir.name / instrument_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_file = target_dir / file.name
            if not target_file.exists():
                shutil.copy2(file, target_file)
                total_synced += 1

        print(f"✓ Synced {total_synced} files to: {GDRIVE_SAMPLES}")


    def show_stats(self):
        """Show statistics"""
        total_combos = len(self.download_log['combinations'])
        total_size = sum(c['total_size_bytes'] for c in self.download_log['combinations'].values())
        total_files = sum(len(c['files']) for c in self.download_log['combinations'].values())

        print(f"\nStatistics:")
        print(f"  Total Combinations: {total_combos}")
        print(f"  Total Files: {total_files}")
        print(f"  Total Size: {total_size / 1024 / 1024:.2f}MB")
        print(f"  Google Drive: {GDRIVE_SAMPLES}")


def main():
    sampler = AutoSampler()

    if len(sys.argv) < 2:
        print("="*70)
        print("AUTOMATIC EMOTION-INSTRUMENT SAMPLER")
        print("="*70)
        print("\nSystematically downloads samples following Music-Brain hierarchy:")
        print("  1. Base Emotions (6): HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST")
        print("  2. Sub-Emotions (36)")
        print("  3. Instruments: piano, guitar, drums, vocals")
        print("\nUSAGE:")
        print("  ./auto_emotion_sampler.py start         # Start automatic fetching")
        print("  ./auto_emotion_sampler.py stats         # Show statistics")
        print("  ./auto_emotion_sampler.py sync          # Sync to Google Drive")
        print("")
        return

    command = sys.argv[1].lower()

    if command == 'start':
        if not sampler.api.api_key:
            print("\n⚠ Freesound API key required!")
            print("\n1. Get free API key: https://freesound.org/apiv2/apply/")
            print("2. Set environment variable:")
            print("   export FREESOUND_API_KEY='your_key_here'")
            print("\nOr add to freesound_config.json:")
            print('   {"freesound_api_key": "your_key_here"}')
            return

        sampler.auto_fetch_all(files_per_combo=5)

    elif command == 'stats':
        sampler.show_stats()

    elif command == 'sync':
        sampler.sync_to_gdrive()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
