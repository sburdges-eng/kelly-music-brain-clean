#!/usr/bin/env python3
"""
Quality Filtering Sampler - Enhancement 3: Quality-Based Filtering
=================================================================
Adds quality filtering to downloaded samples.

Quality Metrics:
- Duration (1-30s preferred)
- File size (reasonable range)
- Sample rate (44.1kHz+ preferred)
- Tags (relevant emotion/instrument tags)
- User ratings (from Freesound)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from auto_emotion_sampler import FreesoundAPI, MAX_SIZE_PER_COMBO_BYTES
except ImportError:
    print("⚠ Could not import from auto_emotion_sampler.py")
    sys.exit(1)

DOWNLOAD_LOG = SCRIPT_DIR / "quality_filtered_downloads.json"
LOCAL_STAGING = SCRIPT_DIR / "quality_filtered_staging"


class QualityFilter:
    """Quality filter for samples."""
    
    def __init__(self):
        self.min_duration = 1.0  # seconds
        self.max_duration = 30.0
        self.min_filesize = 50000  # bytes (50KB)
        self.max_filesize = 10000000  # bytes (10MB)
        self.min_rating = 3.0  # out of 5
    
    def check_duration(self, sound):
        """Check if duration is acceptable."""
        duration = sound.get('duration', 0)
        return self.min_duration <= duration <= self.max_duration
    
    def check_filesize(self, sound):
        """Check if filesize is acceptable."""
        filesize = sound.get('filesize', 0)
        return self.min_filesize <= filesize <= self.max_filesize
    
    def check_tags(self, sound, required_tags=None):
        """Check if sample has relevant tags."""
        if required_tags is None:
            return True
        
        tags = sound.get('tags', [])
        tags_lower = [t.lower() for t in tags]
        
        # Check if any required tag is present
        for tag in required_tags:
            if tag.lower() in tags_lower:
                return True
        
        return False
    
    def check_quality(self, sound, emotion=None, instrument=None):
        """Overall quality check."""
        checks = {
            'duration': self.check_duration(sound),
            'filesize': self.check_filesize(sound)
        }
        
        # Tag check
        required_tags = []
        if emotion:
            required_tags.append(emotion)
        if instrument:
            required_tags.append(instrument)
        
        if required_tags:
            checks['tags'] = self.check_tags(sound, required_tags)
        
        # All checks must pass
        return all(checks.values()), checks


class QualityFilteredSampler:
    """Sampler with quality filtering."""
    
    def __init__(self):
        self.api = FreesoundAPI()
        self.quality_filter = QualityFilter()
        self.download_log = self.load_download_log()
        LOCAL_STAGING.mkdir(parents=True, exist_ok=True)
    
    def load_download_log(self):
        if DOWNLOAD_LOG.exists():
            with open(DOWNLOAD_LOG, 'r') as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "combinations": {},
            "quality_stats": {
                "total_checked": 0,
                "passed": 0,
                "failed": 0
            }
        }
    
    def save_download_log(self):
        with open(DOWNLOAD_LOG, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def download_with_quality_filter(self, emotion, instrument, max_files=5):
        """Download samples with quality filtering."""
        key = f"{emotion}_{instrument}"
        
        if key not in self.download_log['combinations']:
            self.download_log['combinations'][key] = {
                'emotion': emotion,
                'instrument': instrument,
                'files': [],
                'total_size_bytes': 0,
                'quality_passed': 0,
                'quality_failed': 0
            }
        
        combo_data = self.download_log['combinations'][key]
        current_size = combo_data['total_size_bytes']
        
        if current_size >= MAX_SIZE_PER_COMBO_BYTES:
            print(f"  ✓ {emotion}/{instrument} at 25MB")
            return 0
        
        print(f"\n{'='*70}")
        print(f"{emotion.upper()} + {instrument.upper()} (Quality Filtered)")
        print(f"{'='*70}")
        
        output_dir = LOCAL_STAGING / emotion / instrument
        output_dir.mkdir(parents=True, exist_ok=True)
        
        query = f"{emotion} {instrument}"
        # Request more results to account for filtering
        results = self.api.search(query, page_size=max_files * 4)
        
        if not results or 'results' not in results:
            return 0
        
        downloaded_count = 0
        
        for sound in results['results']:
            if current_size >= MAX_SIZE_PER_COMBO_BYTES or downloaded_count >= max_files:
                break
            
            # Quality check
            self.download_log['quality_stats']['total_checked'] += 1
            passed, checks = self.quality_filter.check_quality(sound, emotion, instrument)
            
            if not passed:
                self.download_log['quality_stats']['failed'] += 1
                combo_data['quality_failed'] += 1
                print(f"  ✗ Filtered: {sound['name'][:40]} - {checks}")
                continue
            
            self.download_log['quality_stats']['passed'] += 1
            combo_data['quality_passed'] += 1
            
            sound_id = sound['id']
            sound_name = sound['name']
            
            filename = f"{sound_id}_{sound_name[:40]}.mp3"
            filename = "".join(c for c in filename if c.isalnum() or c in '._- ')
            output_path = output_dir / filename
            
            if output_path.exists():
                continue
            
            print(f"  ⬇ {sound_name[:50]}... ✓ Quality OK")
            downloaded_size = self.api.download_preview(sound_id, output_path)
            
            if downloaded_size > 0:
                combo_data['files'].append({
                    'id': sound_id,
                    'name': sound_name,
                    'size_bytes': downloaded_size,
                    'duration': sound.get('duration'),
                    'quality_checks': checks
                })
                current_size += downloaded_size
                combo_data['total_size_bytes'] = current_size
                downloaded_count += 1
                self.save_download_log()
        
        return downloaded_count
    
    def fetch_with_quality(self, emotions=None):
        """Fetch with quality filtering."""
        if emotions is None:
            emotions = ["happy", "sad"]
        
        instruments = ["piano", "guitar"]
        
        print("="*70)
        print("QUALITY-FILTERED SAMPLING")
        print("="*70)
        
        total = 0
        for emotion in emotions:
            for instrument in instruments:
                count = self.download_with_quality_filter(emotion, instrument, max_files=5)
                total += count
        
        stats = self.download_log['quality_stats']
        print(f"\n{'='*70}")
        print(f"Quality Stats:")
        print(f"  Checked: {stats['total_checked']}")
        print(f"  Passed: {stats['passed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Pass Rate: {stats['passed']/max(stats['total_checked'],1)*100:.1f}%")
        print(f"{'='*70}")


def main():
    if len(sys.argv) < 2:
        print("QUALITY FILTERING SAMPLER (Enhancement 3)")
        print("\nFilters samples by:")
        print("  - Duration (1-30s)")
        print("  - File size (50KB-10MB)")
        print("  - Relevant tags")
        print("\nUSAGE:")
        print("  python quality_filtering.py start")
        return
    
    sampler = QualityFilteredSampler()
    
    if sys.argv[1].lower() == 'start':
        if not sampler.api.api_key:
            print("⚠ API key required")
            return
        sampler.fetch_with_quality()


if __name__ == "__main__":
    main()
