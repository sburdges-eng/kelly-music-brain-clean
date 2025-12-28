#!/usr/bin/env python3
"""
Intensity-Focused Sampler (Variant 3)
=====================================
Samples organized by intensity levels across all emotions.

Intensity Levels (6 tiers):
1. Subtle (pp)
2. Mild (p)
3. Moderate (mp)
4. Strong (mf)
5. Intense (f)
6. Overwhelming (ff)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from auto_emotion_sampler import FreesoundAPI
except ImportError:
    print("⚠ Could not import FreesoundAPI")
    sys.exit(1)

CONFIG_FILE = SCRIPT_DIR / "freesound_config.json"
DOWNLOAD_LOG = SCRIPT_DIR / "intensity_downloads.json"
LOCAL_STAGING = SCRIPT_DIR / "intensity_staging"

INTENSITY_TIERS = {
    1: {"name": "subtle", "dynamics": "pp", "modifiers": ["soft", "gentle", "quiet"]},
    2: {"name": "mild", "dynamics": "p", "modifiers": ["light", "calm", "relaxed"]},
    3: {"name": "moderate", "dynamics": "mp", "modifiers": ["balanced", "medium"]},
    4: {"name": "strong", "dynamics": "mf", "modifiers": ["clear", "bold", "present"]},
    5: {"name": "intense", "dynamics": "f", "modifiers": ["powerful", "dramatic"]},
    6: {"name": "overwhelming", "dynamics": "ff", "modifiers": ["massive", "extreme", "epic"]}
}

BASE_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust"]

MAX_SIZE_PER_COMBO_MB = 25
MAX_SIZE_PER_COMBO_BYTES = MAX_SIZE_PER_COMBO_MB * 1024 * 1024


class IntensitySampler:
    """Intensity-focused sample acquisition."""
    
    def __init__(self):
        self.api = FreesoundAPI()
        self.download_log = self.load_download_log()
        LOCAL_STAGING.mkdir(parents=True, exist_ok=True)
    
    def load_download_log(self):
        if DOWNLOAD_LOG.exists():
            with open(DOWNLOAD_LOG, 'r') as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "combinations": {},
            "intensity_coverage": {}
        }
    
    def save_download_log(self):
        with open(DOWNLOAD_LOG, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def download_for_intensity(self, emotion, tier, max_files=5):
        """Download samples for emotion at specific intensity tier."""
        tier_info = INTENSITY_TIERS[tier]
        key = f"{emotion}_tier{tier}"
        
        if key not in self.download_log['combinations']:
            self.download_log['combinations'][key] = {
                'emotion': emotion,
                'tier': tier,
                'tier_name': tier_info['name'],
                'files': [],
                'total_size_bytes': 0
            }
        
        combo_data = self.download_log['combinations'][key]
        current_size = combo_data['total_size_bytes']
        
        if current_size >= MAX_SIZE_PER_COMBO_BYTES:
            return 0
        
        print(f"\n{'='*70}")
        print(f"Emotion: {emotion.upper()} | Tier {tier}: {tier_info['name']} ({tier_info['dynamics']})")
        print(f"{'='*70}")
        
        output_dir = LOCAL_STAGING / emotion / f"tier_{tier}_{tier_info['name']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build intensity-aware query
        modifier = tier_info['modifiers'][0]
        query = f"{emotion} {modifier} music"
        
        results = self.api.search(query, page_size=max_files * 2)
        
        if not results or 'results' not in results:
            return 0
        
        downloaded_count = 0
        
        for sound in results['results']:
            if current_size >= MAX_SIZE_PER_COMBO_BYTES or downloaded_count >= max_files:
                break
            
            sound_id = sound['id']
            sound_name = sound['name']
            
            filename = f"{sound_id}_{sound_name[:40]}.mp3"
            filename = "".join(c for c in filename if c.isalnum() or c in '._- ')
            output_path = output_dir / filename
            
            if output_path.exists():
                continue
            
            print(f"  ⬇ {sound_name[:50]}...")
            downloaded_size = self.api.download_preview(sound_id, output_path)
            
            if downloaded_size > 0:
                combo_data['files'].append({
                    'id': sound_id,
                    'name': sound_name,
                    'size_bytes': downloaded_size
                })
                current_size += downloaded_size
                combo_data['total_size_bytes'] = current_size
                downloaded_count += 1
                self.save_download_log()
        
        return downloaded_count
    
    def auto_fetch_all(self, files_per_tier=3):
        """Fetch across all intensity levels."""
        print("="*70)
        print("INTENSITY-FOCUSED SAMPLING")
        print("="*70)
        
        total = 0
        for emotion in BASE_EMOTIONS:
            for tier in range(1, 7):  # All 6 tiers
                count = self.download_for_intensity(emotion, tier, max_files=files_per_tier)
                total += count
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {total} files across {len(BASE_EMOTIONS)} emotions × 6 tiers")
        print(f"{'='*70}")


def main():
    if len(sys.argv) < 2:
        print("INTENSITY-FOCUSED SAMPLER (Variant 3)")
        print("USAGE: python intensity_sampler.py start")
        return
    
    sampler = IntensitySampler()
    
    if sys.argv[1].lower() == 'start':
        if not sampler.api.api_key:
            print("⚠ API key required")
            return
        sampler.auto_fetch_all()


if __name__ == "__main__":
    main()
