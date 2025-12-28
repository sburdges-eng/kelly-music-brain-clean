#!/usr/bin/env python3
"""
Blend Emotions Sampler - Enhancement 2: Emotional Blends
=======================================================
Samples blend emotions (combinations of base emotions).

Blend Examples:
- Bittersweet (sad + happy)
- Nostalgic (sad + happy + longing)
- Anxious (fear + anticipation)
- Triumphant (happy + proud)
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

# Blend emotion definitions
BLEND_EMOTIONS = {
    "bittersweet": {
        "components": ["sad", "happy"],
        "ratio": [0.6, 0.4],
        "description": "Mixed sadness and joy"
    },
    "nostalgic": {
        "components": ["sad", "happy"],
        "ratio": [0.5, 0.5],
        "description": "Fond remembrance with melancholy"
    },
    "anxious": {
        "components": ["fear", "anticipation"],
        "ratio": [0.7, 0.3],
        "description": "Worried expectation"
    },
    "triumphant": {
        "components": ["happy", "proud"],
        "ratio": [0.6, 0.4],
        "description": "Victory and achievement"
    },
    "melancholic": {
        "components": ["sad", "reflective"],
        "ratio": [0.7, 0.3],
        "description": "Pensive sadness"
    },
    "romantic": {
        "components": ["love", "longing"],
        "ratio": [0.6, 0.4],
        "description": "Passionate affection"
    }
}

DOWNLOAD_LOG = SCRIPT_DIR / "blend_emotions_downloads.json"
LOCAL_STAGING = SCRIPT_DIR / "blend_emotions_staging"


class BlendEmotionSampler:
    """Sampler for blend emotions."""
    
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
            "blends": {},
            "blend_definitions": BLEND_EMOTIONS
        }
    
    def save_download_log(self):
        with open(DOWNLOAD_LOG, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def download_for_blend(self, blend_name, instrument="piano", max_files=5):
        """Download samples for a blend emotion."""
        if blend_name not in BLEND_EMOTIONS:
            print(f"⚠ Unknown blend: {blend_name}")
            return 0
        
        blend = BLEND_EMOTIONS[blend_name]
        key = f"{blend_name}_{instrument}"
        
        if key not in self.download_log['blends']:
            self.download_log['blends'][key] = {
                'blend': blend_name,
                'components': blend['components'],
                'instrument': instrument,
                'files': [],
                'total_size_bytes': 0
            }
        
        combo_data = self.download_log['blends'][key]
        current_size = combo_data['total_size_bytes']
        
        if current_size >= MAX_SIZE_PER_COMBO_BYTES:
            print(f"  ✓ {blend_name}/{instrument} at 25MB")
            return 0
        
        print(f"\n{'='*70}")
        print(f"Blend: {blend_name.upper()} ({' + '.join(blend['components'])})")
        print(f"Instrument: {instrument}")
        print(f"{'='*70}")
        
        output_dir = LOCAL_STAGING / blend_name / instrument
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Search for blend term
        query = f"{blend_name} {instrument} music"
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
    
    def fetch_all_blends(self, instruments=None):
        """Fetch all blend emotions."""
        if instruments is None:
            instruments = ["piano", "guitar"]
        
        print("="*70)
        print("BLEND EMOTIONS SAMPLING")
        print(f"Blends: {', '.join(BLEND_EMOTIONS.keys())}")
        print("="*70)
        
        total = 0
        for blend_name in BLEND_EMOTIONS.keys():
            for instrument in instruments:
                count = self.download_for_blend(blend_name, instrument, max_files=5)
                total += count
        
        print(f"\n✓ Downloaded {total} files for blend emotions")


def main():
    if len(sys.argv) < 2:
        print("BLEND EMOTIONS SAMPLER (Enhancement 2)")
        print("\nBlends:")
        for name, info in BLEND_EMOTIONS.items():
            print(f"  {name}: {info['description']}")
        print("\nUSAGE:")
        print("  python blend_emotions.py start")
        return
    
    sampler = BlendEmotionSampler()
    
    if sys.argv[1].lower() == 'start':
        if not sampler.api.api_key:
            print("⚠ API key required")
            return
        sampler.fetch_all_blends()


if __name__ == "__main__":
    main()
