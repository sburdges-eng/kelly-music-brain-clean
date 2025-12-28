#!/usr/bin/env python3
"""
Extended Auto Emotion Sampler - Enhancement 1: Additional Instruments
====================================================================
Extends the base sampler with 4 additional instruments beyond the original 4.

Original Instruments: piano, guitar, drums, vocals
New Instruments: bass, synth, strings, brass
"""

import json
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from auto_emotion_sampler import FreesoundAPI, EmotionHierarchy, MAX_SIZE_PER_COMBO_BYTES
except ImportError:
    print("⚠ Could not import from auto_emotion_sampler.py")
    sys.exit(1)

# Extended instrument list
EXTENDED_INSTRUMENTS = ["piano", "guitar", "drums", "vocals", "bass", "synth", "strings", "brass"]

DOWNLOAD_LOG = SCRIPT_DIR / "extended_instruments_downloads.json"
LOCAL_STAGING = SCRIPT_DIR / "extended_instruments_staging"


class ExtendedInstrumentSampler:
    """Extended sampler with additional instruments."""
    
    def __init__(self):
        self.api = FreesoundAPI()
        self.hierarchy = EmotionHierarchy()
        self.download_log = self.load_download_log()
        LOCAL_STAGING.mkdir(parents=True, exist_ok=True)
    
    def load_download_log(self):
        if DOWNLOAD_LOG.exists():
            with open(DOWNLOAD_LOG, 'r') as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "combinations": {},
            "instruments": EXTENDED_INSTRUMENTS
        }
    
    def save_download_log(self):
        with open(DOWNLOAD_LOG, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def download_for_combo(self, emotion, instrument, max_files=5):
        """Download samples for emotion-instrument combination."""
        key = f"{emotion}_{instrument}"
        
        if key not in self.download_log['combinations']:
            self.download_log['combinations'][key] = {
                'emotion': emotion,
                'instrument': instrument,
                'files': [],
                'total_size_bytes': 0
            }
        
        combo_data = self.download_log['combinations'][key]
        current_size = combo_data['total_size_bytes']
        
        if current_size >= MAX_SIZE_PER_COMBO_BYTES:
            print(f"  ✓ {emotion}/{instrument} at 25MB")
            return 0
        
        print(f"\n{'='*70}")
        print(f"{emotion.upper()} + {instrument.upper()}")
        print(f"{'='*70}")
        
        output_dir = LOCAL_STAGING / emotion / instrument
        output_dir.mkdir(parents=True, exist_ok=True)
        
        query = f"{emotion} {instrument}"
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
    
    def fetch_extended_instruments(self, emotions=None):
        """Fetch for new instruments only."""
        if emotions is None:
            emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust"]
        
        new_instruments = ["bass", "synth", "strings", "brass"]
        
        print("="*70)
        print("EXTENDED INSTRUMENTS SAMPLING")
        print(f"New Instruments: {', '.join(new_instruments)}")
        print("="*70)
        
        total = 0
        for emotion in emotions:
            for instrument in new_instruments:
                count = self.download_for_combo(emotion, instrument, max_files=5)
                total += count
        
        print(f"\n✓ Downloaded {total} files for extended instruments")


def main():
    if len(sys.argv) < 2:
        print("EXTENDED INSTRUMENTS SAMPLER (Enhancement 1)")
        print("\nAdds: bass, synth, strings, brass")
        print("\nUSAGE:")
        print("  python extended_instruments.py start")
        return
    
    sampler = ExtendedInstrumentSampler()
    
    if sys.argv[1].lower() == 'start':
        if not sampler.api.api_key:
            print("⚠ API key required")
            return
        sampler.fetch_extended_instruments()


if __name__ == "__main__":
    main()
