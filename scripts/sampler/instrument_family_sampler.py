#!/usr/bin/env python3
"""
Instrument Family Sampler (Variant 2)
=====================================
Alternative sampler organized by instrument families.

Families:
- Strings (violin, cello, guitar, harp)
- Brass (trumpet, trombone, french-horn, tuba)
- Woodwinds (flute, clarinet, saxophone, oboe)
- Percussion (drums, cymbals, marimba, timpani)
- Keyboards (piano, organ, synthesizer, harpsichord)
- Electronic (synth, sampler, drum-machine, modular)
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
DOWNLOAD_LOG = SCRIPT_DIR / "instrument_family_downloads.json"
LOCAL_STAGING = SCRIPT_DIR / "instrument_family_staging"

INSTRUMENT_FAMILIES = {
    "strings": ["violin", "cello", "guitar", "harp", "bass"],
    "brass": ["trumpet", "trombone", "french-horn", "tuba"],
    "woodwinds": ["flute", "clarinet", "saxophone", "oboe"],
    "percussion": ["drums", "cymbals", "marimba", "timpani"],
    "keyboards": ["piano", "organ", "synthesizer", "harpsichord"],
    "electronic": ["synth", "sampler", "drum-machine", "modular"]
}

PLAYING_STYLES = ["melodic", "rhythmic", "harmonic", "solo", "ensemble"]

MAX_SIZE_PER_COMBO_MB = 25
MAX_SIZE_PER_COMBO_BYTES = MAX_SIZE_PER_COMBO_MB * 1024 * 1024


class InstrumentFamilySampler:
    """Instrument family-based sample acquisition."""
    
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
            "total_files": 0
        }
    
    def save_download_log(self):
        with open(DOWNLOAD_LOG, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def download_for_family(self, family, instrument, style=None, max_files=5):
        """Download samples for family-instrument-style combination."""
        key = f"{family}_{instrument}"
        if style:
            key += f"_{style}"
        
        if key not in self.download_log['combinations']:
            self.download_log['combinations'][key] = {
                'family': family,
                'instrument': instrument,
                'style': style,
                'files': [],
                'total_size_bytes': 0
            }
        
        combo_data = self.download_log['combinations'][key]
        current_size = combo_data['total_size_bytes']
        
        if current_size >= MAX_SIZE_PER_COMBO_BYTES:
            print(f"  ✓ {family}/{instrument} at 25MB")
            return 0
        
        print(f"\n{'='*70}")
        print(f"Family: {family.upper()} | Instrument: {instrument}")
        if style:
            print(f"Style: {style}")
        print(f"{'='*70}")
        
        output_dir = LOCAL_STAGING / family / instrument
        if style:
            output_dir = output_dir / style
        output_dir.mkdir(parents=True, exist_ok=True)
        
        query = f"{instrument}"
        if style:
            query += f" {style}"
        
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
    
    def auto_fetch_all(self, files_per_combo=3):
        """Fetch for all families."""
        print("="*70)
        print("INSTRUMENT FAMILY SAMPLING")
        print("="*70)
        
        total = 0
        for family, instruments in INSTRUMENT_FAMILIES.items():
            for instrument in instruments[:2]:  # First 2 per family
                for style in PLAYING_STYLES[:2]:  # First 2 styles
                    count = self.download_for_family(family, instrument, style, max_files=files_per_combo)
                    total += count
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {total} files")
        print(f"{'='*70}")


def main():
    if len(sys.argv) < 2:
        print("INSTRUMENT FAMILY SAMPLER (Variant 2)")
        print("USAGE: python instrument_family_sampler.py start")
        return
    
    sampler = InstrumentFamilySampler()
    
    if sys.argv[1].lower() == 'start':
        if not sampler.api.api_key:
            print("⚠ API key required")
            return
        sampler.auto_fetch_all()


if __name__ == "__main__":
    main()
