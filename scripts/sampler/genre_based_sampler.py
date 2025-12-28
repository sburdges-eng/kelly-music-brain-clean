#!/usr/bin/env python3
"""
Genre-Based Sampler (Variant 1)
================================
Alternative sampler that organizes by musical genre rather than emotion.

Genres:
- Jazz (swing, bebop, smooth)
- Rock (classic, alternative, indie)
- Classical (baroque, romantic, contemporary)
- Electronic (ambient, house, techno)
- Folk (acoustic, traditional, contemporary)
- Hip-Hop (boom-bap, trap, lo-fi)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Reuse FreesoundAPI from auto_emotion_sampler
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from auto_emotion_sampler import FreesoundAPI
except ImportError:
    print("⚠ Could not import FreesoundAPI from auto_emotion_sampler.py")
    sys.exit(1)

# Configuration
CONFIG_FILE = SCRIPT_DIR / "freesound_config.json"
DOWNLOAD_LOG = SCRIPT_DIR / "genre_sampler_downloads.json"
LOCAL_STAGING = SCRIPT_DIR / "genre_staging"

# Genre taxonomy
GENRES = {
    "jazz": ["swing", "bebop", "smooth", "fusion"],
    "rock": ["classic", "alternative", "indie", "progressive"],
    "classical": ["baroque", "romantic", "contemporary", "minimalist"],
    "electronic": ["ambient", "house", "techno", "downtempo"],
    "folk": ["acoustic", "traditional", "contemporary", "world"],
    "hiphop": ["boom-bap", "trap", "lo-fi", "experimental"]
}

# Target moods for each genre
GENRE_MOODS = {
    "jazz": ["sophisticated", "relaxed", "energetic", "melancholic"],
    "rock": ["powerful", "rebellious", "emotional", "energetic"],
    "classical": ["elegant", "dramatic", "peaceful", "majestic"],
    "electronic": ["atmospheric", "rhythmic", "hypnotic", "ethereal"],
    "folk": ["warm", "nostalgic", "intimate", "authentic"],
    "hiphop": ["groovy", "intense", "smooth", "creative"]
}

MAX_SIZE_PER_COMBO_MB = 25
MAX_SIZE_PER_COMBO_BYTES = MAX_SIZE_PER_COMBO_MB * 1024 * 1024


class GenreSampler:
    """Genre-based sample acquisition."""
    
    def __init__(self):
        self.api = FreesoundAPI()
        self.download_log = self.load_download_log()
        LOCAL_STAGING.mkdir(parents=True, exist_ok=True)
    
    def load_download_log(self):
        """Load download history."""
        if DOWNLOAD_LOG.exists():
            with open(DOWNLOAD_LOG, 'r') as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "combinations": {},
            "total_files": 0
        }
    
    def save_download_log(self):
        """Save download history."""
        with open(DOWNLOAD_LOG, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def download_for_genre(self, genre, subgenre, mood=None, max_files=5):
        """Download samples for genre-subgenre-mood combination."""
        key = f"{genre}_{subgenre}"
        if mood:
            key += f"_{mood}"
        
        if key not in self.download_log['combinations']:
            self.download_log['combinations'][key] = {
                'genre': genre,
                'subgenre': subgenre,
                'mood': mood,
                'files': [],
                'total_size_bytes': 0
            }
        
        combo_data = self.download_log['combinations'][key]
        current_size = combo_data['total_size_bytes']
        
        if current_size >= MAX_SIZE_PER_COMBO_BYTES:
            print(f"  ✓ {genre}/{subgenre} already at 25MB")
            return 0
        
        print(f"\n{'='*70}")
        print(f"Genre: {genre.upper()} | Subgenre: {subgenre}")
        if mood:
            print(f"Mood: {mood}")
        print(f"Progress: {current_size / 1024 / 1024:.2f}MB / {MAX_SIZE_PER_COMBO_MB}MB")
        print(f"{'='*70}")
        
        # Create output directory
        output_dir = LOCAL_STAGING / genre / subgenre
        if mood:
            output_dir = output_dir / mood
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build search query
        query = f"{genre} {subgenre}"
        if mood:
            query += f" {mood}"
        
        results = self.api.search(query, page_size=max_files * 2)
        
        if not results or 'results' not in results:
            print(f"  ⚠ No results found")
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
                    'filename': filename,
                    'size_bytes': downloaded_size
                })
                current_size += downloaded_size
                combo_data['total_size_bytes'] = current_size
                downloaded_count += 1
                print(f"    ✓ {downloaded_size / 1024 / 1024:.2f}MB")
                self.save_download_log()
        
        return downloaded_count
    
    def auto_fetch_all(self, files_per_combo=5):
        """Fetch samples for all genre combinations."""
        print("="*70)
        print("GENRE-BASED SAMPLING")
        print("="*70)
        print(f"\nGenres: {len(GENRES)}")
        print(f"Target: {files_per_combo} files per combination\n")
        
        total_downloaded = 0
        
        for genre, subgenres in GENRES.items():
            for subgenre in subgenres:
                # Download samples for each mood
                for mood in GENRE_MOODS[genre][:2]:  # First 2 moods per genre
                    count = self.download_for_genre(genre, subgenre, mood, max_files=files_per_combo)
                    total_downloaded += count
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: {total_downloaded} files downloaded")
        print(f"{'='*70}")


def main():
    if len(sys.argv) < 2:
        print("="*70)
        print("GENRE-BASED SAMPLER (Variant 1)")
        print("="*70)
        print("\nOrganizes samples by musical genre instead of emotion")
        print("\nUSAGE:")
        print("  python genre_based_sampler.py start")
        print("  python genre_based_sampler.py stats")
        return
    
    command = sys.argv[1].lower()
    sampler = GenreSampler()
    
    if command == 'start':
        if not sampler.api.api_key:
            print("\n⚠ Freesound API key required!")
            print("Add to freesound_config.json")
            return
        sampler.auto_fetch_all()
    elif command == 'stats':
        total_files = sum(len(c['files']) for c in sampler.download_log.get('combinations', {}).values())
        total_size = sum(c['total_size_bytes'] for c in sampler.download_log.get('combinations', {}).values())
        print(f"\nGenre Sampler Statistics:")
        print(f"  Total combinations: {len(sampler.download_log.get('combinations', {}))}")
        print(f"  Total files: {total_files}")
        print(f"  Total size: {total_size / 1024 / 1024:.2f}MB")


if __name__ == "__main__":
    main()
