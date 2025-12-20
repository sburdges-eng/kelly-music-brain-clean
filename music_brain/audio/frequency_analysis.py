"""
8-Band Frequency Analysis - Extract frequency distribution for production insights.

Analyzes audio in standard frequency bands to provide mixing/production guidance:
- Sub-bass (20-60 Hz)
- Bass (60-250 Hz)
- Low-mids (250-500 Hz)
- Mids (500-2000 Hz)
- Upper-mids (2000-4000 Hz)
- Presence (4000-6000 Hz)
- Brilliance (6000-12000 Hz)
- Air (12000-20000 Hz)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# Standard frequency bands for music production
FREQUENCY_BANDS = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mids": (250, 500),
    "mids": (500, 2000),
    "upper_mids": (2000, 4000),
    "presence": (4000, 6000),
    "brilliance": (6000, 12000),
    "air": (12000, 20000),
}


@dataclass
class FrequencyProfile:
    """8-band frequency analysis result."""
    # Energy per band (normalized 0.0-1.0)
    sub_bass: float = 0.0
    bass: float = 0.0
    low_mids: float = 0.0
    mids: float = 0.0
    upper_mids: float = 0.0
    presence: float = 0.0
    brilliance: float = 0.0
    air: float = 0.0
    
    # Overall characteristics
    brightness: float = 0.0  # High freq vs low freq ratio
    warmth: float = 0.0      # Low freq energy
    clarity: float = 0.0     # Mid-range definition
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "bands": {
                "sub_bass": self.sub_bass,
                "bass": self.bass,
                "low_mids": self.low_mids,
                "mids": self.mids,
                "upper_mids": self.upper_mids,
                "presence": self.presence,
                "brilliance": self.brilliance,
                "air": self.air,
            },
            "characteristics": {
                "brightness": self.brightness,
                "warmth": self.warmth,
                "clarity": self.clarity,
            },
        }
    
    def get_production_notes(self) -> List[str]:
        """Generate production mixing notes based on frequency profile."""
        notes = []
        
        # Sub-bass
        if self.sub_bass > 0.7:
            notes.append("Heavy sub-bass - consider high-pass filtering non-bass elements")
        elif self.sub_bass < 0.2:
            notes.append("Light sub-bass - could add weight with bass boost below 60Hz")
        
        # Bass
        if self.bass > 0.8:
            notes.append("Bass-heavy mix - may need reduction in 60-250Hz range")
        elif self.bass < 0.3:
            notes.append("Thin low-end - boost bass in 100-200Hz range")
        
        # Low-mids
        if self.low_mids > 0.7:
            notes.append("Muddy low-mids - cut 250-500Hz to clean up mix")
        
        # Mids
        if self.mids > 0.8:
            notes.append("Mid-range dominant - typical of aggressive rock/metal")
        elif self.mids < 0.3:
            notes.append("Scooped mids - boost 500-2kHz for more presence")
        
        # Upper-mids
        if self.upper_mids > 0.7:
            notes.append("Strong upper-mids - adds attack and definition")
        
        # Presence
        if self.presence > 0.8:
            notes.append("Very present mix - may be fatiguing, consider slight cut")
        elif self.presence < 0.2:
            notes.append("Lacking presence - boost 4-6kHz for clarity")
        
        # Brilliance
        if self.brilliance > 0.7:
            notes.append("Bright mix - adds sparkle and shimmer")
        elif self.brilliance < 0.2:
            notes.append("Dark mix - boost 6-12kHz for air")
        
        # Air
        if self.air > 0.6:
            notes.append("Airy high-end - adds openness and space")
        
        # Overall characteristics
        if self.brightness > 0.7:
            notes.append("Overall: Bright, modern production style")
        elif self.brightness < 0.3:
            notes.append("Overall: Warm, vintage production style")
        
        if self.warmth > 0.7:
            notes.append("Overall: Warm, full low-end")
        
        if self.clarity > 0.7:
            notes.append("Overall: Clear, well-defined midrange")
        elif self.clarity < 0.3:
            notes.append("Overall: Needs midrange definition")
        
        return notes


def analyze_frequency_bands(
    audio_path: str,
    n_fft: int = 4096,
    hop_length: int = 512,
) -> FrequencyProfile:
    """
    Analyze audio file and extract 8-band frequency profile.
    
    Args:
        audio_path: Path to audio file
        n_fft: FFT window size (higher = better freq resolution)
        hop_length: Hop length in samples
    
    Returns:
        FrequencyProfile with normalized band energies
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa required for frequency analysis. "
            "Install with: pip install librosa"
        )
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=None)
    
    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    
    # Compute power spectrogram
    power_spec = magnitude ** 2
    
    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Calculate energy per band
    band_energies = {}
    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        # Find frequency bins in this band
        band_mask = (freqs >= low_freq) & (freqs < high_freq)
        
        # Sum power in this band across all time frames
        if np.any(band_mask):
            band_power = np.mean(power_spec[band_mask, :])
            band_energies[band_name] = float(band_power)
        else:
            band_energies[band_name] = 0.0
    
    # Normalize energies (0-1 scale)
    max_energy = max(band_energies.values()) if band_energies else 1.0
    if max_energy > 0:
        normalized_energies = {
            k: v / max_energy for k, v in band_energies.items()
        }
    else:
        normalized_energies = {k: 0.0 for k in band_energies}
    
    # Calculate overall characteristics
    # Brightness: ratio of high freqs to low freqs
    high_energy = (
        normalized_energies["presence"]
        + normalized_energies["brilliance"]
        + normalized_energies["air"]
    ) / 3
    low_energy = (
        normalized_energies["sub_bass"]
        + normalized_energies["bass"]
        + normalized_energies["low_mids"]
    ) / 3
    brightness = high_energy / (high_energy + low_energy + 0.01)
    
    # Warmth: low frequency content
    warmth = (
        normalized_energies["sub_bass"] * 0.4
        + normalized_energies["bass"] * 0.4
        + normalized_energies["low_mids"] * 0.2
    )
    
    # Clarity: midrange definition
    clarity = (
        normalized_energies["mids"] * 0.5
        + normalized_energies["upper_mids"] * 0.3
        + normalized_energies["presence"] * 0.2
    )
    
    return FrequencyProfile(
        sub_bass=normalized_energies["sub_bass"],
        bass=normalized_energies["bass"],
        low_mids=normalized_energies["low_mids"],
        mids=normalized_energies["mids"],
        upper_mids=normalized_energies["upper_mids"],
        presence=normalized_energies["presence"],
        brilliance=normalized_energies["brilliance"],
        air=normalized_energies["air"],
        brightness=brightness,
        warmth=warmth,
        clarity=clarity,
    )


def compare_frequency_profiles(
    profile1: FrequencyProfile,
    profile2: FrequencyProfile,
) -> Dict[str, float]:
    """
    Compare two frequency profiles.
    
    Returns similarity scores for each band and overall similarity.
    """
    similarities = {}
    
    # Compare each band
    for band in ["sub_bass", "bass", "low_mids", "mids", 
                 "upper_mids", "presence", "brilliance", "air"]:
        val1 = getattr(profile1, band)
        val2 = getattr(profile2, band)
        # Similarity as inverse of absolute difference
        similarity = 1.0 - abs(val1 - val2)
        similarities[band] = similarity
    
    # Compare characteristics
    for char in ["brightness", "warmth", "clarity"]:
        val1 = getattr(profile1, char)
        val2 = getattr(profile2, char)
        similarity = 1.0 - abs(val1 - val2)
        similarities[f"{char}_similarity"] = similarity
    
    # Overall similarity (weighted average)
    band_avg = sum(similarities[b] for b in similarities if "_similarity" not in b) / 8
    char_avg = sum(similarities[c] for c in similarities if "_similarity" in c) / 3
    
    similarities["overall"] = (band_avg * 0.7 + char_avg * 0.3)
    
    return similarities


def suggest_eq_adjustments(
    source: FrequencyProfile,
    target: FrequencyProfile,
) -> List[str]:
    """
    Suggest EQ adjustments to make source sound more like target.
    
    Returns list of EQ suggestions in production-friendly language.
    """
    suggestions = []
    
    # Define frequency ranges for each band
    band_freq_centers = {
        "sub_bass": "40 Hz",
        "bass": "150 Hz",
        "low_mids": "375 Hz",
        "mids": "1 kHz",
        "upper_mids": "3 kHz",
        "presence": "5 kHz",
        "brilliance": "9 kHz",
        "air": "16 kHz",
    }
    
    # Compare each band
    threshold = 0.15  # Only suggest changes >15% difference
    
    for band, freq in band_freq_centers.items():
        source_val = getattr(source, band)
        target_val = getattr(target, band)
        diff = target_val - source_val
        
        if abs(diff) > threshold:
            if diff > 0:
                db_change = f"+{int(diff * 6)} dB"
                suggestions.append(f"Boost {freq} ({band.replace('_', ' ')}): {db_change}")
            else:
                db_change = f"{int(diff * 6)} dB"
                suggestions.append(f"Cut {freq} ({band.replace('_', ' ')}): {db_change}")
    
    if not suggestions:
        suggestions.append("Frequency profiles are very similar - no major EQ needed")
    
    return suggestions
