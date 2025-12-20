"""
Chord Detection from Audio - Detect chords and progressions from audio files.

Uses chromagram analysis and chord template matching to identify chords
in real-time or from recorded audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path
import math

if TYPE_CHECKING:
    import numpy as np

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    np = None
    LIBROSA_AVAILABLE = False
    np = None  # type: ignore

if TYPE_CHECKING:
    import numpy as np

from music_brain.structure.chord import Chord, ChordProgression


# =================================================================
# CHORD TEMPLATES
# =================================================================

# Pitch class indices: C=0, C#=1, D=2, etc.
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord templates as binary pitch class sets
CHORD_TEMPLATES = {
    # Triads
    'maj': [0, 4, 7],      # Major triad
    'min': [0, 3, 7],      # Minor triad
    'dim': [0, 3, 6],      # Diminished triad
    'aug': [0, 4, 8],      # Augmented triad
    
    # Seventh chords
    'maj7': [0, 4, 7, 11],    # Major 7th
    'min7': [0, 3, 7, 10],    # Minor 7th
    '7': [0, 4, 7, 10],       # Dominant 7th
    'dim7': [0, 3, 6, 9],     # Diminished 7th
    'm7b5': [0, 3, 6, 10],    # Half-diminished
    
    # Suspended
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
}


@dataclass
class ChordDetection:
    """Result of a single chord detection."""
    chord_name: str  # e.g., "Cmaj", "Am7"
    root: str  # e.g., "C", "A"
    quality: str  # e.g., "maj", "min7"
    confidence: float  # 0.0 - 1.0
    start_time: float  # Seconds
    end_time: float  # Seconds
    chroma_vector: List[float] = field(default_factory=list)  # 12 values
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            "chord": self.chord_name,
            "root": self.root,
            "quality": self.quality,
            "confidence": self.confidence,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class ChordProgressionDetection:
    """Result of chord progression detection."""
    chords: List[ChordDetection]
    estimated_key: Optional[str] = None
    confidence: float = 0.0
    
    @property
    def chord_sequence(self) -> List[str]:
        """Get list of chord names."""
        return [c.chord_name for c in self.chords]
    
    @property
    def unique_chords(self) -> List[str]:
        """Get unique chord names in order of first appearance."""
        seen = set()
        unique = []
        for c in self.chords:
            if c.chord_name not in seen:
                seen.add(c.chord_name)
                unique.append(c.chord_name)
        return unique
    
    def to_dict(self) -> Dict:
        return {
            "chords": [c.to_dict() for c in self.chords],
            "sequence": self.chord_sequence,
            "unique_chords": self.unique_chords,
            "estimated_key": self.estimated_key,
            "confidence": self.confidence,
        }


# =================================================================
# CHORD MATCHING
# =================================================================

def _create_chord_template(root: int, intervals: List[int]) -> "np.ndarray":
    """Create a 12-element chroma template for a chord."""
    template = np.zeros(12)
    for interval in intervals:
        template[(root + interval) % 12] = 1.0
    return template / np.sum(template)  # Normalize


def _match_chord(chroma_vector: "np.ndarray") -> Tuple[str, str, float]:
    """
    Match a chroma vector to the best chord template.
    
    Returns:
        Tuple of (root, quality, confidence)
    """
    best_match = None
    best_score = -1.0
    
    # Normalize input
    chroma_norm = chroma_vector / (np.sum(chroma_vector) + 1e-6)
    
    for root_idx in range(12):
        for quality, intervals in CHORD_TEMPLATES.items():
            template = _create_chord_template(root_idx, intervals)
            
            # Calculate correlation
            score = np.dot(chroma_norm, template)
            
            if score > best_score:
                best_score = score
                best_match = (NOTE_NAMES[root_idx], quality, score)
    
    return best_match


def _format_chord_name(root: str, quality: str) -> str:
    """Format chord name for display."""
    if quality == 'maj':
        return root
    elif quality == 'min':
        return f"{root}m"
    elif quality == 'maj7':
        return f"{root}maj7"
    elif quality == 'min7':
        return f"{root}m7"
    elif quality == '7':
        return f"{root}7"
    elif quality == 'dim':
        return f"{root}dim"
    elif quality == 'dim7':
        return f"{root}dim7"
    elif quality == 'm7b5':
        return f"{root}m7b5"
    elif quality == 'aug':
        return f"{root}aug"
    elif quality == 'sus2':
        return f"{root}sus2"
    elif quality == 'sus4':
        return f"{root}sus4"
    else:
        return f"{root}{quality}"


# =================================================================
# CHORD DETECTOR CLASS
# =================================================================

class ChordDetector:
    """
    Detect chords and progressions from audio.
    
    Uses chromagram analysis and template matching for chord identification.
    """
    
    def __init__(
        self,
        hop_length: int = 512,
        window_size: float = 0.5,  # Seconds
        min_confidence: float = 0.3,
    ):
        """
        Initialize chord detector.
        
        Args:
            hop_length: Analysis hop length in samples
            window_size: Chord detection window in seconds
            min_confidence: Minimum confidence threshold for chord detection
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa required for ChordDetector. "
                "Install with: pip install librosa numpy"
            )
        self.hop_length = hop_length
        self.window_size = window_size
        self.min_confidence = min_confidence
    
    def detect_chord(
        self,
        audio_data: "np.ndarray",
        sr: int,
    ) -> Optional[ChordDetection]:
        """
        Detect the most prominent chord in an audio segment.
        
        Args:
            audio_data: Audio samples (mono)
            sr: Sample rate
        
        Returns:
            ChordDetection or None if confidence too low
        """
        # Extract chromagram
        chroma = librosa.feature.chroma_cqt(
            y=audio_data, sr=sr, hop_length=self.hop_length
        )
        
        # Average across time
        chroma_mean = np.mean(chroma, axis=1)
        
        # Match to chord template
        root, quality, confidence = _match_chord(chroma_mean)
        
        if confidence < self.min_confidence:
            return None
        
        chord_name = _format_chord_name(root, quality)
        duration = len(audio_data) / sr
        
        return ChordDetection(
            chord_name=chord_name,
            root=root,
            quality=quality,
            confidence=float(confidence),
            start_time=0.0,
            end_time=duration,
            chroma_vector=chroma_mean.tolist(),
        )
    
    def detect_progression(
        self,
        filepath: str,
        max_duration: Optional[float] = None,
    ) -> ChordProgressionDetection:
        """
        Detect chord progression from an audio file.
        
        Args:
            filepath: Path to audio file
            max_duration: Maximum duration to analyze (seconds)
        
        Returns:
            ChordProgressionDetection with detected chords
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        # Load audio
        y, sr = librosa.load(str(filepath), sr=None, mono=True, duration=max_duration)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        # Calculate frames per window
        frames_per_window = int(self.window_size * sr / self.hop_length)
        
        # Detect chords in windows
        chords = []
        num_windows = chroma.shape[1] // frames_per_window
        
        for i in range(num_windows):
            start_frame = i * frames_per_window
            end_frame = start_frame + frames_per_window
            
            # Get chroma for this window
            window_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
            
            # Match chord
            root, quality, confidence = _match_chord(window_chroma)
            
            if confidence >= self.min_confidence:
                start_time = start_frame * self.hop_length / sr
                end_time = end_frame * self.hop_length / sr
                
                chord_name = _format_chord_name(root, quality)
                
                chords.append(ChordDetection(
                    chord_name=chord_name,
                    root=root,
                    quality=quality,
                    confidence=float(confidence),
                    start_time=start_time,
                    end_time=end_time,
                    chroma_vector=window_chroma.tolist(),
                ))
        
        # Merge consecutive identical chords
        merged_chords = self._merge_consecutive_chords(chords)
        
        # Estimate key from detected chords
        estimated_key = self._estimate_key_from_chords(merged_chords)
        
        # Calculate overall confidence
        if merged_chords:
            avg_confidence = sum(c.confidence for c in merged_chords) / len(merged_chords)
        else:
            avg_confidence = 0.0
        
        return ChordProgressionDetection(
            chords=merged_chords,
            estimated_key=estimated_key,
            confidence=avg_confidence,
        )
    
    def _merge_consecutive_chords(
        self,
        chords: List[ChordDetection]
    ) -> List[ChordDetection]:
        """Merge consecutive identical chords."""
        if not chords:
            return []
        
        merged = [chords[0]]
        
        for chord in chords[1:]:
            if chord.chord_name == merged[-1].chord_name:
                # Extend previous chord
                merged[-1] = ChordDetection(
                    chord_name=merged[-1].chord_name,
                    root=merged[-1].root,
                    quality=merged[-1].quality,
                    confidence=max(merged[-1].confidence, chord.confidence),
                    start_time=merged[-1].start_time,
                    end_time=chord.end_time,
                    chroma_vector=merged[-1].chroma_vector,
                )
            else:
                merged.append(chord)
        
        return merged
    
    def _estimate_key_from_chords(
        self,
        chords: List[ChordDetection]
    ) -> Optional[str]:
        """Estimate the key from detected chords."""
        if not chords:
            return None
        
        # Count chord roots
        root_counts = {}
        for chord in chords:
            root = chord.root
            root_counts[root] = root_counts.get(root, 0) + 1
        
        # Most common root is likely the key
        most_common = max(root_counts.items(), key=lambda x: x[1])
        
        # Check if minor mode is prevalent
        minor_count = sum(1 for c in chords if 'min' in c.quality or 'm' in c.chord_name)
        major_count = len(chords) - minor_count
        
        mode = "minor" if minor_count > major_count else "major"
        
        return f"{most_common[0]} {mode}"
    
    def confidence_score(self, detection: ChordDetection) -> float:
        """Get confidence score for a detection."""
        return detection.confidence


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def detect_chords_from_audio(
    filepath: str,
    window_size: float = 0.5,
    max_duration: Optional[float] = None,
) -> ChordProgressionDetection:
    """
    Convenience function to detect chords from an audio file.
    
    Args:
        filepath: Path to audio file
        window_size: Chord detection window in seconds
        max_duration: Maximum duration to analyze
    
    Returns:
        ChordProgressionDetection
    """
    detector = ChordDetector(window_size=window_size)
    return detector.detect_progression(filepath, max_duration=max_duration)

