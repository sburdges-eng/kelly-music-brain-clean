"""
Chord Analysis - Detect and analyze chords from MIDI.

Features:
- Chord detection from note clusters
- Key detection
- Roman numeral analysis
- Borrowed chord identification
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from enum import Enum

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


# Note name mappings
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Chord quality definitions (intervals from root)
CHORD_QUALITIES = {
    'maj': (0, 4, 7),
    'min': (0, 3, 7),
    'dim': (0, 3, 6),
    'aug': (0, 4, 8),
    'maj7': (0, 4, 7, 11),
    'min7': (0, 3, 7, 10),
    '7': (0, 4, 7, 10),  # Dominant 7
    'dim7': (0, 3, 6, 9),
    'hdim7': (0, 3, 6, 10),  # Half-diminished
    'sus2': (0, 2, 7),
    'sus4': (0, 5, 7),
    'add9': (0, 4, 7, 14),
    '6': (0, 4, 7, 9),
    'min6': (0, 3, 7, 9),
}

# Major scale degrees for Roman numeral analysis
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # Natural minor

# Diatonic chords in major key
MAJOR_KEY_CHORDS = {
    0: ('I', 'maj'),
    2: ('ii', 'min'),
    4: ('iii', 'min'),
    5: ('IV', 'maj'),
    7: ('V', 'maj'),
    9: ('vi', 'min'),
    11: ('vii°', 'dim'),
}

# Common borrowed chords (from parallel minor in major key)
BORROWED_CHORDS = {
    'bIII': {'source': 'parallel minor', 'interval': 3},
    'bVI': {'source': 'parallel minor', 'interval': 8},
    'bVII': {'source': 'parallel minor/mixolydian', 'interval': 10},
    'iv': {'source': 'parallel minor', 'interval': 5},
    '#IV°': {'source': 'melodic minor', 'interval': 6},
}


class ChordQuality(Enum):
    MAJOR = "maj"
    MINOR = "min"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    DOMINANT7 = "7"
    MAJOR7 = "maj7"
    MINOR7 = "min7"
    HALFDIM7 = "hdim7"
    DIM7 = "dim7"
    SUS2 = "sus2"
    SUS4 = "sus4"


@dataclass
class Chord:
    """Represents a single chord with root and quality."""
    root: int  # MIDI note number (0-11) or note name when created from string
    quality: str  # 'maj', 'min', 'dim', etc.
    bass: Optional[int] = None  # For slash chords
    extensions: List[str] = field(default_factory=list)  # '7', '9', etc.
    
    # Analysis info
    start_tick: int = 0
    duration_ticks: int = 0
    notes: List[int] = field(default_factory=list)  # MIDI pitches
    root_num: int = 0  # Cached pitch class for convenience
    
    def __post_init__(self):
        """
        Normalize root to both string and numeric forms.
        
        When constructed with an integer (analysis paths), we keep the pitch class
        and also expose the note name for human-friendly access. When constructed
        from a chord string, ``root`` may already be a note name.
        """
        if isinstance(self.root, str):
            try:
                self.root_num = NOTE_NAMES.index(self.root)
            except ValueError:
                self.root_num = 0
        else:
            self.root_num = int(self.root) % 12 if self.root is not None else 0
        # Normalize root representation to note name for consistency in tests
        self.root = NOTE_NAMES[self.root_num]
    
    @property
    def name(self) -> str:
        """Get chord name (e.g., 'Am7', 'F#dim')."""
        root_name = self.root if isinstance(self.root, str) else NOTE_NAMES[self.root % 12]
        
        quality_str = ""
        if self.quality == 'maj':
            quality_str = ""  # Implicit
        elif self.quality == 'min':
            quality_str = "m"
        elif self.quality == 'dim':
            quality_str = "dim"
        elif self.quality == 'aug':
            quality_str = "+"
        elif self.quality == '7':
            quality_str = "7"
        elif self.quality == 'maj7':
            quality_str = "maj7"
        elif self.quality == 'min7':
            quality_str = "m7"
        else:
            quality_str = self.quality
        
        ext_str = "".join(self.extensions)
        
        chord_name = f"{root_name}{quality_str}{ext_str}"
        
        if self.bass is not None:
            bass_pc = int(self.bass) % 12
            if bass_pc != self.root_num:
                bass_name = NOTE_NAMES[bass_pc]
                chord_name += f"/{bass_name}"
        
        return chord_name
    
    def get_voicing(self, octave: int = 4, voicing_type: str = "close") -> List[int]:
        """
        Build a simple chord voicing as MIDI note numbers.
        
        Keeps logic lightweight for tests/benchmarks rather than full arranging.
        """
        root_pc = self.root_num % 12 if self.root_num is not None else 0
        base_midi = 12 * (octave + 1) + root_pc  # C4 == 60 when octave=4
        
        quality = (self.quality or "").lower()
        intervals = [0, 4, 7]  # default major triad
        if 'dim' in quality:
            intervals = [0, 3, 6]
        elif 'aug' in quality or quality == '+':
            intervals = [0, 4, 8]
        elif quality.startswith('min'):
            intervals = [0, 3, 7]
        elif quality.startswith('sus2'):
            intervals = [0, 2, 7]
        elif quality.startswith('sus4') or quality.startswith('sus'):
            intervals = [0, 5, 7]
        
        # Seventh quality
        if 'maj7' in quality:
            intervals.append(11)
        elif 'min7' in quality or quality.endswith('m7'):
            intervals.append(10)
        elif quality == '7':
            intervals.append(10)
        
        # Basic handling for common extensions
        for ext in self.extensions:
            if ext in ('9', 'add9'):
                intervals.append(14)
            elif ext == '11':
                intervals.append(17)
            elif ext == '13':
                intervals.append(21)
        
        notes = sorted(base_midi + i for i in intervals)
        
        if voicing_type == "open" and len(notes) >= 3:
            # Spread the middle voice up an octave for a simple open voicing
            notes[1] += 12
            notes = sorted(notes)
        
        return notes
    
    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, chord_str: str, key: str = "") -> "Chord":
        """
        Parse a chord string like \"Am7\" or \"Cmaj9\" into a Chord instance.
        
        Uses the lightweight parser in progression.py to avoid heavy MIDI deps.
        """
        from music_brain.structure.progression import parse_chord

        parsed = parse_chord(chord_str)
        if not parsed:
            raise ValueError(f"Could not parse chord string: {chord_str}")

        bass = None
        if parsed.bass:
            try:
                bass = NOTE_NAMES.index(parsed.bass)
            except ValueError:
                bass = None

        root_value = parsed.root
        if not isinstance(root_value, str):
            try:
                root_value = NOTE_NAMES[int(root_value) % 12]
            except Exception:
                root_value = str(root_value)

        return cls(
            root=root_value,
            quality=parsed.quality,
            bass=bass,
            extensions=parsed.extensions,
            notes=[],
        )

    @classmethod
    def from_pitch_classes(cls, pitch_classes: List[int]) -> "Chord":
        """
        Build a chord from pitch classes (0-11). Intended for quick benchmarks.
        """
        if not pitch_classes:
            raise ValueError("pitch_classes must be non-empty")

        root_pc = pitch_classes[0] % 12
        quality = 'maj'
        intervals = {(pc - root_pc) % 12 for pc in pitch_classes}
        if 3 in intervals and 4 not in intervals:
            quality = 'min'
        elif 4 in intervals and 3 not in intervals:
            quality = 'maj'
        elif 3 in intervals and 4 in intervals:
            quality = 'min7' if 10 in intervals else 'min'
        elif 6 in intervals:
            quality = 'dim'

        return cls(
            root=NOTE_NAMES[root_pc],
            quality=quality,
            extensions=[],
            notes=list(pitch_classes),
        )


@dataclass
class ChordProgression:
    """Complete chord progression with analysis."""
    chords: List[str]  # Chord names
    chord_objects: List[Chord] = field(default_factory=list)
    key: str = "C"
    mode: str = "major"
    roman_numerals: List[str] = field(default_factory=list)
    borrowed_chords: Dict[str, str] = field(default_factory=dict)  # chord -> source
    
    # Source info
    source_file: str = ""
    tempo_bpm: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    
    def __str__(self) -> str:
        return " - ".join(self.chords)


def midi_to_pitch_class(note: int) -> int:
    """Convert MIDI note to pitch class (0-11)."""
    return note % 12


def detect_chord_from_notes(notes: List[int]) -> Optional[Chord]:
    """
    Detect chord from a list of MIDI note numbers.
    
    Uses interval analysis to match against known chord templates.
    """
    if len(notes) < 2:
        return None
    
    # Convert to pitch classes and remove duplicates
    pitch_classes = sorted(set(midi_to_pitch_class(n) for n in notes))
    
    if len(pitch_classes) < 2:
        return None
    
    # Try each pitch class as potential root
    best_match = None
    best_score = 0
    
    for potential_root in pitch_classes:
        # Calculate intervals from this root
        intervals = tuple(sorted((pc - potential_root) % 12 for pc in pitch_classes))
        
        # Try to match against chord templates
        for quality, template in CHORD_QUALITIES.items():
            template_intervals = tuple(sorted(template))
            
            # Check if intervals match (allowing for octave duplications)
            matches = sum(1 for i in intervals if i in template_intervals)
            coverage = matches / len(template_intervals)
            
            if coverage > best_score and coverage >= 0.7:
                best_score = coverage
                best_match = Chord(
                    root=potential_root,
                    quality=quality,
                    notes=notes,
                )
    
    # If no quality match, default to major/minor based on 3rd
    if best_match is None and len(pitch_classes) >= 2:
        root = pitch_classes[0]
        intervals = [(pc - root) % 12 for pc in pitch_classes]
        
        if 3 in intervals:
            quality = 'min'
        elif 4 in intervals:
            quality = 'maj'
        else:
            quality = 'maj'  # Default
        
        best_match = Chord(root=root, quality=quality, notes=notes)
    
    return best_match


def detect_key(chords: List[Chord]) -> Tuple[str, str]:
    """
    Detect key and mode from a list of chords.
    
    Uses chord root distribution and quality analysis.
    
    Returns:
        Tuple of (key_name, mode)
    """
    if not chords:
        return ("C", "major")
    
    def _root_value(chord: Chord) -> int:
        root_val = getattr(chord, "root_num", chord.root)
        if isinstance(root_val, str):
            try:
                root_val = NOTE_NAMES.index(root_val)
            except ValueError:
                root_val = 0
        return int(root_val) % 12
    
    # Count chord roots weighted by position
    root_counts = {}
    for i, chord in enumerate(chords):
        weight = 1.5 if i in [0, len(chords) - 1] else 1.0  # First/last chords weighted more
        root_val = _root_value(chord)
        root_counts[root_val] = root_counts.get(root_val, 0) + weight
    
    # Try each potential key
    best_key = 0
    best_mode = "major"
    best_score = 0
    
    for key in range(12):
        # Test major
        major_score = 0
        for chord in chords:
            root_val = _root_value(chord)
            interval = (root_val - key) % 12
            if interval in MAJOR_SCALE:
                major_score += 1
                # Bonus for tonic and dominant
                if interval == 0:
                    major_score += 0.5
                if interval == 7:
                    major_score += 0.3
        
        if major_score > best_score:
            best_score = major_score
            best_key = key
            best_mode = "major"
        
        # Test minor
        minor_score = 0
        for chord in chords:
            root_val = _root_value(chord)
            interval = (root_val - key) % 12
            if interval in MINOR_SCALE:
                minor_score += 1
                if interval == 0:
                    minor_score += 0.5
        
        if minor_score > best_score:
            best_score = minor_score
            best_key = key
            best_mode = "minor"
    
    return (NOTE_NAMES[best_key], best_mode)


def get_roman_numeral(chord: Chord, key: int, mode: str = "major") -> str:
    """
    Get Roman numeral representation of chord relative to key.
    """
    root_val = getattr(chord, "root_num", chord.root)
    if isinstance(root_val, str):
        try:
            root_val = NOTE_NAMES.index(root_val)
        except ValueError:
            root_val = 0
    interval = (root_val - key) % 12
    
    # Check diatonic chords first
    if mode == "major":
        if interval in MAJOR_KEY_CHORDS:
            numeral, expected_quality = MAJOR_KEY_CHORDS[interval]
            if chord.quality != expected_quality:
                # Quality differs from diatonic - indicate with quality
                if chord.quality == 'min':
                    numeral = numeral.lower()
                elif chord.quality == 'maj' and numeral.islower():
                    numeral = numeral.upper()
            return numeral
    
    # Non-diatonic - use flat/sharp notation
    numeral_map = {
        0: 'I', 1: 'bII', 2: 'II', 3: 'bIII', 4: 'III',
        5: 'IV', 6: '#IV', 7: 'V', 8: 'bVI', 9: 'VI',
        10: 'bVII', 11: 'VII'
    }
    
    numeral = numeral_map.get(interval, '?')
    if chord.quality in ['min', 'min7', 'dim']:
        numeral = numeral.lower()
    
    if chord.quality == 'dim':
        numeral += '°'
    elif chord.quality == '7':
        numeral += '7'
    elif chord.quality == 'maj7':
        numeral += 'M7'
    elif chord.quality == 'min7':
        numeral += '7'
    
    return numeral


def identify_borrowed_chords(chords: List[Chord], key: int, mode: str = "major") -> Dict[str, str]:
    """
    Identify borrowed chords and their sources.
    """
    borrowed = {}
    
    if mode != "major":
        return borrowed  # Only analyze borrowing in major keys for now
    
    for chord in chords:
        root_val = getattr(chord, "root_num", chord.root)
        if isinstance(root_val, str):
            try:
                root_val = NOTE_NAMES.index(root_val)
            except ValueError:
                root_val = 0
        interval = (root_val - key) % 12
        
        # Check common borrowed chord patterns
        if interval == 3 and chord.quality == 'maj':
            borrowed[chord.name] = "parallel minor (bIII)"
        elif interval == 8 and chord.quality == 'maj':
            borrowed[chord.name] = "parallel minor (bVI)"
        elif interval == 10 and chord.quality == 'maj':
            borrowed[chord.name] = "mixolydian/parallel minor (bVII)"
        elif interval == 5 and chord.quality == 'min':
            borrowed[chord.name] = "parallel minor (iv)"
    
    return borrowed


def analyze_chords(midi_path: str, quantize_beats: float = 0.5) -> ChordProgression:
    """
    Analyze chords in a MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        quantize_beats: Quantization window in beats for grouping notes into chords
    
    Returns:
        ChordProgression with detected chords and analysis
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido package required. Install with: pip install mido")
    
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    
    mid = mido.MidiFile(str(midi_path))
    ppq = mid.ticks_per_beat
    
    # Get tempo
    tempo_bpm = 120.0
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_bpm = mido.tempo2bpm(msg.tempo)
                break
    
    # Collect all notes with timing
    all_notes = []
    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                all_notes.append((current_tick, msg.note))
    
    if not all_notes:
        return ChordProgression(chords=[], source_file=str(midi_path), tempo_bpm=tempo_bpm)
    
    # Sort by time
    all_notes.sort(key=lambda x: x[0])
    
    # Group notes into chord windows
    quantize_ticks = int(quantize_beats * ppq)
    chords = []
    
    current_window_start = 0
    current_notes = []
    
    for tick, note in all_notes:
        if tick - current_window_start > quantize_ticks and current_notes:
            # Process current window
            chord = detect_chord_from_notes(current_notes)
            if chord:
                chord.start_tick = current_window_start
                chord.duration_ticks = tick - current_window_start
                chords.append(chord)
            current_notes = []
            current_window_start = tick
        
        current_notes.append(note)
    
    # Process final window
    if current_notes:
        chord = detect_chord_from_notes(current_notes)
        if chord:
            chord.start_tick = current_window_start
            chords.append(chord)
    
    # Remove consecutive duplicates
    unique_chords = []
    for chord in chords:
        if not unique_chords or chord.name != unique_chords[-1].name:
            unique_chords.append(chord)
    
    # Analyze key and mode
    key_name, mode = detect_key(unique_chords)
    key_num = NOTE_NAMES.index(key_name) if key_name in NOTE_NAMES else 0
    
    # Get Roman numerals
    roman_numerals = [get_roman_numeral(c, key_num, mode) for c in unique_chords]
    
    # Identify borrowed chords
    borrowed = identify_borrowed_chords(unique_chords, key_num, mode)
    
    return ChordProgression(
        chords=[c.name for c in unique_chords],
        chord_objects=unique_chords,
        key=key_name,
        mode=mode,
        roman_numerals=roman_numerals,
        borrowed_chords=borrowed,
        source_file=str(midi_path),
        tempo_bpm=tempo_bpm,
    )
