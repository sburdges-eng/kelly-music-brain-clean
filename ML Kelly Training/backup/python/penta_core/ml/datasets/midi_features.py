"""
MIDI Feature Extraction for Kelly ML Training.

Extracts features from MIDI files for training:
- MelodyTransformer: Note sequences, pitch contours
- HarmonyPredictor: Chord progressions, harmonic context
- GroovePredictor: Timing offsets, swing ratios
- DynamicsEngine: Velocity curves, articulation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MIDI Features Dataclass
# =============================================================================


@dataclass
class MIDIFeatures:
    """Extracted features from a MIDI file."""
    
    # File metadata
    file_path: str = ""
    duration_sec: float = 0.0
    tempo_bpm: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    
    # Note-level features
    note_count: int = 0
    pitch_sequence: List[int] = field(default_factory=list)  # MIDI note numbers
    velocity_sequence: List[int] = field(default_factory=list)  # 0-127
    duration_sequence: List[float] = field(default_factory=list)  # seconds
    onset_times: List[float] = field(default_factory=list)  # seconds
    
    # Melodic features (for MelodyTransformer)
    pitch_histogram: List[float] = field(default_factory=list)  # 12 bins (pitch classes)
    pitch_range: int = 0  # max - min pitch
    avg_pitch: float = 0.0
    pitch_std: float = 0.0
    interval_histogram: List[float] = field(default_factory=list)  # Interval distribution
    melodic_contour: List[float] = field(default_factory=list)  # Smoothed pitch curve
    
    # Harmonic features (for HarmonyPredictor)
    chord_sequence: List[str] = field(default_factory=list)  # ["Cmaj", "Am", ...]
    chord_times: List[float] = field(default_factory=list)  # Beat positions
    key_signature: str = ""  # Detected key
    mode: str = ""  # major/minor
    
    # Groove features (for GroovePredictor)
    timing_offsets_ms: List[float] = field(default_factory=list)  # Deviation from grid
    swing_ratio: float = 0.5  # 0.5=straight, 0.67=swing
    groove_type: str = "straight"
    note_density: float = 0.0  # Notes per beat
    syncopation_score: float = 0.0
    
    # Dynamics features (for DynamicsEngine)
    avg_velocity: float = 0.0
    velocity_std: float = 0.0
    velocity_curve: List[float] = field(default_factory=list)  # Smoothed velocity over time
    dynamic_range: float = 0.0  # max - min velocity
    articulation_ratios: Dict[str, float] = field(default_factory=dict)  # legato/staccato %
    
    # Full feature vector (for direct training input)
    feature_vector: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        d = asdict(self)
        d['time_signature'] = list(self.time_signature)
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MIDIFeatures":
        """Create from dict."""
        if 'time_signature' in data and isinstance(data['time_signature'], list):
            data['time_signature'] = tuple(data['time_signature'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Path):
        """Save features to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "MIDIFeatures":
        """Load features from JSON."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# MIDI Feature Extractor
# =============================================================================


class MIDIFeatureExtractor:
    """
    Extracts training features from MIDI files.
    
    Usage:
        extractor = MIDIFeatureExtractor()
        features = extractor.extract("song.mid")
        features.save("song_features.json")
    """
    
    def __init__(
        self,
        quantize_resolution: int = 16,  # 16th notes
        chord_detection: bool = True,
        key_detection: bool = True,
    ):
        self.quantize_resolution = quantize_resolution
        self.chord_detection = chord_detection
        self.key_detection = key_detection
    
    def extract(self, midi_path: Path) -> MIDIFeatures:
        """Extract all features from a MIDI file."""
        try:
            import mido
        except ImportError:
            logger.warning("mido not installed, using fallback MIDI parsing")
            return self._extract_fallback(midi_path)
        
        midi_path = Path(midi_path)
        mid = mido.MidiFile(str(midi_path))
        
        features = MIDIFeatures(file_path=str(midi_path))
        
        # Parse MIDI
        notes = self._parse_notes(mid)
        
        if not notes:
            logger.warning(f"No notes found in {midi_path}")
            return features
        
        # Basic metadata
        features.duration_sec = mid.length
        features.tempo_bpm = self._get_tempo(mid)
        features.time_signature = self._get_time_signature(mid)
        features.note_count = len(notes)
        
        # Extract sequences
        features.pitch_sequence = [n['pitch'] for n in notes]
        features.velocity_sequence = [n['velocity'] for n in notes]
        features.duration_sequence = [n['duration'] for n in notes]
        features.onset_times = [n['onset'] for n in notes]
        
        # Melodic features
        self._extract_melodic_features(features, notes)
        
        # Harmonic features
        if self.chord_detection:
            self._extract_harmonic_features(features, notes)
        
        # Groove features
        self._extract_groove_features(features, notes)
        
        # Dynamics features
        self._extract_dynamics_features(features, notes)
        
        # Build feature vector
        features.feature_vector = self._build_feature_vector(features)
        
        return features
    
    def _parse_notes(self, mid) -> List[Dict[str, Any]]:
        """Parse MIDI into list of notes."""
        import mido
        
        notes = []
        current_time = 0
        active_notes = {}  # pitch -> (onset, velocity)
        
        for msg in mido.merge_tracks(mid.tracks):
            current_time += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_time, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    onset, velocity = active_notes.pop(msg.note)
                    notes.append({
                        'pitch': msg.note,
                        'velocity': velocity,
                        'onset': onset,
                        'duration': current_time - onset,
                        'channel': msg.channel,
                    })
        
        # Sort by onset time
        notes.sort(key=lambda n: n['onset'])
        return notes
    
    def _get_tempo(self, mid) -> float:
        """Extract tempo from MIDI."""
        import mido
        
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    return mido.tempo2bpm(msg.tempo)
        return 120.0  # Default
    
    def _get_time_signature(self, mid) -> Tuple[int, int]:
        """Extract time signature from MIDI."""
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'time_signature':
                    return (msg.numerator, msg.denominator)
        return (4, 4)  # Default
    
    def _extract_melodic_features(self, features: MIDIFeatures, notes: List[Dict]):
        """Extract melodic features."""
        pitches = np.array([n['pitch'] for n in notes])
        
        if len(pitches) == 0:
            return
        
        # Pitch histogram (12 pitch classes)
        pitch_classes = pitches % 12
        features.pitch_histogram = np.bincount(pitch_classes, minlength=12).astype(float).tolist()
        total = sum(features.pitch_histogram)
        if total > 0:
            features.pitch_histogram = [p / total for p in features.pitch_histogram]
        
        # Pitch range and statistics
        features.pitch_range = int(np.max(pitches) - np.min(pitches))
        features.avg_pitch = float(np.mean(pitches))
        features.pitch_std = float(np.std(pitches))
        
        # Interval histogram
        if len(pitches) > 1:
            intervals = np.diff(pitches)
            # Bin intervals from -12 to +12 (25 bins)
            interval_bins = np.clip(intervals, -12, 12) + 12
            features.interval_histogram = np.bincount(interval_bins.astype(int), minlength=25).astype(float).tolist()
            total = sum(features.interval_histogram)
            if total > 0:
                features.interval_histogram = [i / total for i in features.interval_histogram]
        
        # Melodic contour (smoothed pitch over time)
        if len(notes) > 10:
            onsets = np.array([n['onset'] for n in notes])
            # Normalize to 0-1
            if onsets[-1] > 0:
                onsets = onsets / onsets[-1]
            # Resample to 32 points
            from scipy import interpolate
            try:
                f = interpolate.interp1d(onsets, pitches, kind='linear', fill_value='extrapolate')
                x_new = np.linspace(0, 1, 32)
                features.melodic_contour = f(x_new).tolist()
            except Exception:
                features.melodic_contour = pitches[:32].tolist()
    
    def _extract_harmonic_features(self, features: MIDIFeatures, notes: List[Dict]):
        """Extract harmonic features (chord detection)."""
        # Simple chord detection based on simultaneous notes
        features.chord_sequence = []
        features.chord_times = []
        
        # Group notes by beat
        if not notes:
            return
        
        beat_duration = 60.0 / features.tempo_bpm if features.tempo_bpm > 0 else 0.5
        
        # Collect pitch classes per beat
        beat_notes: Dict[int, List[int]] = {}
        for note in notes:
            beat_idx = int(note['onset'] / beat_duration)
            if beat_idx not in beat_notes:
                beat_notes[beat_idx] = []
            beat_notes[beat_idx].append(note['pitch'] % 12)
        
        # Detect chords
        for beat_idx in sorted(beat_notes.keys()):
            pitch_classes = set(beat_notes[beat_idx])
            chord = self._detect_chord(pitch_classes)
            if chord and (not features.chord_sequence or chord != features.chord_sequence[-1]):
                features.chord_sequence.append(chord)
                features.chord_times.append(beat_idx * beat_duration)
        
        # Key detection
        if self.key_detection:
            features.key_signature, features.mode = self._detect_key(features.pitch_histogram)
    
    def _detect_chord(self, pitch_classes: set) -> str:
        """Detect chord from pitch classes."""
        if len(pitch_classes) < 3:
            return ""
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Major chord patterns
        major_pattern = {0, 4, 7}
        minor_pattern = {0, 3, 7}
        dim_pattern = {0, 3, 6}
        aug_pattern = {0, 4, 8}
        
        for root in range(12):
            shifted = {(p - root) % 12 for p in pitch_classes}
            
            if major_pattern.issubset(shifted):
                return f"{note_names[root]}maj"
            if minor_pattern.issubset(shifted):
                return f"{note_names[root]}m"
            if dim_pattern.issubset(shifted):
                return f"{note_names[root]}dim"
            if aug_pattern.issubset(shifted):
                return f"{note_names[root]}aug"
        
        return ""
    
    def _detect_key(self, pitch_histogram: List[float]) -> Tuple[str, str]:
        """Detect key signature from pitch histogram."""
        if not pitch_histogram or sum(pitch_histogram) == 0:
            return "", ""
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Krumhansl-Schmuckler key profiles
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        best_corr = -1
        best_key = "C"
        best_mode = "major"
        
        hist = np.array(pitch_histogram)
        if hist.std() == 0:
            return best_key, best_mode
        
        for i in range(12):
            # Rotate profiles
            major_rot = np.roll(major_profile, i)
            minor_rot = np.roll(minor_profile, i)
            
            # Correlation
            major_corr = np.corrcoef(hist, major_rot)[0, 1]
            minor_corr = np.corrcoef(hist, minor_rot)[0, 1]

            if np.isnan(major_corr):
                major_corr = -np.inf
            if np.isnan(minor_corr):
                minor_corr = -np.inf
            
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = note_names[i]
                best_mode = "major"
            
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = note_names[i]
                best_mode = "minor"
        
        return best_key, best_mode
    
    def _extract_groove_features(self, features: MIDIFeatures, notes: List[Dict]):
        """Extract groove/timing features."""
        if not notes:
            return
        
        beat_duration = 60.0 / features.tempo_bpm if features.tempo_bpm > 0 else 0.5
        
        # Calculate timing offsets from grid
        offsets = []
        for note in notes:
            # Quantize to 16th notes
            grid_position = round(note['onset'] / (beat_duration / 4)) * (beat_duration / 4)
            offset_ms = (note['onset'] - grid_position) * 1000
            offsets.append(offset_ms)
        
        features.timing_offsets_ms = offsets
        
        # Detect swing ratio
        # Look at 8th note pairs (positions 0 and 0.5 of beat)
        downbeat_offsets = []
        upbeat_offsets = []
        
        for note in notes:
            beat_pos = (note['onset'] / beat_duration) % 1
            if beat_pos < 0.1 or beat_pos > 0.9:
                downbeat_offsets.append(note['onset'])
            elif 0.4 < beat_pos < 0.6:
                upbeat_offsets.append(note['onset'])
        
        # Estimate swing from upbeat timing
        if len(downbeat_offsets) > 2 and len(upbeat_offsets) > 2:
            # In swing, upbeats are pushed later (ratio > 0.5)
            # This is a simplified estimation
            avg_upbeat_pos = np.mean([(u / beat_duration) % 1 for u in upbeat_offsets[:10]])
            features.swing_ratio = max(0.5, min(0.67, avg_upbeat_pos))
        
        # Classify groove type
        if features.swing_ratio > 0.58:
            features.groove_type = "swing"
        else:
            avg_offset = np.mean(np.abs(offsets)) if offsets else 0
            if avg_offset > 20:
                features.groove_type = "humanized"
            else:
                features.groove_type = "straight"
        
        # Note density (notes per beat)
        features.note_density = len(notes) / (features.duration_sec / beat_duration) if features.duration_sec > 0 else 0
        
        # Syncopation score (notes on off-beats)
        offbeat_count = 0
        for note in notes:
            beat_pos = (note['onset'] / beat_duration) % 1
            if 0.2 < beat_pos < 0.4 or 0.7 < beat_pos < 0.9:
                offbeat_count += 1
        features.syncopation_score = offbeat_count / len(notes) if notes else 0
    
    def _extract_dynamics_features(self, features: MIDIFeatures, notes: List[Dict]):
        """Extract dynamics features."""
        if not notes:
            return
        
        velocities = np.array([n['velocity'] for n in notes])
        durations = np.array([n['duration'] for n in notes])
        
        # Velocity statistics
        features.avg_velocity = float(np.mean(velocities))
        features.velocity_std = float(np.std(velocities))
        features.dynamic_range = float(np.max(velocities) - np.min(velocities))
        
        # Velocity curve over time (smoothed)
        if len(notes) > 10:
            onsets = np.array([n['onset'] for n in notes])
            # Normalize time to 0-1
            if onsets[-1] > 0:
                norm_onsets = onsets / onsets[-1]
            else:
                norm_onsets = onsets
            
            # Resample to 32 points
            try:
                from scipy import interpolate
                f = interpolate.interp1d(norm_onsets, velocities, kind='linear', fill_value='extrapolate')
                x_new = np.linspace(0, 1, 32)
                features.velocity_curve = f(x_new).tolist()
            except Exception:
                features.velocity_curve = velocities[:32].tolist()
        
        # Articulation ratios
        beat_duration = 60.0 / features.tempo_bpm if features.tempo_bpm > 0 else 0.5
        
        legato_count = 0
        staccato_count = 0
        normal_count = 0
        
        for i, note in enumerate(notes):
            # Legato: duration > 80% of inter-note interval
            # Staccato: duration < 30% of expected duration
            if i < len(notes) - 1:
                inter_note = notes[i + 1]['onset'] - note['onset']
                if inter_note > 0:
                    ratio = note['duration'] / inter_note
                    if ratio > 0.8:
                        legato_count += 1
                    elif ratio < 0.3:
                        staccato_count += 1
                    else:
                        normal_count += 1
        
        total = legato_count + staccato_count + normal_count
        if total > 0:
            features.articulation_ratios = {
                'legato': legato_count / total,
                'staccato': staccato_count / total,
                'normal': normal_count / total,
            }
    
    def _build_feature_vector(self, features: MIDIFeatures) -> List[float]:
        """Build a combined feature vector for ML input."""
        vec = []
        
        # Pitch features (12 + 4 = 16)
        vec.extend(features.pitch_histogram[:12] if features.pitch_histogram else [0] * 12)
        vec.append(features.avg_pitch / 127 if features.avg_pitch else 0)
        vec.append(features.pitch_std / 20 if features.pitch_std else 0)
        vec.append(features.pitch_range / 48 if features.pitch_range else 0)
        vec.append(features.note_density / 8 if features.note_density else 0)
        
        # Interval features (25)
        if features.interval_histogram:
            vec.extend(features.interval_histogram[:25])
        else:
            vec.extend([0] * 25)
        
        # Groove features (6)
        vec.append(features.swing_ratio)
        vec.append(features.syncopation_score)
        vec.append(1.0 if features.groove_type == 'swing' else 0.0)
        vec.append(1.0 if features.groove_type == 'straight' else 0.0)
        vec.append(features.tempo_bpm / 200 if features.tempo_bpm else 0.5)
        vec.append(np.mean(np.abs(features.timing_offsets_ms)) / 50 if features.timing_offsets_ms else 0)
        
        # Dynamics features (6)
        vec.append(features.avg_velocity / 127)
        vec.append(features.velocity_std / 40)
        vec.append(features.dynamic_range / 127)
        vec.append(features.articulation_ratios.get('legato', 0))
        vec.append(features.articulation_ratios.get('staccato', 0))
        vec.append(features.articulation_ratios.get('normal', 0))
        
        # Melodic contour (32) - normalized
        if features.melodic_contour:
            contour = features.melodic_contour[:32]
            if contour:
                min_p, max_p = min(contour), max(contour)
                if max_p > min_p:
                    contour = [(p - min_p) / (max_p - min_p) for p in contour]
            vec.extend(contour + [0] * (32 - len(contour)))
        else:
            vec.extend([0] * 32)
        
        # Velocity curve (32)
        if features.velocity_curve:
            curve = [v / 127 for v in features.velocity_curve[:32]]
            vec.extend(curve + [0] * (32 - len(curve)))
        else:
            vec.extend([0] * 32)
        
        return vec
    
    def _extract_fallback(self, midi_path: Path) -> MIDIFeatures:
        """Fallback extraction without mido."""
        return MIDIFeatures(file_path=str(midi_path))


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_midi_features(midi_path: Path, **kwargs) -> MIDIFeatures:
    """Extract features from a MIDI file."""
    extractor = MIDIFeatureExtractor(**kwargs)
    return extractor.extract(midi_path)


def extract_melody_features(midi_path: Path) -> Dict[str, Any]:
    """Extract melody-specific features for MelodyTransformer."""
    features = extract_midi_features(midi_path)
    return {
        'pitch_sequence': features.pitch_sequence,
        'pitch_histogram': features.pitch_histogram,
        'melodic_contour': features.melodic_contour,
        'interval_histogram': features.interval_histogram,
        'avg_pitch': features.avg_pitch,
        'pitch_range': features.pitch_range,
    }


def extract_groove_features(midi_path: Path) -> Dict[str, Any]:
    """Extract groove-specific features for GroovePredictor."""
    features = extract_midi_features(midi_path)
    return {
        'timing_offsets_ms': features.timing_offsets_ms,
        'swing_ratio': features.swing_ratio,
        'groove_type': features.groove_type,
        'syncopation_score': features.syncopation_score,
        'note_density': features.note_density,
        'tempo_bpm': features.tempo_bpm,
    }


def extract_harmony_features(midi_path: Path) -> Dict[str, Any]:
    """Extract harmony-specific features for HarmonyPredictor."""
    features = extract_midi_features(midi_path)
    return {
        'chord_sequence': features.chord_sequence,
        'chord_times': features.chord_times,
        'key_signature': features.key_signature,
        'mode': features.mode,
        'pitch_histogram': features.pitch_histogram,
    }

