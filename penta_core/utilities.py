"""
Enhanced Python API with caching, utilities, and advanced features.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from functools import lru_cache
import pickle
from pathlib import Path
import json
from datetime import datetime


class ChordCache:
    """LRU cache for chord detection results."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, dict] = {}
        self.access_times: Dict[str, datetime] = {}
    
    def _notes_to_key(self, notes: List[Tuple[int, int]]) -> str:
        """Convert notes to cache key."""
        return str(sorted(notes))
    
    def get(self, notes: List[Tuple[int, int]]) -> Optional[dict]:
        """Get cached chord result."""
        key = self._notes_to_key(notes)
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def put(self, notes: List[Tuple[int, int]], result: dict):
        """Store chord result in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = min(self.access_times.items(), key=lambda x: x[1])
            del self.cache[oldest[0]]
            del self.access_times[oldest[0]]
        
        key = self._notes_to_key(notes)
        self.cache[key] = result
        self.access_times[key] = datetime.now()


class ChordProgressionAnalyzer:
    """Analyze chord progressions for patterns and recommendations."""
    
    COMMON_PROGRESSIONS = {
        (0, 5, 7): "I-IV-V (Classic)",
        (0, 7, 9, 5): "I-V-vi-IV (Pop)",
        (2, 7, 0): "ii-V-I (Jazz)",
        (0, 9, 5, 7): "I-vi-IV-V (50s)",
        (0, 5, 2, 7): "I-IV-ii-V (Circle)",
        (0, 3, 5, 7): "I-iii-IV-V (Doo-wop)",
    }
    
    def __init__(self):
        self.history: List[dict] = []
        self.progression_stats: Dict[tuple, int] = {}
    
    def add_chord(self, chord: dict):
        """Add chord to progression history."""
        self.history.append(chord)
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def analyze_pattern(self, length: int = 4) -> dict:
        """Analyze recent progression pattern."""
        if len(self.history) < length:
            return {"pattern": "insufficient_data", "matches": []}
        
        recent = self.history[-length:]
        roots = tuple(c['root'] for c in recent)
        
        # Normalize to start from 0
        normalized = tuple((r - roots[0]) % 12 for r in roots)
        
        matches = []
        for pattern, name in self.COMMON_PROGRESSIONS.items():
            if normalized == pattern:
                matches.append(name)
        
        return {
            "pattern": normalized,
            "matches": matches if matches else ["Custom progression"],
            "length": length,
            "chords": [c['name'] for c in recent]
        }
    
    def suggest_next_chord(self, current_chord: dict) -> List[dict]:
        """Suggest next chords based on common progressions."""
        suggestions = []
        
        # Common resolutions
        root = current_chord['root']
        quality = current_chord.get('quality', 0)
        
        # Dominant to tonic
        if quality == 4:  # Dominant 7th
            suggestions.append({
                "root": (root - 7) % 12,
                "quality": 0,
                "reason": "V7 -> I resolution"
            })
        
        # Tonic to subdominant or dominant
        if quality == 0:  # Major
            suggestions.append({
                "root": (root + 5) % 12,
                "quality": 0,
                "reason": "I -> IV progression"
            })
            suggestions.append({
                "root": (root + 7) % 12,
                "quality": 0,
                "reason": "I -> V progression"
            })
        
        # Subdominant to dominant
        if quality == 0 and root == 5:
            suggestions.append({
                "root": (root + 2) % 12,
                "quality": 4,
                "reason": "IV -> V progression"
            })
        
        return suggestions


class KeyModulationDetector:
    """Detect key changes and modulations in music."""
    
    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self.scale_history: List[dict] = []
        self.modulations: List[dict] = []
    
    def add_scale(self, scale: dict):
        """Add scale to history."""
        self.scale_history.append({
            **scale,
            "timestamp": datetime.now()
        })
        
        if len(self.scale_history) > self.window_size:
            self.scale_history = self.scale_history[-self.window_size:]
            self._check_modulation()
    
    def _check_modulation(self):
        """Check if a modulation occurred."""
        if len(self.scale_history) < self.window_size:
            return
        
        # Check if recent scales differ from earlier ones
        mid_point = self.window_size // 2
        early_scales = self.scale_history[:mid_point]
        recent_scales = self.scale_history[mid_point:]
        
        early_tonic = max(set(s['tonic'] for s in early_scales), 
                         key=lambda x: sum(1 for s in early_scales if s['tonic'] == x))
        recent_tonic = max(set(s['tonic'] for s in recent_scales), 
                          key=lambda x: sum(1 for s in recent_scales if s['tonic'] == x))
        
        if early_tonic != recent_tonic:
            self.modulations.append({
                "from_key": early_tonic,
                "to_key": recent_tonic,
                "timestamp": datetime.now(),
                "relation": self._get_key_relation(early_tonic, recent_tonic)
            })
    
    def _get_key_relation(self, from_key: int, to_key: int) -> str:
        """Determine the relationship between keys."""
        interval = (to_key - from_key) % 12
        relations = {
            0: "same",
            1: "chromatic",
            2: "supertonic",
            3: "mediant",
            4: "subdominant",
            5: "dominant",
            6: "tritone",
            7: "dominant",
            8: "submediant",
            9: "leading tone",
            10: "subtonic",
            11: "semitone"
        }
        return relations.get(interval, "unknown")
    
    def get_modulations(self) -> List[dict]:
        """Get detected modulations."""
        return self.modulations


class RhythmicPatternLibrary:
    """Library of rhythmic patterns for groove analysis."""
    
    PATTERNS = {
        "straight_8ths": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "straight_16ths": [i * 0.25 for i in range(16)],
        "swing_8ths": [0, 0.67, 1.0, 1.67, 2.0, 2.67, 3.0, 3.67],
        "triplets": [0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67],
        "shuffle": [0, 0.67, 1.0, 1.67, 2.0, 2.67, 3.0, 3.67],
        "bossa_nova": [0, 0.5, 1.5, 2.0, 3.0, 3.5],
        "rumba": [0, 0.5, 1.0, 2.0, 2.5, 3.0],
        "son_clave": [0, 0.5, 1.5, 2.0, 3.0]
    }
    
    @classmethod
    def get_pattern(cls, name: str) -> List[float]:
        """Get pattern by name."""
        return cls.PATTERNS.get(name, [])
    
    @classmethod
    def match_pattern(cls, onsets: List[float], tolerance: float = 0.1) -> str:
        """Match onsets to known pattern."""
        best_match = "unknown"
        best_score = float('inf')
        
        for name, pattern in cls.PATTERNS.items():
            if len(onsets) != len(pattern):
                continue
            
            score = sum(abs(o - p) for o, p in zip(onsets, pattern))
            if score < best_score:
                best_score = score
                best_match = name
        
        return best_match if best_score < tolerance * len(onsets) else "custom"


class MIDIUtilities:
    """Utilities for MIDI file handling."""
    
    @staticmethod
    def note_number_to_name(note: int) -> str:
        """Convert MIDI note number to name."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (note // 12) - 1
        note_name = notes[note % 12]
        return f"{note_name}{octave}"
    
    @staticmethod
    def note_name_to_number(name: str) -> int:
        """Convert note name to MIDI note number."""
        notes = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        
        note_part = name[:-1]
        octave = int(name[-1])
        return (octave + 1) * 12 + notes.get(note_part, 0)
    
    @staticmethod
    def quantize_note(note: int, scale_notes: List[int]) -> int:
        """Quantize note to nearest scale note."""
        min_dist = 12
        closest = note
        
        for scale_note in scale_notes:
            for octave_offset in [-12, 0, 12]:
                candidate = scale_note + octave_offset
                dist = abs(note - candidate)
                if dist < min_dist:
                    min_dist = dist
                    closest = candidate
        
        return closest


class VisualizationData:
    """Generate data structures for visualization."""
    
    @staticmethod
    def chord_circle_positions(chord: dict) -> List[dict]:
        """Generate positions for chord circle visualization."""
        positions = []
        for i, pc in enumerate(chord.get('pitch_classes', [])):
            angle = (pc / 12.0) * 360
            positions.append({
                "pitch_class": pc,
                "angle": angle,
                "x": np.cos(np.radians(angle)),
                "y": np.sin(np.radians(angle)),
                "index": i
            })
        return positions
    
    @staticmethod
    def tempo_curve(tempo_history: List[float], window: int = 10) -> List[float]:
        """Generate smoothed tempo curve."""
        if len(tempo_history) < window:
            return tempo_history
        
        smoothed = []
        for i in range(len(tempo_history) - window + 1):
            avg = sum(tempo_history[i:i+window]) / window
            smoothed.append(avg)
        return smoothed
    
    @staticmethod
    def waveform_data(audio: np.ndarray, samples: int = 1000) -> List[float]:
        """Downsample audio for waveform display."""
        if len(audio) <= samples:
            return audio.tolist()
        
        step = len(audio) // samples
        return audio[::step][:samples].tolist()


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
    
    def record(self, operation: str, duration_ms: float):
        """Record timing for an operation."""
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration_ms)
        
        # Keep only last 1000 measurements
        if len(self.timings[operation]) > 1000:
            self.timings[operation] = self.timings[operation][-1000:]
    
    def get_stats(self, operation: str) -> dict:
        """Get statistics for an operation."""
        if operation not in self.timings:
            return {}
        
        times = self.timings[operation]
        return {
            "count": len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
            "median_ms": sorted(times)[len(times) // 2]
        }
    
    def get_all_stats(self) -> dict:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.timings.keys()}
    
    def export_to_file(self, filepath: str):
        """Export benchmark data to file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_all_stats(), f, indent=2)


class SessionRecorder:
    """Record and replay music sessions."""
    
    def __init__(self):
        self.events: List[dict] = []
        self.start_time: Optional[datetime] = None
        self.is_recording = False
    
    def start_recording(self):
        """Start recording session."""
        self.events = []
        self.start_time = datetime.now()
        self.is_recording = True
    
    def stop_recording(self):
        """Stop recording session."""
        self.is_recording = False
    
    def record_event(self, event_type: str, data: dict):
        """Record an event."""
        if not self.is_recording:
            return
        
        self.events.append({
            "type": event_type,
            "data": data,
            "timestamp": (datetime.now() - self.start_time).total_seconds()
        })
    
    def save_to_file(self, filepath: str):
        """Save recorded session to file."""
        session_data = {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration": self.events[-1]["timestamp"] if self.events else 0,
            "event_count": len(self.events),
            "events": self.events
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SessionRecorder':
        """Load session from file."""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        recorder = cls()
        recorder.events = session_data.get("events", [])
        return recorder


# Export all utilities
__all__ = [
    'ChordCache',
    'ChordProgressionAnalyzer',
    'KeyModulationDetector',
    'RhythmicPatternLibrary',
    'MIDIUtilities',
    'VisualizationData',
    'PerformanceBenchmark',
    'SessionRecorder'
]
