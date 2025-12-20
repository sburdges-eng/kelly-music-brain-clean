"""
Penta Core Python API
High-level Python interface to C++ engines
"""

from typing import List, Optional, Tuple
import numpy as np

import importlib
import sys
import os
native = None
try:
    # Try standard import
    from . import penta_core_native as native
except ImportError:
    # Try direct import from .so file if present
    so_name = None
    pkg_dir = os.path.dirname(__file__)
    for fname in os.listdir(pkg_dir):
        if fname.startswith('penta_core_native') and fname.endswith('.so'):
            so_name = fname
            break
    if so_name:
        import importlib.util
        spec = importlib.util.spec_from_file_location('penta_core_native', os.path.join(pkg_dir, so_name))
        native = importlib.util.module_from_spec(spec)
        sys.modules['penta_core_native'] = native
        spec.loader.exec_module(native)
    else:
        print("Warning: C++ native module not found. Please build the project first.")


class HarmonyEngine:
    """Python wrapper for C++ HarmonyEngine with additional utilities"""
    
    def __init__(self, sample_rate: float = 48000.0, confidence_threshold: float = 0.5):
        if native is None:
            raise RuntimeError("Native C++ module not available")
        
        config = native.harmony.HarmonyConfig()
        config.sample_rate = sample_rate
        config.confidence_threshold = confidence_threshold
        
        self._engine = native.harmony.HarmonyEngine(config)
        self._chord_history = []
        self._scale_history = []
    
    def process_midi_notes(self, notes: List[Tuple[int, int]]) -> None:
        """
        Process MIDI notes for harmony analysis
        
        Args:
            notes: List of (pitch, velocity) tuples
        """
        native_notes = [native.harmony.Note(pitch, vel) for pitch, vel in notes]
        self._engine.process_notes(native_notes)
    
    def get_current_chord(self) -> dict:
        """Get currently detected chord as dictionary"""
        chord = self._engine.get_current_chord()
        return {
            'root': chord.root,
            'quality': chord.quality,
            'confidence': chord.confidence,
            'pitch_classes': chord.pitch_classes,
            'name': self._chord_to_string(chord)
        }
    
    def get_current_scale(self) -> dict:
        """Get currently detected scale as dictionary"""
        scale = self._engine.get_current_scale()
        return {
            'tonic': scale.tonic,
            'mode': scale.mode,
            'confidence': scale.confidence,
            'degrees': scale.degrees,
            'name': self._scale_to_string(scale)
        }
    
    def suggest_voice_leading(self, target_chord_root: int, 
                             current_voices: List[int]) -> List[int]:
        """
        Get voice leading suggestions
        
        Args:
            target_chord_root: Target chord root note (0-11)
            current_voices: List of current voice pitches
            
        Returns:
            List of suggested voice pitches
        """
        # Create target chord (simplified - major triad)
        target_chord = self._create_chord(target_chord_root, quality=0)
        current_notes = [native.harmony.Note(pitch, 64) for pitch in current_voices]
        
        suggested = self._engine.suggest_voice_leading(target_chord, current_notes)
        return [note.pitch for note in suggested]
    
    @staticmethod
    def _chord_to_string(chord) -> str:
        """Convert chord to readable string"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        quality_names = ['', 'm', 'dim', 'aug', '7', 'maj7', 'm7']
        
        if chord.root < 12:
            name = note_names[chord.root]
            if chord.quality < len(quality_names):
                name += quality_names[chord.quality]
            return name
        return "Unknown"
    
    @staticmethod
    def _scale_to_string(scale) -> str:
        """Convert scale to readable string"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        mode_names = ['Ionian', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Aeolian', 'Locrian']
        
        if scale.tonic < 12:
            name = note_names[scale.tonic]
            if scale.mode < len(mode_names):
                name += ' ' + mode_names[scale.mode]
            return name
        return "Unknown"
    
    @staticmethod
    def _create_chord(root: int, quality: int = 0):
        """Helper to create chord object"""
        # This would need proper implementation in C++
        chord = native.harmony.Chord()
        return chord


class GrooveEngine:
    """Python wrapper for C++ GrooveEngine with beat tracking"""
    
    def __init__(self, sample_rate: float = 48000.0, min_tempo: float = 60.0, 
                 max_tempo: float = 180.0):
        if native is None:
            raise RuntimeError("Native C++ module not available")
        
        config = native.groove.GrooveConfig()
        config.sample_rate = sample_rate
        config.min_tempo = min_tempo
        config.max_tempo = max_tempo
        
        self._engine = native.groove.GrooveEngine(config)
    
    def process_audio(self, audio: np.ndarray) -> None:
        """
        Process audio buffer for groove analysis
        
        Args:
            audio: Audio buffer as numpy array (mono)
        """
        if audio.ndim != 1:
            audio = audio.flatten()
        
        self._engine.process_audio(audio.astype(np.float32))
    
    def get_analysis(self) -> dict:
        """Get current groove analysis results"""
        analysis = self._engine.get_analysis()
        return {
            'tempo': analysis.current_tempo,
            'tempo_confidence': analysis.tempo_confidence,
            'time_signature': f"{analysis.time_signature_num}/{analysis.time_signature_den}",
            'swing': analysis.swing,
            'onset_count': len(analysis.onset_positions)
        }
    
    def quantize_timestamp(self, timestamp: int) -> int:
        """Quantize timestamp to rhythmic grid"""
        return self._engine.quantize_to_grid(timestamp)
    
    def get_tempo(self) -> float:
        """Get current tempo estimate in BPM"""
        return self._engine.get_analysis().current_tempo


class DiagnosticsEngine:
    """Python wrapper for C++ DiagnosticsEngine"""
    
    def __init__(self):
        if native is None:
            raise RuntimeError("Native C++ module not available")
        
        config = native.diagnostics.DiagnosticsConfig()
        self._engine = native.diagnostics.DiagnosticsEngine(config)
    
    def get_stats(self) -> dict:
        """Get current system statistics"""
        stats = self._engine.get_stats()
        return {
            'cpu_usage': stats.cpu_usage_percent,
            'average_latency_ms': stats.average_latency_ms,
            'peak_latency_ms': stats.peak_latency_ms,
            'xrun_count': stats.xrun_count,
            'rms_level': stats.rms_level,
            'peak_level': stats.peak_level,
            'clipping': stats.clipping
        }
    
    def get_performance_report(self) -> str:
        """Get detailed performance report"""
        return self._engine.get_performance_report()
    
    def reset(self) -> None:
        """Reset all statistics"""
        self._engine.reset()


class OSCHub:
    """Python wrapper for OSC communication"""
    
    def __init__(self, server_port: int = 8000, client_port: int = 9000):
        if native is None:
            raise RuntimeError("Native C++ module not available")
        
        config = native.osc.OSCConfig()
        config.server_port = server_port
        config.client_port = client_port
        
        self._hub = native.osc.OSCHub(config)
        self._callbacks = {}
    
    def start(self) -> bool:
        """Start OSC server and client"""
        return self._hub.start()
    
    def stop(self) -> None:
        """Stop OSC server and client"""
        self._hub.stop()
    
    def send_message(self, address: str, *args) -> bool:
        """
        Send OSC message
        
        Args:
            address: OSC address pattern (e.g., '/penta/harmony/chord')
            *args: Message arguments (int, float, or str)
        """
        msg = native.osc.create_osc_message(address)
        for arg in args:
            if isinstance(arg, int):
                msg.add_int(arg)
            elif isinstance(arg, float):
                msg.add_float(arg)
            elif isinstance(arg, str):
                msg.add_string(arg)
        
        return self._hub.send_message(msg)
    
    def receive_message(self) -> Optional[dict]:
        """Receive OSC message (non-blocking)"""
        msg = self._hub.receive_message()
        if msg is None:
            return None
        
        return {
            'address': msg.address,
            'arguments': [msg.get_argument(i) for i in range(msg.argument_count)]
        }
    
    def register_callback(self, pattern: str, callback):
        """Register callback for OSC address pattern"""
        self._callbacks[pattern] = callback
        self._hub.register_callback(pattern, callback)


# Convenience function for integrated workflow
class PentaCore:
    """
    Integrated Penta Core engine combining all modules
    This is the main entry point for Python applications
    """
    
    def __init__(self, sample_rate: float = 48000.0):
        self.sample_rate = sample_rate
        
        self.harmony = HarmonyEngine(sample_rate=sample_rate)
        self.groove = GrooveEngine(sample_rate=sample_rate)
        self.diagnostics = DiagnosticsEngine()
        self.osc = OSCHub()
    
    def process(self, audio: np.ndarray, midi_notes: Optional[List[Tuple[int, int]]] = None):
        """
        Process both audio and MIDI in one call
        
        Args:
            audio: Audio buffer (mono or stereo)
            midi_notes: Optional list of (pitch, velocity) tuples
        """
        # Process MIDI for harmony
        if midi_notes:
            self.harmony.process_midi_notes(midi_notes)
        
        # Process audio for groove
        self.groove.process_audio(audio)
    
    def get_state(self) -> dict:
        """Get complete musical state"""
        return {
            'chord': self.harmony.get_current_chord(),
            'scale': self.harmony.get_current_scale(),
            'groove': self.groove.get_analysis(),
            'diagnostics': self.diagnostics.get_stats()
        }
    
    def start_osc(self) -> bool:
        """Start OSC communication"""
        return self.osc.start()
    
    def stop_osc(self) -> None:
        """Stop OSC communication"""
        self.osc.stop()


__all__ = [
    'HarmonyEngine',
    'GrooveEngine',
    'DiagnosticsEngine',
    'OSCHub',
    'PentaCore'
]
