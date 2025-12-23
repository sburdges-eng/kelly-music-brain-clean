"""
Synthetic Data Generation for Kelly ML Training.

Generates training data from music theory rules:
- Emotion samples: Chord progressions with emotional mappings
- Melody samples: Scale-based note sequences
- Harmony samples: Chord progressions with theory relationships
- Groove samples: Timing patterns with swing/straight variations
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Music Theory Constants
# =============================================================================


# Pitch classes
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Scale patterns (semitones from root)
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'blues': [0, 3, 5, 6, 7, 10],
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],
}

# Chord patterns (semitones from root)
CHORDS = {
    'maj': [0, 4, 7],
    'min': [0, 3, 7],
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'dom7': [0, 4, 7, 10],
    'dim7': [0, 3, 6, 9],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
}

# Common chord progressions with emotional associations
PROGRESSIONS = {
    'happy': [
        ['I', 'IV', 'V', 'I'],        # Classic happy
        ['I', 'V', 'vi', 'IV'],       # Pop progression
        ['I', 'IV', 'I', 'V'],        # Simple major
        ['I', 'ii', 'V', 'I'],        # Jazz turnaround
    ],
    'sad': [
        ['vi', 'IV', 'I', 'V'],       # Sad pop
        ['i', 'VI', 'III', 'VII'],    # Minor progression
        ['i', 'iv', 'V', 'i'],        # Minor cadence
        ['i', 'VII', 'VI', 'V'],      # Descending minor
    ],
    'tense': [
        ['i', 'bII', 'V', 'i'],       # Tritone sub
        ['I', 'bVII', 'IV', 'I'],     # Modal mixture
        ['i', 'iv', 'bVI', 'V'],      # Dark progression
        ['i', 'V', 'bVI', 'bVII'],    # Andalusian
    ],
    'peaceful': [
        ['I', 'iii', 'vi', 'IV'],     # Gentle progression
        ['I', 'V/vi', 'vi', 'IV'],    # Borrowed chord
        ['Imaj7', 'IVmaj7', 'I', 'V'],# Jazz voicings
        ['I', 'vi', 'ii', 'V'],       # Standard turnaround
    ],
    'energetic': [
        ['I', 'bVII', 'IV', 'I'],     # Rock progression
        ['I', 'IV', 'bVII', 'IV'],    # Power chord feel
        ['i', 'bVII', 'bVI', 'V'],    # Metal progression
        ['I', 'V', 'I', 'IV'],        # Punk rock
    ],
}

# Emotion to musical feature mappings
EMOTION_FEATURES = {
    'happy': {
        'mode': 'major',
        'tempo_range': (100, 140),
        'velocity_range': (80, 110),
        'articulation': 'staccato',
        'valence': 0.8,
        'arousal': 0.6,
    },
    'sad': {
        'mode': 'minor',
        'tempo_range': (60, 90),
        'velocity_range': (50, 80),
        'articulation': 'legato',
        'valence': -0.7,
        'arousal': -0.4,
    },
    'angry': {
        'mode': 'minor',
        'tempo_range': (120, 180),
        'velocity_range': (100, 127),
        'articulation': 'marcato',
        'valence': -0.6,
        'arousal': 0.9,
    },
    'peaceful': {
        'mode': 'major',
        'tempo_range': (60, 90),
        'velocity_range': (40, 70),
        'articulation': 'legato',
        'valence': 0.5,
        'arousal': -0.6,
    },
    'tense': {
        'mode': 'minor',
        'tempo_range': (90, 130),
        'velocity_range': (70, 100),
        'articulation': 'staccato',
        'valence': -0.3,
        'arousal': 0.5,
    },
    'melancholic': {
        'mode': 'minor',
        'tempo_range': (70, 100),
        'velocity_range': (60, 90),
        'articulation': 'legato',
        'valence': -0.5,
        'arousal': -0.2,
    },
    'energetic': {
        'mode': 'major',
        'tempo_range': (120, 160),
        'velocity_range': (90, 120),
        'articulation': 'staccato',
        'valence': 0.6,
        'arousal': 0.8,
    },
    'neutral': {
        'mode': 'major',
        'tempo_range': (90, 120),
        'velocity_range': (70, 90),
        'articulation': 'normal',
        'valence': 0.0,
        'arousal': 0.0,
    },
}

# Groove patterns
GROOVE_PATTERNS = {
    'straight': {
        'swing_ratio': 0.5,
        'offsets': [0, 0, 0, 0],
        'accents': [1.0, 0.7, 0.8, 0.7],
    },
    'swing': {
        'swing_ratio': 0.67,
        'offsets': [0, 20, 0, 20],  # ms late on upbeats
        'accents': [1.0, 0.6, 0.9, 0.5],
    },
    'shuffle': {
        'swing_ratio': 0.58,
        'offsets': [0, 15, 0, 15],
        'accents': [1.0, 0.5, 0.9, 0.5],
    },
    'laid_back': {
        'swing_ratio': 0.52,
        'offsets': [5, 10, 5, 10],  # Everything slightly late
        'accents': [0.9, 0.7, 0.85, 0.65],
    },
    'rushed': {
        'swing_ratio': 0.48,
        'offsets': [-5, -5, -5, -5],  # Everything slightly early
        'accents': [1.0, 0.8, 0.95, 0.75],
    },
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation."""
    
    # Target counts
    samples_per_category: int = 1000
    
    # Variation
    keys: List[str] = field(default_factory=lambda: PITCH_NAMES.copy())
    tempo_range: Tuple[int, int] = (60, 180)
    bar_lengths: List[int] = field(default_factory=lambda: [4, 8, 16])
    time_signatures: List[str] = field(default_factory=lambda: ['4/4', '3/4', '6/8'])
    
    # Randomization
    random_seed: Optional[int] = None
    add_variations: bool = True
    variation_factor: float = 0.1  # How much to vary parameters


# =============================================================================
# Synthetic Data Generator
# =============================================================================


class SyntheticGenerator:
    """
    Generates synthetic training data from music theory rules.
    
    Creates:
    - Emotion samples: MIDI + labels for EmotionRecognizer
    - Melody samples: Note sequences for MelodyTransformer
    - Harmony samples: Chord progressions for HarmonyPredictor
    - Groove samples: Timing patterns for GroovePredictor
    
    Usage:
        generator = SyntheticGenerator()
        
        # Generate emotion dataset
        samples = generator.generate_emotion_samples(1000)
        
        # Generate groove dataset
        samples = generator.generate_groove_samples(1000)
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
    
    def generate_emotion_samples(
        self,
        num_samples: int,
        output_dir: Optional[Path] = None,
        emotions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate emotion-labeled samples.
        
        Each sample contains:
        - MIDI data (note events)
        - Emotion labels (valence, arousal, category)
        - Audio features (simulated)
        """
        emotions = emotions or list(EMOTION_FEATURES.keys())
        samples_per_emotion = num_samples // len(emotions)
        
        samples = []
        
        for emotion in emotions:
            features = EMOTION_FEATURES[emotion]
            
            for i in range(samples_per_emotion):
                sample = self._generate_emotion_sample(emotion, features, i)
                samples.append(sample)
                
                # Save MIDI if output dir provided
                if output_dir:
                    self._save_sample_midi(sample, output_dir / emotion)
        
        logger.info(f"Generated {len(samples)} emotion samples")
        return samples
    
    def _generate_emotion_sample(
        self,
        emotion: str,
        features: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        """Generate a single emotion sample."""
        # Choose random key
        root = random.choice(self.config.keys)
        root_midi = PITCH_NAMES.index(root) + 60  # Middle C range
        
        # Get scale
        mode = features['mode']
        scale = SCALES['major'] if mode == 'major' else SCALES['natural_minor']
        
        # Get tempo
        tempo_min, tempo_max = features['tempo_range']
        tempo = random.randint(tempo_min, tempo_max)
        
        # Get velocity
        vel_min, vel_max = features['velocity_range']
        
        # Choose progression
        progression_name = emotion if emotion in PROGRESSIONS else 'happy'
        progression = random.choice(PROGRESSIONS[progression_name])
        
        # Generate notes
        notes = self._generate_notes_from_progression(
            progression, root_midi, scale, tempo, vel_min, vel_max
        )
        
        # Create sample
        sample = {
            'id': f"synth_{emotion}_{index:05d}",
            'emotion': emotion,
            'valence': features['valence'] + random.uniform(-0.1, 0.1),
            'arousal': features['arousal'] + random.uniform(-0.1, 0.1),
            'key': root,
            'mode': mode,
            'tempo_bpm': tempo,
            'progression': progression,
            'articulation': features['articulation'],
            'notes': notes,
            'is_synthetic': True,
            'feature_vector': self._generate_emotion_feature_vector(features),
        }
        
        return sample
    
    def _generate_notes_from_progression(
        self,
        progression: List[str],
        root_midi: int,
        scale: List[int],
        tempo: int,
        vel_min: int,
        vel_max: int,
    ) -> List[Dict[str, Any]]:
        """Generate note events from a chord progression."""
        notes = []
        beat_duration = 60.0 / tempo
        
        current_time = 0.0
        bars = random.choice(self.config.bar_lengths)
        
        for bar in range(bars):
            chord_idx = bar % len(progression)
            chord_symbol = progression[chord_idx]
            
            # Parse chord (simplified)
            chord_root = self._parse_chord_root(chord_symbol, scale)
            chord_type = 'min' if 'i' in chord_symbol.lower() and chord_symbol[0].islower() else 'maj'
            
            # Get chord notes
            chord_pattern = CHORDS[chord_type]
            
            # Generate bass note
            bass_pitch = root_midi + chord_root - 12
            notes.append({
                'pitch': bass_pitch,
                'velocity': random.randint(vel_min, vel_max),
                'onset': current_time,
                'duration': beat_duration * 4,
                'channel': 0,
            })
            
            # Generate chord notes
            for i in range(4):
                onset = current_time + i * beat_duration
                
                for interval in chord_pattern:
                    pitch = root_midi + chord_root + interval
                    velocity = random.randint(vel_min, vel_max)
                    
                    notes.append({
                        'pitch': pitch,
                        'velocity': velocity,
                        'onset': onset,
                        'duration': beat_duration * 0.9,
                        'channel': 1,
                    })
            
            # Add melody note
            melody_pitch = root_midi + chord_root + random.choice([0, 4, 7, 12])
            notes.append({
                'pitch': melody_pitch,
                'velocity': random.randint(vel_min + 10, min(127, vel_max + 10)),
                'onset': current_time,
                'duration': beat_duration * random.choice([1, 2, 4]),
                'channel': 2,
            })
            
            current_time += beat_duration * 4
        
        return notes
    
    def _parse_chord_root(self, chord_symbol: str, scale: List[int]) -> int:
        """Parse chord symbol to get root interval."""
        # Simplified Roman numeral parsing
        numeral_map = {
            'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6,
            'i': 0, 'ii': 1, 'iii': 2, 'iv': 3, 'v': 4, 'vi': 5, 'vii': 6,
        }
        
        # Extract base numeral
        base = ''
        for c in chord_symbol.upper():
            if c in 'IVX':
                base += c
            else:
                break
        
        if base in numeral_map:
            scale_degree = numeral_map[base]
            if scale_degree < len(scale):
                return scale[scale_degree]
        
        return 0
    
    def _generate_emotion_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Generate a feature vector for emotion classification."""
        vec = []
        
        # Valence/arousal (2)
        vec.append(features['valence'])
        vec.append(features['arousal'])
        
        # Mode (2)
        vec.append(1.0 if features['mode'] == 'major' else 0.0)
        vec.append(1.0 if features['mode'] == 'minor' else 0.0)
        
        # Tempo (normalized, 1)
        tempo_mid = (features['tempo_range'][0] + features['tempo_range'][1]) / 2
        vec.append(tempo_mid / 180)
        
        # Velocity (normalized, 1)
        vel_mid = (features['velocity_range'][0] + features['velocity_range'][1]) / 2
        vec.append(vel_mid / 127)
        
        # Articulation (one-hot, 4)
        articulations = ['staccato', 'legato', 'marcato', 'normal']
        for art in articulations:
            vec.append(1.0 if features.get('articulation') == art else 0.0)
        
        # Pad to 128
        while len(vec) < 128:
            vec.append(random.gauss(0, 0.1))
        
        return vec[:128]
    
    def generate_melody_samples(
        self,
        num_samples: int,
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Generate melody samples for MelodyTransformer."""
        samples = []
        
        for i in range(num_samples):
            # Random parameters
            root = random.choice(self.config.keys)
            scale_type = random.choice(list(SCALES.keys()))
            tempo = random.randint(*self.config.tempo_range)
            
            # Generate melody
            melody = self._generate_melody(root, scale_type, tempo)
            
            sample = {
                'id': f"synth_melody_{i:05d}",
                'key': root,
                'scale': scale_type,
                'tempo_bpm': tempo,
                'notes': melody,
                'pitch_histogram': self._compute_pitch_histogram(melody),
                'interval_histogram': self._compute_interval_histogram(melody),
                'is_synthetic': True,
            }
            samples.append(sample)
        
        logger.info(f"Generated {len(samples)} melody samples")
        return samples
    
    def _generate_melody(
        self,
        root: str,
        scale_type: str,
        tempo: int,
    ) -> List[Dict[str, Any]]:
        """Generate a melodic sequence."""
        root_midi = PITCH_NAMES.index(root) + 60
        scale = SCALES[scale_type]
        
        notes = []
        beat_duration = 60.0 / tempo
        current_time = 0.0
        
        num_notes = random.randint(16, 64)
        current_pitch_idx = random.randint(0, len(scale) - 1)
        
        for _ in range(num_notes):
            # Melodic movement (mostly stepwise)
            step = random.choice([-2, -1, -1, 0, 1, 1, 2])
            current_pitch_idx = max(0, min(len(scale) - 1, current_pitch_idx + step))
            
            pitch = root_midi + scale[current_pitch_idx]
            
            # Random octave shift
            if random.random() < 0.1:
                pitch += random.choice([-12, 12])
            
            pitch = max(48, min(84, pitch))  # Keep in reasonable range
            
            duration = beat_duration * random.choice([0.25, 0.5, 0.5, 1, 1, 2])
            velocity = random.randint(60, 100)
            
            notes.append({
                'pitch': pitch,
                'velocity': velocity,
                'onset': current_time,
                'duration': duration,
            })
            
            current_time += duration
        
        return notes
    
    def generate_harmony_samples(
        self,
        num_samples: int,
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Generate harmony samples for HarmonyPredictor."""
        samples = []
        
        emotions = list(PROGRESSIONS.keys())
        
        for i in range(num_samples):
            emotion = random.choice(emotions)
            progression = random.choice(PROGRESSIONS[emotion])
            root = random.choice(self.config.keys)
            
            sample = {
                'id': f"synth_harmony_{i:05d}",
                'key': root,
                'emotion': emotion,
                'progression': progression,
                'chord_sequence': self._progression_to_chords(progression, root),
                'is_synthetic': True,
            }
            samples.append(sample)
        
        logger.info(f"Generated {len(samples)} harmony samples")
        return samples
    
    def _progression_to_chords(self, progression: List[str], root: str) -> List[str]:
        """Convert Roman numerals to chord names."""
        root_idx = PITCH_NAMES.index(root)
        major_intervals = [0, 2, 4, 5, 7, 9, 11]
        
        chords = []
        for numeral in progression:
            # Parse numeral
            is_minor = numeral[0].islower()
            
            numeral_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
            base = ''
            for c in numeral.upper():
                if c in 'IVX':
                    base += c
                else:
                    break
            
            degree = numeral_map.get(base, 0)
            chord_root_idx = (root_idx + major_intervals[degree]) % 12
            chord_root = PITCH_NAMES[chord_root_idx]
            
            suffix = 'm' if is_minor else 'maj'
            chords.append(f"{chord_root}{suffix}")
        
        return chords
    
    def generate_groove_samples(
        self,
        num_samples: int,
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Generate groove samples for GroovePredictor."""
        samples = []
        
        groove_types = list(GROOVE_PATTERNS.keys())
        samples_per_type = num_samples // len(groove_types)
        
        for groove_type in groove_types:
            pattern = GROOVE_PATTERNS[groove_type]
            
            for i in range(samples_per_type):
                tempo = random.randint(80, 160)
                
                # Generate timing data
                timing_data = self._generate_groove_timing(pattern, tempo)
                
                sample = {
                    'id': f"synth_groove_{groove_type}_{i:05d}",
                    'groove_type': groove_type,
                    'swing_ratio': pattern['swing_ratio'] + random.uniform(-0.02, 0.02),
                    'tempo_bpm': tempo,
                    'timing_offsets_ms': timing_data['offsets'],
                    'velocity_accents': timing_data['velocities'],
                    'syncopation_score': timing_data['syncopation'],
                    'is_synthetic': True,
                    'feature_vector': self._generate_groove_feature_vector(pattern, tempo),
                }
                samples.append(sample)
        
        logger.info(f"Generated {len(samples)} groove samples")
        return samples
    
    def _generate_groove_timing(
        self,
        pattern: Dict[str, Any],
        tempo: int,
    ) -> Dict[str, Any]:
        """Generate groove timing data."""
        num_beats = random.randint(16, 64)
        
        offsets = []
        velocities = []
        syncopation_count = 0
        
        for beat in range(num_beats):
            # Base offset from pattern (with variation)
            base_offset = pattern['offsets'][beat % len(pattern['offsets'])]
            offset = base_offset + random.gauss(0, 3)
            offsets.append(offset)
            
            # Velocity from pattern
            base_accent = pattern['accents'][beat % len(pattern['accents'])]
            velocity = int(base_accent * 100 + random.gauss(0, 10))
            velocity = max(40, min(127, velocity))
            velocities.append(velocity)
            
            # Check syncopation
            if beat % 4 in [1, 3] and velocity > 80:
                syncopation_count += 1
        
        return {
            'offsets': offsets,
            'velocities': velocities,
            'syncopation': syncopation_count / num_beats,
        }
    
    def _generate_groove_feature_vector(
        self,
        pattern: Dict[str, Any],
        tempo: int,
    ) -> List[float]:
        """Generate feature vector for groove classification."""
        vec = []
        
        # Swing ratio (1)
        vec.append(pattern['swing_ratio'])
        
        # Tempo (normalized, 1)
        vec.append(tempo / 200)
        
        # Pattern offsets (4)
        vec.extend([o / 50 for o in pattern['offsets'][:4]])
        
        # Pattern accents (4)
        vec.extend(pattern['accents'][:4])
        
        # Groove type one-hot (5)
        for groove in ['straight', 'swing', 'shuffle', 'laid_back', 'rushed']:
            vec.append(1.0 if groove in str(pattern) else 0.0)
        
        # Pad to 64
        while len(vec) < 64:
            vec.append(random.gauss(0, 0.1))
        
        return vec[:64]
    
    def _compute_pitch_histogram(self, notes: List[Dict]) -> List[float]:
        """Compute pitch class histogram."""
        histogram = [0] * 12
        for note in notes:
            pc = note['pitch'] % 12
            histogram[pc] += 1
        
        total = sum(histogram)
        if total > 0:
            histogram = [h / total for h in histogram]
        
        return histogram
    
    def _compute_interval_histogram(self, notes: List[Dict]) -> List[float]:
        """Compute interval histogram."""
        if len(notes) < 2:
            return [0] * 25
        
        histogram = [0] * 25  # -12 to +12
        pitches = [n['pitch'] for n in notes]
        
        for i in range(1, len(pitches)):
            interval = pitches[i] - pitches[i-1]
            interval = max(-12, min(12, interval))
            histogram[interval + 12] += 1
        
        total = sum(histogram)
        if total > 0:
            histogram = [h / total for h in histogram]
        
        return histogram
    
    def _save_sample_midi(self, sample: Dict[str, Any], output_dir: Path):
        """Save sample as MIDI file."""
        try:
            import mido
        except ImportError:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo
        tempo = sample.get('tempo_bpm', 120)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
        
        # Sort notes by onset
        notes = sorted(sample.get('notes', []), key=lambda n: n['onset'])
        
        # Convert to MIDI messages
        ticks_per_beat = 480
        last_time = 0
        
        for note in notes:
            onset_ticks = int(note['onset'] * tempo / 60 * ticks_per_beat)
            duration_ticks = int(note['duration'] * tempo / 60 * ticks_per_beat)
            
            delta = onset_ticks - last_time
            track.append(mido.Message('note_on', 
                note=note['pitch'], 
                velocity=note['velocity'],
                time=max(0, delta)
            ))
            track.append(mido.Message('note_off',
                note=note['pitch'],
                velocity=0,
                time=duration_ticks
            ))
            last_time = onset_ticks + duration_ticks
        
        output_path = output_dir / f"{sample['id']}.mid"
        mid.save(str(output_path))


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_emotion_samples(
    num_samples: int = 1000,
    output_dir: Optional[Path] = None,
    config: Optional[GeneratorConfig] = None,
) -> List[Dict[str, Any]]:
    """Generate emotion samples."""
    generator = SyntheticGenerator(config)
    return generator.generate_emotion_samples(num_samples, output_dir)


def generate_melody_samples(
    num_samples: int = 1000,
    output_dir: Optional[Path] = None,
    config: Optional[GeneratorConfig] = None,
) -> List[Dict[str, Any]]:
    """Generate melody samples."""
    generator = SyntheticGenerator(config)
    return generator.generate_melody_samples(num_samples, output_dir)


def generate_harmony_samples(
    num_samples: int = 1000,
    output_dir: Optional[Path] = None,
    config: Optional[GeneratorConfig] = None,
) -> List[Dict[str, Any]]:
    """Generate harmony samples."""
    generator = SyntheticGenerator(config)
    return generator.generate_harmony_samples(num_samples, output_dir)


def generate_groove_samples(
    num_samples: int = 1000,
    output_dir: Optional[Path] = None,
    config: Optional[GeneratorConfig] = None,
) -> List[Dict[str, Any]]:
    """Generate groove samples."""
    generator = SyntheticGenerator(config)
    return generator.generate_groove_samples(num_samples, output_dir)

