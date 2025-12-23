"""
Data Augmentation for Kelly ML Training.

Expands datasets through transformations:
- MIDI: Transpose, time stretch, velocity scaling
- Audio: Pitch shift, time stretch, noise addition

Goal: Turn 100 examples into 1000+ variations.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    
    # MIDI augmentations
    transpose_semitones: List[int] = field(default_factory=lambda: list(range(-6, 7)))
    time_stretch_factors: List[float] = field(default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])
    velocity_scale_factors: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    velocity_offset_range: Tuple[int, int] = (-10, 10)
    timing_jitter_ms: float = 5.0  # Random timing offset
    drop_note_probability: float = 0.02  # Randomly drop notes
    
    # Audio augmentations
    pitch_shift_semitones: List[float] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    audio_time_stretch_factors: List[float] = field(default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.005])
    gain_factors: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    
    # General
    max_augmentations_per_sample: int = 10
    preserve_original: bool = True
    random_seed: Optional[int] = None


# =============================================================================
# MIDI Augmenter
# =============================================================================


class MIDIAugmenter:
    """
    Augments MIDI data through various transformations.
    
    Transformations:
    - Transpose: Shift all notes by semitones
    - Time stretch: Speed up or slow down
    - Velocity scaling: Adjust dynamics
    - Timing jitter: Add human-like timing variations
    - Note dropout: Randomly remove notes
    
    Usage:
        augmenter = MIDIAugmenter()
        variations = augmenter.augment("song.mid", num_variations=10)
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
    
    def augment(
        self,
        midi_path: Path,
        output_dir: Path,
        num_variations: int = 10,
    ) -> List[Path]:
        """
        Generate augmented variations of a MIDI file.
        
        Returns list of paths to generated files.
        """
        try:
            import mido
        except ImportError:
            logger.error("mido required for MIDI augmentation")
            return []
        
        midi_path = Path(midi_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original
        try:
            original = mido.MidiFile(str(midi_path))
        except Exception as e:
            logger.error(f"Failed to load {midi_path}: {e}")
            return []
        
        outputs = []
        
        # Keep original if configured
        if self.config.preserve_original:
            orig_out = output_dir / f"{midi_path.stem}_original.mid"
            original.save(str(orig_out))
            outputs.append(orig_out)
        
        # Generate variations
        for i in range(num_variations):
            # Choose random augmentations
            aug_type = random.choice(['transpose', 'time_stretch', 'velocity', 'combined'])
            
            try:
                if aug_type == 'transpose':
                    semitones = random.choice(self.config.transpose_semitones)
                    augmented = self._transpose(original, semitones)
                    suffix = f"_trans_{semitones:+d}"
                
                elif aug_type == 'time_stretch':
                    factor = random.choice(self.config.time_stretch_factors)
                    augmented = self._time_stretch(original, factor)
                    suffix = f"_stretch_{factor:.2f}"
                
                elif aug_type == 'velocity':
                    scale = random.choice(self.config.velocity_scale_factors)
                    augmented = self._velocity_scale(original, scale)
                    suffix = f"_vel_{scale:.2f}"
                
                else:  # combined
                    semitones = random.choice(self.config.transpose_semitones)
                    factor = random.choice(self.config.time_stretch_factors)
                    scale = random.choice(self.config.velocity_scale_factors)
                    
                    augmented = self._transpose(original, semitones)
                    augmented = self._time_stretch(augmented, factor)
                    augmented = self._velocity_scale(augmented, scale)
                    
                    suffix = f"_aug_{i:03d}"
                
                # Add timing jitter
                if self.config.timing_jitter_ms > 0:
                    augmented = self._add_timing_jitter(augmented)
                
                # Random note dropout
                if self.config.drop_note_probability > 0:
                    augmented = self._drop_notes(augmented)
                
                # Save
                out_path = output_dir / f"{midi_path.stem}{suffix}.mid"
                augmented.save(str(out_path))
                outputs.append(out_path)
                
            except Exception as e:
                logger.warning(f"Augmentation failed for variation {i}: {e}")
                continue
        
        logger.info(f"Generated {len(outputs)} variations of {midi_path.name}")
        return outputs
    
    def _transpose(self, midi, semitones: int):
        """Transpose all notes by semitones."""
        import mido
        
        new_midi = mido.MidiFile()
        new_midi.ticks_per_beat = midi.ticks_per_beat
        
        for track in midi.tracks:
            new_track = mido.MidiTrack()
            for msg in track:
                if msg.type in ['note_on', 'note_off']:
                    new_note = max(0, min(127, msg.note + semitones))
                    new_msg = msg.copy(note=new_note)
                    new_track.append(new_msg)
                else:
                    new_track.append(msg.copy())
            new_midi.tracks.append(new_track)
        
        return new_midi
    
    def _time_stretch(self, midi, factor: float):
        """Stretch/compress time by factor."""
        import mido
        
        new_midi = mido.MidiFile()
        new_midi.ticks_per_beat = midi.ticks_per_beat
        
        for track in midi.tracks:
            new_track = mido.MidiTrack()
            for msg in track:
                if hasattr(msg, 'time'):
                    new_time = int(msg.time / factor)
                    new_msg = msg.copy(time=new_time)
                    new_track.append(new_msg)
                else:
                    new_track.append(msg.copy())
            new_midi.tracks.append(new_track)
        
        return new_midi
    
    def _velocity_scale(self, midi, scale: float):
        """Scale all velocities."""
        import mido
        
        new_midi = mido.MidiFile()
        new_midi.ticks_per_beat = midi.ticks_per_beat
        
        for track in midi.tracks:
            new_track = mido.MidiTrack()
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    new_vel = max(1, min(127, int(msg.velocity * scale)))
                    new_msg = msg.copy(velocity=new_vel)
                    new_track.append(new_msg)
                else:
                    new_track.append(msg.copy())
            new_midi.tracks.append(new_track)
        
        return new_midi
    
    def _add_timing_jitter(self, midi):
        """Add random timing variations."""
        import mido
        
        jitter_ticks = int(self.config.timing_jitter_ms * midi.ticks_per_beat / 500)
        
        new_midi = mido.MidiFile()
        new_midi.ticks_per_beat = midi.ticks_per_beat
        
        for track in midi.tracks:
            new_track = mido.MidiTrack()
            for msg in track:
                if msg.type in ['note_on', 'note_off'] and hasattr(msg, 'time'):
                    jitter = random.randint(-jitter_ticks, jitter_ticks)
                    new_time = max(0, msg.time + jitter)
                    new_msg = msg.copy(time=new_time)
                    new_track.append(new_msg)
                else:
                    new_track.append(msg.copy())
            new_midi.tracks.append(new_track)
        
        return new_midi
    
    def _drop_notes(self, midi):
        """Randomly drop notes."""
        import mido
        
        new_midi = mido.MidiFile()
        new_midi.ticks_per_beat = midi.ticks_per_beat
        
        dropped_notes = set()
        
        for track in midi.tracks:
            new_track = mido.MidiTrack()
            accumulated_time = 0
            
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    if random.random() < self.config.drop_note_probability:
                        dropped_notes.add(msg.note)
                        accumulated_time += msg.time
                        continue
                    elif msg.note in dropped_notes:
                        dropped_notes.discard(msg.note)
                        accumulated_time += msg.time
                        continue
                
                if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in dropped_notes:
                        dropped_notes.discard(msg.note)
                        accumulated_time += msg.time
                        continue
                
                # Add accumulated time from dropped notes
                if hasattr(msg, 'time'):
                    new_msg = msg.copy(time=msg.time + accumulated_time)
                    accumulated_time = 0
                else:
                    new_msg = msg.copy()
                new_track.append(new_msg)
            
            new_midi.tracks.append(new_track)
        
        return new_midi
    
    def augment_notes(
        self,
        notes: List[Dict[str, Any]],
        augmentation_type: str = 'random',
    ) -> List[Dict[str, Any]]:
        """
        Augment a list of note dictionaries (in-memory).
        
        Each note: {'pitch': int, 'velocity': int, 'onset': float, 'duration': float}
        """
        notes = copy.deepcopy(notes)
        
        if augmentation_type == 'transpose' or augmentation_type == 'random':
            semitones = random.choice(self.config.transpose_semitones)
            for note in notes:
                note['pitch'] = max(0, min(127, note['pitch'] + semitones))
        
        if augmentation_type == 'velocity' or augmentation_type == 'random':
            scale = random.choice(self.config.velocity_scale_factors)
            for note in notes:
                note['velocity'] = max(1, min(127, int(note['velocity'] * scale)))
        
        if augmentation_type == 'time_stretch' or augmentation_type == 'random':
            factor = random.choice(self.config.time_stretch_factors)
            for note in notes:
                note['onset'] /= factor
                note['duration'] /= factor
        
        return notes


# =============================================================================
# Audio Augmenter
# =============================================================================


class AudioAugmenter:
    """
    Augments audio data through various transformations.
    
    Transformations:
    - Pitch shift: Change pitch without changing speed
    - Time stretch: Speed up or slow down without changing pitch
    - Noise addition: Add background noise
    - Gain adjustment: Change volume
    
    Usage:
        augmenter = AudioAugmenter()
        variations = augmenter.augment("song.wav", output_dir, num_variations=5)
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
    
    def augment(
        self,
        audio_path: Path,
        output_dir: Path,
        num_variations: int = 5,
        sample_rate: int = 22050,
    ) -> List[Path]:
        """
        Generate augmented variations of an audio file.
        """
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            logger.error("librosa and soundfile required for audio augmentation")
            return []
        
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original
        try:
            y, sr = librosa.load(str(audio_path), sr=sample_rate)
        except Exception as e:
            logger.error(f"Failed to load {audio_path}: {e}")
            return []
        
        outputs = []
        
        # Keep original
        if self.config.preserve_original:
            orig_out = output_dir / f"{audio_path.stem}_original.wav"
            sf.write(str(orig_out), y, sr)
            outputs.append(orig_out)
        
        # Generate variations
        for i in range(num_variations):
            try:
                aug_type = random.choice(['pitch_shift', 'time_stretch', 'noise', 'gain', 'combined'])
                
                if aug_type == 'pitch_shift':
                    semitones = random.choice(self.config.pitch_shift_semitones)
                    y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
                    suffix = f"_pitch_{semitones:+.1f}"
                
                elif aug_type == 'time_stretch':
                    factor = random.choice(self.config.audio_time_stretch_factors)
                    y_aug = librosa.effects.time_stretch(y, rate=factor)
                    suffix = f"_stretch_{factor:.2f}"
                
                elif aug_type == 'noise':
                    noise_level = random.choice(self.config.noise_levels)
                    if noise_level > 0:
                        noise = np.random.randn(len(y)) * noise_level
                        y_aug = y + noise
                    else:
                        y_aug = y.copy()
                    suffix = f"_noise_{noise_level:.4f}"
                
                elif aug_type == 'gain':
                    gain = random.choice(self.config.gain_factors)
                    y_aug = y * gain
                    y_aug = np.clip(y_aug, -1, 1)
                    suffix = f"_gain_{gain:.2f}"
                
                else:  # combined
                    y_aug = y.copy()
                    
                    # Random pitch shift
                    if random.random() > 0.5:
                        semitones = random.choice(self.config.pitch_shift_semitones)
                        y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=semitones)
                    
                    # Random time stretch
                    if random.random() > 0.5:
                        factor = random.choice(self.config.audio_time_stretch_factors)
                        y_aug = librosa.effects.time_stretch(y_aug, rate=factor)
                    
                    # Random noise
                    noise_level = random.choice(self.config.noise_levels)
                    if noise_level > 0:
                        noise = np.random.randn(len(y_aug)) * noise_level
                        y_aug = y_aug + noise
                    
                    # Random gain
                    gain = random.choice(self.config.gain_factors)
                    y_aug = y_aug * gain
                    y_aug = np.clip(y_aug, -1, 1)
                    
                    suffix = f"_aug_{i:03d}"
                
                # Save
                out_path = output_dir / f"{audio_path.stem}{suffix}.wav"
                sf.write(str(out_path), y_aug, sr)
                outputs.append(out_path)
                
            except Exception as e:
                logger.warning(f"Audio augmentation failed for variation {i}: {e}")
                continue
        
        logger.info(f"Generated {len(outputs)} audio variations of {audio_path.name}")
        return outputs
    
    def augment_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = 22050,
        augmentation_type: str = 'random',
    ) -> np.ndarray:
        """
        Augment a waveform in memory.
        """
        try:
            import librosa
        except ImportError:
            return waveform
        
        y = waveform.copy()
        
        if augmentation_type in ['pitch_shift', 'random']:
            semitones = random.choice(self.config.pitch_shift_semitones)
            y = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=semitones)
        
        if augmentation_type in ['time_stretch', 'random']:
            factor = random.choice(self.config.audio_time_stretch_factors)
            y = librosa.effects.time_stretch(y, rate=factor)
        
        if augmentation_type in ['noise', 'random']:
            noise_level = random.choice(self.config.noise_levels)
            if noise_level > 0:
                y = y + np.random.randn(len(y)) * noise_level
        
        if augmentation_type in ['gain', 'random']:
            gain = random.choice(self.config.gain_factors)
            y = np.clip(y * gain, -1, 1)
        
        return y


# =============================================================================
# Convenience Functions
# =============================================================================


def augment_midi(
    midi_path: Path,
    output_dir: Path,
    num_variations: int = 10,
    config: Optional[AugmentationConfig] = None,
) -> List[Path]:
    """Augment a MIDI file."""
    augmenter = MIDIAugmenter(config)
    return augmenter.augment(midi_path, output_dir, num_variations)


def augment_audio(
    audio_path: Path,
    output_dir: Path,
    num_variations: int = 5,
    config: Optional[AugmentationConfig] = None,
) -> List[Path]:
    """Augment an audio file."""
    augmenter = AudioAugmenter(config)
    return augmenter.augment(audio_path, output_dir, num_variations)


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    variations_per_file: int = 10,
    file_types: List[str] = ['midi', 'audio'],
    config: Optional[AugmentationConfig] = None,
) -> Dict[str, int]:
    """
    Augment all files in a dataset directory.
    
    Returns counts of generated files by type.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    config = config or AugmentationConfig()
    midi_aug = MIDIAugmenter(config)
    audio_aug = AudioAugmenter(config)
    
    counts = {'midi': 0, 'audio': 0}
    
    # MIDI files
    if 'midi' in file_types:
        for midi_file in input_dir.glob("**/*.mid"):
            outputs = midi_aug.augment(midi_file, output_dir / "midi", variations_per_file)
            counts['midi'] += len(outputs)
        for midi_file in input_dir.glob("**/*.midi"):
            outputs = midi_aug.augment(midi_file, output_dir / "midi", variations_per_file)
            counts['midi'] += len(outputs)
    
    # Audio files
    if 'audio' in file_types:
        for ext in ['*.wav', '*.mp3', '*.flac']:
            for audio_file in input_dir.glob(f"**/{ext}"):
                outputs = audio_aug.augment(audio_file, output_dir / "audio", variations_per_file)
                counts['audio'] += len(outputs)
    
    logger.info(f"Dataset augmentation complete: {counts}")
    return counts

