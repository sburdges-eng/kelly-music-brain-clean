#!/usr/bin/env python3
"""
Kelly MIDI Generator for ML Training

Generates synthetic MIDI training data from:
1. Chord progressions (static data)
2. Scale patterns
3. Emotion mappings
4. Groove templates

Integrates with:
- Freesound audio downloader (audio-MIDI alignment)
- ML OSC patterns (control sequence generation)
- CUDA trainer (training-ready output)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib


# MIDI note mappings
NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

# Chord quality intervals
CHORD_INTERVALS = {
    '': [0, 4, 7],           # Major
    'm': [0, 3, 7],          # Minor
    'dim': [0, 3, 6],        # Diminished
    'aug': [0, 4, 8],        # Augmented
    '7': [0, 4, 7, 10],      # Dominant 7
    'maj7': [0, 4, 7, 11],   # Major 7
    'm7': [0, 3, 7, 10],     # Minor 7
    'dim7': [0, 3, 6, 9],    # Diminished 7
    'sus2': [0, 2, 7],       # Suspended 2
    'sus4': [0, 5, 7],       # Suspended 4
    'add9': [0, 4, 7, 14],   # Add 9
}


class EmotionProfile(Enum):
    """Emotion-based generation profiles."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


@dataclass
class MIDINote:
    """A single MIDI note."""
    pitch: int
    velocity: int
    start_time: float  # In beats
    duration: float    # In beats
    channel: int = 0


@dataclass
class MIDISequence:
    """A sequence of MIDI notes."""
    notes: List[MIDINote] = field(default_factory=list)
    tempo: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    key: str = "C"
    mode: str = "major"
    emotion: Optional[str] = None
    source_progression: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notes": [
                {"pitch": n.pitch, "velocity": n.velocity,
                 "start": n.start_time, "duration": n.duration, "channel": n.channel}
                for n in self.notes
            ],
            "tempo": self.tempo,
            "time_signature": list(self.time_signature),
            "key": self.key,
            "mode": self.mode,
            "emotion": self.emotion,
            "source_progression": self.source_progression,
        }


@dataclass
class GenerationConfig:
    """Configuration for MIDI generation."""
    # Tempo range
    min_tempo: int = 60
    max_tempo: int = 180

    # Velocity range
    min_velocity: int = 40
    max_velocity: int = 127

    # Note duration range (in beats)
    min_duration: float = 0.25
    max_duration: float = 4.0

    # Octave range
    min_octave: int = 3
    max_octave: int = 6

    # Humanization
    timing_variance: float = 0.02  # Timing jitter (beats)
    velocity_variance: int = 10   # Velocity variation

    # Generation counts
    variations_per_progression: int = 5
    bars_per_sequence: int = 8


class ChordParser:
    """Parse chord symbols into MIDI notes."""

    @staticmethod
    def parse(chord_symbol: str, octave: int = 4) -> List[int]:
        """Parse chord symbol to MIDI note numbers."""
        if not chord_symbol or chord_symbol in ['N', 'NC', 'N.C.']:
            return []

        # Handle slash chords (C/G -> use C chord)
        if '/' in chord_symbol:
            chord_symbol = chord_symbol.split('/')[0]

        # Extract root and quality
        root = chord_symbol[0].upper()
        rest = chord_symbol[1:]

        # Check for accidental
        if rest and rest[0] in '#b':
            root += rest[0]
            rest = rest[1:]

        # Get root MIDI note
        if root not in NOTE_TO_MIDI:
            return []

        root_midi = NOTE_TO_MIDI[root] + (octave * 12)

        # Find matching quality
        quality = ''
        for q in sorted(CHORD_INTERVALS.keys(), key=len, reverse=True):
            if rest.startswith(q):
                quality = q
                break

        # Get intervals
        intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[''])

        # Build chord notes
        return [root_midi + i for i in intervals]


class EmotionModulator:
    """Modulate generation parameters based on emotion."""

    EMOTION_PARAMS = {
        EmotionProfile.HAPPY: {
            "tempo_bias": 20,      # Faster
            "velocity_bias": 10,   # Louder
            "major_weight": 0.9,   # Prefer major
            "staccato_weight": 0.3,
            "octave_bias": 1,      # Higher
        },
        EmotionProfile.SAD: {
            "tempo_bias": -20,     # Slower
            "velocity_bias": -15,  # Softer
            "major_weight": 0.2,   # Prefer minor
            "staccato_weight": 0.1,
            "octave_bias": -1,     # Lower
        },
        EmotionProfile.ANGRY: {
            "tempo_bias": 30,      # Faster
            "velocity_bias": 20,   # Louder
            "major_weight": 0.3,
            "staccato_weight": 0.6,
            "octave_bias": 0,
        },
        EmotionProfile.FEAR: {
            "tempo_bias": 10,
            "velocity_bias": -5,
            "major_weight": 0.2,
            "staccato_weight": 0.4,
            "octave_bias": -1,
        },
        EmotionProfile.SURPRISE: {
            "tempo_bias": 15,
            "velocity_bias": 15,
            "major_weight": 0.5,
            "staccato_weight": 0.5,
            "octave_bias": 1,
        },
        EmotionProfile.DISGUST: {
            "tempo_bias": -10,
            "velocity_bias": 5,
            "major_weight": 0.3,
            "staccato_weight": 0.2,
            "octave_bias": -1,
        },
        EmotionProfile.NEUTRAL: {
            "tempo_bias": 0,
            "velocity_bias": 0,
            "major_weight": 0.5,
            "staccato_weight": 0.3,
            "octave_bias": 0,
        },
    }

    @classmethod
    def modulate(cls, config: GenerationConfig, emotion: EmotionProfile) -> GenerationConfig:
        """Modulate config based on emotion."""
        params = cls.EMOTION_PARAMS.get(emotion, cls.EMOTION_PARAMS[EmotionProfile.NEUTRAL])

        # Create modulated config
        modulated = GenerationConfig(
            min_tempo=config.min_tempo + params["tempo_bias"],
            max_tempo=config.max_tempo + params["tempo_bias"],
            min_velocity=max(1, config.min_velocity + params["velocity_bias"]),
            max_velocity=min(127, config.max_velocity + params["velocity_bias"]),
            min_duration=config.min_duration * (0.5 if params["staccato_weight"] > 0.4 else 1.0),
            max_duration=config.max_duration * (0.7 if params["staccato_weight"] > 0.4 else 1.0),
            min_octave=config.min_octave + params["octave_bias"],
            max_octave=config.max_octave + params["octave_bias"],
            timing_variance=config.timing_variance,
            velocity_variance=config.velocity_variance,
            variations_per_progression=config.variations_per_progression,
            bars_per_sequence=config.bars_per_sequence,
        )
        return modulated


class MIDIGenerator:
    """Generate MIDI sequences from chord progressions."""

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.parser = ChordParser()

    def generate_from_progression(
        self,
        chords: List[str],
        emotion: EmotionProfile = EmotionProfile.NEUTRAL,
        num_variations: int = None
    ) -> List[MIDISequence]:
        """Generate MIDI sequences from chord progression."""
        if num_variations is None:
            num_variations = self.config.variations_per_progression

        # Modulate config for emotion
        config = EmotionModulator.modulate(self.config, emotion)
        sequences = []

        for i in range(num_variations):
            # Randomize parameters within config bounds
            tempo = random.randint(config.min_tempo, config.max_tempo)
            base_velocity = random.randint(config.min_velocity, config.max_velocity)
            octave = random.randint(config.min_octave, config.max_octave)

            notes = []
            beat_position = 0.0
            beats_per_chord = 4.0  # One chord per bar

            for chord_symbol in chords:
                chord_notes = self.parser.parse(chord_symbol, octave)
                if not chord_notes:
                    beat_position += beats_per_chord
                    continue

                # Add chord notes
                for pitch in chord_notes:
                    # Humanize timing
                    start = beat_position + random.gauss(0, config.timing_variance)
                    start = max(0, start)

                    # Humanize velocity
                    velocity = base_velocity + random.randint(-config.velocity_variance, config.velocity_variance)
                    velocity = max(1, min(127, velocity))

                    # Randomize duration
                    duration = random.uniform(config.min_duration, config.max_duration)
                    duration = min(duration, beats_per_chord - 0.1)

                    notes.append(MIDINote(
                        pitch=pitch,
                        velocity=velocity,
                        start_time=start,
                        duration=duration
                    ))

                beat_position += beats_per_chord

            # Create sequence
            mode = "major" if random.random() < EmotionModulator.EMOTION_PARAMS[emotion]["major_weight"] else "minor"
            seq = MIDISequence(
                notes=notes,
                tempo=tempo,
                key=chords[0][0] if chords else "C",
                mode=mode,
                emotion=emotion.value,
                source_progression=chords
            )
            sequences.append(seq)

        return sequences

    def generate_from_scale(
        self,
        scale_name: str,
        scale_notes: List[int],
        emotion: EmotionProfile = EmotionProfile.NEUTRAL,
        num_sequences: int = 5
    ) -> List[MIDISequence]:
        """Generate melodic sequences from scale."""
        config = EmotionModulator.modulate(self.config, emotion)
        sequences = []

        for _ in range(num_sequences):
            tempo = random.randint(config.min_tempo, config.max_tempo)
            octave = random.randint(config.min_octave, config.max_octave)

            notes = []
            beat_position = 0.0

            # Generate 16-32 notes
            num_notes = random.randint(16, 32)

            for _ in range(num_notes):
                # Pick scale degree
                degree = random.choice(scale_notes)
                pitch = degree + (octave * 12)

                # Random duration
                duration = random.choice([0.25, 0.5, 1.0, 2.0])

                velocity = random.randint(config.min_velocity, config.max_velocity)

                notes.append(MIDINote(
                    pitch=pitch,
                    velocity=velocity,
                    start_time=beat_position,
                    duration=duration
                ))

                beat_position += duration

            sequences.append(MIDISequence(
                notes=notes,
                tempo=tempo,
                key=scale_name.split()[0] if scale_name else "C",
                emotion=emotion.value,
            ))

        return sequences


class MIDIDatasetGenerator:
    """Generate complete MIDI dataset from all sources."""

    def __init__(self, data_dir: Path = None, output_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.output_dir = output_dir or Path("datasets") / "midi_generated"
        self.generator = MIDIGenerator()

    def generate_from_static_data(self) -> Dict[str, List[MIDISequence]]:
        """Generate MIDI from all static data sources."""
        all_sequences = {
            "chord_progressions": [],
            "scales": [],
            "emotions": [],
        }

        # Load chord progressions
        prog_files = [
            self.data_dir / "chord_progressions_db.json",
            self.data_dir / "common_progressions.json",
            self.data_dir / "json" / "chord_progressions.json",
        ]

        for pfile in prog_files:
            if pfile.exists():
                with open(pfile) as f:
                    data = json.load(f)
                sequences = self._generate_from_progressions(data)
                all_sequences["chord_progressions"].extend(sequences)

        # Load scales and generate melodic sequences
        scales_path = self.data_dir / "scales" / "scales_database.json"
        if scales_path.exists():
            with open(scales_path) as f:
                scales = json.load(f)
            for scale_name, scale_data in scales.items():
                if isinstance(scale_data, dict) and "intervals" in scale_data:
                    intervals = scale_data["intervals"]
                    if isinstance(intervals, list):
                        seqs = self.generator.generate_from_scale(
                            scale_name, intervals,
                            emotion=random.choice(list(EmotionProfile)),
                            num_sequences=3
                        )
                        all_sequences["scales"].extend(seqs)

        # Generate emotion-specific sequences
        for emotion in EmotionProfile:
            if emotion == EmotionProfile.NEUTRAL:
                continue

            # Use common progressions for this emotion
            common_progs = {
                EmotionProfile.HAPPY: [["C", "G", "Am", "F"], ["G", "D", "Em", "C"]],
                EmotionProfile.SAD: [["Am", "F", "C", "G"], ["Em", "C", "G", "D"]],
                EmotionProfile.ANGRY: [["Em", "D", "C", "B7"], ["Am", "G", "F", "E"]],
                EmotionProfile.FEAR: [["Dm", "Bb", "Gm", "A"], ["Cm", "Fm", "Bb", "G"]],
                EmotionProfile.SURPRISE: [["C", "E", "Am", "F"], ["G", "B", "Em", "D"]],
                EmotionProfile.DISGUST: [["Dm", "Am", "E", "Dm"], ["Gm", "Dm", "A", "Dm"]],
            }

            for prog in common_progs.get(emotion, []):
                seqs = self.generator.generate_from_progression(prog, emotion, num_variations=3)
                all_sequences["emotions"].extend(seqs)

        return all_sequences

    def _generate_from_progressions(self, data: Any, path: str = "") -> List[MIDISequence]:
        """Recursively extract and generate from progression data."""
        sequences = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(x, str) for x in value):
                    # This is a chord progression
                    emotion = self._infer_emotion_from_context(key, value)
                    seqs = self.generator.generate_from_progression(value, emotion)
                    sequences.extend(seqs)
                elif isinstance(value, (dict, list)):
                    sequences.extend(self._generate_from_progressions(value, f"{path}/{key}"))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    sequences.extend(self._generate_from_progressions(item, path))

        return sequences

    def _infer_emotion_from_context(self, name: str, chords: List[str]) -> EmotionProfile:
        """Infer emotion from progression name or chord content."""
        name_lower = name.lower()

        # Check name for emotion keywords
        if any(w in name_lower for w in ["happy", "joy", "bright", "upbeat"]):
            return EmotionProfile.HAPPY
        if any(w in name_lower for w in ["sad", "melancholy", "dark", "minor"]):
            return EmotionProfile.SAD
        if any(w in name_lower for w in ["angry", "aggressive", "intense"]):
            return EmotionProfile.ANGRY
        if any(w in name_lower for w in ["fear", "scary", "tense", "suspense"]):
            return EmotionProfile.FEAR

        # Analyze chord content
        minor_count = sum(1 for c in chords if 'm' in c and 'maj' not in c.lower())
        if minor_count > len(chords) / 2:
            return EmotionProfile.SAD

        return EmotionProfile.NEUTRAL

    def save_dataset(self, sequences: Dict[str, List[MIDISequence]]) -> Path:
        """Save generated sequences to dataset format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        all_samples = []
        for category, seqs in sequences.items():
            for seq in seqs:
                sample_id = hashlib.md5(str(seq.to_dict()).encode()).hexdigest()[:12]
                all_samples.append({
                    "sample_id": sample_id,
                    "dataset_type": "midi_sequences",
                    "category": category,
                    "data": seq.to_dict(),
                    "labels": {
                        "emotion": seq.emotion,
                        "key": seq.key,
                        "mode": seq.mode,
                        "tempo": seq.tempo,
                    }
                })

        # Save main dataset
        output_path = self.output_dir / "midi_generated_v1.json"
        with open(output_path, "w") as f:
            json.dump({
                "name": "kelly_midi_generated",
                "version": "1.0.0",
                "total_samples": len(all_samples),
                "categories": {k: len(v) for k, v in sequences.items()},
                "samples": all_samples
            }, f, indent=2)

        print(f"Saved {len(all_samples)} MIDI sequences to {output_path}")
        return output_path

    def generate_and_save(self) -> Path:
        """Generate full dataset and save."""
        print("Generating MIDI dataset from static data...")
        sequences = self.generate_from_static_data()

        for cat, seqs in sequences.items():
            print(f"  {cat}: {len(seqs)} sequences")

        return self.save_dataset(sequences)


def main():
    """Generate MIDI training dataset."""
    generator = MIDIDatasetGenerator()
    output = generator.generate_and_save()
    print(f"\nDataset saved to: {output}")


if __name__ == "__main__":
    main()
