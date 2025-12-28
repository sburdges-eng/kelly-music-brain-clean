#!/usr/bin/env python3
"""
Kelly Music Labeler - Automatic MIDI/Audio Feature Extraction & Labeling

Extracts features and generates training labels from:
- MIDI files (tempo, key, instruments, chord progressions)
- Audio files (emotion, energy, chroma, spectral features)

Output: JSONL format for ML training (DiffSinger, RVC, emotion models)

Usage:
    python scripts/label_music.py --data-dir data/
    python scripts/label_music.py --midi-dir datasets/midi --audio-dir datasets/audio
    python scripts/label_music.py --therapy-tags  # Add therapeutic intent labels
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmotionFeatures:
    """Emotion/arousal features from audio."""
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # -1 (calm) to 1 (energetic)
    energy: float   # Overall energy level

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class MusicMetadata:
    """Complete music metadata for training."""
    file: str
    file_type: str  # "midi" or "audio"

    # Core features
    tempo: Optional[float] = None
    key: Optional[str] = None
    mode: Optional[str] = None  # "major" or "minor"
    time_signature: Optional[str] = None
    duration_seconds: Optional[float] = None

    # MIDI-specific
    instruments: Optional[List[str]] = None
    instrument_programs: Optional[List[int]] = None
    note_count: Optional[int] = None
    pitch_range: Optional[Tuple[int, int]] = None

    # Audio-specific
    emotion: Optional[Dict[str, float]] = None
    chroma: Optional[List[float]] = None
    spectral_centroid: Optional[float] = None
    spectral_rolloff: Optional[float] = None
    zero_crossing_rate: Optional[float] = None

    # Labels
    genre: str = "unknown"
    style: str = "neutral"
    voicing: str = "unknown"  # solo, ensemble, orchestral

    # Therapy tags
    therapy_tag: Optional[str] = None
    mood_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Remove None values
        return {k: v for k, v in d.items() if v is not None}


class EmotionEstimator:
    """Estimate emotion from audio features."""

    THERAPY_MAPPINGS = {
        # (valence_range, arousal_range): tag
        ((-1.0, -0.3), (-1.0, -0.3)): "melancholic",
        ((-1.0, -0.3), (-0.3, 0.3)): "contemplative",
        ((-1.0, -0.3), (0.3, 1.0)): "tense",
        ((-0.3, 0.3), (-1.0, -0.3)): "soothing",
        ((-0.3, 0.3), (-0.3, 0.3)): "neutral",
        ((-0.3, 0.3), (0.3, 1.0)): "driving",
        ((0.3, 1.0), (-1.0, -0.3)): "peaceful",
        ((0.3, 1.0), (-0.3, 0.3)): "pleasant",
        ((0.3, 1.0), (0.3, 1.0)): "energizing",
    }

    @classmethod
    def estimate(cls, y, sr) -> EmotionFeatures:
        """Estimate emotion from audio signal."""
        try:
            import librosa
            import numpy as np

            # Energy from RMS
            rms = librosa.feature.rms(y=y)[0]
            energy = float(np.mean(rms))

            # Tempo for arousal estimation
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Spectral centroid for brightness (correlates with valence)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            brightness = float(np.mean(centroid) / 5000)  # Normalize

            # Estimate valence from brightness and major/minor mode
            valence = float(min(1.0, max(-1.0, brightness - 0.5)))

            # Estimate arousal from tempo and energy
            tempo_norm = (tempo - 60) / 120  # Normalize: 60bpm=0, 180bpm=1
            arousal = float(min(1.0, max(-1.0, (tempo_norm + energy * 5) / 2)))

            return EmotionFeatures(
                valence=round(valence, 3),
                arousal=round(arousal, 3),
                energy=round(energy, 4)
            )
        except Exception as e:
            logger.warning(f"Emotion estimation failed: {e}")
            return EmotionFeatures(valence=0.0, arousal=0.0, energy=0.0)

    @classmethod
    def get_therapy_tag(cls, valence: float, arousal: float) -> str:
        """Map valence/arousal to therapy tag."""
        for (v_range, a_range), tag in cls.THERAPY_MAPPINGS.items():
            if v_range[0] <= valence <= v_range[1] and a_range[0] <= arousal <= a_range[1]:
                return tag
        return "neutral"

    @classmethod
    def get_mood_category(cls, valence: float, arousal: float) -> str:
        """Categorize mood for training labels."""
        if valence > 0.3:
            if arousal > 0.3:
                return "happy_excited"
            elif arousal < -0.3:
                return "happy_calm"
            else:
                return "happy_neutral"
        elif valence < -0.3:
            if arousal > 0.3:
                return "sad_agitated"
            elif arousal < -0.3:
                return "sad_calm"
            else:
                return "sad_neutral"
        else:
            if arousal > 0.3:
                return "neutral_energetic"
            elif arousal < -0.3:
                return "neutral_calm"
            else:
                return "neutral"


class MIDIExtractor:
    """Extract features from MIDI files."""

    # General MIDI program to instrument name mapping
    GM_INSTRUMENTS = {
        0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano",
        24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)",
        26: "Electric Guitar (jazz)", 27: "Electric Guitar (clean)",
        32: "Acoustic Bass", 33: "Electric Bass (finger)",
        40: "Violin", 41: "Viola", 42: "Cello",
        48: "String Ensemble 1", 52: "Choir Aahs", 53: "Voice Oohs",
        56: "Trumpet", 57: "Trombone", 60: "French Horn",
        73: "Flute", 74: "Recorder", 80: "Lead 1 (square)",
    }

    @classmethod
    def extract(cls, path: Path) -> Optional[MusicMetadata]:
        """Extract features from MIDI file."""
        try:
            import pretty_midi

            pm = pretty_midi.PrettyMIDI(str(path))

            # Basic features
            tempo = pm.estimate_tempo()
            duration = pm.get_end_time()

            # Instruments
            instruments = []
            programs = []
            for inst in pm.instruments:
                if not inst.is_drum:
                    name = cls.GM_INSTRUMENTS.get(inst.program, f"Program {inst.program}")
                    instruments.append(name)
                    programs.append(inst.program)
                else:
                    instruments.append("Drums")
                    programs.append(-1)

            # Note statistics
            all_notes = []
            for inst in pm.instruments:
                all_notes.extend(inst.notes)

            note_count = len(all_notes)
            if all_notes:
                pitches = [n.pitch for n in all_notes]
                pitch_range = (min(pitches), max(pitches))
            else:
                pitch_range = None

            # Key detection using music21
            key, mode = cls._detect_key(path)

            # Time signature
            time_sig = None
            if pm.time_signature_changes:
                ts = pm.time_signature_changes[0]
                time_sig = f"{ts.numerator}/{ts.denominator}"

            # Voicing estimation
            if len(instruments) == 1:
                voicing = "solo"
            elif len(instruments) <= 4:
                voicing = "ensemble"
            else:
                voicing = "orchestral"

            return MusicMetadata(
                file=path.name,
                file_type="midi",
                tempo=round(tempo, 1),
                key=key,
                mode=mode,
                time_signature=time_sig,
                duration_seconds=round(duration, 2),
                instruments=instruments,
                instrument_programs=programs,
                note_count=note_count,
                pitch_range=pitch_range,
                voicing=voicing,
            )

        except ImportError:
            logger.error("pretty_midi not installed. pip install pretty_midi")
            return None
        except Exception as e:
            logger.error(f"MIDI extraction failed for {path}: {e}")
            return None

    @classmethod
    def _detect_key(cls, path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Detect key using music21."""
        try:
            import music21
            stream = music21.converter.parse(str(path))
            key = stream.analyze('key')
            return key.tonic.name, key.mode
        except:
            return None, None


class AudioExtractor:
    """Extract features from audio files."""

    @classmethod
    def extract(cls, path: Path, add_therapy: bool = False) -> Optional[MusicMetadata]:
        """Extract features from audio file."""
        try:
            import librosa
            import numpy as np

            # Load audio
            y, sr = librosa.load(str(path), sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Emotion
            emotion = EmotionEstimator.estimate(y, sr)

            # Chroma (key/harmony features)
            chroma = librosa.feature.chroma_cens(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1).tolist()

            # Spectral features
            centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
            rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
            zcr = float(librosa.feature.zero_crossing_rate(y).mean())

            # Key detection from chroma
            key, mode = cls._detect_key_from_chroma(chroma_mean)

            metadata = MusicMetadata(
                file=path.name,
                file_type="audio",
                tempo=round(float(tempo), 1),
                key=key,
                mode=mode,
                duration_seconds=round(duration, 2),
                emotion=emotion.to_dict(),
                chroma=[round(c, 4) for c in chroma_mean],
                spectral_centroid=round(centroid, 2),
                spectral_rolloff=round(rolloff, 2),
                zero_crossing_rate=round(zcr, 4),
            )

            # Add therapy tags
            if add_therapy:
                metadata.therapy_tag = EmotionEstimator.get_therapy_tag(
                    emotion.valence, emotion.arousal
                )
                metadata.mood_category = EmotionEstimator.get_mood_category(
                    emotion.valence, emotion.arousal
                )

            return metadata

        except ImportError:
            logger.error("librosa not installed. pip install librosa")
            return None
        except Exception as e:
            logger.error(f"Audio extraction failed for {path}: {e}")
            return None

    @classmethod
    def _detect_key_from_chroma(cls, chroma: List[float]) -> Tuple[str, str]:
        """Estimate key from chroma features."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Major and minor key profiles (Krumhansl-Schmuckler)
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        import numpy as np
        chroma = np.array(chroma)

        best_corr = -1
        best_key = 'C'
        best_mode = 'major'

        for i, note in enumerate(note_names):
            # Rotate profiles
            major_rot = np.roll(major_profile, i)
            minor_rot = np.roll(minor_profile, i)

            # Correlate with chroma
            major_corr = np.corrcoef(chroma, major_rot)[0, 1]
            minor_corr = np.corrcoef(chroma, minor_rot)[0, 1]

            if major_corr > best_corr:
                best_corr = major_corr
                best_key = note
                best_mode = 'major'
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = note
                best_mode = 'minor'

        return best_key, best_mode


class MusicLabeler:
    """Main music labeling pipeline."""

    def __init__(
        self,
        midi_dir: Optional[Path] = None,
        audio_dir: Optional[Path] = None,
        output_file: Path = Path("output/metadata.jsonl"),
        add_therapy_tags: bool = False,
    ):
        self.midi_dir = midi_dir
        self.audio_dir = audio_dir
        self.output_file = output_file
        self.add_therapy_tags = add_therapy_tags

    def label_all(self) -> List[MusicMetadata]:
        """Label all MIDI and audio files."""
        records = []

        # Process MIDI files
        if self.midi_dir and self.midi_dir.exists():
            logger.info(f"Processing MIDI files from {self.midi_dir}")
            for midi_file in self.midi_dir.glob("**/*.mid"):
                metadata = MIDIExtractor.extract(midi_file)
                if metadata:
                    # Try to find matching audio
                    if self.audio_dir:
                        audio_path = self.audio_dir / midi_file.stem
                        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
                            audio_file = audio_path.with_suffix(ext)
                            if audio_file.exists():
                                audio_meta = AudioExtractor.extract(
                                    audio_file, self.add_therapy_tags
                                )
                                if audio_meta:
                                    # Merge audio features into MIDI metadata
                                    metadata.emotion = audio_meta.emotion
                                    metadata.chroma = audio_meta.chroma
                                    metadata.therapy_tag = audio_meta.therapy_tag
                                    metadata.mood_category = audio_meta.mood_category
                                break
                    records.append(metadata)

            # Also check for .midi extension
            for midi_file in self.midi_dir.glob("**/*.midi"):
                metadata = MIDIExtractor.extract(midi_file)
                if metadata:
                    records.append(metadata)

        # Process standalone audio files
        if self.audio_dir and self.audio_dir.exists():
            logger.info(f"Processing audio files from {self.audio_dir}")
            for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
                for audio_file in self.audio_dir.glob(f"**/{ext}"):
                    # Skip if already processed with MIDI
                    if self.midi_dir:
                        midi_path = self.midi_dir / audio_file.stem
                        if midi_path.with_suffix(".mid").exists():
                            continue
                        if midi_path.with_suffix(".midi").exists():
                            continue

                    metadata = AudioExtractor.extract(audio_file, self.add_therapy_tags)
                    if metadata:
                        records.append(metadata)

        return records

    def save(self, records: List[MusicMetadata]):
        """Save records to JSONL file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, "w") as f:
            for record in records:
                f.write(json.dumps(record.to_dict()) + "\n")

        logger.info(f"✅ Saved {len(records)} labeled entries to {self.output_file}")

    def run(self) -> int:
        """Run full labeling pipeline."""
        records = self.label_all()
        if records:
            self.save(records)
        return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Kelly Music Labeler - Extract features and labels from MIDI/audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Label files in data/ directory
    python scripts/label_music.py --data-dir data/

    # Specify separate MIDI and audio directories
    python scripts/label_music.py --midi-dir datasets/midi --audio-dir datasets/audio

    # Add therapeutic intent labels
    python scripts/label_music.py --data-dir data/ --therapy-tags

    # Custom output file
    python scripts/label_music.py --data-dir data/ -o training_labels.jsonl
        """
    )

    parser.add_argument("--data-dir", type=Path, default=None,
                       help="Data directory containing midi/ and audio/ subdirs")
    parser.add_argument("--midi-dir", type=Path, default=None,
                       help="Directory with MIDI files")
    parser.add_argument("--audio-dir", type=Path, default=None,
                       help="Directory with audio files")
    parser.add_argument("-o", "--output", type=Path,
                       default=Path("output/metadata.jsonl"),
                       help="Output JSONL file")
    parser.add_argument("--therapy-tags", action="store_true",
                       help="Add therapeutic intent labels")

    args = parser.parse_args()

    # Determine directories
    midi_dir = args.midi_dir
    audio_dir = args.audio_dir

    if args.data_dir:
        if midi_dir is None:
            midi_dir = args.data_dir / "midi"
        if audio_dir is None:
            audio_dir = args.data_dir / "audio"

    if not midi_dir and not audio_dir:
        # Default to data/ in project root
        data_dir = Path(__file__).parent.parent / "data"
        midi_dir = data_dir / "midi"
        audio_dir = data_dir / "audio"

    logger.info("=" * 60)
    logger.info("Kelly Music Labeler")
    logger.info("=" * 60)
    logger.info(f"MIDI directory: {midi_dir}")
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Therapy tags: {args.therapy_tags}")

    labeler = MusicLabeler(
        midi_dir=midi_dir,
        audio_dir=audio_dir,
        output_file=args.output,
        add_therapy_tags=args.therapy_tags,
    )

    count = labeler.run()

    if count == 0:
        logger.warning("No files found to label. Check your directories.")
        logger.info(f"  Looking for MIDI in: {midi_dir}")
        logger.info(f"  Looking for audio in: {audio_dir}")

    return 0 if count > 0 else 1


if __name__ == "__main__":
    exit(main())
