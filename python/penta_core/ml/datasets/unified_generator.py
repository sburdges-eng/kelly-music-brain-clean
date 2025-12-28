#!/usr/bin/env python3
"""
Kelly Unified Dataset Generator

Combines all data sources into training-ready datasets:
1. Static data (chords, scales, theory, emotions)
2. Downloaded audio samples (Freesound)
3. Generated MIDI (chord progressions, melodies)
4. ML OSC patterns

Output: Unified dataset format for NVIDIA CUDA training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib


class DatasetType(Enum):
    """Types of training data."""
    CHORD_PROGRESSIONS = "chord_progressions"
    SCALES = "scales"
    EMOTIONS = "emotions"
    GROOVES = "grooves"
    AUDIO_SAMPLES = "audio_samples"
    MIDI_SEQUENCES = "midi_sequences"
    OSC_PATTERNS = "osc_patterns"
    THEORY_CONCEPTS = "theory_concepts"


@dataclass
class DataSample:
    """A single training sample."""
    sample_id: str
    dataset_type: DatasetType
    data: Dict[str, Any]
    labels: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dataset_type"] = self.dataset_type.value
        return d


@dataclass
class UnifiedDataset:
    """Unified dataset containing all training data."""
    name: str
    version: str
    samples: List[DataSample] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def add_sample(self, sample: DataSample):
        self.samples.append(sample)
        dtype = sample.dataset_type.value
        self.stats[dtype] = self.stats.get(dtype, 0) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def iter_by_type(self, dtype: DatasetType) -> Iterator[DataSample]:
        for s in self.samples:
            if s.dataset_type == dtype:
                yield s


class StaticDataLoader:
    """Load static JSON data files."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent.parent / "data"

    def load_chord_progressions(self) -> List[DataSample]:
        """Load chord progression datasets."""
        samples = []

        # Load main chord progressions
        for fname in ["chord_progressions.json", "chord_progression_families.json",
                      "common_progressions.json", "chord_progressions_db.json"]:
            fpath = self.data_dir / "json" / fname
            if not fpath.exists():
                fpath = self.data_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                samples.extend(self._parse_progressions(data, fname))

        return samples

    def _parse_progressions(self, data: Any, source: str) -> List[DataSample]:
        """Parse chord progression data into samples."""
        samples = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(x, str) for x in value):
                    # This is a chord progression
                    sample_id = hashlib.md5(f"{key}:{','.join(value)}".encode()).hexdigest()[:12]
                    samples.append(DataSample(
                        sample_id=sample_id,
                        dataset_type=DatasetType.CHORD_PROGRESSIONS,
                        data={"chords": value, "name": key},
                        labels={"progression_name": key},
                        source_file=source
                    ))
                elif isinstance(value, dict):
                    # Nested structure
                    samples.extend(self._parse_progressions(value, source))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    samples.extend(self._parse_progressions(item, source))

        return samples

    def load_scales(self) -> List[DataSample]:
        """Load scale datasets."""
        samples = []
        scales_path = self.data_dir / "scales" / "scales_database.json"

        if scales_path.exists():
            with open(scales_path) as f:
                data = json.load(f)

            for scale_name, scale_data in data.items():
                if isinstance(scale_data, dict):
                    sample_id = hashlib.md5(scale_name.encode()).hexdigest()[:12]
                    samples.append(DataSample(
                        sample_id=sample_id,
                        dataset_type=DatasetType.SCALES,
                        data=scale_data,
                        labels={"scale_name": scale_name},
                        source_file="scales_database.json"
                    ))

        # Load emotional scale mappings
        emotional_path = self.data_dir / "scales" / "scale_emotional_map.json"
        if emotional_path.exists():
            with open(emotional_path) as f:
                emotional_data = json.load(f)

            for scale, emotions in emotional_data.items():
                sample_id = hashlib.md5(f"emo_{scale}".encode()).hexdigest()[:12]
                samples.append(DataSample(
                    sample_id=sample_id,
                    dataset_type=DatasetType.SCALES,
                    data={"scale": scale, "emotional_mapping": emotions},
                    labels={"scale_name": scale, "has_emotion": True},
                    source_file="scale_emotional_map.json"
                ))

        return samples

    def load_emotions(self) -> List[DataSample]:
        """Load emotion thesaurus data."""
        samples = []
        emotions_dir = self.data_dir / "emotion_thesaurus"

        if emotions_dir.exists():
            for emo_file in emotions_dir.glob("*.json"):
                if emo_file.name == "metadata.json":
                    continue

                with open(emo_file) as f:
                    data = json.load(f)

                emotion_name = emo_file.stem
                sample_id = hashlib.md5(f"emo_{emotion_name}".encode()).hexdigest()[:12]
                samples.append(DataSample(
                    sample_id=sample_id,
                    dataset_type=DatasetType.EMOTIONS,
                    data=data,
                    labels={"emotion": emotion_name, "category": "thesaurus"},
                    source_file=emo_file.name
                ))

        return samples

    def load_grooves(self) -> List[DataSample]:
        """Load groove and genre data."""
        samples = []
        grooves_dir = self.data_dir / "grooves"

        for fname in ["genre_pocket_maps.json", "genre_mix_fingerprints.json", "humanize_presets.json"]:
            fpath = grooves_dir / fname
            if not fpath.exists():
                fpath = self.data_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)

                for key, value in data.items():
                    sample_id = hashlib.md5(f"groove_{key}".encode()).hexdigest()[:12]
                    samples.append(DataSample(
                        sample_id=sample_id,
                        dataset_type=DatasetType.GROOVES,
                        data={"name": key, "params": value},
                        labels={"groove_name": key},
                        source_file=fname
                    ))

        return samples

    def load_theory(self) -> List[DataSample]:
        """Load music theory concepts."""
        samples = []
        theory_dir = self.data_dir / "music_theory"

        for fname in ["concepts.json", "exercises.json", "learning_paths.json"]:
            fpath = theory_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    for key, value in data.items():
                        sample_id = hashlib.md5(f"theory_{key}".encode()).hexdigest()[:12]
                        samples.append(DataSample(
                            sample_id=sample_id,
                            dataset_type=DatasetType.THEORY_CONCEPTS,
                            data=value if isinstance(value, dict) else {"content": value},
                            labels={"concept": key},
                            source_file=fname
                        ))

        return samples

    def load_all(self) -> UnifiedDataset:
        """Load all static data into unified dataset."""
        dataset = UnifiedDataset(name="kelly_static", version="1.0.0")

        # Load each data type
        for sample in self.load_chord_progressions():
            dataset.add_sample(sample)

        for sample in self.load_scales():
            dataset.add_sample(sample)

        for sample in self.load_emotions():
            dataset.add_sample(sample)

        for sample in self.load_grooves():
            dataset.add_sample(sample)

        for sample in self.load_theory():
            dataset.add_sample(sample)

        return dataset


class DownloadedSamplesLoader:
    """Load downloaded audio samples from emotion sampler."""

    def __init__(self, staging_dir: Path = None):
        self.staging_dir = staging_dir or Path.home() / ".kelly" / "emotion_samples"

    def load_samples(self) -> List[DataSample]:
        """Load all downloaded audio samples."""
        samples = []

        if not self.staging_dir.exists():
            return samples

        # Check download log
        log_path = self.staging_dir.parent / "emotion_sampler_downloads.json"
        if log_path.exists():
            with open(log_path) as f:
                download_log = json.load(f)

            for entry in download_log.get("downloads", []):
                sample_id = entry.get("sound_id", hashlib.md5(str(entry).encode()).hexdigest()[:12])
                samples.append(DataSample(
                    sample_id=str(sample_id),
                    dataset_type=DatasetType.AUDIO_SAMPLES,
                    data={
                        "file_path": entry.get("file_path"),
                        "duration": entry.get("duration"),
                        "format": entry.get("format", "mp3"),
                    },
                    labels={
                        "emotion": entry.get("emotion"),
                        "instrument": entry.get("instrument"),
                        "source": "freesound",
                    },
                    metadata=entry
                ))

        return samples


class OSCPatternLoader:
    """Load ML OSC training patterns."""

    def __init__(self, patterns_dir: Path = None):
        self.patterns_dir = patterns_dir or Path.home() / ".kelly" / "osc_patterns"

    def load_patterns(self) -> List[DataSample]:
        """Load recorded OSC patterns for training."""
        samples = []

        if not self.patterns_dir.exists():
            return samples

        for pattern_file in self.patterns_dir.glob("*.json"):
            with open(pattern_file) as f:
                data = json.load(f)

            sample_id = pattern_file.stem
            samples.append(DataSample(
                sample_id=sample_id,
                dataset_type=DatasetType.OSC_PATTERNS,
                data=data.get("sequence", data),
                labels=data.get("labels", {}),
                metadata=data.get("metadata", {}),
                source_file=pattern_file.name
            ))

        return samples


class UnifiedDatasetGenerator:
    """Generate unified dataset from all sources."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("datasets") / "unified"
        self.static_loader = StaticDataLoader()
        self.samples_loader = DownloadedSamplesLoader()
        self.osc_loader = OSCPatternLoader()

    def generate(self, include_audio: bool = True, include_osc: bool = True) -> UnifiedDataset:
        """Generate complete unified dataset."""
        dataset = UnifiedDataset(name="kelly_unified", version="1.0.0")

        # Load static data
        static = self.static_loader.load_all()
        for sample in static.samples:
            dataset.add_sample(sample)

        # Load downloaded audio samples
        if include_audio:
            for sample in self.samples_loader.load_samples():
                dataset.add_sample(sample)

        # Load OSC patterns
        if include_osc:
            for sample in self.osc_loader.load_patterns():
                dataset.add_sample(sample)

        return dataset

    def save(self, dataset: UnifiedDataset, format: str = "json"):
        """Save dataset to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if format == "json":
            output_path = self.output_dir / f"{dataset.name}_v{dataset.version}.json"
            with open(output_path, "w") as f:
                json.dump({
                    "name": dataset.name,
                    "version": dataset.version,
                    "stats": dataset.stats,
                    "total_samples": len(dataset),
                    "samples": [s.to_dict() for s in dataset.samples]
                }, f, indent=2)
            print(f"Saved {len(dataset)} samples to {output_path}")

        # Also save split files by type
        for dtype in DatasetType:
            type_samples = list(dataset.iter_by_type(dtype))
            if type_samples:
                type_path = self.output_dir / f"{dtype.value}.json"
                with open(type_path, "w") as f:
                    json.dump({
                        "type": dtype.value,
                        "count": len(type_samples),
                        "samples": [s.to_dict() for s in type_samples]
                    }, f, indent=2)
                print(f"  {dtype.value}: {len(type_samples)} samples")

        return self.output_dir


def main():
    """Generate unified dataset."""
    print("=" * 60)
    print("Kelly Unified Dataset Generator")
    print("=" * 60)

    generator = UnifiedDatasetGenerator()
    dataset = generator.generate()

    print(f"\nDataset Statistics:")
    print("-" * 40)
    for dtype, count in sorted(dataset.stats.items()):
        print(f"  {dtype}: {count}")
    print(f"  TOTAL: {len(dataset)}")

    output_dir = generator.save(dataset)
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
