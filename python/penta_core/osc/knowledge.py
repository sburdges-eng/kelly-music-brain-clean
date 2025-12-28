"""OSC and sound card knowledge base utilities for audio learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import importlib
import json

from python.penta_core.ml.datasets.audio_features import AudioFeatureExtractor


@dataclass(frozen=True)
class EQBand:
    frequency_hz: float
    gain_db: float
    q_factor: float


@dataclass(frozen=True)
class EQProfile:
    name: str
    bands: List[EQBand] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class CompressionProfile:
    name: str
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    makeup_gain_db: float


@dataclass(frozen=True)
class AudioReference:
    artist: str
    engineer: str
    project: str
    notes: str = ""


@dataclass(frozen=True)
class SoundCardProfile:
    name: str
    input_channels: int
    output_channels: int
    sample_rate: int
    latency_ms: Optional[float] = None


class OSCAudioKnowledgeBase:
    """Lightweight knowledge base for OSC/audio references."""

    def __init__(self) -> None:
        self.eq_profiles: Dict[str, EQProfile] = {}
        self.compression_profiles: Dict[str, CompressionProfile] = {}
        self.references: List[AudioReference] = []
        self.sound_cards: List[SoundCardProfile] = []

    def add_eq_profile(self, profile: EQProfile) -> None:
        self.eq_profiles[profile.name] = profile

    def add_compression_profile(self, profile: CompressionProfile) -> None:
        self.compression_profiles[profile.name] = profile

    def add_reference(self, reference: AudioReference) -> None:
        self.references.append(reference)

    def add_sound_card(self, profile: SoundCardProfile) -> None:
        self.sound_cards.append(profile)

    def learn_eq_from_audio(self, audio_path: str, *, name: Optional[str] = None) -> EQProfile:
        extractor = AudioFeatureExtractor()
        features = extractor.extract(Path(audio_path))
        centroid = features.spectral_centroid_mean
        bandwidth_values = features.spectral_bandwidth if features.spectral_bandwidth else []
        bandwidth = sum(bandwidth_values) / len(bandwidth_values) if bandwidth_values else 0.0
        bands = _eq_bands_from_features(centroid, bandwidth)
        profile_name = name or f"EQ:{Path(audio_path).stem}"
        profile = EQProfile(name=profile_name, bands=bands, notes="Auto-learned EQ from spectral features")
        self.add_eq_profile(profile)
        return profile

    def serialize(self) -> str:
        payload = {
            "eq_profiles": {name: _eq_profile_to_dict(profile) for name, profile in self.eq_profiles.items()},
            "compression_profiles": {
                name: profile.__dict__ for name, profile in self.compression_profiles.items()
            },
            "references": [ref.__dict__ for ref in self.references],
            "sound_cards": [card.__dict__ for card in self.sound_cards],
        }
        return json.dumps(payload, indent=2)

    def detect_sound_cards(self) -> List[SoundCardProfile]:
        if not importlib.util.find_spec("sounddevice"):
            return []
        sounddevice = importlib.import_module("sounddevice")
        devices = sounddevice.query_devices()
        profiles = []
        for device in devices:
            profiles.append(
                SoundCardProfile(
                    name=device["name"],
                    input_channels=device.get("max_input_channels", 0),
                    output_channels=device.get("max_output_channels", 0),
                    sample_rate=int(device.get("default_samplerate", 0)),
                )
            )
        self.sound_cards.extend(profiles)
        return profiles


def default_osc_references() -> List[AudioReference]:
    """Provide a starting set of reference metadata."""
    return [
        AudioReference(artist="Daft Punk", engineer="Mick Guzauski", project="Random Access Memories"),
        AudioReference(artist="Billie Eilish", engineer="Finneas O'Connell", project="When We All Fall Asleep, Where Do We Go?"),
        AudioReference(artist="Adele", engineer="Tom Elmhirst", project="21"),
        AudioReference(artist="Radiohead", engineer="Nigel Godrich", project="In Rainbows"),
    ]


def _eq_bands_from_features(centroid: float, bandwidth: float) -> List[EQBand]:
    base_freq = max(60.0, centroid)
    width = max(200.0, bandwidth)
    return [
        EQBand(frequency_hz=base_freq * 0.5, gain_db=2.0, q_factor=1.0),
        EQBand(frequency_hz=base_freq, gain_db=1.0, q_factor=0.8),
        EQBand(frequency_hz=base_freq + width, gain_db=-1.5, q_factor=1.2),
    ]


def _eq_profile_to_dict(profile: EQProfile) -> Dict[str, object]:
    return {
        "name": profile.name,
        "bands": [band.__dict__ for band in profile.bands],
        "notes": profile.notes,
    }
