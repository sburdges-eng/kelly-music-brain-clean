#!/usr/bin/env python3
"""
Kelly Emotion-Annotated Music Dataset Integration

Downloads and processes publicly available emotion-annotated music datasets:
- DEAM (MediaEval Database for Emotional Analysis of Music)
- EMOPIA (Emotional Pop Music Dataset)
- Emotify+ (Pop-Rock Music Emotion Dataset)
- Memo2496 (Instrumental Music Emotion Dataset)
- MERP (Music Emotion Recognition and Prediction)

Also includes:
- GEMS (Geneva Emotional Music Scale) annotation
- Big Five personality trait mapping
- openSMILE feature extraction integration
- Therapeutic music recommendation

Usage:
    from penta_core.ml.datasets.emotion_datasets import (
        EmotionDatasetDownloader,
        GEMSAnnotator,
        PersonalityMapper,
        TherapeuticRecommender,
    )
"""

import os
import json
import logging
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Definitions
# ============================================================================

class EmotionDataset(Enum):
    """Supported emotion-annotated music datasets."""
    DEAM = "deam"                    # MediaEval valence/arousal
    EMOPIA = "emopia"               # Emotional pop piano MIDI
    EMOTIFY_PLUS = "emotify_plus"   # Pop-rock emotion
    MEMO2496 = "memo2496"           # Instrumental emotion
    MERP = "merp"                   # Music emotion recognition
    VGMIDI = "vgmidi"               # Video game music emotion


DATASET_INFO = {
    EmotionDataset.DEAM: {
        "name": "DEAM - MediaEval Database for Emotional Analysis",
        "url": "https://cvml.unige.ch/databases/DEAM/",
        "format": "audio + annotations",
        "annotations": "valence, arousal (continuous)",
        "size": "2,058 excerpts",
        "zenodo_id": "10.5281/zenodo.2629459",
        "license": "CC BY-NC-SA 4.0",
    },
    EmotionDataset.EMOPIA: {
        "name": "EMOPIA - Emotional Pop Music Dataset",
        "url": "https://zenodo.org/records/5090631",
        "format": "MIDI + annotations",
        "annotations": "4-quadrant emotion (HVHA, HVLA, LVHA, LVLA)",
        "size": "1,087 clips from 387 songs",
        "zenodo_id": "5090631",
        "license": "CC BY 4.0",
    },
    EmotionDataset.EMOTIFY_PLUS: {
        "name": "Emotify+ - Pop-Rock Emotion Dataset",
        "url": "https://github.com/juansgomez87/emotify-plus",
        "format": "audio features + annotations",
        "annotations": "valence, arousal, Big Five correlations",
        "size": "400+ songs",
        "license": "Research use",
    },
    EmotionDataset.MEMO2496: {
        "name": "Memo2496 - Instrumental Music Emotion",
        "url": "https://github.com/Memo2496/Dataset",
        "format": "audio + annotations",
        "annotations": "emotion categories",
        "size": "2,496 clips",
        "license": "Research use",
    },
    EmotionDataset.MERP: {
        "name": "MERP - Music Emotion Recognition",
        "url": "https://github.com/comp-music-lab/MERP",
        "format": "audio features + annotations",
        "annotations": "valence, arousal, emotion labels",
        "size": "5,000+ annotations",
        "license": "Research use",
    },
    EmotionDataset.VGMIDI: {
        "name": "VGMIDI - Video Game Music Emotion Dataset",
        "url": "https://github.com/lucasnfe/vgmidi",
        "format": "MIDI + annotations",
        "annotations": "valence, arousal (per-note)",
        "size": "200 MIDI files",
        "license": "CC BY 4.0",
    },
}


# ============================================================================
# GEMS - Geneva Emotional Music Scale
# ============================================================================

@dataclass
class GEMSRating:
    """Geneva Emotional Music Scale (9 factors)."""
    wonder: float = 0.0        # Awe, amazement, fascination
    transcendence: float = 0.0  # Spirituality, inspired feelings
    tenderness: float = 0.0    # Love, affection, warmth
    nostalgia: float = 0.0     # Longing, sentimental, bittersweet
    peacefulness: float = 0.0  # Calm, relaxed, serene
    power: float = 0.0         # Strong, triumphant, energetic
    joyful_activation: float = 0.0  # Happy, dancing, animated
    tension: float = 0.0       # Nervous, agitated, anxious
    sadness: float = 0.0       # Sad, sorrowful, melancholic

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_valence_arousal(cls, valence: float, arousal: float) -> "GEMSRating":
        """Approximate GEMS from valence/arousal model."""
        # Mapping based on research correlations
        return cls(
            wonder=max(0, valence * 0.5 + arousal * 0.3),
            transcendence=max(0, valence * 0.4 + (1 - abs(arousal)) * 0.3),
            tenderness=max(0, valence * 0.6 - arousal * 0.2),
            nostalgia=max(0, -valence * 0.2 + (1 - arousal) * 0.5),
            peacefulness=max(0, valence * 0.3 - arousal * 0.6),
            power=max(0, arousal * 0.7 + valence * 0.2),
            joyful_activation=max(0, valence * 0.6 + arousal * 0.5),
            tension=max(0, -valence * 0.3 + arousal * 0.6),
            sadness=max(0, -valence * 0.7 - arousal * 0.3),
        )

    def dominant_factor(self) -> str:
        """Get the dominant GEMS factor."""
        factors = self.to_dict()
        return max(factors, key=factors.get)


class GEMSAnnotator:
    """Annotate music with Geneva Emotional Music Scale."""

    # GEMS to valence/arousal approximate mappings
    FACTOR_VA_MAP = {
        "wonder": (0.5, 0.4),
        "transcendence": (0.6, 0.1),
        "tenderness": (0.7, -0.3),
        "nostalgia": (-0.2, -0.4),
        "peacefulness": (0.4, -0.7),
        "power": (0.3, 0.8),
        "joyful_activation": (0.8, 0.7),
        "tension": (-0.3, 0.6),
        "sadness": (-0.7, -0.5),
    }

    @classmethod
    def annotate_from_features(
        cls,
        valence: float,
        arousal: float,
        tempo: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> GEMSRating:
        """Generate GEMS annotation from audio features."""
        gems = GEMSRating.from_valence_arousal(valence, arousal)

        # Refine based on additional features
        if tempo is not None:
            # High tempo increases power and joyful activation
            if tempo > 120:
                gems.power = min(1.0, gems.power + 0.2)
                gems.joyful_activation = min(1.0, gems.joyful_activation + 0.15)
            elif tempo < 70:
                gems.peacefulness = min(1.0, gems.peacefulness + 0.2)
                gems.nostalgia = min(1.0, gems.nostalgia + 0.1)

        if mode is not None:
            # Minor mode increases sadness and nostalgia
            if mode.lower() == "minor":
                gems.sadness = min(1.0, gems.sadness + 0.15)
                gems.nostalgia = min(1.0, gems.nostalgia + 0.1)
                gems.tenderness = min(1.0, gems.tenderness + 0.1)
            elif mode.lower() == "major":
                gems.joyful_activation = min(1.0, gems.joyful_activation + 0.1)
                gems.power = min(1.0, gems.power + 0.05)

        return gems


# ============================================================================
# Big Five Personality Mapping
# ============================================================================

@dataclass
class BigFiveProfile:
    """Big Five personality traits."""
    openness: float = 0.5          # Openness to experience
    conscientiousness: float = 0.5  # Organization, dependability
    extraversion: float = 0.5       # Sociability, assertiveness
    agreeableness: float = 0.5      # Cooperation, trust
    neuroticism: float = 0.5        # Emotional instability, anxiety

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class PersonalityMapper:
    """
    Map music preferences to Big Five personality traits.

    Based on research:
    - Rentfrow & Gosling (2003) - STOMP
    - Greenberg et al. (2016) - Musical Universe
    - Langmeyer et al. (2012) - Big Five and music preferences
    """

    # Music style to personality trait correlations
    STYLE_CORRELATIONS = {
        # (openness, conscientiousness, extraversion, agreeableness, neuroticism)
        "classical": (0.7, 0.5, 0.3, 0.5, 0.4),
        "jazz": (0.8, 0.4, 0.5, 0.5, 0.4),
        "blues": (0.6, 0.4, 0.5, 0.5, 0.5),
        "folk": (0.7, 0.5, 0.4, 0.6, 0.4),
        "rock": (0.6, 0.4, 0.6, 0.4, 0.5),
        "alternative": (0.7, 0.3, 0.5, 0.4, 0.6),
        "heavy_metal": (0.6, 0.4, 0.5, 0.3, 0.5),
        "punk": (0.5, 0.3, 0.6, 0.3, 0.6),
        "pop": (0.4, 0.5, 0.7, 0.6, 0.5),
        "country": (0.4, 0.6, 0.5, 0.6, 0.4),
        "electronic": (0.6, 0.4, 0.6, 0.4, 0.5),
        "hip_hop": (0.5, 0.4, 0.7, 0.4, 0.5),
        "soul_rnb": (0.5, 0.5, 0.6, 0.6, 0.5),
        "ambient": (0.7, 0.5, 0.3, 0.5, 0.5),
        "world": (0.8, 0.4, 0.5, 0.6, 0.4),
    }

    # Emotion preference correlations
    EMOTION_CORRELATIONS = {
        # High valence + high arousal -> extraversion
        # Low valence + high arousal -> neuroticism
        # High complexity -> openness
        # Structured -> conscientiousness
        # Warm/tender -> agreeableness
    }

    @classmethod
    def infer_from_preferences(
        cls,
        preferred_styles: List[str],
        valence_preference: float = 0.0,  # -1 to 1
        arousal_preference: float = 0.0,  # -1 to 1
        complexity_preference: float = 0.5,  # 0 to 1
    ) -> BigFiveProfile:
        """Infer personality from music preferences."""
        # Start with neutral profile
        o, c, e, a, n = 0.5, 0.5, 0.5, 0.5, 0.5

        # Aggregate style correlations
        style_count = 0
        for style in preferred_styles:
            style_key = style.lower().replace(" ", "_").replace("-", "_")
            if style_key in cls.STYLE_CORRELATIONS:
                corr = cls.STYLE_CORRELATIONS[style_key]
                o += corr[0] - 0.5
                c += corr[1] - 0.5
                e += corr[2] - 0.5
                a += corr[3] - 0.5
                n += corr[4] - 0.5
                style_count += 1

        if style_count > 0:
            # Average the deviations
            factor = 1.0 / style_count
            o = 0.5 + (o - 0.5) * factor
            c = 0.5 + (c - 0.5) * factor
            e = 0.5 + (e - 0.5) * factor
            a = 0.5 + (a - 0.5) * factor
            n = 0.5 + (n - 0.5) * factor

        # Adjust based on valence/arousal preferences
        # High valence + high arousal -> extraversion
        if valence_preference > 0.3 and arousal_preference > 0.3:
            e = min(1.0, e + 0.15)
        # Low valence + high arousal -> neuroticism
        elif valence_preference < -0.3 and arousal_preference > 0.3:
            n = min(1.0, n + 0.15)
        # Low arousal -> introversion
        elif arousal_preference < -0.3:
            e = max(0.0, e - 0.1)

        # Complexity preference -> openness
        if complexity_preference > 0.7:
            o = min(1.0, o + 0.15)
        elif complexity_preference < 0.3:
            o = max(0.0, o - 0.1)

        return BigFiveProfile(
            openness=round(max(0.0, min(1.0, o)), 2),
            conscientiousness=round(max(0.0, min(1.0, c)), 2),
            extraversion=round(max(0.0, min(1.0, e)), 2),
            agreeableness=round(max(0.0, min(1.0, a)), 2),
            neuroticism=round(max(0.0, min(1.0, n)), 2),
        )

    @classmethod
    def recommend_styles_for_personality(
        cls, profile: BigFiveProfile, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Recommend music styles based on personality profile."""
        scores = []

        for style, corr in cls.STYLE_CORRELATIONS.items():
            # Calculate similarity score
            diff = (
                abs(profile.openness - corr[0]) +
                abs(profile.conscientiousness - corr[1]) +
                abs(profile.extraversion - corr[2]) +
                abs(profile.agreeableness - corr[3]) +
                abs(profile.neuroticism - corr[4])
            )
            similarity = 1.0 - (diff / 5.0)
            scores.append((style, round(similarity, 3)))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]


# ============================================================================
# Therapeutic Music Recommendation
# ============================================================================

@dataclass
class TherapeuticGoal:
    """Therapeutic music recommendation goal."""
    goal_type: str  # relaxation, energization, mood_elevation, etc.
    target_valence: Optional[float] = None
    target_arousal: Optional[float] = None
    target_gems: Optional[str] = None
    notes: str = ""


class TherapeuticRecommender:
    """
    Recommend music for therapeutic purposes.

    Based on:
    - Iso-principle (matching current state before transitioning)
    - Music therapy research
    - GEMS emotional outcomes
    """

    THERAPEUTIC_PRESETS = {
        "relaxation": {
            "target_valence": (0.2, 0.6),
            "target_arousal": (-0.8, -0.2),
            "target_gems": ["peacefulness", "tenderness"],
            "tempo_range": (50, 80),
            "mode_preference": "major",
        },
        "energization": {
            "target_valence": (0.4, 0.9),
            "target_arousal": (0.5, 0.9),
            "target_gems": ["power", "joyful_activation"],
            "tempo_range": (110, 140),
            "mode_preference": "major",
        },
        "mood_elevation": {
            "target_valence": (0.5, 0.9),
            "target_arousal": (0.0, 0.5),
            "target_gems": ["joyful_activation", "wonder"],
            "tempo_range": (90, 120),
            "mode_preference": "major",
        },
        "emotional_processing": {
            "target_valence": (-0.5, 0.3),
            "target_arousal": (-0.4, 0.3),
            "target_gems": ["nostalgia", "sadness", "tenderness"],
            "tempo_range": (60, 90),
            "mode_preference": None,  # Any mode
        },
        "focus_concentration": {
            "target_valence": (0.0, 0.5),
            "target_arousal": (-0.3, 0.3),
            "target_gems": ["transcendence", "peacefulness"],
            "tempo_range": (70, 100),
            "mode_preference": None,
        },
        "anxiety_reduction": {
            "target_valence": (0.1, 0.5),
            "target_arousal": (-0.7, -0.1),
            "target_gems": ["peacefulness", "tenderness"],
            "tempo_range": (60, 80),
            "mode_preference": "major",
        },
        "sleep_induction": {
            "target_valence": (0.0, 0.4),
            "target_arousal": (-0.9, -0.5),
            "target_gems": ["peacefulness"],
            "tempo_range": (40, 65),
            "mode_preference": None,
        },
        "catharsis": {
            "target_valence": (-0.6, 0.0),
            "target_arousal": (0.3, 0.8),
            "target_gems": ["tension", "power", "sadness"],
            "tempo_range": (90, 130),
            "mode_preference": "minor",
        },
    }

    @classmethod
    def get_recommendation_profile(
        cls, goal: str, current_state: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Get music profile for therapeutic goal.

        If current_state (valence, arousal) is provided, uses iso-principle
        to start from current emotional state and transition to target.
        """
        if goal not in cls.THERAPEUTIC_PRESETS:
            available = list(cls.THERAPEUTIC_PRESETS.keys())
            raise ValueError(f"Unknown goal: {goal}. Available: {available}")

        preset = cls.THERAPEUTIC_PRESETS[goal]
        result = {
            "goal": goal,
            "target_valence_range": preset["target_valence"],
            "target_arousal_range": preset["target_arousal"],
            "target_gems_factors": preset["target_gems"],
            "tempo_range": preset["tempo_range"],
            "mode_preference": preset["mode_preference"],
        }

        # Apply iso-principle if current state provided
        if current_state is not None:
            curr_v, curr_a = current_state
            target_v = sum(preset["target_valence"]) / 2
            target_a = sum(preset["target_arousal"]) / 2

            # Create transition path
            result["iso_principle"] = {
                "start_valence": curr_v,
                "start_arousal": curr_a,
                "end_valence": target_v,
                "end_arousal": target_a,
                "suggested_tracks": 3,  # Number of tracks for transition
            }

        return result

    @classmethod
    def score_track(
        cls,
        goal: str,
        track_valence: float,
        track_arousal: float,
        track_tempo: Optional[float] = None,
        track_mode: Optional[str] = None,
    ) -> float:
        """Score how well a track fits a therapeutic goal."""
        if goal not in cls.THERAPEUTIC_PRESETS:
            return 0.0

        preset = cls.THERAPEUTIC_PRESETS[goal]
        score = 1.0

        # Valence fit
        v_min, v_max = preset["target_valence"]
        if track_valence < v_min:
            score -= (v_min - track_valence) * 0.5
        elif track_valence > v_max:
            score -= (track_valence - v_max) * 0.5

        # Arousal fit
        a_min, a_max = preset["target_arousal"]
        if track_arousal < a_min:
            score -= (a_min - track_arousal) * 0.5
        elif track_arousal > a_max:
            score -= (track_arousal - a_max) * 0.5

        # Tempo fit
        if track_tempo is not None:
            t_min, t_max = preset["tempo_range"]
            if track_tempo < t_min:
                score -= (t_min - track_tempo) / 100 * 0.3
            elif track_tempo > t_max:
                score -= (track_tempo - t_max) / 100 * 0.3

        # Mode fit
        if track_mode is not None and preset["mode_preference"] is not None:
            if track_mode.lower() != preset["mode_preference"].lower():
                score -= 0.1

        return max(0.0, min(1.0, score))


# ============================================================================
# openSMILE Feature Extraction
# ============================================================================

class OpenSMILEExtractor:
    """
    Extract features using openSMILE toolkit.

    Requires openSMILE to be installed:
    https://github.com/audeering/opensmile

    Alternatively, use opensmile-python:
    pip install opensmile
    """

    # Standard feature sets
    FEATURE_SETS = {
        "egemaps": "eGeMAPSv02",  # Extended Geneva Minimalistic Acoustic Parameter Set
        "compare": "ComParE_2016",  # Computational Paralinguistics Challenge
        "is09": "IS09_emotion",  # InterSpeech 2009 emotion challenge
        "is10": "IS10_paraling",  # InterSpeech 2010 paralinguistics
    }

    def __init__(self, feature_set: str = "egemaps"):
        self.feature_set = self.FEATURE_SETS.get(feature_set, feature_set)
        self._smile = None

    def _init_opensmile(self):
        """Initialize openSMILE Python wrapper."""
        if self._smile is not None:
            return True

        try:
            import opensmile
            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            return True
        except ImportError:
            logger.warning("opensmile not installed. pip install opensmile")
            return False

    def extract(self, audio_path: Path) -> Optional[Dict[str, float]]:
        """Extract features from audio file."""
        if not self._init_opensmile():
            return self._extract_fallback(audio_path)

        try:
            features = self._smile.process_file(str(audio_path))
            return features.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"openSMILE extraction failed: {e}")
            return self._extract_fallback(audio_path)

    def _extract_fallback(self, audio_path: Path) -> Optional[Dict[str, float]]:
        """Fallback to librosa-based features."""
        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(str(audio_path), sr=16000)

            # eGeMAPS-like features using librosa
            features = {}

            # F0 (fundamental frequency) statistics
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=50, fmax=500, sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                features["F0_mean"] = float(np.mean(f0_valid))
                features["F0_std"] = float(np.std(f0_valid))
                features["F0_range"] = float(np.ptp(f0_valid))
            else:
                features["F0_mean"] = 0.0
                features["F0_std"] = 0.0
                features["F0_range"] = 0.0

            # Energy/loudness
            rms = librosa.feature.rms(y=y)[0]
            features["loudness_mean"] = float(np.mean(rms))
            features["loudness_std"] = float(np.std(rms))

            # Spectral features
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = float(np.mean(spec_cent))

            spec_flux = librosa.onset.onset_strength(y=y, sr=sr)
            features["spectral_flux_mean"] = float(np.mean(spec_flux))

            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(min(4, mfccs.shape[0])):
                features[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
                features[f"mfcc{i+1}_std"] = float(np.std(mfccs[i]))

            # Jitter/shimmer approximations (voice quality)
            if len(f0_valid) > 1:
                jitter = np.mean(np.abs(np.diff(f0_valid))) / np.mean(f0_valid)
                features["jitter_local"] = float(jitter)

            return features

        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return None

    def extract_batch(
        self, audio_paths: List[Path], show_progress: bool = True
    ) -> List[Optional[Dict[str, float]]]:
        """Extract features from multiple files."""
        results = []
        total = len(audio_paths)

        for i, path in enumerate(audio_paths):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing {i+1}/{total}: {path.name}")
            results.append(self.extract(path))

        return results


# ============================================================================
# Dataset Downloader
# ============================================================================

class EmotionDatasetDownloader:
    """Download and process emotion-annotated music datasets."""

    def __init__(
        self,
        output_dir: Path = Path("datasets/emotion"),
        cache_dir: Optional[Path] = None,
    ):
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir or Path.home() / ".cache" / "kelly-datasets"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> Dict[str, Dict[str, str]]:
        """List available datasets with info."""
        return {ds.value: info for ds, info in DATASET_INFO.items()}

    def download_emopia(self) -> Path:
        """
        Download EMOPIA dataset from Zenodo.

        Contains 1,087 clips with 4-quadrant emotion labels.
        """
        dataset_dir = self.output_dir / "emopia"
        if (dataset_dir / "annotations.json").exists():
            logger.info("EMOPIA already downloaded")
            return dataset_dir

        logger.info("Downloading EMOPIA from Zenodo...")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try zenodo_get if available
            import subprocess
            zenodo_url = "https://zenodo.org/records/5090631"

            try:
                result = subprocess.run(
                    ["zenodo_get", "5090631", "-o", str(dataset_dir)],
                    capture_output=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    logger.info(f"EMOPIA downloaded to {dataset_dir}")
                    return dataset_dir
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Fallback: provide instructions
            readme = dataset_dir / "README.txt"
            readme.write_text(f"""EMOPIA Dataset Download Instructions
=====================================

1. Visit: {zenodo_url}
2. Download the dataset files
3. Extract to this directory: {dataset_dir}

Or install zenodo_get: pip install zenodo_get
Then run: zenodo_get 5090631 -o {dataset_dir}

Dataset contains:
- MIDI files with emotion annotations
- 4-quadrant labels (HVHA, HVLA, LVHA, LVLA)
- 1,087 clips from 387 pop songs
""")
            logger.info(f"See download instructions: {readme}")
            return dataset_dir

        except Exception as e:
            logger.error(f"EMOPIA download failed: {e}")
            return dataset_dir

    def download_deam(self) -> Path:
        """
        Download DEAM (MediaEval) dataset.

        Contains 2,058 excerpts with continuous valence/arousal.
        """
        dataset_dir = self.output_dir / "deam"
        if (dataset_dir / "annotations").exists():
            logger.info("DEAM already downloaded")
            return dataset_dir

        logger.info("Downloading DEAM annotations...")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # DEAM annotations are available from MediaEval
        readme = dataset_dir / "README.txt"
        readme.write_text("""DEAM Dataset Download Instructions
===================================

The DEAM dataset requires registration:

1. Visit: https://cvml.unige.ch/databases/DEAM/
2. Request access to the dataset
3. Download annotations and audio excerpts
4. Extract to this directory

Dataset contains:
- 2,058 music excerpts (45 seconds each)
- Continuous valence and arousal annotations
- Static (per-song) and dynamic (per-second) annotations

Annotations format:
- song_id, valence_mean, valence_std, arousal_mean, arousal_std
""")
        logger.info(f"See download instructions: {readme}")
        return dataset_dir

    def download_vgmidi(self) -> Path:
        """
        Download VGMIDI dataset from GitHub.

        Contains 200 MIDI files with per-note valence/arousal.
        """
        dataset_dir = self.output_dir / "vgmidi"
        if (dataset_dir / "midi").exists():
            logger.info("VGMIDI already downloaded")
            return dataset_dir

        logger.info("Downloading VGMIDI from GitHub...")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            import subprocess

            # Clone the repository
            repo_url = "https://github.com/lucasnfe/vgmidi.git"
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(dataset_dir / "repo")],
                capture_output=True,
                timeout=120,
            )

            if result.returncode == 0:
                # Move relevant files
                repo_dir = dataset_dir / "repo"
                if (repo_dir / "data").exists():
                    shutil.move(str(repo_dir / "data"), str(dataset_dir / "midi"))
                logger.info(f"VGMIDI downloaded to {dataset_dir}")
            else:
                raise RuntimeError(result.stderr.decode())

            return dataset_dir

        except Exception as e:
            logger.error(f"VGMIDI download failed: {e}")
            readme = dataset_dir / "README.txt"
            readme.write_text(f"""VGMIDI Dataset Download Instructions
=====================================

Manual download:
1. git clone https://github.com/lucasnfe/vgmidi.git
2. Copy data/ folder to {dataset_dir}/midi/

Dataset contains:
- 200 MIDI files from video game soundtracks
- Per-note valence and arousal annotations
- Great for training emotion-aware MIDI models
""")
            return dataset_dir

    def download_all(self) -> Dict[str, Path]:
        """Download all available datasets."""
        results = {}

        results["emopia"] = self.download_emopia()
        results["deam"] = self.download_deam()
        results["vgmidi"] = self.download_vgmidi()

        return results


# ============================================================================
# Combined Emotion Dataset Schema
# ============================================================================

@dataclass
class EmotionAnnotatedSample:
    """Unified schema for emotion-annotated music samples."""
    # Identification
    id: str
    source_dataset: str
    file_path: Optional[str] = None
    file_type: str = "audio"  # audio, midi, features

    # Core emotion (valence/arousal model)
    valence: Optional[float] = None  # -1 to 1
    arousal: Optional[float] = None  # -1 to 1

    # Categorical emotion
    emotion_quadrant: Optional[str] = None  # HVHA, HVLA, LVHA, LVLA
    emotion_label: Optional[str] = None  # happy, sad, angry, etc.

    # GEMS annotation
    gems: Optional[Dict[str, float]] = None

    # Musical features
    tempo: Optional[float] = None
    key: Optional[str] = None
    mode: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Audio features (openSMILE or librosa)
    audio_features: Optional[Dict[str, float]] = None

    # Personality correlation (if available)
    personality_correlation: Optional[Dict[str, float]] = None

    # Therapeutic tags
    therapy_goal: Optional[str] = None
    therapy_score: Optional[float] = None

    # Metadata
    genre: Optional[str] = None
    artist: Optional[str] = None
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class EmotionDatasetProcessor:
    """Process and unify emotion-annotated datasets."""

    def __init__(self, datasets_dir: Path = Path("datasets/emotion")):
        self.datasets_dir = Path(datasets_dir)
        self.gems_annotator = GEMSAnnotator()
        self.personality_mapper = PersonalityMapper()
        self.therapeutic_recommender = TherapeuticRecommender()

    def process_emopia(self) -> List[EmotionAnnotatedSample]:
        """Process EMOPIA dataset into unified schema."""
        samples = []
        emopia_dir = self.datasets_dir / "emopia"

        if not emopia_dir.exists():
            logger.warning(f"EMOPIA not found at {emopia_dir}")
            return samples

        # Map quadrants to valence/arousal
        quadrant_va = {
            "HVHA": (0.6, 0.6),   # High Valence, High Arousal
            "HVLA": (0.6, -0.6),  # High Valence, Low Arousal
            "LVHA": (-0.6, 0.6),  # Low Valence, High Arousal
            "LVLA": (-0.6, -0.6), # Low Valence, Low Arousal
        }

        # Look for annotation files
        for anno_file in emopia_dir.glob("**/*.json"):
            try:
                with open(anno_file) as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        sample = self._process_emopia_item(item, quadrant_va)
                        if sample:
                            samples.append(sample)
                elif isinstance(data, dict):
                    sample = self._process_emopia_item(data, quadrant_va)
                    if sample:
                        samples.append(sample)

            except Exception as e:
                logger.warning(f"Error processing {anno_file}: {e}")

        logger.info(f"Processed {len(samples)} EMOPIA samples")
        return samples

    def _process_emopia_item(
        self, item: Dict, quadrant_va: Dict
    ) -> Optional[EmotionAnnotatedSample]:
        """Process single EMOPIA item."""
        try:
            quadrant = item.get("emotion", item.get("quadrant", ""))
            valence, arousal = quadrant_va.get(quadrant, (0.0, 0.0))

            # Get GEMS from valence/arousal
            gems = GEMSRating.from_valence_arousal(valence, arousal)

            return EmotionAnnotatedSample(
                id=str(item.get("id", item.get("midi_id", ""))),
                source_dataset="emopia",
                file_path=item.get("path", item.get("midi_path")),
                file_type="midi",
                valence=valence,
                arousal=arousal,
                emotion_quadrant=quadrant,
                gems=gems.to_dict(),
                tempo=item.get("tempo"),
                key=item.get("key"),
                mode=item.get("mode"),
                genre="pop",
            )
        except Exception as e:
            logger.warning(f"Error processing EMOPIA item: {e}")
            return None

    def process_deam(self) -> List[EmotionAnnotatedSample]:
        """Process DEAM dataset into unified schema."""
        samples = []
        deam_dir = self.datasets_dir / "deam"

        if not deam_dir.exists():
            logger.warning(f"DEAM not found at {deam_dir}")
            return samples

        # Look for annotation files
        anno_files = list(deam_dir.glob("**/*.csv")) + list(deam_dir.glob("**/*.txt"))

        for anno_file in anno_files:
            try:
                # Parse annotation file (format varies)
                with open(anno_file) as f:
                    lines = f.readlines()

                for line in lines[1:]:  # Skip header
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        sample = EmotionAnnotatedSample(
                            id=parts[0],
                            source_dataset="deam",
                            file_type="audio",
                            valence=float(parts[1]) if len(parts) > 1 else None,
                            arousal=float(parts[2]) if len(parts) > 2 else None,
                        )
                        # Add GEMS
                        if sample.valence and sample.arousal:
                            gems = GEMSRating.from_valence_arousal(
                                sample.valence, sample.arousal
                            )
                            sample.gems = gems.to_dict()
                        samples.append(sample)

            except Exception as e:
                logger.warning(f"Error processing {anno_file}: {e}")

        logger.info(f"Processed {len(samples)} DEAM samples")
        return samples

    def process_all(self) -> List[EmotionAnnotatedSample]:
        """Process all datasets into unified format."""
        all_samples = []

        all_samples.extend(self.process_emopia())
        all_samples.extend(self.process_deam())

        logger.info(f"Total samples: {len(all_samples)}")
        return all_samples

    def add_therapeutic_scores(
        self, samples: List[EmotionAnnotatedSample], goal: str
    ) -> List[EmotionAnnotatedSample]:
        """Add therapeutic fitness scores to samples."""
        for sample in samples:
            if sample.valence is not None and sample.arousal is not None:
                score = self.therapeutic_recommender.score_track(
                    goal=goal,
                    track_valence=sample.valence,
                    track_arousal=sample.arousal,
                    track_tempo=sample.tempo,
                    track_mode=sample.mode,
                )
                sample.therapy_goal = goal
                sample.therapy_score = round(score, 3)

        return samples

    def save(
        self, samples: List[EmotionAnnotatedSample], output_path: Path
    ) -> None:
        """Save processed samples to JSONL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        logger.info(f"Saved {len(samples)} samples to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI for emotion dataset tools."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Kelly Emotion Dataset Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available datasets
    python -m penta_core.ml.datasets.emotion_datasets --list

    # Download datasets
    python -m penta_core.ml.datasets.emotion_datasets --download emopia
    python -m penta_core.ml.datasets.emotion_datasets --download all

    # Process datasets
    python -m penta_core.ml.datasets.emotion_datasets --process

    # Get therapeutic recommendations
    python -m penta_core.ml.datasets.emotion_datasets --recommend relaxation

    # Infer personality from preferences
    python -m penta_core.ml.datasets.emotion_datasets --personality jazz classical ambient
        """
    )

    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    parser.add_argument("--download", type=str, default=None,
                       help="Download dataset (emopia, deam, vgmidi, all)")
    parser.add_argument("--process", action="store_true",
                       help="Process downloaded datasets")
    parser.add_argument("--recommend", type=str, default=None,
                       help="Get therapeutic music profile")
    parser.add_argument("--personality", nargs="+", default=None,
                       help="Infer personality from music style preferences")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("datasets/emotion"),
                       help="Output directory")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Emotion-Annotated Music Datasets:")
        print("=" * 60)
        for ds, info in DATASET_INFO.items():
            print(f"\n{ds.value}: {info['name']}")
            print(f"  URL: {info['url']}")
            print(f"  Format: {info['format']}")
            print(f"  Annotations: {info['annotations']}")
            print(f"  Size: {info['size']}")
            print(f"  License: {info['license']}")
        return

    if args.download:
        downloader = EmotionDatasetDownloader(output_dir=args.output_dir)
        if args.download == "all":
            paths = downloader.download_all()
            for name, path in paths.items():
                print(f"  {name}: {path}")
        elif args.download == "emopia":
            print(downloader.download_emopia())
        elif args.download == "deam":
            print(downloader.download_deam())
        elif args.download == "vgmidi":
            print(downloader.download_vgmidi())
        else:
            print(f"Unknown dataset: {args.download}")
        return

    if args.process:
        processor = EmotionDatasetProcessor(args.output_dir)
        samples = processor.process_all()
        output_path = args.output_dir / "unified_emotion_dataset.jsonl"
        processor.save(samples, output_path)
        return

    if args.recommend:
        try:
            profile = TherapeuticRecommender.get_recommendation_profile(args.recommend)
            print(f"\nTherapeutic Profile for: {args.recommend}")
            print("=" * 40)
            for key, value in profile.items():
                print(f"  {key}: {value}")
        except ValueError as e:
            print(f"Error: {e}")
        return

    if args.personality:
        profile = PersonalityMapper.infer_from_preferences(args.personality)
        print(f"\nInferred Personality from: {', '.join(args.personality)}")
        print("=" * 40)
        for trait, value in profile.to_dict().items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {trait:20s} [{bar}] {value:.2f}")

        print("\nRecommended additional styles:")
        recs = PersonalityMapper.recommend_styles_for_personality(profile)
        for style, score in recs:
            print(f"  {style}: {score:.2f}")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
