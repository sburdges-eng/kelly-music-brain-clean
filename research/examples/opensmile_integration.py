"""
OpenSMILE Integration Example for miDiKompanion

This module demonstrates how to integrate OpenSMILE feature extraction
to enhance emotion detection with industry-standard acoustic features.
"""

import warnings
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import opensmile

    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    opensmile = None  # type: ignore[assignment, misc]
    warnings.warn(
        "OpenSMILE not installed. Install with: pip install opensmile",
        stacklevel=2,
    )

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None  # type: ignore[assignment, misc]
    warnings.warn(
        "Librosa not installed. Install with: pip install librosa",
        stacklevel=2,
    )


class OpenSMILEFeatureExtractor:
    """
    Extract acoustic features using OpenSMILE.

    Provides eGeMAPS (88 features) and ComParE (6,373 features) sets
    optimized for emotion recognition.
    """

    def __init__(
        self,
        feature_set: str = "eGeMAPSv02",
        feature_level: str = "Functionals",
    ):
        """
        Initialize OpenSMILE feature extractor.

        Args:
            feature_set: "eGeMAPSv02" (88 features) or "ComParE_2016"
                (6,373 features)
            feature_level: "Functionals" (summary) or "LowLevelDescriptors"
                (frame-level)
        """
        if not OPENSMILE_AVAILABLE or opensmile is None:
            msg = (
                "OpenSMILE not available. "
                "Install with: pip install opensmile"
            )
            raise ImportError(msg)

        # Map string to OpenSMILE enum
        feature_set_map = {
            "eGeMAPSv02": opensmile.FeatureSet.eGeMAPSv02,
            "ComParE_2016": opensmile.FeatureSet.ComParE_2016,
        }

        feature_level_map = {
            "Functionals": opensmile.FeatureLevel.Functionals,
            "LowLevelDescriptors": (
                opensmile.FeatureLevel.LowLevelDescriptors
            ),
        }

        default_set = opensmile.FeatureSet.eGeMAPSv02
        default_level = opensmile.FeatureLevel.Functionals
        self.smile = opensmile.Smile(
            feature_set=feature_set_map.get(feature_set, default_set),
            feature_level=feature_level_map.get(feature_level, default_level),
        )
        self.feature_set_name = feature_set
        self.feature_level_name = feature_level

    def extract_features(self, audio_path: str | Path) -> np.ndarray:
        """
        Extract features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            NumPy array of features
        """
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            msg = f"Audio file not found: {audio_path_obj}"
            raise FileNotFoundError(msg)

        # Extract features (returns pandas DataFrame)
        features_df = self.smile.process_file(str(audio_path_obj))

        # Convert to numpy array (first row if multiple rows)
        if len(features_df) > 0:
            features_array = features_df.values[0]
        else:
            features_array = features_df.values

        return features_array

    def extract_features_from_signal(
        self,
        audio_signal: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Extract features from audio signal array.

        Args:
            audio_signal: Audio signal as numpy array
            sample_rate: Sample rate of audio

        Returns:
            NumPy array of features
        """
        if np is None:
            msg = "numpy is required"
            raise ImportError(msg)

        # OpenSMILE expects mono signal
        if len(audio_signal.shape) > 1:
            audio_signal = np.mean(audio_signal, axis=0)

        # Extract features
        features_df = self.smile.process_signal(
            audio_signal, sampling_rate=sample_rate
        )

        # Convert to numpy array
        if len(features_df) > 0:
            features_array = features_df.values[0]
        else:
            features_array = features_df.values

        return features_array

    def get_feature_names(self) -> list[str]:
        """
        Get names of extracted features.

        Returns:
            List of feature names
        """
        if np is None:
            msg = "numpy is required"
            raise ImportError(msg)

        # Create dummy signal to get feature names
        dummy_signal = np.zeros(16000)  # 1 second at 16kHz
        features_df = self.smile.process_signal(
            dummy_signal, sampling_rate=16000
        )
        return list(features_df.columns)


class EnhancedFeatureExtractor:
    """
    Combine OpenSMILE and librosa features for comprehensive emotion analysis.

    This provides the best of both worlds:
    - OpenSMILE: Industry-standard acoustic features
    - Librosa: Music-specific features (chroma, tonnetz, etc.)
    """

    def __init__(
        self,
        opensmile_set: str = "eGeMAPSv02",
        include_librosa: bool = True,
    ):
        """
        Initialize enhanced feature extractor.

        Args:
            opensmile_set: OpenSMILE feature set ("eGeMAPSv02" or
                "ComParE_2016")
            include_librosa: Whether to include librosa features
        """
        self.opensmile_extractor = OpenSMILEFeatureExtractor(
            feature_set=opensmile_set,
            feature_level="Functionals",
        )
        self.include_librosa = include_librosa and LIBROSA_AVAILABLE

    def extract_all_features(
        self, audio_path: str | Path
    ) -> dict[str, np.ndarray]:
        """
        Extract all features (OpenSMILE + librosa).

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with 'opensmile', 'librosa', and 'combined' features
        """
        if np is None:
            msg = "numpy is required"
            raise ImportError(msg)

        audio_path_obj = Path(audio_path)

        # Extract OpenSMILE features
        opensmile_features = (
            self.opensmile_extractor.extract_features(audio_path_obj)
        )

        result = {
            "opensmile": opensmile_features,
            "opensmile_names": (
                self.opensmile_extractor.get_feature_names()
            ),
        }

        # Extract librosa features if requested
        if self.include_librosa:
            librosa_features = self._extract_librosa_features(audio_path_obj)
            result["librosa"] = librosa_features

            # Combine features
            combined = np.concatenate(
                [opensmile_features, librosa_features]
            )
            result["combined"] = combined
            result["combined_dim"] = len(combined)
        else:
            result["combined"] = opensmile_features
            result["combined_dim"] = len(opensmile_features)

        return result

    def _extract_librosa_features(self, audio_path: Path) -> np.ndarray:
        """
        Extract librosa features (similar to existing system).

        Args:
            audio_path: Path to audio file

        Returns:
            NumPy array of librosa features
        """
        if not LIBROSA_AVAILABLE or librosa is None or np is None:
            return np.array([])

        y, sr = librosa.load(str(audio_path), sr=44100, duration=3.0)
        features = []

        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(mfccs.mean(axis=1))

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(spectral_centroids.mean())

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(spectral_rolloff.mean())

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        features.append(zero_crossing_rate.mean())

        # Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma(y=y, sr=sr)
        features.extend(chroma.mean(axis=1))

        # Tonnetz (6 features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features.extend(tonnetz.mean(axis=1))

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo / 200.0)  # Normalize

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features.append(rms.mean())

        # Spectral contrast (7 bands)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(spectral_contrast.mean(axis=1))

        # Harmonic/percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features.append(np.mean(np.abs(y_harmonic)))
        features.append(np.mean(np.abs(y_percussive)))

        return np.array(features)


# Integration with existing emotional_mapping.py
# Import at top level to avoid PLC0415 warning
try:
    from data.emotional_mapping import EmotionalState
except ImportError:
    EmotionalState = None  # type: ignore[assignment, misc]


def estimate_emotion_from_features(
    features: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Estimate valence and arousal from OpenSMILE features using heuristics.

    Note: This is a basic heuristic mapping. For production use, train a
    classifier (e.g., SVM, Random Forest) on labeled emotion data.

    Args:
        features: Extracted feature array
        feature_names: Optional list of feature names (for eGeMAPSv02)

    Returns:
        Dictionary with valence, arousal, and primary_emotion estimates
    """
    if np is None:
        msg = "numpy is required"
        raise ImportError(msg)

    # Default to neutral
    valence = 0.0
    arousal = 0.5
    primary_emotion = "neutral"

    if feature_names is None or len(features) != len(feature_names):
        # Fallback: use simple statistics if feature names unavailable
        # Higher energy -> higher arousal
        # More spectral variation -> potentially negative valence
        if len(features) > 0:
            mean_feature = np.mean(np.abs(features))
            std_feature = np.std(features)
            arousal = min(1.0, max(0.0, mean_feature * 2.0))
            valence = np.clip(std_feature - 0.5, -1.0, 1.0)
    else:
        # Use eGeMAPSv02 feature names for better mapping
        feature_dict = dict(zip(feature_names, features))

        # Arousal indicators (energy, loudness, F0)
        arousal_features = [
            "loudness_sma3_amean",
            "F0semitoneFrom27.5Hz_sma3nz_amean",
            "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
            "jitterLocal_sma3nz_amean",
            "shimmerLocaldB_sma3nz_amean",
        ]

        # Valence indicators (spectral features, formants)
        valence_features = [
            "spectralFlux_sma3_amean",
            "mfcc1_sma3_amean",
            "mfcc2_sma3_amean",
            "F1frequency_sma3nz_amean",
            "F2frequency_sma3nz_amean",
        ]

        # Calculate arousal (energy-based)
        arousal_values = [
            feature_dict.get(name, 0.0)
            for name in arousal_features
            if name in feature_dict
        ]
        if arousal_values:
            arousal = np.clip(
                np.mean(arousal_values) * 0.5 + 0.5, 0.0, 1.0
            )

        # Calculate valence (spectral-based)
        valence_values = [
            feature_dict.get(name, 0.0)
            for name in valence_features
            if name in feature_dict
        ]
        if valence_values:
            # Normalize and map to [-1, 1]
            valence = np.clip(np.mean(valence_values), -1.0, 1.0)

    # Map to primary emotion based on quadrant
    if valence > 0.3 and arousal > 0.6:
        primary_emotion = "excited"
    elif valence > 0.3 and arousal <= 0.6:
        primary_emotion = "calm"
    elif valence <= -0.3 and arousal > 0.6:
        primary_emotion = "anger"
    elif valence <= -0.3 and arousal <= 0.6:
        primary_emotion = "grief"
    else:
        primary_emotion = "neutral"

    return {
        "valence": float(valence),
        "arousal": float(arousal),
        "primary_emotion": primary_emotion,
    }


def create_emotional_state_from_opensmile(
    extractor: OpenSMILEFeatureExtractor | EnhancedFeatureExtractor,
    audio_path: str | Path,
) -> EmotionalState:
    """
    Create EmotionalState from OpenSMILE features.

    Compatible with data/emotional_mapping.py EmotionalState class.

    Note: This uses heuristic mapping. For better accuracy, train a
    classifier on labeled emotion data using OpenSMILE features.

    Args:
        extractor: OpenSMILEFeatureExtractor or EnhancedFeatureExtractor
        audio_path: Path to audio file

    Returns:
        EmotionalState object
    """
    if EmotionalState is None:
        msg = "data.emotional_mapping.EmotionalState not available"
        raise ImportError(msg)

    if isinstance(extractor, EnhancedFeatureExtractor):
        # Use combined features
        result = extractor.extract_all_features(audio_path)
        features = result["combined"]
        feature_names = result.get("opensmile_names", None)
    else:
        # Use OpenSMILE features only
        features = extractor.extract_features(audio_path)
        feature_names = extractor.get_feature_names()

    # Estimate emotion from features
    emotion_estimate = estimate_emotion_from_features(features, feature_names)

    return EmotionalState(
        valence=emotion_estimate["valence"],
        arousal=emotion_estimate["arousal"],
        primary_emotion=emotion_estimate["primary_emotion"],
        secondary_emotions=[],
        has_intrusions=False,
        intrusion_probability=0.0,
    )


# Example usage
if __name__ == "__main__":
    if not OPENSMILE_AVAILABLE:
        print(
            "OpenSMILE not installed. "
            "Install with: pip install opensmile"
        )
    else:
        # Example usage (commented out for demonstration)
        # Uncomment to test with your audio files:
        #
        # Example 1: Extract OpenSMILE features only
        # extractor = OpenSMILEFeatureExtractor(feature_set="eGeMAPSv02")
        # features = extractor.extract_features("path/to/audio.wav")
        # print(f"Extracted {len(features)} features")
        # names = extractor.get_feature_names()[:5]
        # print(f"Feature names: {names}...")  # First 5
        #
        # Example 2: Extract combined features
        # enhanced = EnhancedFeatureExtractor(
        #     opensmile_set="eGeMAPSv02", include_librosa=True
        # )
        # result = enhanced.extract_all_features("path/to/audio.wav")
        # print(f"OpenSMILE features: {len(result['opensmile'])}")
        # print(f"Librosa features: {len(result['librosa'])}")
        # print(f"Combined features: {len(result['combined'])}")
        #
        # Example 3: Integration with existing system
        # extractor = OpenSMILEFeatureExtractor(feature_set="eGeMAPSv02")
        # state = create_emotional_state_from_opensmile(
        #     extractor, "path/to/audio.wav"
        # )
        # from data.emotional_mapping import get_parameters_for_state
        # params = get_parameters_for_state(state)
        # print(f"Estimated valence: {state.valence:.2f}")
        # print(f"Estimated arousal: {state.arousal:.2f}")
        # print(f"Primary emotion: {state.primary_emotion}")

        print("OpenSMILE integration example loaded successfully!")
        print("Uncomment examples above to test with your audio files.")
        print(
            "Note: Emotion estimation uses heuristics. "
            "For production, train a classifier on labeled data."
        )
