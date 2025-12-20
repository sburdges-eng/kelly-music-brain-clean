"""
SpeechBrain Integration Example for miDiKompanion

This module demonstrates how to integrate SpeechBrain emotion recognition
into the existing miDiKompanion system.
"""

import warnings
from pathlib import Path
from typing import ClassVar

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import torch
    import torchaudio
    from speechbrain.inference.interfaces import foreign_class

    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    foreign_class = None  # type: ignore[assignment, misc]
    torch = None  # type: ignore[assignment, misc]
    torchaudio = None  # type: ignore[assignment, misc]
    warnings.warn(
        "SpeechBrain not installed. Install with: pip install speechbrain",
        stacklevel=2,
    )


class SpeechBrainEmotionDetector:
    """
    Emotion detector using SpeechBrain's pre-trained wav2vec2 model.

    Maps detected emotions to valence-arousal space compatible with
    the existing emotional_mapping.py system.
    """

    # Emotion to valence-arousal mapping
    # Based on Russell's circumplex model of emotion
    # Expanded to cover more emotions that SpeechBrain models may detect
    EMOTION_MAPPING: ClassVar[dict[str, dict[str, float | str]]] = {
        "happy": {"valence": 0.7, "arousal": 0.8, "primary": "calm"},
        "sad": {"valence": -0.7, "arousal": 0.3, "primary": "grief"},
        "angry": {"valence": -0.6, "arousal": 0.9, "primary": "anger"},
        "neutral": {"valence": 0.0, "arousal": 0.4, "primary": "calm"},
        # Additional emotions that may be detected
        "fearful": {"valence": -0.6, "arousal": 0.9, "primary": "anxiety"},
        "fear": {"valence": -0.6, "arousal": 0.9, "primary": "anxiety"},
        "surprised": {"valence": 0.3, "arousal": 0.8, "primary": "calm"},
        "surprise": {"valence": 0.3, "arousal": 0.8, "primary": "calm"},
        "disgusted": {"valence": -0.7, "arousal": 0.5, "primary": "anger"},
        "disgust": {"valence": -0.7, "arousal": 0.5, "primary": "anger"},
        "excited": {"valence": 0.7, "arousal": 0.9, "primary": "calm"},
        "calm": {"valence": 0.3, "arousal": 0.2, "primary": "calm"},
        "happiness": {"valence": 0.7, "arousal": 0.8, "primary": "calm"},
        "sadness": {"valence": -0.7, "arousal": 0.3, "primary": "grief"},
        "anger": {"valence": -0.6, "arousal": 0.9, "primary": "anger"},
    }

    def __init__(
        self,
        model_source: str = (
            "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
        ),
    ):
        """
        Initialize SpeechBrain emotion detector.

        Args:
            model_source: HuggingFace model identifier or local path
        """
        if not SPEECHBRAIN_AVAILABLE or foreign_class is None:
            msg = (
                "SpeechBrain not available. "
                "Install with: pip install speechbrain"
            )
            raise ImportError(msg)

        self.model_source = model_source
        self.classifier = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the pre-trained emotion recognition model."""
        if foreign_class is None:
            return

        try:
            self.classifier = foreign_class(
                source=self.model_source,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
            )
        except Exception as e:
            # Fallback: try standard SpeechBrain EncoderClassifier
            warnings.warn(
                f"Could not load custom interface: {e}\n"
                "Trying alternative loading method...",
                stacklevel=2,
            )
            try:
                # Alternative: try EncoderClassifier (may have different interface)
                from speechbrain.pretrained import EncoderClassifier

                savedir = (
                    f"pretrained_models/{self.model_source.replace('/', '_')}"
                )
                self.classifier = EncoderClassifier.from_hparams(
                    source=self.model_source,
                    savedir=savedir,
                )
                warnings.warn(
                    "Using EncoderClassifier fallback. "
                    "Some methods may not work as expected. "
                    "For best results, ensure custom_interface.py is available.",
                    stacklevel=2,
                )
            except Exception as e2:
                msg = (
                    f"Could not load model: {e}\n"
                    f"Fallback also failed: {e2}\n"
                    "Please check that SpeechBrain is properly installed, "
                    "the model source is correct, and "
                    "custom_interface.py exists."
                )
                warnings.warn(msg, stacklevel=2)
                raise RuntimeError(msg) from e2

    def detect_emotion(self, audio_path: str) -> dict:
        """
        Detect emotion from audio file.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)

        Returns:
            Dictionary with emotion, confidence, valence, arousal,
            and primary_emotion
        """
        if self.classifier is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            msg = f"Audio file not found: {audio_path_obj}"
            raise FileNotFoundError(msg)

        # Classify audio file
        # Note: Interface may vary between foreign_class and EncoderClassifier
        EXPECTED_TUPLE_LENGTH = 4
        try:
            result = self.classifier.classify_file(str(audio_path_obj))
            if (
                isinstance(result, tuple)
                and len(result) >= EXPECTED_TUPLE_LENGTH
            ):
                out_prob, score, _index, text_lab = result
            else:
                # Handle different return formats
                text_lab = str(result).lower() if result else "neutral"
                score = 0.5
                out_prob = None
        except AttributeError:
            # EncoderClassifier might use different method
            msg = (
                "Classifier interface not supported. "
                "Please use foreign_class with custom_interface.py"
            )
            raise RuntimeError(msg) from None

        # Map to valence-arousal
        emotion_info = self.EMOTION_MAPPING.get(
            text_lab.lower(),
            {"valence": 0.0, "arousal": 0.5, "primary": "calm"},
        )

        probabilities = (
            out_prob.tolist() if hasattr(out_prob, "tolist") else None
        )

        return {
            "emotion": text_lab,
            "confidence": float(score),
            "valence": emotion_info["valence"],
            "arousal": emotion_info["arousal"],
            "primary_emotion": emotion_info["primary"],
            "probabilities": probabilities,
        }

    def detect_emotion_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Detect emotion from audio array (for real-time processing).

        Args:
            audio_array: Audio signal as numpy array
            sample_rate: Sample rate of audio (will be resampled to 16kHz
                if needed)

        Returns:
            Dictionary with emotion, confidence, valence, arousal,
            and primary_emotion
        """
        if self.classifier is None or torch is None or torchaudio is None:
            msg = "Model not loaded or torch/torchaudio unavailable"
            raise RuntimeError(msg)

        if np is None:
            msg = "numpy is required"
            raise ImportError(msg)

        # Convert to torch tensor
        if isinstance(audio_array, np.ndarray):
            waveform = torch.from_numpy(audio_array).float()
        else:
            waveform = audio_array

        # Ensure correct shape: (channels, samples)
        MIN_DIMENSIONS = 1
        MAX_DIMENSIONS = 2
        if len(waveform.shape) == MIN_DIMENSIONS:
            waveform = waveform.unsqueeze(0)
        elif (
            len(waveform.shape) == MAX_DIMENSIONS
            and waveform.shape[0] > waveform.shape[1]
        ):
            # Assume (samples, channels) -> (channels, samples)
            waveform = waveform.transpose(0, 1)

        # Resample if needed (SpeechBrain models typically expect 16kHz)
        TARGET_SAMPLE_RATE = 16000
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate, TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Classify
        # Note: Interface may vary between foreign_class and EncoderClassifier
        EXPECTED_TUPLE_LENGTH = 4
        try:
            result = self.classifier.classify_batch(waveform)
            if (
                isinstance(result, tuple)
                and len(result) >= EXPECTED_TUPLE_LENGTH
            ):
                out_prob, score, _index, text_lab = result
            else:
                # Handle different return formats
                text_lab = str(result).lower() if result else "neutral"
                score = 0.5
                out_prob = None
        except AttributeError:
            # EncoderClassifier might use different method
            msg = (
                "Classifier interface not supported for batch processing. "
                "Please use foreign_class with custom_interface.py"
            )
            raise RuntimeError(msg) from None

        # Map to valence-arousal
        emotion_info = self.EMOTION_MAPPING.get(
            text_lab.lower(),
            {"valence": 0.0, "arousal": 0.5, "primary": "calm"},
        )

        probabilities = (
            out_prob.tolist() if hasattr(out_prob, "tolist") else None
        )

        return {
            "emotion": text_lab,
            "confidence": float(score),
            "valence": emotion_info["valence"],
            "arousal": emotion_info["arousal"],
            "primary_emotion": emotion_info["primary"],
            "probabilities": probabilities,
        }

    def batch_detect(self, audio_paths: list[str]) -> list[dict]:
        """
        Detect emotions from multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of emotion detection results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.detect_emotion(audio_path)
                result["file"] = str(audio_path)
                results.append(result)
            except Exception as e:
                warnings.warn(
                    f"Error processing {audio_path}: {e}",
                    stacklevel=2,
                )
                results.append({"file": str(audio_path), "error": str(e)})
        return results


# Integration with existing emotional_mapping.py
# Import at top level to avoid PLC0415 warning
try:
    from data.emotional_mapping import EmotionalState
except ImportError:
    EmotionalState = None  # type: ignore[assignment, misc]


def create_emotional_state_from_speechbrain(
    detector: SpeechBrainEmotionDetector,
    audio_path: str,
):
    """
    Create EmotionalState from SpeechBrain detection.

    Compatible with data/emotional_mapping.py EmotionalState class.
    """
    if EmotionalState is None:
        msg = "data.emotional_mapping.EmotionalState not available"
        raise ImportError(msg)

    result = detector.detect_emotion(audio_path)

    return EmotionalState(
        valence=result["valence"],
        arousal=result["arousal"],
        primary_emotion=result["primary_emotion"],
        secondary_emotions=[],
        has_intrusions=False,
        intrusion_probability=0.0,
    )


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = SpeechBrainEmotionDetector()

    # Example usage (commented out for demonstration)
    # Uncomment to test with your audio files:
    #
    # Example 1: Detect emotion from file
    # result = detector.detect_emotion("path/to/audio.wav")
    # print(
    #     f"Emotion: {result['emotion']}, "
    #     f"Confidence: {result['confidence']:.2f}"
    # )
    # print(
    #     f"Valence: {result['valence']:.2f}, "
    #     f"Arousal: {result['arousal']:.2f}"
    # )
    #
    # Example 2: Real-time processing
    # import librosa
    # audio, sr = librosa.load("path/to/audio.wav", sr=16000)
    # result = detector.detect_emotion_from_array(audio, sr)
    # print(result)
    #
    # Example 3: Integration with existing system
    # state = create_emotional_state_from_speechbrain(
    #     detector, "path/to/audio.wav"
    # )
    # from data.emotional_mapping import get_parameters_for_state
    # params = get_parameters_for_state(state)
    # print(f"Suggested tempo: {params.tempo_suggested} BPM")

    print("SpeechBrain integration example loaded successfully!")
    print("Uncomment examples above to test with your audio files.")
