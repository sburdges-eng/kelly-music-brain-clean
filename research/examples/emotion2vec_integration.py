"""
emotion2vec Integration Example for miDiKompanion

This module demonstrates how to integrate emotion2vec for emotion embeddings
and recognition using self-supervised pre-trained models.
"""

import warnings
from pathlib import Path
from typing import ClassVar

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from funasr import AutoModel

    EMOTION2VEC_AVAILABLE = True
except ImportError:
    EMOTION2VEC_AVAILABLE = False
    AutoModel = None  # type: ignore[assignment, misc]
    warnings.warn(
        "emotion2vec not installed. Install with: "
        "pip install -U funasr modelscope",
        stacklevel=2,
    )

try:
    import librosa
    import soundfile as sf

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None  # type: ignore[assignment, misc]
    sf = None  # type: ignore[assignment, misc]


# Integration with existing emotional_mapping.py
# Import at top level to avoid PLC0415 warning
try:
    from data.emotional_mapping import EmotionalState
except ImportError:
    EmotionalState = None  # type: ignore[assignment, misc]


class Emotion2VecExtractor:
    """
    Extract emotion embeddings using emotion2vec models.

    Provides two modes:
    1. Feature extraction (emotion2vec_base) - for custom downstream tasks
    2. Emotion recognition (emotion2vec_plus) - direct 9-class emotion
       prediction
    """

    # emotion2vec+ emotion classes
    EMOTION_CLASSES: ClassVar[list[str]] = [
        "happy",
        "sad",
        "angry",
        "fearful",
        "surprised",
        "disgusted",
        "neutral",
        "excited",
        "calm",
    ]

    # Emotion to valence-arousal mapping (9-class)
    EMOTION_MAPPING: ClassVar[dict[str, dict[str, float | str]]] = {
        "happy": {"valence": 0.8, "arousal": 0.7, "primary": "calm"},
        "sad": {"valence": -0.8, "arousal": 0.2, "primary": "grief"},
        "angry": {"valence": -0.7, "arousal": 0.9, "primary": "anger"},
        "fearful": {"valence": -0.6, "arousal": 0.9, "primary": "anxiety"},
        "surprised": {"valence": 0.3, "arousal": 0.8, "primary": "calm"},
        "disgusted": {"valence": -0.7, "arousal": 0.5, "primary": "anger"},
        "neutral": {"valence": 0.0, "arousal": 0.4, "primary": "calm"},
        "excited": {"valence": 0.7, "arousal": 0.9, "primary": "calm"},
        "calm": {"valence": 0.3, "arousal": 0.2, "primary": "calm"},
    }

    def __init__(
        self,
        model_type: str = "base",
        output_dir: str = "./emotion2vec_outputs",
    ):
        """
        Initialize emotion2vec extractor.

        Args:
            model_type: "base" (embeddings) or "plus" (emotion recognition)
            output_dir: Directory to save outputs
        """
        if not EMOTION2VEC_AVAILABLE:
            msg = (
                "emotion2vec not available. "
                "Install with: pip install -U funasr modelscope"
            )
            raise ImportError(msg)

        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Select model
        if model_type == "base":
            model_name = "iic/emotion2vec_base"
        elif model_type == "plus":
            model_name = "iic/emotion2vec_plus_large"
        else:
            msg = f"Unknown model_type: {model_type}. Use 'base' or 'plus'"
            raise ValueError(msg)

        if AutoModel is None:
            msg = "AutoModel not available"
            raise ImportError(msg)
        self.model = AutoModel(model=model_name)
        self.model_name = model_name

    def extract_embeddings(self, audio_path: str | Path) -> dict:
        """
        Extract emotion embeddings from audio file.

        Args:
            audio_path: Path to audio file (must be 16kHz mono WAV)

        Returns:
            Dictionary with embeddings and metadata
        """
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            msg = f"Audio file not found: {audio_path_obj}"
            raise FileNotFoundError(msg)

        # Ensure audio is 16kHz mono WAV (emotion2vec requirement)
        self._validate_audio_format(audio_path_obj)

        # Extract embeddings
        result = self.model(
            input=str(audio_path_obj),
            output_dir=str(self.output_dir),
            granularity="utterance",
            extract_embedding=True,
        )

        # Load embeddings from output directory
        # Note: Actual file path depends on funasr output structure
        # You may need to adjust this based on actual output
        embedding_file = self.output_dir / f"{audio_path_obj.stem}.npy"

        embeddings = None
        if embedding_file.exists() and np is not None:
            embeddings = np.load(embedding_file)
        else:
            # Try to find embedding file in result
            # This may vary based on funasr version
            warnings.warn(
                f"Embedding file not found at {embedding_file}",
                stacklevel=2,
            )

        return {
            "embeddings": embeddings,
            "audio_path": str(audio_path_obj),
            "model": self.model_name,
            "result": result,
        }

    def recognize_emotion(self, audio_path: str | Path) -> dict:
        """
        Recognize emotion directly (using emotion2vec+).

        Args:
            audio_path: Path to audio file (must be 16kHz mono WAV)

        Returns:
            Dictionary with emotion, confidence, valence, arousal
        """
        if self.model_type != "plus":
            msg = "Use model_type='plus' for emotion recognition"
            raise ValueError(msg)

        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            msg = f"Audio file not found: {audio_path_obj}"
            raise FileNotFoundError(msg)

        # Ensure audio is 16kHz mono WAV
        self._validate_audio_format(audio_path_obj)

        # Recognize emotion
        result = self.model(
            input=str(audio_path_obj),
            output_dir=str(self.output_dir),
            granularity="utterance",
            extract_embedding=False,
        )

        # Parse result (structure may vary)
        # This is a placeholder - adjust based on actual funasr output
        emotion = "neutral"
        confidence = 0.5

        if isinstance(result, dict):
            emotion = result.get("emotion", "neutral")
            confidence = result.get("confidence", 0.5)
        elif isinstance(result, list) and len(result) > 0:
            # Handle list output format
            emotion = result[0].get("emotion", "neutral")
            confidence = result[0].get("confidence", 0.5)

        # Map to valence-arousal
        emotion_info = self.EMOTION_MAPPING.get(
            emotion.lower(),
            {"valence": 0.0, "arousal": 0.5, "primary": "calm"},
        )

        return {
            "emotion": emotion,
            "confidence": confidence,
            "valence": emotion_info["valence"],
            "arousal": emotion_info["arousal"],
            "primary_emotion": emotion_info["primary"],
            "raw_result": result,
        }

    def _validate_audio_format(self, audio_path: Path) -> None:
        """
        Validate audio format (should be 16kHz mono WAV).

        Args:
            audio_path: Path to audio file
        """
        # Check extension
        if audio_path.suffix.lower() != ".wav":
            warnings.warn(
                f"Audio file {audio_path} is not WAV format. "
                "emotion2vec works best with 16kHz mono WAV files.",
                stacklevel=2,
            )

        # Note: Actual format validation would require loading the file
        # This is a placeholder - you may want to add actual validation
        # using librosa or soundfile to check sample rate and channels

    def convert_to_16khz_mono(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Convert audio to 16kHz mono WAV format required by emotion2vec.

        Args:
            input_path: Path to input audio file
            output_path: Optional output path (default: input_path with
                _16khz.wav suffix)

        Returns:
            Path to converted file
        """
        if not LIBROSA_AVAILABLE or librosa is None or sf is None:
            msg = (
                "librosa and soundfile required for audio conversion. "
                "Install with: pip install librosa soundfile"
            )
            raise ImportError(msg)

        input_path_obj = Path(input_path)
        if output_path is None:
            output_path_obj = (
                input_path_obj.parent / f"{input_path_obj.stem}_16khz.wav"
            )
        else:
            output_path_obj = Path(output_path)

        # Load and convert
        y, _ = librosa.load(str(input_path_obj), sr=16000, mono=True)

        # Save as WAV
        sf.write(str(output_path_obj), y, 16000)

        return output_path_obj


# Integration helper
def create_emotional_state_from_emotion2vec(
    extractor: Emotion2VecExtractor,
    audio_path: str | Path,
):
    """
    Create EmotionalState from emotion2vec recognition.

    Compatible with data/emotional_mapping.py EmotionalState class.
    """
    if EmotionalState is None:
        msg = "data.emotional_mapping.EmotionalState not available"
        raise ImportError(msg)

    if extractor.model_type != "plus":
        msg = "Use model_type='plus' for emotion recognition"
        raise ValueError(msg)

    result = extractor.recognize_emotion(audio_path)

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
    if not EMOTION2VEC_AVAILABLE:
        print(
            "emotion2vec not installed. "
            "Install with: pip install -U funasr modelscope"
        )
    else:
        # Example usage (commented out for demonstration)
        # Uncomment to test with your audio files:
        #
        # Example 1: Extract embeddings
        # extractor_base = Emotion2VecExtractor(model_type="base")
        # result = extractor_base.extract_embeddings("path/to/audio.wav")
        # shape = (
        #     result['embeddings'].shape
        #     if result['embeddings'] is not None
        #     else 'None'
        # )
        # print(f"Embedding shape: {shape}")
        #
        # Example 2: Recognize emotion
        # extractor_plus = Emotion2VecExtractor(model_type="plus")
        # result = extractor_plus.recognize_emotion("path/to/audio.wav")
        # print(
        #     f"Emotion: {result['emotion']}, "
        #     f"Confidence: {result['confidence']:.2f}"
        # )
        # print(
        #     f"Valence: {result['valence']:.2f}, "
        #     f"Arousal: {result['arousal']:.2f}"
        # )
        #
        # Example 3: Convert audio format
        # extractor = Emotion2VecExtractor(model_type="plus")
        # converted = extractor.convert_to_16khz_mono("path/to/audio.mp3")
        # result = extractor.recognize_emotion(converted)
        #
        # Example 4: Integration with existing system
        # extractor = Emotion2VecExtractor(model_type="plus")
        # state = create_emotional_state_from_emotion2vec(
        #     extractor, "path/to/audio.wav"
        # )
        # from data.emotional_mapping import get_parameters_for_state
        # params = get_parameters_for_state(state)
        # print(f"Estimated valence: {state.valence:.2f}")
        # print(f"Estimated arousal: {state.arousal:.2f}")
        # print(f"Primary emotion: {state.primary_emotion}")

        print("emotion2vec integration example loaded successfully!")
        print("Uncomment examples above to test with your audio files.")
        print("Note: emotion2vec requires 16kHz mono WAV files.")
