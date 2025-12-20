"""
Audio Emotion Detection using SpeechBrain

Provides state-of-the-art emotion detection from audio files using
pre-trained SpeechBrain models. Integrates seamlessly with the existing
emotional_mapping.py system.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

try:
    from speechbrain.inference.interfaces import foreign_class
    import torch
    import torchaudio
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    foreign_class = None
    torch = None
    torchaudio = None

import numpy as np

from data.emotional_mapping import EmotionalState


class EmotionDetector:
    """
    Emotion detector using SpeechBrain's pre-trained wav2vec2 model.
    
    Maps detected emotions to valence-arousal space compatible with
    the existing emotional_mapping.py system.
    
    Example:
        >>> detector = EmotionDetector()
        >>> result = detector.detect_emotion("path/to/audio.wav")
        >>> print(f"Emotion: {result['emotion']}, Confidence: {result['confidence']:.2f}")
        >>> 
        >>> # Get EmotionalState for use with existing system
        >>> state = detector.get_emotional_state("path/to/audio.wav")
        >>> from data.emotional_mapping import get_parameters_for_state
        >>> params = get_parameters_for_state(state)
    """
    
    # Emotion to valence-arousal mapping based on Russell's circumplex model
    EMOTION_MAPPING = {
        "happy": {"valence": 0.7, "arousal": 0.8, "primary": "calm"},
        "sad": {"valence": -0.7, "arousal": 0.3, "primary": "grief"},
        "angry": {"valence": -0.6, "arousal": 0.9, "primary": "anger"},
        "neutral": {"valence": 0.0, "arousal": 0.4, "primary": "calm"},
    }
    
    def __init__(
        self,
        model_source: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        auto_load: bool = True
    ):
        """
        Initialize SpeechBrain emotion detector.
        
        Args:
            model_source: HuggingFace model identifier or local path.
                         Default uses IEMOCAP-trained model.
            auto_load: If True, load model immediately. If False, load on first use.
        
        Raises:
            ImportError: If SpeechBrain is not installed.
        """
        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError(
                "SpeechBrain is not installed. Install with:\n"
                "  pip install speechbrain\n"
                "  pip install torch torchaudio"
            )
        
        self.model_source = model_source
        self.classifier = None
        
        if auto_load:
            self._load_model()
    
    def _load_model(self):
        """Load the pre-trained emotion recognition model."""
        if self.classifier is not None:
            return
        
        try:
            # Try loading with custom interface first
            self.classifier = foreign_class(
                source=self.model_source,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier"
            )
        except Exception as e:
            # Fallback: try standard SpeechBrain interface
            warnings.warn(
                f"Could not load model with custom interface: {e}\n"
                "Trying alternative loading method..."
            )
            try:
                # Alternative: use SpeechBrain's standard emotion recognition
                from speechbrain.pretrained import EncoderClassifier
                self.classifier = EncoderClassifier.from_hparams(
                    source=self.model_source,
                    savedir=f"pretrained_models/{self.model_source.replace('/', '_')}"
                )
            except Exception as e2:
                warnings.warn(
                    f"Could not load model with standard interface: {e2}\n"
                    "Model will be loaded on first use."
                )
                self.classifier = None
    
    def detect_emotion(self, audio_path: Union[str, Path]) -> Dict:
        """
        Detect emotion from audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
        
        Returns:
            Dictionary with:
            - emotion: Detected emotion label (str)
            - confidence: Confidence score (float, 0-1)
            - valence: Valence value (float, -1 to 1)
            - arousal: Arousal value (float, 0 to 1)
            - primary_emotion: Mapped primary emotion (str)
            - probabilities: Full probability distribution (optional)
        
        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If model failed to load.
        """
        if self.classifier is None:
            self._load_model()
            if self.classifier is None:
                raise RuntimeError(
                    "Failed to load emotion detection model. "
                    "Check your internet connection and SpeechBrain installation."
                )
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Classify audio file
            out_prob, score, index, text_lab = self.classifier.classify_file(str(audio_path))
        except Exception as e:
            # Try alternative API if first attempt fails
            try:
                # Some SpeechBrain models use different API
                prediction = self.classifier.classify_file(str(audio_path))
                if isinstance(prediction, tuple):
                    out_prob, score, index, text_lab = prediction
                else:
                    # Handle dict or other formats
                    text_lab = str(prediction).lower()
                    score = 0.8  # Default confidence
                    out_prob = None
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to classify audio file: {e}. "
                    f"Alternative method also failed: {e2}"
                ) from e2
        
        # Normalize emotion label
        emotion_label = str(text_lab).lower().strip()
        
        # Map to valence-arousal
        emotion_info = self.EMOTION_MAPPING.get(
            emotion_label,
            {"valence": 0.0, "arousal": 0.5, "primary": "calm"}
        )
        
        # Convert score to float if needed
        confidence = float(score) if score is not None else 0.5
        
        result = {
            "emotion": emotion_label,
            "confidence": confidence,
            "valence": emotion_info["valence"],
            "arousal": emotion_info["arousal"],
            "primary_emotion": emotion_info["primary"],
        }
        
        # Add probabilities if available
        if out_prob is not None:
            try:
                if hasattr(out_prob, 'tolist'):
                    result["probabilities"] = out_prob.tolist()
                elif isinstance(out_prob, (list, tuple)):
                    result["probabilities"] = list(out_prob)
            except Exception:
                pass  # Skip if conversion fails
        
        return result
    
    def detect_emotion_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict:
        """
        Detect emotion from audio array (for real-time processing).
        
        Args:
            audio_array: Audio signal as numpy array (mono or stereo)
            sample_rate: Sample rate of audio (will be resampled to 16kHz if needed)
        
        Returns:
            Dictionary with emotion detection results (same format as detect_emotion)
        
        Raises:
            RuntimeError: If model failed to load.
        """
        if self.classifier is None:
            self._load_model()
            if self.classifier is None:
                raise RuntimeError("Failed to load emotion detection model")
        
        # Convert to torch tensor
        if isinstance(audio_array, np.ndarray):
            waveform = torch.from_numpy(audio_array).float()
        else:
            waveform = audio_array
        
        # Ensure correct shape: (channels, samples)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif len(waveform.shape) == 2:
            if waveform.shape[0] > waveform.shape[1]:
                # Assume (samples, channels) -> (channels, samples)
                waveform = waveform.transpose(0, 1)
            # If stereo, convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed (SpeechBrain models typically expect 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        try:
            # Classify
            out_prob, score, index, text_lab = self.classifier.classify_batch(waveform)
        except Exception as e:
            raise RuntimeError(f"Failed to classify audio array: {e}") from e
        
        # Normalize emotion label
        emotion_label = str(text_lab).lower().strip()
        
        # Map to valence-arousal
        emotion_info = self.EMOTION_MAPPING.get(
            emotion_label,
            {"valence": 0.0, "arousal": 0.5, "primary": "calm"}
        )
        
        confidence = float(score) if score is not None else 0.5
        
        result = {
            "emotion": emotion_label,
            "confidence": confidence,
            "valence": emotion_info["valence"],
            "arousal": emotion_info["arousal"],
            "primary_emotion": emotion_info["primary"],
        }
        
        if out_prob is not None:
            try:
                if hasattr(out_prob, 'tolist'):
                    result["probabilities"] = out_prob.tolist()
            except Exception:
                pass
        
        return result
    
    def get_emotional_state(
        self,
        audio_path: Union[str, Path],
        secondary_emotions: Optional[list] = None,
        has_intrusions: bool = False,
        intrusion_probability: float = 0.0
    ) -> EmotionalState:
        """
        Get EmotionalState from audio file (compatible with existing system).
        
        Args:
            audio_path: Path to audio file
            secondary_emotions: Optional list of secondary emotions
            has_intrusions: Whether PTSD/trauma intrusions are present
            intrusion_probability: Probability of intrusion events (0-1)
        
        Returns:
            EmotionalState instance ready for use with get_parameters_for_state()
        
        Example:
            >>> detector = EmotionDetector()
            >>> state = detector.get_emotional_state("audio.wav")
            >>> from data.emotional_mapping import get_parameters_for_state
            >>> params = get_parameters_for_state(state)
            >>> print(f"Suggested tempo: {params.tempo_suggested} BPM")
        """
        result = self.detect_emotion(audio_path)
        
        return EmotionalState(
            valence=result["valence"],
            arousal=result["arousal"],
            primary_emotion=result["primary_emotion"],
            secondary_emotions=secondary_emotions or [],
            has_intrusions=has_intrusions,
            intrusion_probability=intrusion_probability,
        )
    
    def batch_detect(self, audio_paths: list) -> list:
        """
        Detect emotions from multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
        
        Returns:
            List of emotion detection results, one per file
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.detect_emotion(audio_path)
                result["file"] = str(audio_path)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Error processing {audio_path}: {e}")
                results.append({
                    "file": str(audio_path),
                    "error": str(e),
                    "status": "error"
                })
        return results


# Convenience function for quick usage
def detect_emotion_from_audio(
    audio_path: Union[str, Path],
    model_source: Optional[str] = None
) -> Dict:
    """
    Quick function to detect emotion from audio file.
    
    Args:
        audio_path: Path to audio file
        model_source: Optional model source (uses default if None)
    
    Returns:
        Dictionary with emotion detection results
    
    Example:
        >>> result = detect_emotion_from_audio("song.wav")
        >>> print(f"Emotion: {result['emotion']}")
    """
    detector = EmotionDetector(
        model_source=model_source,
        auto_load=True
    )
    return detector.detect_emotion(audio_path)


def get_emotional_state_from_audio(
    audio_path: Union[str, Path],
    **kwargs
) -> EmotionalState:
    """
    Quick function to get EmotionalState from audio file.
    
    Args:
        audio_path: Path to audio file
        **kwargs: Additional arguments passed to get_emotional_state()
    
    Returns:
        EmotionalState instance
    
    Example:
        >>> state = get_emotional_state_from_audio("song.wav")
        >>> from data.emotional_mapping import get_parameters_for_state
        >>> params = get_parameters_for_state(state)
    """
    detector = EmotionDetector(auto_load=True)
    return detector.get_emotional_state(audio_path, **kwargs)
