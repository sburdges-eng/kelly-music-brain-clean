"""
Advanced Audio Data Augmentation for Kelly ML Training.

Provides music-aware augmentation techniques that preserve musical properties
while increasing dataset diversity.

Augmentations:
- Time stretching (tempo change without pitch shift)
- Pitch shifting (transposition)
- Noise injection (various noise types)
- Room simulation (reverb/acoustics)
- Dynamic range manipulation
- Spectral augmentation (SpecAugment)
- Mixup and CutMix for audio
- Music-specific: key transposition, tempo scaling

Usage:
    from python.penta_core.ml.training.augmentation import AudioAugmentor
    
    augmentor = AudioAugmentor()
    augmented = augmentor.augment(waveform, sr=16000)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation."""
    
    # Time domain
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)
    pitch_shift_range: Tuple[int, int] = (-4, 4)  # semitones
    
    # Noise
    noise_snr_range: Tuple[float, float] = (10, 30)  # dB
    noise_types: List[str] = field(default_factory=lambda: ["white", "pink", "brown"])
    
    # Dynamics
    gain_range: Tuple[float, float] = (-6, 6)  # dB
    compression_threshold: float = -20  # dB
    
    # Spectral
    spec_freq_mask_param: int = 20  # max frequency bands to mask
    spec_time_mask_param: int = 40  # max time steps to mask
    spec_num_masks: int = 2
    
    # Mixup
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Probabilities
    p_time_stretch: float = 0.3
    p_pitch_shift: float = 0.3
    p_noise: float = 0.2
    p_gain: float = 0.3
    p_reverb: float = 0.2
    p_spec_augment: float = 0.5


class AudioAugmentor:
    """
    Advanced audio augmentation for music ML training.
    
    Supports both waveform and spectrogram augmentations.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self._librosa_available = self._check_librosa()
    
    def _check_librosa(self) -> bool:
        try:
            import librosa
            return True
        except ImportError:
            logger.warning("librosa not available, some augmentations disabled")
            return False
    
    # =========================================================================
    # Time Domain Augmentations
    # =========================================================================
    
    def time_stretch(
        self,
        y: np.ndarray,
        sr: int,
        rate: Optional[float] = None,
    ) -> np.ndarray:
        """
        Time stretch without changing pitch.
        
        Args:
            y: Audio waveform
            sr: Sample rate
            rate: Stretch factor (>1 = faster, <1 = slower)
        
        Returns:
            Time-stretched audio
        """
        if rate is None:
            rate = random.uniform(*self.config.time_stretch_range)
        
        if not self._librosa_available:
            # Simple resampling fallback
            target_length = int(len(y) / rate)
            indices = np.linspace(0, len(y) - 1, target_length).astype(int)
            return y[indices]
        
        import librosa
        return librosa.effects.time_stretch(y, rate=rate)
    
    def pitch_shift(
        self,
        y: np.ndarray,
        sr: int,
        n_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Pitch shift (transpose) audio.
        
        Args:
            y: Audio waveform
            sr: Sample rate
            n_steps: Semitones to shift
        
        Returns:
            Pitch-shifted audio
        """
        if n_steps is None:
            n_steps = random.randint(*self.config.pitch_shift_range)
        
        if not self._librosa_available:
            logger.warning("Pitch shift requires librosa")
            return y
        
        import librosa
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    def add_noise(
        self,
        y: np.ndarray,
        noise_type: str = "white",
        snr_db: Optional[float] = None,
    ) -> np.ndarray:
        """
        Add noise to audio.
        
        Args:
            y: Audio waveform
            noise_type: "white", "pink", "brown", or "environmental"
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            Noisy audio
        """
        if snr_db is None:
            snr_db = random.uniform(*self.config.noise_snr_range)
        
        # Generate noise
        if noise_type == "white":
            noise = np.random.randn(len(y))
        elif noise_type == "pink":
            noise = self._generate_pink_noise(len(y))
        elif noise_type == "brown":
            noise = self._generate_brown_noise(len(y))
        else:
            noise = np.random.randn(len(y))
        
        # Scale noise to achieve target SNR
        signal_power = np.mean(y ** 2)
        noise_power = np.mean(noise ** 2)
        
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise_scale = np.sqrt(target_noise_power / (noise_power + 1e-10))
        
        return y + noise * noise_scale
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink (1/f) noise."""
        white = np.random.randn(length)
        
        # Simple approximation using cumulative filter
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        
        from scipy.signal import lfilter
        return lfilter(b, a, white)
    
    def _generate_brown_noise(self, length: int) -> np.ndarray:
        """Generate brown (1/f²) noise via integration."""
        white = np.random.randn(length)
        brown = np.cumsum(white)
        return brown / np.max(np.abs(brown) + 1e-10)
    
    def apply_gain(
        self,
        y: np.ndarray,
        gain_db: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply gain (volume change) to audio.
        
        Args:
            y: Audio waveform
            gain_db: Gain in decibels
        
        Returns:
            Gained audio
        """
        if gain_db is None:
            gain_db = random.uniform(*self.config.gain_range)
        
        gain_linear = 10 ** (gain_db / 20)
        return y * gain_linear
    
    def apply_compression(
        self,
        y: np.ndarray,
        threshold_db: float = -20,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        sr: int = 16000,
    ) -> np.ndarray:
        """
        Apply dynamic range compression.
        
        Args:
            y: Audio waveform
            threshold_db: Compression threshold
            ratio: Compression ratio
            attack_ms: Attack time
            release_ms: Release time
            sr: Sample rate
        
        Returns:
            Compressed audio
        """
        # Convert to dB
        eps = 1e-10
        y_db = 20 * np.log10(np.abs(y) + eps)
        
        # Calculate gain reduction
        over_threshold = y_db - threshold_db
        over_threshold = np.maximum(over_threshold, 0)
        gain_reduction = over_threshold * (1 - 1/ratio)
        
        # Apply smoothing (simple exponential)
        attack_samples = int(attack_ms * sr / 1000)
        release_samples = int(release_ms * sr / 1000)
        
        smoothed_gain = np.zeros_like(gain_reduction)
        for i in range(1, len(gain_reduction)):
            if gain_reduction[i] > smoothed_gain[i-1]:
                # Attack
                alpha = 1 - np.exp(-1 / attack_samples)
            else:
                # Release
                alpha = 1 - np.exp(-1 / release_samples)
            smoothed_gain[i] = alpha * gain_reduction[i] + (1 - alpha) * smoothed_gain[i-1]
        
        # Apply gain reduction
        gain_linear = 10 ** (-smoothed_gain / 20)
        return y * gain_linear
    
    def add_reverb(
        self,
        y: np.ndarray,
        sr: int,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3,
    ) -> np.ndarray:
        """
        Add reverb effect (simple convolution-based).
        
        Args:
            y: Audio waveform
            sr: Sample rate
            room_size: Room size (0-1)
            damping: High frequency damping (0-1)
            wet_level: Wet/dry mix (0-1)
        
        Returns:
            Reverberant audio
        """
        # Generate simple impulse response
        ir_length = int(room_size * sr)  # Up to 1 second
        ir = np.zeros(ir_length)
        
        # Early reflections
        n_reflections = int(10 + room_size * 20)
        for i in range(n_reflections):
            delay = int(random.random() * ir_length * 0.3)
            amplitude = random.random() * 0.5 * (1 - i / n_reflections)
            if delay < ir_length:
                ir[delay] += amplitude
        
        # Diffuse tail (exponential decay)
        decay = np.exp(-np.linspace(0, 5 * (1 - room_size + 0.1), ir_length))
        noise = np.random.randn(ir_length) * decay
        
        # Apply damping (low-pass filter)
        if damping > 0:
            from scipy.ndimage import uniform_filter1d
            kernel_size = int(1 + damping * 10)
            noise = uniform_filter1d(noise, kernel_size)
        
        ir += noise * 0.3
        ir = ir / (np.max(np.abs(ir)) + 1e-10)
        
        # Convolve
        from scipy.signal import fftconvolve
        wet = fftconvolve(y, ir, mode='same')
        
        # Mix
        return (1 - wet_level) * y + wet_level * wet
    
    # =========================================================================
    # Spectral Augmentations
    # =========================================================================
    
    def spec_augment(
        self,
        spec: np.ndarray,
        freq_mask_param: Optional[int] = None,
        time_mask_param: Optional[int] = None,
        num_masks: Optional[int] = None,
    ) -> np.ndarray:
        """
        SpecAugment: Frequency and time masking for spectrograms.
        
        Args:
            spec: Spectrogram (freq x time)
            freq_mask_param: Max frequency bands to mask
            time_mask_param: Max time steps to mask
            num_masks: Number of masks to apply
        
        Returns:
            Augmented spectrogram
        """
        freq_mask_param = freq_mask_param or self.config.spec_freq_mask_param
        time_mask_param = time_mask_param or self.config.spec_time_mask_param
        num_masks = num_masks or self.config.spec_num_masks
        
        spec = spec.copy()
        n_freq, n_time = spec.shape
        
        # Frequency masking
        for _ in range(num_masks):
            f = random.randint(0, min(freq_mask_param, n_freq))
            f0 = random.randint(0, n_freq - f)
            spec[f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(num_masks):
            t = random.randint(0, min(time_mask_param, n_time))
            t0 = random.randint(0, n_time - t)
            spec[:, t0:t0 + t] = 0
        
        return spec
    
    def frequency_warp(
        self,
        spec: np.ndarray,
        warp_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Warp frequency axis of spectrogram.
        
        Args:
            spec: Spectrogram (freq x time)
            warp_factor: Warping factor
        
        Returns:
            Warped spectrogram
        """
        n_freq, n_time = spec.shape
        
        # Create warped frequency indices
        orig_indices = np.arange(n_freq)
        warped_indices = orig_indices ** warp_factor
        warped_indices = warped_indices / warped_indices[-1] * (n_freq - 1)
        
        # Interpolate
        from scipy.interpolate import interp1d
        warped_spec = np.zeros_like(spec)
        
        for t in range(n_time):
            f = interp1d(warped_indices, spec[:, t], fill_value="extrapolate")
            warped_spec[:, t] = f(orig_indices)
        
        return warped_spec
    
    # =========================================================================
    # Mixup and CutMix
    # =========================================================================
    
    def mixup(
        self,
        y1: np.ndarray,
        y2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation: linear interpolation of samples.
        
        Args:
            y1, y2: Audio waveforms
            label1, label2: One-hot labels
            alpha: Beta distribution parameter
        
        Returns:
            Mixed audio and labels
        """
        alpha = alpha or self.config.mixup_alpha
        
        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Ensure same length
        min_len = min(len(y1), len(y2))
        y1, y2 = y1[:min_len], y2[:min_len]
        
        # Mix
        mixed_y = lam * y1 + (1 - lam) * y2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_y, mixed_label
    
    def cutmix(
        self,
        spec1: np.ndarray,
        spec2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutMix for spectrograms: cut and paste regions.
        
        Args:
            spec1, spec2: Spectrograms
            label1, label2: One-hot labels
            alpha: Beta distribution parameter
        
        Returns:
            Mixed spectrogram and labels
        """
        alpha = alpha or self.config.cutmix_alpha
        
        lam = np.random.beta(alpha, alpha)
        n_freq, n_time = spec1.shape
        
        # Calculate cut size
        cut_ratio = np.sqrt(1 - lam)
        cut_freq = int(n_freq * cut_ratio)
        cut_time = int(n_time * cut_ratio)
        
        # Random position
        f0 = random.randint(0, n_freq - cut_freq)
        t0 = random.randint(0, n_time - cut_time)
        
        # Apply cut
        mixed_spec = spec1.copy()
        mixed_spec[f0:f0+cut_freq, t0:t0+cut_time] = spec2[f0:f0+cut_freq, t0:t0+cut_time]
        
        # Adjust lambda based on actual area
        lam = 1 - (cut_freq * cut_time) / (n_freq * n_time)
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_spec, mixed_label
    
    # =========================================================================
    # Music-Specific Augmentations
    # =========================================================================
    
    def transpose_key(
        self,
        y: np.ndarray,
        sr: int,
        semitones: Optional[int] = None,
    ) -> np.ndarray:
        """
        Transpose audio by a number of semitones (musical key change).
        
        Args:
            y: Audio waveform
            sr: Sample rate
            semitones: Number of semitones to transpose
        
        Returns:
            Transposed audio
        """
        if semitones is None:
            # Random transposition within an octave
            semitones = random.randint(-6, 6)
        
        return self.pitch_shift(y, sr, semitones)
    
    def tempo_augment(
        self,
        y: np.ndarray,
        sr: int,
        tempo_factor: Optional[float] = None,
    ) -> np.ndarray:
        """
        Change tempo without changing pitch.
        
        Args:
            y: Audio waveform
            sr: Sample rate
            tempo_factor: Tempo multiplier (>1 = faster)
        
        Returns:
            Tempo-adjusted audio
        """
        if tempo_factor is None:
            tempo_factor = random.uniform(0.8, 1.2)
        
        return self.time_stretch(y, sr, tempo_factor)
    
    # =========================================================================
    # Main Augmentation Pipeline
    # =========================================================================
    
    def augment(
        self,
        y: np.ndarray,
        sr: int = 16000,
        augmentations: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Apply random augmentations to audio.
        
        Args:
            y: Audio waveform
            sr: Sample rate
            augmentations: List of augmentations to apply, or None for random
        
        Returns:
            Augmented audio
        """
        if augmentations is None:
            # Apply random augmentations based on probabilities
            if random.random() < self.config.p_time_stretch:
                y = self.time_stretch(y, sr)
            
            if random.random() < self.config.p_pitch_shift:
                y = self.pitch_shift(y, sr)
            
            if random.random() < self.config.p_noise:
                noise_type = random.choice(self.config.noise_types)
                y = self.add_noise(y, noise_type)
            
            if random.random() < self.config.p_gain:
                y = self.apply_gain(y)
            
            if random.random() < self.config.p_reverb:
                y = self.add_reverb(y, sr)
        else:
            # Apply specified augmentations
            for aug in augmentations:
                if aug == "time_stretch":
                    y = self.time_stretch(y, sr)
                elif aug == "pitch_shift":
                    y = self.pitch_shift(y, sr)
                elif aug == "noise":
                    y = self.add_noise(y)
                elif aug == "gain":
                    y = self.apply_gain(y)
                elif aug == "reverb":
                    y = self.add_reverb(y, sr)
                elif aug == "compression":
                    y = self.apply_compression(y, sr=sr)
        
        return y
    
    def augment_spectrogram(
        self,
        spec: np.ndarray,
        augmentations: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Apply augmentations to spectrogram.
        
        Args:
            spec: Spectrogram (freq x time)
            augmentations: List of augmentations
        
        Returns:
            Augmented spectrogram
        """
        if augmentations is None:
            if random.random() < self.config.p_spec_augment:
                spec = self.spec_augment(spec)
        else:
            for aug in augmentations:
                if aug == "spec_augment":
                    spec = self.spec_augment(spec)
                elif aug == "freq_warp":
                    spec = self.frequency_warp(spec)
        
        return spec


class AugmentationPipeline:
    """
    Composable augmentation pipeline.
    
    Usage:
        pipeline = AugmentationPipeline([
            ("time_stretch", {"rate": 1.1}),
            ("pitch_shift", {"n_steps": 2}),
            ("noise", {"snr_db": 20}),
        ])
        augmented = pipeline(waveform, sr=16000)
    """
    
    def __init__(
        self,
        augmentations: List[Tuple[str, Dict]],
        p: float = 1.0,
    ):
        """
        Args:
            augmentations: List of (name, kwargs) tuples
            p: Probability of applying the pipeline
        """
        self.augmentations = augmentations
        self.p = p
        self.augmentor = AudioAugmentor()
    
    def __call__(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        if random.random() > self.p:
            return y
        
        for name, kwargs in self.augmentations:
            method = getattr(self.augmentor, name, None)
            if method:
                if "sr" in method.__code__.co_varnames:
                    y = method(y, sr=sr, **kwargs)
                else:
                    y = method(y, **kwargs)
        
        return y


# Convenience functions
def augment_audio(
    y: np.ndarray,
    sr: int = 16000,
    config: Optional[AugmentationConfig] = None,
) -> np.ndarray:
    """Apply random augmentations to audio."""
    augmentor = AudioAugmentor(config)
    return augmentor.augment(y, sr)


def create_augmentation_pipeline(
    preset: str = "default",
) -> AugmentationPipeline:
    """
    Create a preset augmentation pipeline.
    
    Presets:
        - "default": Balanced augmentations
        - "aggressive": Strong augmentations for small datasets
        - "light": Subtle augmentations
        - "music": Music-specific augmentations
    """
    presets = {
        "default": [
            ("time_stretch", {}),
            ("pitch_shift", {}),
            ("add_noise", {"snr_db": 25}),
        ],
        "aggressive": [
            ("time_stretch", {"rate": random.uniform(0.7, 1.3)}),
            ("pitch_shift", {"n_steps": random.randint(-6, 6)}),
            ("add_noise", {"snr_db": random.uniform(10, 20)}),
            ("apply_gain", {"gain_db": random.uniform(-10, 10)}),
            ("add_reverb", {"wet_level": 0.4}),
        ],
        "light": [
            ("apply_gain", {"gain_db": random.uniform(-3, 3)}),
            ("add_noise", {"snr_db": 30}),
        ],
        "music": [
            ("transpose_key", {}),
            ("tempo_augment", {}),
            ("add_reverb", {"room_size": 0.3, "wet_level": 0.2}),
        ],
    }
    
    return AugmentationPipeline(presets.get(preset, presets["default"]))

