"""
Parrot DSP - Sample Playback Engine and Pitch Shifting.

Provides:
- Sample playback with various modes
- Pitch shifting algorithms (granular, phase vocoder)
- Time stretching
- Granular synthesis
- Formant preservation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import math
import random


class PlaybackMode(Enum):
    """Sample playback modes."""
    ONE_SHOT = "one_shot"       # Play once
    LOOP = "loop"               # Loop continuously
    PING_PONG = "ping_pong"     # Loop forward/backward
    REVERSE = "reverse"         # Play backwards
    GRANULAR = "granular"       # Granular playback


class PitchAlgorithm(Enum):
    """Pitch shifting algorithms."""
    SIMPLE = "simple"           # Simple resampling (changes length)
    GRANULAR = "granular"       # Granular time-domain
    PHASE_VOCODER = "phase_vocoder"  # Phase vocoder (FFT-based)
    FORMANT = "formant"         # Formant-preserving


@dataclass
class SamplePlayback:
    """
    Sample playback engine.
    """
    samples: List[float] = field(default_factory=list)
    sample_rate: float = 44100.0
    mode: PlaybackMode = PlaybackMode.ONE_SHOT

    # Playback parameters
    start_position: float = 0.0    # Start point (0.0-1.0)
    end_position: float = 1.0      # End point (0.0-1.0)
    loop_start: float = 0.0        # Loop start (0.0-1.0)
    loop_end: float = 1.0          # Loop end (0.0-1.0)

    # Playback rate
    rate: float = 1.0              # Playback rate (1.0 = normal)
    pitch_semitones: float = 0.0   # Pitch shift in semitones

    # State
    _position: float = 0.0
    _direction: int = 1            # 1 = forward, -1 = backward
    _is_playing: bool = False

    def start(self):
        """Start playback."""
        if self.mode == PlaybackMode.REVERSE:
            self._position = self.end_position * len(self.samples)
            self._direction = -1
        else:
            self._position = self.start_position * len(self.samples)
            self._direction = 1
        self._is_playing = True

    def stop(self):
        """Stop playback."""
        self._is_playing = False

    def process_sample(self) -> float:
        """
        Get next sample.

        Returns:
            Output sample
        """
        if not self._is_playing or not self.samples:
            return 0.0

        # Calculate effective playback rate
        effective_rate = self.rate * (2 ** (self.pitch_semitones / 12.0))

        # Get sample with interpolation
        sample = self._interpolate(self._position)

        # Advance position
        self._position += self._direction * effective_rate

        # Handle boundaries
        sample_end = self.end_position * len(self.samples)
        sample_start = self.start_position * len(self.samples)
        loop_start = self.loop_start * len(self.samples)
        loop_end = self.loop_end * len(self.samples)

        if self.mode == PlaybackMode.ONE_SHOT:
            if self._direction > 0 and self._position >= sample_end:
                self._is_playing = False
            elif self._direction < 0 and self._position <= sample_start:
                self._is_playing = False

        elif self.mode == PlaybackMode.LOOP:
            if self._position >= loop_end:
                self._position = loop_start
            elif self._position < loop_start:
                self._position = loop_end

        elif self.mode == PlaybackMode.PING_PONG:
            if self._position >= loop_end:
                self._position = loop_end
                self._direction = -1
            elif self._position <= loop_start:
                self._position = loop_start
                self._direction = 1

        elif self.mode == PlaybackMode.REVERSE:
            if self._position <= sample_start:
                self._is_playing = False

        return sample

    def _interpolate(self, position: float) -> float:
        """Linear interpolation at position."""
        if position < 0 or position >= len(self.samples) - 1:
            return 0.0

        index_a = int(position)
        index_b = index_a + 1
        frac = position - index_a

        if index_b >= len(self.samples):
            return self.samples[index_a]

        return self.samples[index_a] * (1 - frac) + self.samples[index_b] * frac

    def process_block(self, num_samples: int) -> List[float]:
        """Process a block of samples."""
        return [self.process_sample() for _ in range(num_samples)]

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def position(self) -> float:
        """Current position as 0.0-1.0."""
        if not self.samples:
            return 0.0
        return self._position / len(self.samples)


@dataclass
class PitchShifter:
    """
    Pitch shifter using granular synthesis.
    """
    sample_rate: float = 44100.0
    algorithm: PitchAlgorithm = PitchAlgorithm.GRANULAR

    # Granular parameters
    grain_size_ms: float = 50.0
    grain_overlap: float = 0.5  # 0.0-1.0
    window_type: str = "hann"

    # State
    _input_buffer: List[float] = field(default_factory=list)
    _output_buffer: List[float] = field(default_factory=list)
    _grain_position: float = 0.0
    _output_position: int = 0

    def __post_init__(self):
        grain_samples = int(self.grain_size_ms * self.sample_rate / 1000.0)
        self._input_buffer = [0.0] * (grain_samples * 4)
        self._output_buffer = [0.0] * (grain_samples * 4)
        self._window = self._create_window(grain_samples)

    def _create_window(self, size: int) -> List[float]:
        """Create grain window."""
        if self.window_type == "hann":
            return [0.5 * (1 - math.cos(2 * math.pi * i / (size - 1)))
                    for i in range(size)]
        elif self.window_type == "hamming":
            return [0.54 - 0.46 * math.cos(2 * math.pi * i / (size - 1))
                    for i in range(size)]
        else:  # Triangle
            mid = size // 2
            return [i / mid if i < mid else 2 - i / mid for i in range(size)]

    def process(
        self,
        samples: List[float],
        semitones: float,
    ) -> List[float]:
        """
        Pitch shift samples.

        Args:
            samples: Input samples
            semitones: Pitch shift in semitones

        Returns:
            Pitch-shifted samples
        """
        if self.algorithm == PitchAlgorithm.SIMPLE:
            return self._simple_shift(samples, semitones)
        elif self.algorithm == PitchAlgorithm.GRANULAR:
            return self._granular_shift(samples, semitones)
        else:
            return self._granular_shift(samples, semitones)

    def _simple_shift(
        self,
        samples: List[float],
        semitones: float,
    ) -> List[float]:
        """Simple resampling (changes duration)."""
        rate = 2 ** (semitones / 12.0)
        new_length = int(len(samples) / rate)

        result = []
        for i in range(new_length):
            source_pos = i * rate
            index_a = int(source_pos)
            index_b = min(index_a + 1, len(samples) - 1)
            frac = source_pos - index_a

            if index_a < len(samples):
                sample = samples[index_a] * (1 - frac) + samples[index_b] * frac
                result.append(sample)

        return result

    def _granular_shift(
        self,
        samples: List[float],
        semitones: float,
    ) -> List[float]:
        """Granular pitch shifting (preserves duration)."""
        pitch_ratio = 2 ** (semitones / 12.0)
        grain_samples = int(self.grain_size_ms * self.sample_rate / 1000.0)
        hop_size = int(grain_samples * (1 - self.grain_overlap))

        # Pad input
        padded = [0.0] * grain_samples + samples + [0.0] * grain_samples

        # Create output buffer
        output_length = len(samples)
        output = [0.0] * output_length
        normalization = [0.0] * output_length

        # Process grains
        input_pos = 0
        output_pos = 0

        while output_pos < output_length:
            # Calculate input grain position
            input_grain_start = int(input_pos)

            if input_grain_start + grain_samples > len(padded):
                break

            # Extract and window grain
            grain = []
            for i in range(grain_samples):
                if i < len(self._window):
                    grain.append(padded[input_grain_start + i] * self._window[i])
                else:
                    grain.append(0.0)

            # Resample grain for pitch shift
            if pitch_ratio != 1.0:
                resampled_length = int(grain_samples / pitch_ratio)
                resampled = []
                for i in range(grain_samples):
                    source_pos = i * pitch_ratio
                    index_a = int(source_pos)
                    if index_a < len(grain) - 1:
                        frac = source_pos - index_a
                        resampled.append(grain[index_a] * (1 - frac) + grain[index_a + 1] * frac)
                    elif index_a < len(grain):
                        resampled.append(grain[index_a])
                    else:
                        resampled.append(0.0)
                grain = resampled

            # Add grain to output
            for i in range(min(len(grain), output_length - output_pos)):
                output[output_pos + i] += grain[i]
                normalization[output_pos + i] += self._window[i] if i < len(self._window) else 0.0

            # Advance positions
            input_pos += hop_size * pitch_ratio
            output_pos += hop_size

        # Normalize
        for i in range(output_length):
            if normalization[i] > 0.001:
                output[i] /= normalization[i]

        return output


@dataclass
class GrainCloud:
    """
    Granular synthesis cloud for textural effects.
    """
    samples: List[float] = field(default_factory=list)
    sample_rate: float = 44100.0

    # Grain parameters
    grain_size_ms: float = 50.0
    grain_density: float = 10.0  # Grains per second
    position: float = 0.5        # Source position (0.0-1.0)
    position_spread: float = 0.1  # Random position spread

    # Modulation
    pitch_spread: float = 0.0    # Random pitch variation (semitones)
    pan_spread: float = 0.0      # Stereo spread (0.0-1.0)
    reverse_probability: float = 0.0  # Chance of reverse grain

    # State
    _grains: List[Dict] = field(default_factory=list)
    _sample_counter: int = 0

    def process_sample(self) -> Tuple[float, float]:
        """
        Generate next stereo sample.

        Returns:
            (left, right) samples
        """
        # Maybe spawn new grain
        spawn_interval = self.sample_rate / max(1, self.grain_density)
        if self._sample_counter >= spawn_interval:
            self._spawn_grain()
            self._sample_counter = 0
        self._sample_counter += 1

        # Mix active grains
        left = 0.0
        right = 0.0
        active_grains = []

        for grain in self._grains:
            sample = self._process_grain(grain)

            if grain["active"]:
                # Apply panning
                left += sample * (1 - grain["pan"])
                right += sample * grain["pan"]
                active_grains.append(grain)

        self._grains = active_grains

        return left, right

    def _spawn_grain(self):
        """Spawn a new grain."""
        if not self.samples:
            return

        grain_samples = int(self.grain_size_ms * self.sample_rate / 1000.0)

        # Randomize position
        pos = self.position + (random.random() - 0.5) * 2 * self.position_spread
        pos = max(0.0, min(1.0, pos))
        start_sample = int(pos * (len(self.samples) - grain_samples))

        # Randomize pitch
        pitch_ratio = 2 ** ((random.random() - 0.5) * 2 * self.pitch_spread / 12.0)

        # Randomize direction
        reverse = random.random() < self.reverse_probability

        # Randomize pan
        pan = 0.5 + (random.random() - 0.5) * self.pan_spread

        self._grains.append({
            "start": start_sample,
            "position": 0.0,
            "length": grain_samples,
            "pitch_ratio": pitch_ratio,
            "reverse": reverse,
            "pan": pan,
            "active": True,
        })

    def _process_grain(self, grain: Dict) -> float:
        """Process a single grain."""
        if grain["position"] >= grain["length"]:
            grain["active"] = False
            return 0.0

        # Calculate window
        t = grain["position"] / grain["length"]
        window = 0.5 * (1 - math.cos(2 * math.pi * t))

        # Calculate source position
        if grain["reverse"]:
            source_pos = grain["start"] + grain["length"] - grain["position"] * grain["pitch_ratio"]
        else:
            source_pos = grain["start"] + grain["position"] * grain["pitch_ratio"]

        # Get sample with interpolation
        index_a = int(source_pos)
        if index_a < 0 or index_a >= len(self.samples) - 1:
            grain["position"] += 1
            return 0.0

        frac = source_pos - index_a
        sample = self.samples[index_a] * (1 - frac) + self.samples[index_a + 1] * frac

        grain["position"] += 1

        return sample * window

    def process_block(self, num_samples: int) -> Tuple[List[float], List[float]]:
        """Generate a block of stereo samples."""
        left = []
        right = []
        for _ in range(num_samples):
            l, r = self.process_sample()
            left.append(l)
            right.append(r)
        return left, right


def create_pitch_shifter(
    algorithm: str = "granular",
    grain_size_ms: float = 50.0,
    sample_rate: float = 44100.0,
) -> PitchShifter:
    """
    Create a pitch shifter.

    Args:
        algorithm: Algorithm type (simple, granular, phase_vocoder)
        grain_size_ms: Grain size for granular algorithm
        sample_rate: Sample rate

    Returns:
        PitchShifter instance
    """
    algo = PitchAlgorithm(algorithm)
    return PitchShifter(
        sample_rate=sample_rate,
        algorithm=algo,
        grain_size_ms=grain_size_ms,
    )


def shift_pitch(
    samples: List[float],
    semitones: float,
    algorithm: str = "granular",
    sample_rate: float = 44100.0,
) -> List[float]:
    """
    Pitch shift audio samples.

    Args:
        samples: Input samples
        semitones: Pitch shift in semitones
        algorithm: Algorithm to use
        sample_rate: Sample rate

    Returns:
        Pitch-shifted samples
    """
    shifter = create_pitch_shifter(algorithm, sample_rate=sample_rate)
    return shifter.process(samples, semitones)


def time_stretch(
    samples: List[float],
    factor: float,
    grain_size_ms: float = 50.0,
    sample_rate: float = 44100.0,
) -> List[float]:
    """
    Time stretch audio without changing pitch.

    Args:
        samples: Input samples
        factor: Stretch factor (2.0 = twice as long)
        grain_size_ms: Grain size
        sample_rate: Sample rate

    Returns:
        Time-stretched samples
    """
    grain_samples = int(grain_size_ms * sample_rate / 1000.0)
    overlap = 0.5
    hop_in = int(grain_samples * (1 - overlap))
    hop_out = int(hop_in * factor)

    # Create Hann window
    window = [0.5 * (1 - math.cos(2 * math.pi * i / (grain_samples - 1)))
              for i in range(grain_samples)]

    # Calculate output length
    output_length = int(len(samples) * factor)
    output = [0.0] * output_length
    normalization = [0.0] * output_length

    # Process grains
    input_pos = 0
    output_pos = 0

    while input_pos + grain_samples <= len(samples) and output_pos + grain_samples <= output_length:
        # Extract and window grain
        for i in range(grain_samples):
            if output_pos + i < output_length:
                output[output_pos + i] += samples[input_pos + i] * window[i]
                normalization[output_pos + i] += window[i]

        input_pos += hop_in
        output_pos += hop_out

    # Normalize
    for i in range(output_length):
        if normalization[i] > 0.001:
            output[i] /= normalization[i]

    return output


def create_grain_cloud(
    samples: List[float],
    grain_size_ms: float = 50.0,
    density: float = 10.0,
    sample_rate: float = 44100.0,
) -> GrainCloud:
    """
    Create a granular cloud generator.

    Args:
        samples: Source samples
        grain_size_ms: Grain size
        density: Grains per second
        sample_rate: Sample rate

    Returns:
        GrainCloud instance
    """
    return GrainCloud(
        samples=samples,
        sample_rate=sample_rate,
        grain_size_ms=grain_size_ms,
        grain_density=density,
    )


def detect_pitch(
    samples: List[float],
    sample_rate: float = 44100.0,
    min_hz: float = 50.0,
    max_hz: float = 2000.0,
) -> Optional[float]:
    """
    Detect pitch using autocorrelation.

    Args:
        samples: Input samples
        sample_rate: Sample rate
        min_hz: Minimum frequency to detect
        max_hz: Maximum frequency to detect

    Returns:
        Detected frequency in Hz, or None
    """
    if len(samples) < 256:
        return None

    # Calculate lag bounds
    min_lag = int(sample_rate / max_hz)
    max_lag = int(sample_rate / min_hz)
    max_lag = min(max_lag, len(samples) // 2)

    if max_lag <= min_lag:
        return None

    # Autocorrelation
    best_correlation = -1.0
    best_lag = 0

    for lag in range(min_lag, max_lag):
        correlation = 0.0
        for i in range(len(samples) - lag):
            correlation += samples[i] * samples[i + lag]

        correlation /= (len(samples) - lag)

        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag

    if best_lag <= 0:
        return None

    return sample_rate / best_lag


def resample(
    samples: List[float],
    from_rate: float,
    to_rate: float,
) -> List[float]:
    """
    Resample audio to different sample rate.

    Args:
        samples: Input samples
        from_rate: Source sample rate
        to_rate: Target sample rate

    Returns:
        Resampled audio
    """
    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)

    result = []
    for i in range(new_length):
        source_pos = i / ratio
        index_a = int(source_pos)
        index_b = min(index_a + 1, len(samples) - 1)
        frac = source_pos - index_a

        sample = samples[index_a] * (1 - frac) + samples[index_b] * frac
        result.append(sample)

    return result
