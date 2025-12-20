"""
Trace DSP - Envelope Follower and Pattern Automation.

Provides:
- Envelope following with multiple detection modes
- Pattern-based automation generation
- LFO patterns with various waveforms
- Sidechain-style ducking
- Automation curve generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import math


class EnvelopeMode(Enum):
    """Envelope detection modes."""
    PEAK = "peak"           # Peak detection
    RMS = "rms"             # RMS (average power)
    PEAK_RMS = "peak_rms"   # Hybrid peak/RMS
    TRUE_PEAK = "true_peak" # Oversampled peak detection


class AutomationCurve(Enum):
    """Automation curve shapes."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    S_CURVE = "s_curve"
    STEP = "step"


class LFOShape(Enum):
    """LFO waveform shapes."""
    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    SAW_UP = "saw_up"
    SAW_DOWN = "saw_down"
    RANDOM = "random"
    SAMPLE_HOLD = "sample_hold"


@dataclass
class EnvelopeFollower:
    """
    Envelope follower for extracting amplitude contours.
    """
    mode: EnvelopeMode = EnvelopeMode.RMS
    attack_ms: float = 10.0
    release_ms: float = 100.0
    sample_rate: float = 44100.0

    # Internal state
    _envelope: float = 0.0
    _attack_coeff: float = 0.0
    _release_coeff: float = 0.0
    _rms_window: List[float] = field(default_factory=list)
    _rms_window_size: int = 256

    def __post_init__(self):
        self._update_coefficients()

    def _update_coefficients(self):
        """Calculate time constant coefficients."""
        if self.attack_ms > 0:
            self._attack_coeff = math.exp(-1.0 / (self.attack_ms * self.sample_rate / 1000.0))
        else:
            self._attack_coeff = 0.0

        if self.release_ms > 0:
            self._release_coeff = math.exp(-1.0 / (self.release_ms * self.sample_rate / 1000.0))
        else:
            self._release_coeff = 0.0

    def process_sample(self, sample: float) -> float:
        """
        Process a single sample and return envelope value.

        Args:
            sample: Input sample

        Returns:
            Envelope value (0.0-1.0)
        """
        input_level = abs(sample)

        if self.mode == EnvelopeMode.RMS:
            # RMS with sliding window
            self._rms_window.append(sample * sample)
            if len(self._rms_window) > self._rms_window_size:
                self._rms_window.pop(0)
            input_level = math.sqrt(sum(self._rms_window) / len(self._rms_window))

        elif self.mode == EnvelopeMode.PEAK_RMS:
            # Hybrid: peak for attack, RMS for release
            peak = abs(sample)
            self._rms_window.append(sample * sample)
            if len(self._rms_window) > self._rms_window_size:
                self._rms_window.pop(0)
            rms = math.sqrt(sum(self._rms_window) / len(self._rms_window))

            if peak > self._envelope:
                input_level = peak
            else:
                input_level = rms

        # Attack/release envelope
        if input_level > self._envelope:
            self._envelope = input_level + self._attack_coeff * (self._envelope - input_level)
        else:
            self._envelope = input_level + self._release_coeff * (self._envelope - input_level)

        return self._envelope

    def process_block(self, samples: List[float]) -> List[float]:
        """
        Process a block of samples.

        Args:
            samples: Input samples

        Returns:
            Envelope values
        """
        return [self.process_sample(s) for s in samples]

    def reset(self):
        """Reset envelope state."""
        self._envelope = 0.0
        self._rms_window.clear()


@dataclass
class PatternAutomation:
    """
    Pattern-based automation generator.
    """
    pattern: List[float] = field(default_factory=list)  # 0.0-1.0 values
    pattern_length_beats: float = 4.0
    tempo_bpm: float = 120.0
    curve: AutomationCurve = AutomationCurve.LINEAR
    sample_rate: float = 44100.0

    # Modulation
    depth: float = 1.0       # Modulation depth (0.0-1.0)
    offset: float = 0.0      # DC offset (-1.0 to 1.0)
    phase_offset: float = 0.0  # Phase offset (0.0-1.0)

    # State
    _position: float = 0.0

    def set_pattern(self, pattern: List[float]):
        """Set automation pattern."""
        self.pattern = pattern

    def set_tempo(self, bpm: float):
        """Set tempo for sync."""
        self.tempo_bpm = bpm

    def process_sample(self) -> float:
        """
        Generate next automation value.

        Returns:
            Automation value (0.0-1.0)
        """
        if not self.pattern:
            return 0.5

        # Calculate position in pattern
        beat_duration = 60.0 / self.tempo_bpm
        pattern_duration_samples = self.pattern_length_beats * beat_duration * self.sample_rate

        normalized_pos = (self._position / pattern_duration_samples + self.phase_offset) % 1.0

        # Get pattern value with interpolation
        pattern_index = normalized_pos * len(self.pattern)
        index_a = int(pattern_index) % len(self.pattern)
        index_b = (index_a + 1) % len(self.pattern)
        frac = pattern_index - int(pattern_index)

        # Apply curve
        frac = self._apply_curve(frac)

        # Interpolate
        value = self.pattern[index_a] * (1 - frac) + self.pattern[index_b] * frac

        # Apply depth and offset
        value = value * self.depth + self.offset

        # Advance position
        self._position = (self._position + 1) % pattern_duration_samples

        return max(0.0, min(1.0, value))

    def _apply_curve(self, t: float) -> float:
        """Apply curve shape to interpolation factor."""
        if self.curve == AutomationCurve.LINEAR:
            return t
        elif self.curve == AutomationCurve.EXPONENTIAL:
            return t * t
        elif self.curve == AutomationCurve.LOGARITHMIC:
            return math.sqrt(t)
        elif self.curve == AutomationCurve.S_CURVE:
            return t * t * (3 - 2 * t)
        elif self.curve == AutomationCurve.STEP:
            return 0.0 if t < 0.5 else 1.0
        return t

    def process_block(self, num_samples: int) -> List[float]:
        """Generate a block of automation values."""
        return [self.process_sample() for _ in range(num_samples)]

    def reset(self):
        """Reset position."""
        self._position = 0.0


@dataclass
class SidechainDucker:
    """
    Sidechain-style ducking/pumping effect.
    """
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 1.0
    release_ms: float = 100.0
    hold_ms: float = 10.0
    sample_rate: float = 44100.0

    # State
    _envelope: float = 0.0
    _hold_counter: int = 0
    _envelope_follower: Optional[EnvelopeFollower] = None

    def __post_init__(self):
        self._envelope_follower = EnvelopeFollower(
            mode=EnvelopeMode.PEAK,
            attack_ms=self.attack_ms,
            release_ms=self.release_ms,
            sample_rate=self.sample_rate,
        )
        self._hold_samples = int(self.hold_ms * self.sample_rate / 1000.0)

    def process(
        self,
        input_sample: float,
        sidechain_sample: float,
    ) -> float:
        """
        Process a sample with sidechain ducking.

        Args:
            input_sample: Main input to duck
            sidechain_sample: Sidechain signal to trigger ducking

        Returns:
            Ducked output
        """
        # Get sidechain envelope
        sidechain_level = self._envelope_follower.process_sample(sidechain_sample)

        # Convert to dB
        if sidechain_level > 0:
            level_db = 20 * math.log10(sidechain_level)
        else:
            level_db = -96.0

        # Calculate gain reduction
        if level_db > self.threshold_db:
            over_db = level_db - self.threshold_db
            reduction_db = over_db - (over_db / self.ratio)
            gain = 10 ** (-reduction_db / 20.0)
        else:
            gain = 1.0

        return input_sample * gain


def create_envelope_follower(
    mode: str = "rms",
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    sample_rate: float = 44100.0,
) -> EnvelopeFollower:
    """
    Create an envelope follower.

    Args:
        mode: Detection mode (peak, rms, peak_rms, true_peak)
        attack_ms: Attack time in ms
        release_ms: Release time in ms
        sample_rate: Sample rate in Hz

    Returns:
        EnvelopeFollower instance
    """
    mode_enum = EnvelopeMode(mode)
    return EnvelopeFollower(
        mode=mode_enum,
        attack_ms=attack_ms,
        release_ms=release_ms,
        sample_rate=sample_rate,
    )


def follow_envelope(
    samples: List[float],
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    mode: str = "rms",
    sample_rate: float = 44100.0,
) -> List[float]:
    """
    Extract envelope from audio samples.

    Args:
        samples: Input samples
        attack_ms: Attack time
        release_ms: Release time
        mode: Detection mode
        sample_rate: Sample rate

    Returns:
        Envelope values
    """
    follower = create_envelope_follower(mode, attack_ms, release_ms, sample_rate)
    return follower.process_block(samples)


def apply_pattern_automation(
    parameter_values: List[float],
    pattern: List[float],
    depth: float = 1.0,
) -> List[float]:
    """
    Apply automation pattern to parameter values.

    Args:
        parameter_values: Base parameter values
        pattern: Automation pattern (0.0-1.0)
        depth: Modulation depth

    Returns:
        Modulated parameter values
    """
    if not pattern:
        return parameter_values

    result = []
    pattern_len = len(pattern)

    for i, value in enumerate(parameter_values):
        pattern_idx = i % pattern_len
        modulation = pattern[pattern_idx] * depth
        result.append(value * modulation)

    return result


def generate_lfo_pattern(
    shape: str = "sine",
    length: int = 64,
    frequency: float = 1.0,
    phase: float = 0.0,
    duty_cycle: float = 0.5,
) -> List[float]:
    """
    Generate LFO pattern.

    Args:
        shape: Waveform shape (sine, triangle, square, saw_up, saw_down)
        length: Pattern length in samples
        frequency: Cycles per pattern
        phase: Phase offset (0.0-1.0)
        duty_cycle: Duty cycle for square wave

    Returns:
        List of values (0.0-1.0)
    """
    import random

    pattern = []

    for i in range(length):
        t = (i / length * frequency + phase) % 1.0

        if shape == "sine":
            value = (math.sin(t * 2 * math.pi) + 1) / 2
        elif shape == "triangle":
            if t < 0.5:
                value = t * 2
            else:
                value = 2 - t * 2
        elif shape == "square":
            value = 1.0 if t < duty_cycle else 0.0
        elif shape == "saw_up":
            value = t
        elif shape == "saw_down":
            value = 1 - t
        elif shape == "random":
            value = random.random()
        elif shape == "sample_hold":
            # Change value at frequency rate
            if i == 0 or int(t * 4) != int(((i-1) / length * frequency + phase) % 1.0 * 4):
                value = random.random()
            else:
                value = pattern[-1] if pattern else 0.5
        else:
            value = 0.5

        pattern.append(value)

    return pattern


def create_sidechain_pattern(
    pattern: str = "four_on_floor",
    resolution: int = 16,
) -> List[float]:
    """
    Create common sidechain patterns.

    Args:
        pattern: Pattern name
        resolution: Steps per bar

    Returns:
        Pattern values (0.0-1.0)
    """
    patterns = {
        "four_on_floor": [1.0, 0.2, 0.4, 0.2] * 4,  # Kick on every beat
        "half_time": [1.0, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1] * 2,  # Kick on 1 and 3
        "pumping": [1.0, 0.0, 0.3, 0.0, 0.8, 0.0, 0.3, 0.0] * 2,  # EDM pump
        "trap": [1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.0, 0.3, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0],
        "broken": [1.0, 0.0, 0.0, 0.6, 0.0, 0.8, 0.0, 0.0, 0.5, 0.0, 0.0, 0.7, 0.0, 0.0, 0.4, 0.0],
    }

    base_pattern = patterns.get(pattern, patterns["four_on_floor"])

    # Resample to requested resolution
    if len(base_pattern) == resolution:
        return base_pattern

    result = []
    ratio = len(base_pattern) / resolution
    for i in range(resolution):
        idx = int(i * ratio)
        result.append(base_pattern[idx % len(base_pattern)])

    return result


def envelope_to_midi_cc(
    envelope: List[float],
    cc_number: int = 1,
    min_value: int = 0,
    max_value: int = 127,
    threshold: float = 0.01,
) -> List[Dict]:
    """
    Convert envelope to MIDI CC messages.

    Args:
        envelope: Envelope values (0.0-1.0)
        cc_number: MIDI CC number
        min_value: Minimum CC value
        max_value: Maximum CC value
        threshold: Change threshold to emit message

    Returns:
        List of MIDI CC events
    """
    events = []
    last_value = -1

    for i, env_value in enumerate(envelope):
        cc_value = int(min_value + env_value * (max_value - min_value))
        cc_value = max(min_value, min(max_value, cc_value))

        if last_value < 0 or abs(cc_value - last_value) >= threshold * 127:
            events.append({
                "type": "cc",
                "time": i,
                "cc": cc_number,
                "value": cc_value,
            })
            last_value = cc_value

    return events
