"""
Groove Style Transfer - ML-based groove style transformation.

Provides neural network-based groove style transfer using:
- CycleGAN-style adversarial networks
- Variational autoencoders (VAE)
- Template-based fallback
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from enum import Enum

from python.penta_core.ml.model_registry import (
    ModelInfo,
    ModelBackend,
    ModelTask,
    get_model,
    register_model,
)
from python.penta_core.ml.inference import create_engine, InferenceEngine


class GrooveStyle(Enum):
    """Predefined groove styles."""
    FUNK = "funk"
    JAZZ = "jazz"
    ROCK = "rock"
    HIP_HOP = "hip_hop"
    LATIN = "latin"
    REGGAE = "reggae"
    ELECTRONIC = "electronic"
    SOUL = "soul"
    BLUES = "blues"
    POP = "pop"


@dataclass
class GrooveFeatures:
    """Extracted groove features for style transfer."""
    # Timing features
    timing_offsets: List[float]  # Offset from grid in ms
    velocity_curve: List[int]  # 0-127 velocities
    swing_amount: float  # 0.0-1.0

    # Articulation
    note_durations: List[float]  # As fraction of beat
    ghost_note_ratio: float  # Fraction of ghost notes

    # Pattern features
    density: float  # Notes per beat
    syncopation: float  # 0.0-1.0

    # Optional: raw MIDI data
    midi_notes: Optional[List[Dict]] = None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        features = []

        # Pad/truncate timing offsets to fixed length
        offsets = self.timing_offsets[:16] + [0.0] * (16 - len(self.timing_offsets))
        features.extend(offsets)

        # Pad/truncate velocities
        vels = [v / 127.0 for v in self.velocity_curve[:16]]
        vels = vels + [0.5] * (16 - len(vels))
        features.extend(vels)

        # Scalar features
        features.extend([
            self.swing_amount,
            self.ghost_note_ratio,
            self.density / 4.0,  # Normalize assuming max 4 notes per beat
            self.syncopation,
        ])

        return np.array(features, dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "GrooveFeatures":
        """Create from numpy array."""
        timing_offsets = arr[:16].tolist()
        velocity_curve = [int(v * 127) for v in arr[16:32]]

        return cls(
            timing_offsets=timing_offsets,
            velocity_curve=velocity_curve,
            swing_amount=float(arr[32]),
            note_durations=[0.5] * 16,  # Default
            ghost_note_ratio=float(arr[33]),
            density=float(arr[34]) * 4.0,
            syncopation=float(arr[35]),
        )


@dataclass
class StyleTransferResult:
    """Result from groove style transfer."""
    original_features: GrooveFeatures
    transferred_features: GrooveFeatures
    source_style: Optional[GrooveStyle]
    target_style: GrooveStyle
    confidence: float

    # Transformed MIDI data
    transformed_notes: Optional[List[Dict]] = None

    def get_timing_adjustments(self) -> List[float]:
        """Get timing adjustments in ms."""
        orig = self.original_features.timing_offsets
        trans = self.transferred_features.timing_offsets
        return [t - o for o, t in zip(orig, trans)]

    def get_velocity_adjustments(self) -> List[int]:
        """Get velocity adjustments."""
        orig = self.original_features.velocity_curve
        trans = self.transferred_features.velocity_curve
        return [t - o for o, t in zip(orig, trans)]


class GrooveStyleTransfer:
    """
    ML-based groove style transfer.

    Transforms the "feel" of a groove from one style to another
    while preserving the musical content.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fallback: bool = True,
    ):
        """
        Initialize style transfer.

        Args:
            model_name: Name of registered model (None for template-based)
            use_fallback: Use template-based transfer when model unavailable
        """
        self._engine: Optional[InferenceEngine] = None
        self._use_fallback = use_fallback
        self._style_templates: Dict[GrooveStyle, GrooveFeatures] = {}

        if model_name:
            model_info = get_model(model_name)
            if model_info:
                self._engine = create_engine(model_info)
                self._engine.load()

        # Initialize templates
        if use_fallback:
            self._init_style_templates()

    def _init_style_templates(self) -> None:
        """Initialize style templates for fallback."""
        # Funk: heavy backbeat, syncopated
        self._style_templates[GrooveStyle.FUNK] = GrooveFeatures(
            timing_offsets=[0, 5, 0, -5] * 4,  # Slight push on 2 and 4
            velocity_curve=[80, 100, 80, 110] * 4,  # Emphasize backbeat
            swing_amount=0.1,
            note_durations=[0.8, 0.9, 0.8, 0.95] * 4,
            ghost_note_ratio=0.3,
            density=3.0,
            syncopation=0.7,
        )

        # Jazz: swing, dynamic
        self._style_templates[GrooveStyle.JAZZ] = GrooveFeatures(
            timing_offsets=[0, 10, 0, 5] * 4,  # Swing timing
            velocity_curve=[90, 70, 80, 75] * 4,  # Varied dynamics
            swing_amount=0.67,  # Triplet swing
            note_durations=[0.9, 0.5, 0.8, 0.6] * 4,
            ghost_note_ratio=0.2,
            density=2.5,
            syncopation=0.5,
        )

        # Hip Hop: laid back, pocket
        self._style_templates[GrooveStyle.HIP_HOP] = GrooveFeatures(
            timing_offsets=[0, -10, 0, -15] * 4,  # Laid back feel
            velocity_curve=[100, 60, 90, 70] * 4,  # Strong 1, ghost 2
            swing_amount=0.2,
            note_durations=[1.0, 0.4, 0.9, 0.5] * 4,
            ghost_note_ratio=0.4,
            density=2.0,
            syncopation=0.4,
        )

        # Rock: straight, driving
        self._style_templates[GrooveStyle.ROCK] = GrooveFeatures(
            timing_offsets=[0, 0, 0, 0] * 4,  # Straight time
            velocity_curve=[100, 80, 100, 80] * 4,  # 1 and 3 emphasis
            swing_amount=0.0,
            note_durations=[0.95, 0.95, 0.95, 0.95] * 4,
            ghost_note_ratio=0.1,
            density=2.0,
            syncopation=0.2,
        )

        # Latin: syncopated, clave
        self._style_templates[GrooveStyle.LATIN] = GrooveFeatures(
            timing_offsets=[0, 3, -5, 2] * 4,
            velocity_curve=[100, 60, 90, 70] * 4,
            swing_amount=0.0,
            note_durations=[0.5, 0.3, 0.5, 0.4] * 4,
            ghost_note_ratio=0.25,
            density=4.0,
            syncopation=0.8,
        )

        # Reggae: offbeat emphasis
        self._style_templates[GrooveStyle.REGGAE] = GrooveFeatures(
            timing_offsets=[0, 5, 0, 5] * 4,
            velocity_curve=[60, 100, 60, 100] * 4,  # Offbeat emphasis
            swing_amount=0.15,
            note_durations=[0.3, 0.8, 0.3, 0.8] * 4,
            ghost_note_ratio=0.15,
            density=2.0,
            syncopation=0.6,
        )

        # Electronic: quantized, consistent
        self._style_templates[GrooveStyle.ELECTRONIC] = GrooveFeatures(
            timing_offsets=[0, 0, 0, 0] * 4,
            velocity_curve=[100, 80, 90, 80] * 4,
            swing_amount=0.0,
            note_durations=[1.0, 1.0, 1.0, 1.0] * 4,
            ghost_note_ratio=0.05,
            density=4.0,
            syncopation=0.3,
        )

        # Soul: warm, groovy
        self._style_templates[GrooveStyle.SOUL] = GrooveFeatures(
            timing_offsets=[0, 3, 0, 5] * 4,
            velocity_curve=[85, 95, 80, 100] * 4,
            swing_amount=0.3,
            note_durations=[0.9, 0.8, 0.85, 0.9] * 4,
            ghost_note_ratio=0.25,
            density=2.5,
            syncopation=0.5,
        )

        # Blues: shuffle, expressive
        self._style_templates[GrooveStyle.BLUES] = GrooveFeatures(
            timing_offsets=[0, 8, 0, 8] * 4,  # Shuffle timing
            velocity_curve=[100, 70, 85, 75] * 4,
            swing_amount=0.5,
            note_durations=[0.8, 0.6, 0.7, 0.65] * 4,
            ghost_note_ratio=0.2,
            density=2.0,
            syncopation=0.4,
        )

        # Pop: clean, modern
        self._style_templates[GrooveStyle.POP] = GrooveFeatures(
            timing_offsets=[0, 2, 0, 2] * 4,
            velocity_curve=[100, 75, 90, 80] * 4,
            swing_amount=0.05,
            note_durations=[0.9, 0.85, 0.9, 0.85] * 4,
            ghost_note_ratio=0.1,
            density=2.0,
            syncopation=0.3,
        )

    def extract_features(
        self,
        midi_notes: List[Dict],
        ppq: int = 480,
    ) -> GrooveFeatures:
        """
        Extract groove features from MIDI notes.

        Args:
            midi_notes: List of note dicts with start_tick, velocity, duration
            ppq: Pulses per quarter note

        Returns:
            Extracted groove features
        """
        if not midi_notes:
            return GrooveFeatures(
                timing_offsets=[0.0] * 16,
                velocity_curve=[64] * 16,
                swing_amount=0.0,
                note_durations=[0.5] * 16,
                ghost_note_ratio=0.0,
                density=0.0,
                syncopation=0.0,
            )

        # Sort by time
        sorted_notes = sorted(midi_notes, key=lambda n: n.get("start_tick", 0))

        # Calculate grid positions
        beat_ticks = ppq  # One beat = ppq ticks
        sixteenth_ticks = ppq // 4

        timing_offsets = []
        velocities = []
        durations = []

        for note in sorted_notes[:16]:  # First 16 notes
            start = note.get("start_tick", 0)
            vel = note.get("velocity", 64)
            dur = note.get("duration_ticks", ppq // 2)

            # Find nearest grid position (16th notes)
            grid_pos = round(start / sixteenth_ticks) * sixteenth_ticks
            offset_ms = (start - grid_pos) / ppq * 500  # Convert to ms at 120 BPM

            timing_offsets.append(offset_ms)
            velocities.append(vel)
            durations.append(dur / beat_ticks)

        # Pad to 16
        while len(timing_offsets) < 16:
            timing_offsets.append(0.0)
            velocities.append(64)
            durations.append(0.5)

        # Calculate swing
        even_offsets = timing_offsets[0::2]
        odd_offsets = timing_offsets[1::2]
        swing_diff = np.mean(odd_offsets) - np.mean(even_offsets) if odd_offsets else 0
        swing_amount = max(0, min(1, swing_diff / 20))  # Normalize

        # Calculate ghost note ratio
        ghost_threshold = 50
        ghost_count = sum(1 for v in velocities if v < ghost_threshold)
        ghost_ratio = ghost_count / len(velocities)

        # Calculate density and syncopation
        total_beats = max(n.get("start_tick", 0) for n in sorted_notes) / ppq + 1
        density = len(sorted_notes) / total_beats if total_beats > 0 else 0

        # Syncopation: notes not on strong beats
        syncopated = 0
        for note in sorted_notes:
            pos_in_beat = (note.get("start_tick", 0) % ppq) / ppq
            if pos_in_beat > 0.1 and abs(pos_in_beat - 0.5) > 0.1:
                syncopated += 1
        syncopation = syncopated / len(sorted_notes) if sorted_notes else 0

        return GrooveFeatures(
            timing_offsets=timing_offsets,
            velocity_curve=velocities,
            swing_amount=swing_amount,
            note_durations=durations,
            ghost_note_ratio=ghost_ratio,
            density=density,
            syncopation=syncopation,
            midi_notes=midi_notes,
        )

    def transfer(
        self,
        features: GrooveFeatures,
        target_style: GrooveStyle,
        intensity: float = 1.0,
    ) -> StyleTransferResult:
        """
        Transfer groove to target style.

        Args:
            features: Source groove features
            target_style: Target style to transfer to
            intensity: Transfer intensity (0.0 = no change, 1.0 = full)

        Returns:
            Style transfer result
        """
        # Try neural network first
        if self._engine and self._engine.is_loaded():
            return self._transfer_neural(features, target_style, intensity)

        # Fall back to template-based transfer
        if self._use_fallback:
            return self._transfer_template(features, target_style, intensity)

        # No transfer available
        return StyleTransferResult(
            original_features=features,
            transferred_features=features,
            source_style=None,
            target_style=target_style,
            confidence=0.0,
        )

    def _transfer_neural(
        self,
        features: GrooveFeatures,
        target_style: GrooveStyle,
        intensity: float,
    ) -> StyleTransferResult:
        """Transfer using neural network."""
        # Encode features
        input_arr = features.to_array().reshape(1, -1)

        # One-hot encode target style
        style_idx = list(GrooveStyle).index(target_style)
        style_one_hot = np.zeros(len(GrooveStyle), dtype=np.float32)
        style_one_hot[style_idx] = 1.0

        # Concatenate inputs
        full_input = np.concatenate([input_arr, style_one_hot.reshape(1, -1)], axis=1)

        # Run inference
        result = self._engine.infer({"input": full_input})
        output = result.get_output()

        # Decode output
        transferred = GrooveFeatures.from_array(output.flatten())

        # Blend based on intensity
        if intensity < 1.0:
            transferred = self._blend_features(features, transferred, intensity)

        return StyleTransferResult(
            original_features=features,
            transferred_features=transferred,
            source_style=None,
            target_style=target_style,
            confidence=float(result.confidence or 0.8),
        )

    def _transfer_template(
        self,
        features: GrooveFeatures,
        target_style: GrooveStyle,
        intensity: float,
    ) -> StyleTransferResult:
        """Transfer using template blending."""
        template = self._style_templates.get(target_style)
        if not template:
            template = self._style_templates[GrooveStyle.POP]  # Default

        # Blend features
        transferred = self._blend_features(features, template, intensity)

        return StyleTransferResult(
            original_features=features,
            transferred_features=transferred,
            source_style=None,
            target_style=target_style,
            confidence=0.7,
        )

    def _blend_features(
        self,
        source: GrooveFeatures,
        target: GrooveFeatures,
        intensity: float,
    ) -> GrooveFeatures:
        """Blend two groove features."""
        def blend_list(a: List, b: List, t: float) -> List:
            return [
                a_val * (1 - t) + b_val * t
                for a_val, b_val in zip(a, b)
            ]

        def blend_scalar(a: float, b: float, t: float) -> float:
            return a * (1 - t) + b * t

        return GrooveFeatures(
            timing_offsets=blend_list(
                source.timing_offsets,
                target.timing_offsets,
                intensity,
            ),
            velocity_curve=[
                int(round(v))
                for v in blend_list(
                    source.velocity_curve,
                    target.velocity_curve,
                    intensity,
                )
            ],
            swing_amount=blend_scalar(
                source.swing_amount,
                target.swing_amount,
                intensity,
            ),
            note_durations=blend_list(
                source.note_durations,
                target.note_durations,
                intensity,
            ),
            ghost_note_ratio=blend_scalar(
                source.ghost_note_ratio,
                target.ghost_note_ratio,
                intensity,
            ),
            density=blend_scalar(source.density, target.density, intensity),
            syncopation=blend_scalar(source.syncopation, target.syncopation, intensity),
        )

    def apply_transfer(
        self,
        midi_notes: List[Dict],
        result: StyleTransferResult,
        ppq: int = 480,
    ) -> List[Dict]:
        """
        Apply style transfer result to MIDI notes.

        Args:
            midi_notes: Original MIDI notes
            result: Style transfer result
            ppq: Pulses per quarter note

        Returns:
            Transformed MIDI notes
        """
        if not midi_notes:
            return []

        transferred = result.transferred_features
        transformed = []

        for i, note in enumerate(midi_notes):
            new_note = dict(note)

            # Apply timing offset
            if i < len(transferred.timing_offsets):
                offset_ticks = int(transferred.timing_offsets[i] * ppq / 500)
                new_note["start_tick"] = max(0, note["start_tick"] + offset_ticks)

            # Apply velocity
            if i < len(transferred.velocity_curve):
                new_note["velocity"] = max(1, min(127, transferred.velocity_curve[i]))

            # Apply duration
            if i < len(transferred.note_durations):
                beat_ticks = ppq
                new_note["duration_ticks"] = int(
                    transferred.note_durations[i] * beat_ticks
                )

            transformed.append(new_note)

        return transformed


# Convenience functions
def transfer_groove_style(
    midi_notes: List[Dict],
    target_style: GrooveStyle,
    intensity: float = 1.0,
    ppq: int = 480,
) -> List[Dict]:
    """
    Transfer MIDI notes to a target groove style.

    Args:
        midi_notes: Input MIDI notes
        target_style: Target groove style
        intensity: Transfer intensity (0.0-1.0)
        ppq: Pulses per quarter note

    Returns:
        Transformed MIDI notes
    """
    transfer = GrooveStyleTransfer()
    features = transfer.extract_features(midi_notes, ppq)
    result = transfer.transfer(features, target_style, intensity)
    return transfer.apply_transfer(midi_notes, result, ppq)


def get_style_template(style: GrooveStyle) -> GrooveFeatures:
    """Get the template features for a style."""
    transfer = GrooveStyleTransfer()
    return transfer._style_templates.get(style, transfer._style_templates[GrooveStyle.POP])
