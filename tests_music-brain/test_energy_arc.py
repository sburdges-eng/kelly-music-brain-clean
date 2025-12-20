"""
Tests for music_brain.arrangement.energy_arc module.

Tests cover:
- NarrativeArc enum
- EnergyArc dataclass functionality
- Energy curve calculation for all arc types
- Energy interpolation
- Emotion to arc mapping
- Curve smoothing
"""

import pytest
import math


class TestNarrativeArcEnum:
    """Tests for NarrativeArc enum."""

    def test_all_narrative_arcs_defined(self):
        """Verify all expected narrative arcs exist."""
        from music_brain.arrangement.energy_arc import NarrativeArc

        expected_arcs = [
            "CLIMB_TO_CLIMAX",
            "SLOW_REVEAL",
            "REPETITIVE_DESPAIR",
            "SUDDEN_BREAK",
            "QUIET_ACCEPTANCE",
            "EXPLOSIVE_START",
            "WAVE_PATTERN",
        ]

        for arc_name in expected_arcs:
            assert hasattr(NarrativeArc, arc_name), f"Missing arc: {arc_name}"

    def test_narrative_arc_values(self):
        """Test that arc values are lowercase strings with hyphens."""
        from music_brain.arrangement.energy_arc import NarrativeArc

        assert NarrativeArc.CLIMB_TO_CLIMAX.value == "climb-to-climax"
        assert NarrativeArc.SLOW_REVEAL.value == "slow-reveal"
        assert NarrativeArc.REPETITIVE_DESPAIR.value == "repetitive-despair"
        assert NarrativeArc.SUDDEN_BREAK.value == "sudden-break"
        assert NarrativeArc.QUIET_ACCEPTANCE.value == "quiet-acceptance"
        assert NarrativeArc.EXPLOSIVE_START.value == "explosive-start"
        assert NarrativeArc.WAVE_PATTERN.value == "wave-pattern"


class TestEnergyArc:
    """Tests for EnergyArc dataclass."""

    def test_energy_arc_creation(self):
        """Test creating an EnergyArc instance."""
        from music_brain.arrangement.energy_arc import EnergyArc, NarrativeArc

        arc = EnergyArc(
            narrative_arc=NarrativeArc.CLIMB_TO_CLIMAX,
            energy_curve=[0.3, 0.5, 0.7, 0.9, 0.85],
            peak_position=0.8,
            intensity_range=(0.2, 0.9),
        )

        assert arc.narrative_arc == NarrativeArc.CLIMB_TO_CLIMAX
        assert len(arc.energy_curve) == 5
        assert arc.peak_position == 0.8
        assert arc.intensity_range == (0.2, 0.9)

    def test_energy_arc_defaults(self):
        """Test default values."""
        from music_brain.arrangement.energy_arc import EnergyArc, NarrativeArc

        arc = EnergyArc(
            narrative_arc=NarrativeArc.CLIMB_TO_CLIMAX,
            energy_curve=[0.5, 0.6, 0.7],
        )

        assert arc.peak_position == 0.7  # Default
        assert arc.intensity_range == (0.2, 0.9)  # Default

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from music_brain.arrangement.energy_arc import EnergyArc, NarrativeArc

        arc = EnergyArc(
            narrative_arc=NarrativeArc.SLOW_REVEAL,
            energy_curve=[0.2, 0.25, 0.3, 0.8],
            peak_position=0.85,
        )

        result = arc.to_dict()

        assert result["narrative_arc"] == "slow-reveal"
        assert result["energy_curve"] == [0.2, 0.25, 0.3, 0.8]
        assert result["peak_position"] == 0.85


class TestEnergyInterpolation:
    """Tests for get_energy_at_position method."""

    def test_get_energy_at_start(self):
        """Test energy at position 0.0."""
        from music_brain.arrangement.energy_arc import EnergyArc, NarrativeArc

        arc = EnergyArc(
            narrative_arc=NarrativeArc.CLIMB_TO_CLIMAX,
            energy_curve=[0.3, 0.5, 0.7, 0.9],
        )

        energy = arc.get_energy_at_position(0.0)
        assert energy == 0.3

    def test_get_energy_at_end(self):
        """Test energy at position 1.0."""
        from music_brain.arrangement.energy_arc import EnergyArc, NarrativeArc

        arc = EnergyArc(
            narrative_arc=NarrativeArc.CLIMB_TO_CLIMAX,
            energy_curve=[0.3, 0.5, 0.7, 0.9],
        )

        energy = arc.get_energy_at_position(1.0)
        assert energy == 0.9

    def test_get_energy_at_midpoint(self):
        """Test energy interpolation at midpoint."""
        from music_brain.arrangement.energy_arc import EnergyArc, NarrativeArc

        arc = EnergyArc(
            narrative_arc=NarrativeArc.CLIMB_TO_CLIMAX,
            energy_curve=[0.0, 1.0],  # Linear from 0 to 1
        )

        energy = arc.get_energy_at_position(0.5)
        assert abs(energy - 0.5) < 0.01  # Should interpolate to 0.5

    def test_get_energy_interpolation_between_points(self):
        """Test linear interpolation between curve points."""
        from music_brain.arrangement.energy_arc import EnergyArc, NarrativeArc

        arc = EnergyArc(
            narrative_arc=NarrativeArc.CLIMB_TO_CLIMAX,
            energy_curve=[0.2, 0.4, 0.6, 0.8, 1.0],  # 5 points
        )

        # At 0.125 (between index 0 and 1)
        energy = arc.get_energy_at_position(0.125)
        # Index = 0.125 * 4 = 0.5, so halfway between 0.2 and 0.4
        assert abs(energy - 0.3) < 0.01

    def test_get_energy_empty_curve(self):
        """Test energy with empty curve returns default."""
        from music_brain.arrangement.energy_arc import EnergyArc, NarrativeArc

        arc = EnergyArc(
            narrative_arc=NarrativeArc.CLIMB_TO_CLIMAX,
            energy_curve=[],
        )

        energy = arc.get_energy_at_position(0.5)
        assert energy == 0.5  # Default fallback


class TestCalculateEnergyCurve:
    """Tests for calculate_energy_curve function."""

    def test_climb_to_climax(self):
        """Test CLIMB_TO_CLIMAX arc generation."""
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc,
        )

        arc = calculate_energy_curve(
            NarrativeArc.CLIMB_TO_CLIMAX,
            num_sections=8,
            base_intensity=0.6,
        )

        assert arc.narrative_arc == NarrativeArc.CLIMB_TO_CLIMAX
        assert len(arc.energy_curve) == 8

        # Energy should generally increase to peak
        curve = arc.energy_curve
        assert curve[-2] > curve[0]  # Climax higher than start

    def test_slow_reveal(self):
        """Test SLOW_REVEAL arc stays low then rises."""
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc,
        )

        arc = calculate_energy_curve(
            NarrativeArc.SLOW_REVEAL,
            num_sections=10,
            base_intensity=0.6,
        )

        curve = arc.energy_curve

        # First 70% should be relatively low
        early_avg = sum(curve[:7]) / 7
        late_avg = sum(curve[7:]) / 3

        assert late_avg > early_avg, "Late sections should have higher energy"

    def test_repetitive_despair(self):
        """Test REPETITIVE_DESPAIR oscillates around base."""
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc,
        )

        arc = calculate_energy_curve(
            NarrativeArc.REPETITIVE_DESPAIR,
            num_sections=8,
            base_intensity=0.5,
        )

        curve = arc.energy_curve

        # Should oscillate (not monotonic)
        min_val = min(curve)
        max_val = max(curve)

        # Range should be small (no dramatic build)
        assert max_val - min_val < 0.3

    def test_sudden_break(self):
        """Test SUDDEN_BREAK has sharp transition."""
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc,
        )

        arc = calculate_energy_curve(
            NarrativeArc.SUDDEN_BREAK,
            num_sections=10,
            base_intensity=0.6,
        )

        curve = arc.energy_curve

        # First ~40% should be low
        early_sections = curve[:4]
        late_sections = curve[5:]

        assert all(e < 0.5 for e in early_sections), "Early sections should be calm"
        assert all(e > 0.7 for e in late_sections), "Late sections should be intense"

    def test_quiet_acceptance(self):
        """Test QUIET_ACCEPTANCE stays gentle throughout."""
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc,
        )

        arc = calculate_energy_curve(
            NarrativeArc.QUIET_ACCEPTANCE,
            num_sections=8,
            base_intensity=0.5,
        )

        curve = arc.energy_curve

        # Should never get very intense
        assert all(e <= 0.5 for e in curve), "All sections should be gentle"

    def test_explosive_start(self):
        """Test EXPLOSIVE_START begins high and fades."""
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc,
        )

        arc = calculate_energy_curve(
            NarrativeArc.EXPLOSIVE_START,
            num_sections=8,
            base_intensity=0.6,
        )

        curve = arc.energy_curve

        # First section should be highest
        assert curve[0] >= max(curve[1:]), "First section should be peak"
        # Last should be lower
        assert curve[-1] < curve[0], "Energy should decay"

    def test_wave_pattern(self):
        """Test WAVE_PATTERN has multiple peaks."""
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc,
        )

        arc = calculate_energy_curve(
            NarrativeArc.WAVE_PATTERN,
            num_sections=12,
            base_intensity=0.5,
        )

        curve = arc.energy_curve

        # Look for local maxima (peaks)
        peaks = []
        for i in range(1, len(curve) - 1):
            if curve[i] > curve[i - 1] and curve[i] > curve[i + 1]:
                peaks.append(i)

        # Should have multiple peaks (wave pattern)
        assert len(peaks) >= 1, "Wave pattern should have peaks"

    def test_custom_peak_position(self):
        """Test custom peak position is respected."""
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc,
        )

        arc = calculate_energy_curve(
            NarrativeArc.CLIMB_TO_CLIMAX,
            num_sections=10,
            peak_position=0.5,  # Peak at midpoint
        )

        assert arc.peak_position == 0.5

    def test_intensity_range_calculation(self):
        """Test intensity range is calculated from base intensity."""
        from music_brain.arrangement.energy_arc import (
            calculate_energy_curve,
            NarrativeArc,
        )

        arc = calculate_energy_curve(
            NarrativeArc.CLIMB_TO_CLIMAX,
            num_sections=8,
            base_intensity=0.7,
        )

        min_e, max_e = arc.intensity_range
        assert min_e >= 0.1
        assert max_e <= 1.0
        assert max_e > min_e


class TestDefaultPeakPosition:
    """Tests for _get_default_peak_position function."""

    def test_climb_to_climax_peak(self):
        """CLIMB_TO_CLIMAX should peak at 75%."""
        from music_brain.arrangement.energy_arc import (
            _get_default_peak_position,
            NarrativeArc,
        )

        pos = _get_default_peak_position(NarrativeArc.CLIMB_TO_CLIMAX)
        assert pos == 0.75

    def test_slow_reveal_peak(self):
        """SLOW_REVEAL should peak late at 85%."""
        from music_brain.arrangement.energy_arc import (
            _get_default_peak_position,
            NarrativeArc,
        )

        pos = _get_default_peak_position(NarrativeArc.SLOW_REVEAL)
        assert pos == 0.85

    def test_explosive_start_peak(self):
        """EXPLOSIVE_START should peak early at 10%."""
        from music_brain.arrangement.energy_arc import (
            _get_default_peak_position,
            NarrativeArc,
        )

        pos = _get_default_peak_position(NarrativeArc.EXPLOSIVE_START)
        assert pos == 0.1


class TestSmoothEnergyCurve:
    """Tests for smooth_energy_curve function."""

    def test_smooth_empty_curve(self):
        """Empty curve should return empty."""
        from music_brain.arrangement.energy_arc import smooth_energy_curve

        result = smooth_energy_curve([])
        assert result == []

    def test_smooth_short_curve(self):
        """Short curves (< 3) should return unchanged."""
        from music_brain.arrangement.energy_arc import smooth_energy_curve

        result = smooth_energy_curve([0.5, 0.7])
        assert result == [0.5, 0.7]

    def test_smooth_no_smoothing(self):
        """Zero smoothing should return original."""
        from music_brain.arrangement.energy_arc import smooth_energy_curve

        curve = [0.2, 0.8, 0.3, 0.9, 0.4]
        result = smooth_energy_curve(curve, smoothing=0.0)
        assert result == curve

    def test_smooth_preserves_endpoints(self):
        """Smoothing should preserve first and last values."""
        from music_brain.arrangement.energy_arc import smooth_energy_curve

        curve = [0.1, 0.9, 0.2, 0.8, 0.3]
        result = smooth_energy_curve(curve, smoothing=0.5)

        assert result[0] == 0.1
        assert result[-1] == 0.3

    def test_smooth_reduces_variance(self):
        """Smoothing should reduce variance in middle values."""
        from music_brain.arrangement.energy_arc import smooth_energy_curve

        # Spiky curve
        curve = [0.5, 0.9, 0.2, 0.8, 0.5]
        result = smooth_energy_curve(curve, smoothing=0.5)

        # Middle values should be closer to neighbors
        original_range = max(curve[1:-1]) - min(curve[1:-1])
        smoothed_range = max(result[1:-1]) - min(result[1:-1])

        assert smoothed_range <= original_range


class TestMapEmotionToArc:
    """Tests for map_emotion_to_arc function."""

    def test_grief_maps_to_slow_reveal(self):
        """Grief should map to SLOW_REVEAL."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        arc = map_emotion_to_arc("grief")
        assert arc == NarrativeArc.SLOW_REVEAL

    def test_anxiety_maps_to_wave(self):
        """Anxiety should map to WAVE_PATTERN."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        arc = map_emotion_to_arc("anxiety")
        assert arc == NarrativeArc.WAVE_PATTERN

    def test_anger_maps_to_explosive(self):
        """Anger should map to EXPLOSIVE_START."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        arc = map_emotion_to_arc("anger")
        assert arc == NarrativeArc.EXPLOSIVE_START

    def test_hope_maps_to_climb(self):
        """Hope should map to CLIMB_TO_CLIMAX."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        arc = map_emotion_to_arc("hope")
        assert arc == NarrativeArc.CLIMB_TO_CLIMAX

    def test_despair_maps_to_repetitive(self):
        """Despair should map to REPETITIVE_DESPAIR."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        arc = map_emotion_to_arc("despair")
        assert arc == NarrativeArc.REPETITIVE_DESPAIR

    def test_fear_maps_to_sudden_break(self):
        """Fear should map to SUDDEN_BREAK."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        arc = map_emotion_to_arc("fear")
        assert arc == NarrativeArc.SUDDEN_BREAK

    def test_case_insensitivity(self):
        """Emotion matching should be case-insensitive."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        assert map_emotion_to_arc("GRIEF") == NarrativeArc.SLOW_REVEAL
        assert map_emotion_to_arc("Anger") == NarrativeArc.EXPLOSIVE_START

    def test_unknown_emotion_default(self):
        """Unknown emotions should default to CLIMB_TO_CLIMAX."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        arc = map_emotion_to_arc("unknown_emotion")
        assert arc == NarrativeArc.CLIMB_TO_CLIMAX

    def test_high_tension_modifies_arc(self):
        """High secondary tension should modify arc choice."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        # Quiet acceptance + high tension = sudden break
        arc = map_emotion_to_arc("nostalgia", secondary_tension=0.8)
        assert arc == NarrativeArc.SUDDEN_BREAK

        # Slow reveal + high tension = climb to climax
        arc = map_emotion_to_arc("grief", secondary_tension=0.9)
        assert arc == NarrativeArc.CLIMB_TO_CLIMAX

    def test_low_tension_preserves_arc(self):
        """Low secondary tension should not modify arc."""
        from music_brain.arrangement.energy_arc import (
            map_emotion_to_arc,
            NarrativeArc,
        )

        arc = map_emotion_to_arc("nostalgia", secondary_tension=0.3)
        assert arc == NarrativeArc.QUIET_ACCEPTANCE  # Original mapping


class TestArcShapeFunctions:
    """Tests for individual arc shape generation functions."""

    def test_linear_growth(self):
        """Test _linear_growth produces linear curve."""
        from music_brain.arrangement.energy_arc import _linear_growth

        curve = _linear_growth(5, 0.2, 0.8)

        assert len(curve) == 5
        assert curve[0] == 0.2
        assert curve[-1] == 0.8

        # Should be approximately linear
        for i in range(1, len(curve)):
            assert curve[i] >= curve[i - 1], "Should be monotonically increasing"

    def test_climb_to_climax_shape(self):
        """Test _climb_to_climax produces correct shape."""
        from music_brain.arrangement.energy_arc import _climb_to_climax

        curve = _climb_to_climax(10, 0.2, 1.0, peak_pos=0.8)

        assert len(curve) == 10
        # Peak should be near position 8 (80%)
        peak_idx = curve.index(max(curve))
        assert 7 <= peak_idx <= 9

    def test_explosive_start_decays(self):
        """Test _explosive_start produces decaying curve."""
        from music_brain.arrangement.energy_arc import _explosive_start

        curve = _explosive_start(8, 0.2, 1.0)

        assert curve[0] > curve[-1], "Should decay from start to end"
        assert curve[0] == max(curve), "First value should be maximum"
