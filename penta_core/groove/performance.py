"""
Live Performance Analysis - Analyze timing and expression in performances.

Provides:
- Timing deviation analysis
- Tempo variation detection
- Expression extraction (dynamics, phrasing)
- Performance scoring and feedback
- Comparison with reference performances
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math


class PerformanceRating(Enum):
    """Performance quality ratings."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    NEEDS_WORK = "needs_work"


@dataclass
class TimingProfile:
    """
    Detailed timing analysis of a performance.
    """
    # Overall metrics
    mean_deviation_ms: float = 0.0
    std_deviation_ms: float = 0.0
    max_deviation_ms: float = 0.0

    # Directional bias
    ahead_percentage: float = 50.0  # % of notes ahead of beat
    behind_percentage: float = 50.0  # % of notes behind beat

    # Consistency over time
    timing_stability: float = 1.0   # 0.0-1.0, lower = more drift
    rushing_tendency: float = 0.0   # Tendency to speed up (-1 to 1)

    # Per-beat analysis
    beat_deviations: List[float] = field(default_factory=list)

    # Rating
    overall_rating: PerformanceRating = PerformanceRating.GOOD

    def get_summary(self) -> str:
        """Get human-readable summary."""
        tendency = ""
        if self.ahead_percentage > 60:
            tendency = "tends to rush"
        elif self.behind_percentage > 60:
            tendency = "tends to drag"
        else:
            tendency = "balanced timing"

        return (
            f"Timing: {self.mean_deviation_ms:.1f}ms avg deviation, "
            f"±{self.std_deviation_ms:.1f}ms variance, {tendency}"
        )


@dataclass
class PerformanceAnalysis:
    """
    Complete analysis of a live performance.
    """
    timing: TimingProfile
    tempo_bpm: float = 120.0
    tempo_stability: float = 1.0  # 0.0-1.0

    # Dynamics analysis
    velocity_mean: float = 80.0
    velocity_std: float = 15.0
    dynamic_range: float = 60.0  # Max - min velocity

    # Phrasing
    phrase_lengths: List[int] = field(default_factory=list)
    breath_points: List[float] = field(default_factory=list)  # Times of pauses

    # Expression markers
    accents: List[Tuple[float, float]] = field(default_factory=list)  # (time, intensity)
    crescendos: List[Tuple[float, float, float]] = field(default_factory=list)  # (start, end, intensity)
    decrescendos: List[Tuple[float, float, float]] = field(default_factory=list)

    # Overall
    expressiveness_score: float = 0.5  # 0.0-1.0
    consistency_score: float = 0.5     # 0.0-1.0
    musicality_score: float = 0.5      # 0.0-1.0

    def get_overall_rating(self) -> PerformanceRating:
        """Calculate overall performance rating."""
        avg_score = (
            self.expressiveness_score +
            self.consistency_score +
            self.musicality_score
        ) / 3

        if avg_score >= 0.85:
            return PerformanceRating.EXCELLENT
        elif avg_score >= 0.7:
            return PerformanceRating.GOOD
        elif avg_score >= 0.5:
            return PerformanceRating.FAIR
        else:
            return PerformanceRating.NEEDS_WORK

    def get_feedback(self) -> List[str]:
        """Generate constructive feedback."""
        feedback = []

        # Timing feedback
        if self.timing.mean_deviation_ms > 15:
            feedback.append(
                f"Work on timing precision - average deviation is {self.timing.mean_deviation_ms:.0f}ms"
            )
        if self.timing.ahead_percentage > 65:
            feedback.append("Try to relax tempo - tendency to rush")
        elif self.timing.behind_percentage > 65:
            feedback.append("Work on forward momentum - tendency to drag")

        # Dynamics feedback
        if self.dynamic_range < 30:
            feedback.append("Add more dynamic contrast for expressiveness")
        elif self.dynamic_range > 100:
            feedback.append("Consider moderating dynamic extremes for consistency")

        # Expression feedback
        if self.expressiveness_score < 0.5:
            feedback.append("Add more expression through dynamics and timing variations")

        if not feedback:
            feedback.append("Great performance! Keep up the good work.")

        return feedback


def analyze_live_performance(
    events: List[Dict],
    reference_tempo: Optional[float] = None,
    grid_resolution: int = 16,
) -> PerformanceAnalysis:
    """
    Analyze a live performance from MIDI events.

    Args:
        events: List of MIDI events with 'time', 'velocity'
        reference_tempo: Expected tempo (auto-detected if None)
        grid_resolution: Grid resolution for timing analysis

    Returns:
        PerformanceAnalysis object
    """
    if not events:
        return PerformanceAnalysis(timing=TimingProfile())

    # Sort by time
    sorted_events = sorted(events, key=lambda e: e.get("time", 0))

    # Detect tempo if not provided
    if reference_tempo is None:
        reference_tempo = _estimate_tempo(sorted_events)

    beat_duration = 60.0 / reference_tempo

    # Analyze timing
    timing_profile = _analyze_timing(sorted_events, beat_duration, grid_resolution)

    # Analyze dynamics
    velocities = [e.get("velocity", 100) for e in sorted_events]
    velocity_mean = sum(velocities) / len(velocities)
    velocity_std = math.sqrt(
        sum((v - velocity_mean) ** 2 for v in velocities) / len(velocities)
    )
    dynamic_range = max(velocities) - min(velocities)

    # Detect expression markers
    accents = _detect_accents(sorted_events, velocity_mean)
    crescendos, decrescendos = _detect_dynamics_changes(sorted_events)

    # Detect phrase boundaries
    breath_points = _detect_breath_points(sorted_events, beat_duration)

    # Calculate scores
    expressiveness = _calculate_expressiveness(
        velocity_std, dynamic_range, accents, crescendos, decrescendos
    )
    consistency = _calculate_consistency(timing_profile, velocity_std)
    musicality = _calculate_musicality(
        expressiveness, consistency, timing_profile
    )

    # Detect tempo stability
    tempo_stability = _analyze_tempo_stability(sorted_events, reference_tempo)

    return PerformanceAnalysis(
        timing=timing_profile,
        tempo_bpm=reference_tempo,
        tempo_stability=tempo_stability,
        velocity_mean=velocity_mean,
        velocity_std=velocity_std,
        dynamic_range=dynamic_range,
        breath_points=breath_points,
        accents=accents,
        crescendos=crescendos,
        decrescendos=decrescendos,
        expressiveness_score=expressiveness,
        consistency_score=consistency,
        musicality_score=musicality,
    )


def _estimate_tempo(events: List[Dict]) -> float:
    """Estimate tempo from events using onset intervals."""
    if len(events) < 2:
        return 120.0

    times = [e.get("time", 0) for e in events]
    # More efficient: use zip to create pairs
    intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]

    # Filter out very short intervals (grace notes) and long ones (rests)
    valid_intervals = [i for i in intervals if 0.1 < i < 2.0]

    if not valid_intervals:
        return 120.0

    # Find most common interval cluster
    avg_interval = sum(valid_intervals) / len(valid_intervals)

    # Assume this is a beat or subdivision
    # Try to find the beat-level interval
    if avg_interval < 0.3:  # Likely 16th notes at moderate tempo
        beat_interval = avg_interval * 4
    elif avg_interval < 0.5:  # Likely 8th notes
        beat_interval = avg_interval * 2
    else:
        beat_interval = avg_interval

    return 60.0 / beat_interval


def _analyze_timing(
    events: List[Dict],
    beat_duration: float,
    grid_resolution: int,
) -> TimingProfile:
    """Analyze timing deviations."""
    if not events:
        return TimingProfile()

    grid_step = beat_duration / grid_resolution
    deviations = []
    ahead_count = 0
    behind_count = 0

    for event in events:
        time = event.get("time", 0)

        # Find nearest grid position
        grid_position = round(time / grid_step)
        expected_time = grid_position * grid_step
        deviation_ms = (time - expected_time) * 1000

        deviations.append(deviation_ms)

        if deviation_ms < -1:
            ahead_count += 1
        elif deviation_ms > 1:
            behind_count += 1

    total = len(deviations)
    mean_deviation = sum(deviations) / total
    variance = sum((d - mean_deviation) ** 2 for d in deviations) / total
    std_deviation = math.sqrt(variance)

    # Calculate timing stability (less drift = more stable)
    first_half_mean = sum(deviations[:total//2]) / max(1, total//2)
    second_half_mean = sum(deviations[total//2:]) / max(1, total - total//2)
    drift = abs(second_half_mean - first_half_mean)
    stability = max(0.0, 1.0 - drift / 50.0)

    # Calculate rushing tendency
    rushing = (second_half_mean - first_half_mean) / max(1, std_deviation)
    rushing = max(-1.0, min(1.0, rushing))

    # Rating
    if std_deviation < 8:
        rating = PerformanceRating.EXCELLENT
    elif std_deviation < 15:
        rating = PerformanceRating.GOOD
    elif std_deviation < 25:
        rating = PerformanceRating.FAIR
    else:
        rating = PerformanceRating.NEEDS_WORK

    return TimingProfile(
        mean_deviation_ms=abs(mean_deviation),
        std_deviation_ms=std_deviation,
        max_deviation_ms=max(abs(d) for d in deviations),
        ahead_percentage=(ahead_count / total) * 100,
        behind_percentage=(behind_count / total) * 100,
        timing_stability=stability,
        rushing_tendency=rushing,
        beat_deviations=deviations,
        overall_rating=rating,
    )


def _detect_accents(
    events: List[Dict],
    mean_velocity: float,
) -> List[Tuple[float, float]]:
    """Detect accented notes."""
    accents = []
    threshold = mean_velocity + 20

    for event in events:
        velocity = event.get("velocity", 100)
        if velocity > threshold:
            intensity = (velocity - mean_velocity) / 127.0
            accents.append((event.get("time", 0), intensity))

    return accents


def _detect_dynamics_changes(
    events: List[Dict],
    window_size: int = 5,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """Detect crescendos and decrescendos."""
    if len(events) < window_size:
        return [], []

    crescendos = []
    decrescendos = []

    velocities = [e.get("velocity", 100) for e in events]
    times = [e.get("time", 0) for e in events]

    i = 0
    while i < len(velocities) - window_size:
        window = velocities[i:i + window_size]
        trend = window[-1] - window[0]

        if trend > 20:  # Crescendo
            start_time = times[i]
            end_time = times[i + window_size - 1]
            intensity = trend / 127.0
            crescendos.append((start_time, end_time, intensity))
            i += window_size
        elif trend < -20:  # Decrescendo
            start_time = times[i]
            end_time = times[i + window_size - 1]
            intensity = abs(trend) / 127.0
            decrescendos.append((start_time, end_time, intensity))
            i += window_size
        else:
            i += 1

    return crescendos, decrescendos


def _detect_breath_points(
    events: List[Dict],
    beat_duration: float,
    min_gap: float = 0.5,
) -> List[float]:
    """Detect natural pause/breath points."""
    if len(events) < 2:
        return []

    breath_points = []
    times = sorted(e.get("time", 0) for e in events)

    # More efficient: use zip to create pairs
    for t1, t2 in zip(times[:-1], times[1:]):
        gap = t2 - t1
        if gap > min_gap * beat_duration:
            breath_points.append(t2)

    return breath_points


def _calculate_expressiveness(
    velocity_std: float,
    dynamic_range: float,
    accents: List,
    crescendos: List,
    decrescendos: List,
) -> float:
    """Calculate expressiveness score."""
    # Dynamic variety contributes to expressiveness
    dynamic_score = min(1.0, velocity_std / 25.0)

    # Accents contribute
    accent_score = min(1.0, len(accents) / 10.0)

    # Dynamic changes contribute
    change_score = min(1.0, (len(crescendos) + len(decrescendos)) / 5.0)

    return (dynamic_score * 0.4 + accent_score * 0.3 + change_score * 0.3)


def _calculate_consistency(
    timing: TimingProfile,
    velocity_std: float,
) -> float:
    """Calculate consistency score."""
    # Lower timing variance = more consistent
    timing_score = max(0.0, 1.0 - timing.std_deviation_ms / 30.0)

    # Moderate velocity variance is good
    velocity_score = 1.0 if 10 < velocity_std < 25 else 0.7

    return timing_score * 0.7 + velocity_score * 0.3


def _calculate_musicality(
    expressiveness: float,
    consistency: float,
    timing: TimingProfile,
) -> float:
    """Calculate overall musicality score."""
    # Balance of expression and control
    balance_score = 1.0 - abs(expressiveness - consistency)

    # Timing stability matters
    stability_score = timing.timing_stability

    return (expressiveness * 0.3 + consistency * 0.3 +
            balance_score * 0.2 + stability_score * 0.2)


def detect_tempo_variations(
    events: List[Dict],
    window_beats: int = 4,
) -> List[Tuple[float, float]]:
    """
    Detect tempo variations over time.

    Args:
        events: List of MIDI events
        window_beats: Window size in beats for tempo calculation

    Returns:
        List of (time, tempo_bpm) tuples
    """
    if len(events) < 4:
        return []

    tempo_curve = []
    times = sorted(e.get("time", 0) for e in events)

    # Calculate local tempo at each point
    for i, _ in enumerate(times[:-window_beats]):
        window_times = times[i:i + window_beats + 1]
        # More efficient: use zip to create pairs
        intervals = [t2 - t1 for t1, t2 in zip(window_times[:-1], window_times[1:])]

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval > 0:
                local_tempo = 60.0 / avg_interval
                tempo_curve.append((window_times[0], local_tempo))

    return tempo_curve


def extract_expression(
    events: List[Dict],
    tempo_bpm: float = 120.0,
) -> Dict:
    """
    Extract expression parameters from performance.

    Args:
        events: List of MIDI events
        tempo_bpm: Reference tempo

    Returns:
        Dict with expression parameters
    """
    analysis = analyze_live_performance(events, tempo_bpm)

    return {
        "timing": {
            "mean_deviation_ms": analysis.timing.mean_deviation_ms,
            "variance_ms": analysis.timing.std_deviation_ms,
            "ahead_bias": analysis.timing.ahead_percentage > 55,
            "behind_bias": analysis.timing.behind_percentage > 55,
        },
        "dynamics": {
            "mean_velocity": analysis.velocity_mean,
            "variance": analysis.velocity_std,
            "range": analysis.dynamic_range,
            "accents": len(analysis.accents),
        },
        "phrasing": {
            "breath_points": len(analysis.breath_points),
            "crescendos": len(analysis.crescendos),
            "decrescendos": len(analysis.decrescendos),
        },
        "scores": {
            "expressiveness": analysis.expressiveness_score,
            "consistency": analysis.consistency_score,
            "musicality": analysis.musicality_score,
        },
    }


def _analyze_tempo_stability(
    events: List[Dict],
    reference_tempo: float,
) -> float:
    """Analyze how stable the tempo is throughout the performance."""
    tempo_variations = detect_tempo_variations(events)

    if not tempo_variations:
        return 1.0

    tempos = [t[1] for t in tempo_variations]
    mean_tempo = sum(tempos) / len(tempos)

    # Calculate variance from reference
    variance = sum((t - reference_tempo) ** 2 for t in tempos) / len(tempos)
    std = math.sqrt(variance)

    # Normalize to 0-1 (0 = unstable, 1 = stable)
    stability = max(0.0, 1.0 - std / reference_tempo)

    return stability
