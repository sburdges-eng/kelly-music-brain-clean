"""
Tests for Penta-Core Rules Module.

Tests severity, context, base rules, emotion mappings, and timing pockets.
"""

import pytest
from unittest.mock import patch, MagicMock

from penta_core.rules.severity import RuleSeverity
from penta_core.rules.context import (
    MusicalContext,
    CONTEXT_GROUPS,
    get_context_group,
)
from penta_core.rules.base import (
    Rule,
    RuleViolation,
    RuleBreakSuggestion,
)
from penta_core.rules.emotion import (
    Emotion,
    EmotionalMapping,
    EMOTION_TO_TECHNIQUES,
    TECHNIQUE_TO_EMOTIONS,
    get_techniques_for_emotion,
    get_emotions_for_technique,
)
from penta_core.rules.timing import (
    SwingType,
    TimingPocket,
    GENRE_POCKETS,
    get_genre_pocket,
    apply_pocket_to_midi,
)


class TestRuleSeverity:
    """Tests for RuleSeverity enum."""

    def test_severity_values(self):
        """Test that all severity values are defined."""
        assert RuleSeverity.STRICT is not None
        assert RuleSeverity.GUIDELINE is not None
        assert RuleSeverity.STYLISTIC is not None
        assert RuleSeverity.MODERN is not None

    def test_severity_str(self):
        """Test string representation of severity."""
        assert str(RuleSeverity.STRICT) == "strict"
        assert str(RuleSeverity.GUIDELINE) == "guideline"
        assert str(RuleSeverity.STYLISTIC) == "stylistic"
        assert str(RuleSeverity.MODERN) == "modern"

    def test_severity_repr(self):
        """Test repr of severity."""
        assert "STRICT" in repr(RuleSeverity.STRICT)
        assert "RuleSeverity" in repr(RuleSeverity.STRICT)

    def test_severity_ordering(self):
        """Test that severities are distinct."""
        severities = [RuleSeverity.STRICT, RuleSeverity.GUIDELINE,
                      RuleSeverity.STYLISTIC, RuleSeverity.MODERN]
        assert len(set(severities)) == 4


class TestMusicalContext:
    """Tests for MusicalContext enum."""

    def test_context_values(self):
        """Test that all context values are defined."""
        assert MusicalContext.RENAISSANCE is not None
        assert MusicalContext.BAROQUE is not None
        assert MusicalContext.CLASSICAL is not None
        assert MusicalContext.ROMANTIC is not None
        assert MusicalContext.JAZZ is not None
        assert MusicalContext.CONTEMPORARY is not None

    def test_context_str(self):
        """Test string representation of context."""
        assert str(MusicalContext.CLASSICAL) == "classical"
        assert str(MusicalContext.JAZZ) == "jazz"

    def test_context_repr(self):
        """Test repr of context."""
        assert "BAROQUE" in repr(MusicalContext.BAROQUE)


class TestContextGroups:
    """Tests for context groupings."""

    def test_common_practice_group(self):
        """Test common practice period group."""
        group = CONTEXT_GROUPS["common_practice"]
        assert MusicalContext.BAROQUE in group
        assert MusicalContext.CLASSICAL in group
        assert MusicalContext.ROMANTIC in group
        assert MusicalContext.JAZZ not in group

    def test_early_music_group(self):
        """Test early music group."""
        group = CONTEXT_GROUPS["early_music"]
        assert MusicalContext.RENAISSANCE in group
        assert MusicalContext.BAROQUE in group
        assert MusicalContext.CLASSICAL not in group

    def test_tonal_group(self):
        """Test tonal music group."""
        group = CONTEXT_GROUPS["tonal"]
        assert MusicalContext.BAROQUE in group
        assert MusicalContext.JAZZ in group
        assert MusicalContext.CONTEMPORARY not in group

    def test_modern_group(self):
        """Test modern music group."""
        group = CONTEXT_GROUPS["modern"]
        assert MusicalContext.JAZZ in group
        assert MusicalContext.CONTEMPORARY in group

    def test_all_group(self):
        """Test all contexts group."""
        group = CONTEXT_GROUPS["all"]
        assert len(group) == len(MusicalContext)

    def test_get_context_group(self):
        """Test get_context_group function."""
        group = get_context_group("common_practice")
        assert MusicalContext.CLASSICAL in group

    def test_get_context_group_invalid(self):
        """Test get_context_group with invalid name."""
        with pytest.raises(KeyError):
            get_context_group("invalid_group")


class TestRule:
    """Tests for Rule dataclass."""

    @pytest.fixture
    def sample_rule(self):
        """Create a sample rule for testing."""
        return Rule(
            name="no_parallel_fifths",
            description="Avoid parallel perfect fifths between voices",
            severity=RuleSeverity.STRICT,
            contexts={MusicalContext.CLASSICAL, MusicalContext.BAROQUE},
            reason="Parallel fifths reduce voice independence",
            exceptions=["Power chords in rock music"],
            category="parallel_motion",
        )

    def test_rule_creation(self, sample_rule):
        """Test creating a rule."""
        assert sample_rule.name == "no_parallel_fifths"
        assert sample_rule.severity == RuleSeverity.STRICT
        assert len(sample_rule.contexts) == 2

    def test_rule_defaults(self):
        """Test rule default values."""
        rule = Rule(
            name="test",
            description="Test rule",
            severity=RuleSeverity.GUIDELINE,
            contexts={MusicalContext.CLASSICAL},
            reason="Testing",
        )
        assert rule.exceptions == []
        assert rule.category == ""

    def test_applies_to_context(self, sample_rule):
        """Test applies_to_context method."""
        assert sample_rule.applies_to_context(MusicalContext.CLASSICAL) is True
        assert sample_rule.applies_to_context(MusicalContext.JAZZ) is False

    def test_get_severity_for_context_simple(self, sample_rule):
        """Test getting severity for a context (simple case)."""
        severity = sample_rule.get_severity_for_context(MusicalContext.CLASSICAL)
        assert severity == RuleSeverity.STRICT

    def test_get_severity_for_context_dict(self):
        """Test getting severity when severity is context-dependent."""
        rule = Rule(
            name="parallel_fifths",
            description="Parallel fifths",
            severity={
                MusicalContext.CLASSICAL: RuleSeverity.STRICT,
                MusicalContext.JAZZ: RuleSeverity.STYLISTIC,
            },
            contexts={MusicalContext.CLASSICAL, MusicalContext.JAZZ},
            reason="Context-dependent rule",
        )
        assert rule.get_severity_for_context(MusicalContext.CLASSICAL) == RuleSeverity.STRICT
        assert rule.get_severity_for_context(MusicalContext.JAZZ) == RuleSeverity.STYLISTIC
        # Unknown context should return GUIDELINE
        assert rule.get_severity_for_context(MusicalContext.CONTEMPORARY) == RuleSeverity.GUIDELINE

    def test_is_strict(self, sample_rule):
        """Test is_strict method."""
        assert sample_rule.is_strict() is True

    def test_is_strict_dict(self):
        """Test is_strict with context-dependent severity."""
        rule = Rule(
            name="test",
            description="Test",
            severity={
                MusicalContext.CLASSICAL: RuleSeverity.STRICT,
                MusicalContext.JAZZ: RuleSeverity.STYLISTIC,
            },
            contexts=set(MusicalContext),
            reason="Test",
        )
        assert rule.is_strict() is True

    def test_is_strict_not_strict(self):
        """Test is_strict when rule is not strict."""
        rule = Rule(
            name="test",
            description="Test",
            severity=RuleSeverity.GUIDELINE,
            contexts={MusicalContext.CLASSICAL},
            reason="Test",
        )
        assert rule.is_strict() is False

    def test_to_dict(self, sample_rule):
        """Test converting rule to dictionary."""
        data = sample_rule.to_dict()
        assert data["name"] == "no_parallel_fifths"
        assert data["description"] == "Avoid parallel perfect fifths between voices"
        assert "strict" in str(data["severity"]).lower()
        assert "classical" in [c.lower() for c in data["context"]]

    def test_to_dict_with_dict_severity(self):
        """Test to_dict with context-dependent severity."""
        rule = Rule(
            name="test",
            description="Test",
            severity={
                MusicalContext.CLASSICAL: RuleSeverity.STRICT,
            },
            contexts={MusicalContext.CLASSICAL},
            reason="Test",
        )
        data = rule.to_dict()
        assert isinstance(data["severity"], dict)


class TestRuleViolation:
    """Tests for RuleViolation dataclass."""

    @pytest.fixture
    def sample_rule(self):
        return Rule(
            name="no_parallel_fifths",
            description="Avoid parallel fifths",
            severity=RuleSeverity.STRICT,
            contexts={MusicalContext.CLASSICAL},
            reason="Voice independence",
        )

    @pytest.fixture
    def sample_violation(self, sample_rule):
        return RuleViolation(
            rule=sample_rule,
            location="measure 4, beat 2",
            pitches=[60, 67, 62, 69],  # C4-G4 to D4-A4
            explanation="Parallel fifth motion from C-G to D-A",
        )

    def test_violation_creation(self, sample_violation):
        """Test creating a violation."""
        assert sample_violation.location == "measure 4, beat 2"
        assert len(sample_violation.pitches) == 4
        assert sample_violation.severity_override is None

    def test_effective_severity(self, sample_violation):
        """Test effective_severity property."""
        assert sample_violation.effective_severity == RuleSeverity.STRICT

    def test_effective_severity_with_override(self, sample_rule):
        """Test effective_severity with override."""
        violation = RuleViolation(
            rule=sample_rule,
            location="measure 1",
            pitches=[60],
            explanation="Test",
            severity_override=RuleSeverity.STYLISTIC,
        )
        assert violation.effective_severity == RuleSeverity.STYLISTIC

    def test_violation_str(self, sample_violation):
        """Test string representation of violation."""
        string = str(sample_violation)
        assert "no_parallel_fifths" in string
        assert "measure 4" in string
        assert "Parallel fifth" in string


class TestRuleBreakSuggestion:
    """Tests for RuleBreakSuggestion dataclass."""

    @pytest.fixture
    def sample_rule(self):
        return Rule(
            name="no_parallel_fifths",
            description="Avoid parallel fifths",
            severity=RuleSeverity.STRICT,
            contexts={MusicalContext.CLASSICAL},
            reason="Voice independence",
        )

    def test_suggestion_creation(self, sample_rule):
        """Test creating a suggestion."""
        suggestion = RuleBreakSuggestion(
            rule=sample_rule,
            context=MusicalContext.JAZZ,
            musical_example=[60, 67, 62, 69],
            explanation="Parallel fifths create power in rock/jazz",
            difficulty=4,
        )
        assert suggestion.rule == sample_rule
        assert suggestion.context == MusicalContext.JAZZ
        assert suggestion.difficulty == 4

    def test_suggestion_default_difficulty(self, sample_rule):
        """Test default difficulty value."""
        suggestion = RuleBreakSuggestion(
            rule=sample_rule,
            context=MusicalContext.JAZZ,
            musical_example=[60],
            explanation="Test",
        )
        assert suggestion.difficulty == 3

    def test_suggestion_str(self, sample_rule):
        """Test string representation of suggestion."""
        suggestion = RuleBreakSuggestion(
            rule=sample_rule,
            context=MusicalContext.JAZZ,
            musical_example=[60],
            explanation="Power and drive",
            difficulty=4,
        )
        string = str(suggestion)
        assert "no_parallel_fifths" in string
        assert "jazz" in string.lower()
        assert "4/5" in string


class TestEmotion:
    """Tests for Emotion enum."""

    def test_core_emotions(self):
        """Test core emotions are defined."""
        assert Emotion.JOY is not None
        assert Emotion.SADNESS is not None
        assert Emotion.ANGER is not None
        assert Emotion.FEAR is not None

    def test_musical_emotions(self):
        """Test musical emotions are defined."""
        assert Emotion.TENSION is not None
        assert Emotion.RESOLUTION is not None
        assert Emotion.NOSTALGIA is not None

    def test_complex_emotions(self):
        """Test complex emotional states."""
        assert Emotion.GRIEF is not None
        assert Emotion.TRIUMPH is not None
        assert Emotion.YEARNING is not None

    def test_emotion_str(self):
        """Test string representation of emotion."""
        assert str(Emotion.GRIEF) == "grief"
        assert str(Emotion.POWER) == "power"


class TestEmotionalMapping:
    """Tests for EmotionalMapping dataclass."""

    def test_mapping_creation(self):
        """Test creating an emotional mapping."""
        mapping = EmotionalMapping(
            rule_name="parallel_fifths",
            emotion=Emotion.POWER,
            intensity=8,
            context_dependencies={"rock", "metal"},
            explanation="Parallel fifths create powerful sound",
        )
        assert mapping.rule_name == "parallel_fifths"
        assert mapping.emotion == Emotion.POWER
        assert mapping.intensity == 8


class TestEmotionToTechniques:
    """Tests for emotion-to-technique mappings."""

    def test_grief_techniques(self):
        """Test techniques for grief emotion."""
        techniques = EMOTION_TO_TECHNIQUES.get(Emotion.GRIEF, [])
        assert len(techniques) > 0
        assert "non_resolution" in techniques

    def test_power_techniques(self):
        """Test techniques for power emotion."""
        techniques = EMOTION_TO_TECHNIQUES.get(Emotion.POWER, [])
        assert "parallel_fifths" in techniques
        assert "parallel_octaves" in techniques

    def test_tension_techniques(self):
        """Test techniques for tension emotion."""
        techniques = EMOTION_TO_TECHNIQUES.get(Emotion.TENSION, [])
        assert "unprepared_dissonance" in techniques

    def test_resolution_techniques(self):
        """Test techniques for resolution emotion."""
        techniques = EMOTION_TO_TECHNIQUES.get(Emotion.RESOLUTION, [])
        assert "authentic_cadence" in techniques

    def test_get_techniques_for_emotion(self):
        """Test get_techniques_for_emotion function."""
        techniques = get_techniques_for_emotion(Emotion.YEARNING)
        assert len(techniques) > 0
        assert "dominant_prolongation" in techniques

    def test_get_techniques_for_unknown_emotion(self):
        """Test getting techniques for less-covered emotions."""
        # Should return empty list for emotions without mappings
        techniques = get_techniques_for_emotion(Emotion.SURPRISE)
        assert isinstance(techniques, list)


class TestTechniqueToEmotions:
    """Tests for technique-to-emotion reverse mappings."""

    def test_reverse_mapping_exists(self):
        """Test that reverse mapping is populated."""
        assert len(TECHNIQUE_TO_EMOTIONS) > 0

    def test_parallel_fifths_emotions(self):
        """Test emotions for parallel fifths."""
        mappings = get_emotions_for_technique("parallel_fifths")
        assert len(mappings) > 0
        emotions = [m.emotion for m in mappings]
        assert Emotion.POWER in emotions

    def test_get_emotions_for_unknown_technique(self):
        """Test getting emotions for unknown technique."""
        mappings = get_emotions_for_technique("nonexistent_technique")
        assert mappings == []


class TestSwingType:
    """Tests for SwingType enum."""

    def test_swing_types_defined(self):
        """Test that all swing types are defined."""
        assert SwingType.STRAIGHT is not None
        assert SwingType.LIGHT_SWING is not None
        assert SwingType.MEDIUM_SWING is not None
        assert SwingType.HARD_SWING is not None
        assert SwingType.SHUFFLE is not None
        assert SwingType.DILLA_SWING is not None


class TestTimingPocket:
    """Tests for TimingPocket dataclass."""

    def test_timing_pocket_creation(self):
        """Test creating a timing pocket."""
        pocket = TimingPocket(
            swing_ratio=0.62,
            kick_offset_ms=20,
            snare_offset_ms=-12,
        )
        assert pocket.swing_ratio == 0.62
        assert pocket.kick_offset_ms == 20
        assert pocket.snare_offset_ms == -12

    def test_timing_pocket_defaults(self):
        """Test timing pocket default values."""
        pocket = TimingPocket(swing_ratio=0.5)
        assert pocket.kick_offset_ms == 0.0
        assert pocket.snare_offset_ms == 0.0
        assert pocket.humanization_variance == 5.0
        assert pocket.push_pull_tendency == 0.0


class TestGenrePockets:
    """Tests for genre-specific timing pockets."""

    def test_dilla_pocket(self):
        """Test J Dilla timing pocket."""
        pocket = GENRE_POCKETS.get("dilla")
        assert pocket is not None
        assert pocket.swing_ratio == 0.62
        assert pocket.kick_offset_ms == 20  # Laid-back

    def test_bebop_pocket(self):
        """Test bebop timing pocket."""
        pocket = GENRE_POCKETS.get("bebop")
        assert pocket is not None
        assert pocket.swing_ratio == 0.54  # Light swing
        assert pocket.push_pull_tendency < 0  # Tends to rush

    def test_techno_pocket(self):
        """Test techno timing pocket."""
        pocket = GENRE_POCKETS.get("techno")
        assert pocket is not None
        assert pocket.swing_ratio == 0.5  # Straight
        assert pocket.humanization_variance == 0.0  # No humanization

    def test_funk_pocket(self):
        """Test funk timing pocket."""
        pocket = GENRE_POCKETS.get("funk")
        assert pocket is not None
        assert pocket.snare_offset_ms < 0  # Snare on top

    def test_reggae_pocket(self):
        """Test reggae timing pocket."""
        pocket = GENRE_POCKETS.get("reggae")
        assert pocket is not None
        assert pocket.push_pull_tendency > 0  # Drags behind

    def test_shuffle_pocket(self):
        """Test shuffle timing pocket."""
        pocket = GENRE_POCKETS.get("shuffle")
        assert pocket is not None
        assert pocket.swing_ratio == 0.67  # Triplet feel

    def test_get_genre_pocket(self):
        """Test get_genre_pocket function."""
        pocket = get_genre_pocket("dilla")
        assert pocket is not None
        assert pocket.swing_ratio == 0.62

    def test_get_genre_pocket_case_insensitive(self):
        """Test that get_genre_pocket is case-insensitive."""
        pocket = get_genre_pocket("DILLA")
        assert pocket is not None

    def test_get_genre_pocket_unknown(self):
        """Test get_genre_pocket with unknown genre."""
        pocket = get_genre_pocket("unknown_genre")
        assert pocket is None


class TestApplyPocketToMidi:
    """Tests for apply_pocket_to_midi function."""

    @pytest.fixture
    def sample_midi(self):
        """Sample MIDI note data."""
        return [
            (36, 0, 100),      # Kick at time 0
            (36, 500, 100),   # Kick at 500ms
            (36, 1000, 100),  # Kick at 1000ms
        ]

    @pytest.fixture
    def sample_pocket(self):
        """Sample timing pocket."""
        return TimingPocket(
            swing_ratio=0.5,
            kick_offset_ms=10,
            snare_offset_ms=-5,
            humanization_variance=0,  # No randomness for testing
            push_pull_tendency=0,
        )

    def test_apply_pocket_kicks(self, sample_midi, sample_pocket):
        """Test applying pocket to kick drums."""
        result = apply_pocket_to_midi(sample_midi, sample_pocket, "kick")

        assert len(result) == 3
        # Kicks should be offset by 10ms (plus potential variance)
        for i, (pitch, time, vel) in enumerate(result):
            assert pitch == 36
            assert vel == 100

    def test_apply_pocket_snare(self, sample_pocket):
        """Test applying pocket to snare drums."""
        midi = [(38, 250, 100), (38, 750, 100)]
        result = apply_pocket_to_midi(midi, sample_pocket, "snare")

        assert len(result) == 2

    def test_apply_pocket_hihat(self, sample_pocket):
        """Test applying pocket to hi-hats."""
        midi = [(42, 0, 80), (42, 125, 80)]
        result = apply_pocket_to_midi(midi, sample_pocket, "hihat")

        assert len(result) == 2

    def test_apply_pocket_preserves_pitches(self, sample_midi, sample_pocket):
        """Test that apply_pocket preserves pitch values."""
        result = apply_pocket_to_midi(sample_midi, sample_pocket, "kick")

        for i, (pitch, _, _) in enumerate(result):
            assert pitch == sample_midi[i][0]

    def test_apply_pocket_unknown_instrument(self, sample_midi, sample_pocket):
        """Test applying pocket with unknown instrument type."""
        result = apply_pocket_to_midi(sample_midi, sample_pocket, "unknown")

        # Should still work with 0 offset
        assert len(result) == 3


class TestTimingPocketEdgeCases:
    """Edge case tests for timing functionality."""

    def test_extreme_swing_ratio(self):
        """Test pocket with extreme swing ratio."""
        pocket = TimingPocket(swing_ratio=0.9)  # Very heavy swing
        assert pocket.swing_ratio == 0.9

    def test_negative_offsets(self):
        """Test pocket with negative offsets (pushing ahead)."""
        pocket = TimingPocket(
            swing_ratio=0.5,
            kick_offset_ms=-20,
            snare_offset_ms=-15,
        )
        assert pocket.kick_offset_ms == -20

    def test_high_humanization(self):
        """Test pocket with high humanization variance."""
        pocket = TimingPocket(
            swing_ratio=0.5,
            humanization_variance=50,  # High variance
        )
        midi = [(36, 1000, 100)]

        # With high variance, results should vary
        results = []
        for _ in range(10):
            result = apply_pocket_to_midi(midi, pocket, "kick")
            results.append(result[0][1])

        # Should have some variation in times
        assert len(set(results)) > 1

    def test_empty_midi_input(self):
        """Test applying pocket to empty MIDI list."""
        pocket = TimingPocket(swing_ratio=0.5)
        result = apply_pocket_to_midi([], pocket, "kick")
        assert result == []


class TestIntegration:
    """Integration tests across rules modules."""

    def test_rule_with_context_and_severity(self):
        """Test creating rules with contexts and checking severity."""
        rule = Rule(
            name="parallel_fifths",
            description="Parallel perfect fifths",
            severity={
                MusicalContext.BAROQUE: RuleSeverity.STRICT,
                MusicalContext.CLASSICAL: RuleSeverity.STRICT,
                MusicalContext.JAZZ: RuleSeverity.STYLISTIC,
                MusicalContext.CONTEMPORARY: RuleSeverity.MODERN,
            },
            contexts=CONTEXT_GROUPS["all"],
            reason="Voice independence vs power",
        )

        # Should be strict in common practice
        assert rule.get_severity_for_context(MusicalContext.BAROQUE) == RuleSeverity.STRICT
        assert rule.get_severity_for_context(MusicalContext.CLASSICAL) == RuleSeverity.STRICT

        # More flexible in modern contexts
        assert rule.get_severity_for_context(MusicalContext.JAZZ) == RuleSeverity.STYLISTIC
        assert rule.get_severity_for_context(MusicalContext.CONTEMPORARY) == RuleSeverity.MODERN

    def test_emotion_technique_roundtrip(self):
        """Test that emotion->technique->emotion is consistent."""
        for emotion, techniques in EMOTION_TO_TECHNIQUES.items():
            for technique in techniques:
                # Each technique should map back to include original emotion
                mappings = get_emotions_for_technique(technique)
                mapped_emotions = [m.emotion for m in mappings]
                assert emotion in mapped_emotions, f"{technique} should map to {emotion}"

    def test_pocket_application_workflow(self):
        """Test complete timing pocket application workflow."""
        # Get a genre pocket
        pocket = get_genre_pocket("funk")
        assert pocket is not None

        # Create MIDI data
        kick = [(36, 0, 100), (36, 500, 100)]
        snare = [(38, 250, 100), (38, 750, 100)]
        hihat = [(42, 0, 80), (42, 125, 80), (42, 250, 80)]

        # Apply pocket to each instrument
        kick_timed = apply_pocket_to_midi(kick, pocket, "kick")
        snare_timed = apply_pocket_to_midi(snare, pocket, "snare")
        hihat_timed = apply_pocket_to_midi(hihat, pocket, "hihat")

        # All should be processed
        assert len(kick_timed) == 2
        assert len(snare_timed) == 2
        assert len(hihat_timed) == 3
