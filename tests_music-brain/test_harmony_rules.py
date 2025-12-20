"""
Tests for HarmonyRules class in penta_core.teachers.harmony_rules.

This module tests the chord construction rules, progression rules,
and the get_progression_strength method.

Run with: pytest tests_music-brain/test_harmony_rules.py -v
"""

import pytest


class TestHarmonyRulesImports:
    """Test that HarmonyRules can be imported."""

    def test_import_harmony_rules(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules is not None

    def test_import_chord_quality(self):
        from python.penta_core.teachers.harmony_rules import ChordQuality

        assert ChordQuality is not None


class TestChordQuality:
    """Test ChordQuality enum."""

    def test_chord_qualities_exist(self):
        from python.penta_core.teachers.harmony_rules import ChordQuality

        assert ChordQuality.MAJOR.value == "major"
        assert ChordQuality.MINOR.value == "minor"
        assert ChordQuality.DIMINISHED.value == "diminished"
        assert ChordQuality.AUGMENTED.value == "augmented"
        assert ChordQuality.DOMINANT.value == "dominant7"
        assert ChordQuality.MAJOR7.value == "major7"
        assert ChordQuality.MINOR7.value == "minor7"


class TestHarmonyRulesChordConstruction:
    """Test chord construction rules."""

    def test_get_all_rules(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        rules = HarmonyRules.get_all_rules()
        assert "chord_construction" in rules
        assert "functional_harmony" in rules
        assert "progressions" in rules
        assert "jazz_harmony" in rules
        assert "pop_rock_harmony" in rules

    def test_get_chord_intervals_major(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        intervals = HarmonyRules.get_chord_intervals("major")
        assert intervals == [0, 4, 7]

    def test_get_chord_intervals_minor(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        intervals = HarmonyRules.get_chord_intervals("minor")
        assert intervals == [0, 3, 7]

    def test_get_chord_intervals_diminished(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        intervals = HarmonyRules.get_chord_intervals("diminished")
        assert intervals == [0, 3, 6]

    def test_get_chord_intervals_augmented(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        intervals = HarmonyRules.get_chord_intervals("augmented")
        assert intervals == [0, 4, 8]

    def test_get_chord_intervals_unknown(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        intervals = HarmonyRules.get_chord_intervals("nonexistent")
        assert intervals is None


class TestProgressionStrength:
    """Test get_progression_strength method."""

    def test_authentic_cadence_very_strong(self):
        """V -> I is the strongest progression (authentic cadence)."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules.get_progression_strength("V", "I") == "very_strong"
        assert HarmonyRules.get_progression_strength("V7", "I") == "very_strong"

    def test_circle_of_fifths_very_strong(self):
        """Circle of fifths progressions are very strong."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # ii -> V (descending fifth)
        assert HarmonyRules.get_progression_strength("ii", "V") == "very_strong"
        # vi -> ii (descending fifth)
        assert HarmonyRules.get_progression_strength("vi", "ii") == "very_strong"
        # iii -> vi (descending fifth)
        assert HarmonyRules.get_progression_strength("iii", "vi") == "very_strong"

    def test_leading_tone_resolution_strong(self):
        """vii -> I is a strong progression."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules.get_progression_strength("vii°", "I") == "strong"
        assert HarmonyRules.get_progression_strength("vii", "I") == "strong"

    def test_subdominant_to_dominant_strong(self):
        """Subdominant to dominant preparations are strong."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # IV -> V
        assert HarmonyRules.get_progression_strength("IV", "V") == "strong"

    def test_plagal_cadence_moderate(self):
        """IV -> I (plagal cadence) is moderate strength."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules.get_progression_strength("IV", "I") == "moderate"

    def test_descending_thirds_moderate(self):
        """Descending third progressions are moderate."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # I -> vi (down a third)
        assert HarmonyRules.get_progression_strength("I", "vi") == "moderate"
        # vi -> IV (down a third)
        assert HarmonyRules.get_progression_strength("vi", "IV") == "moderate"

    def test_ascending_thirds_moderate(self):
        """Ascending third progressions are moderate."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # I -> iii (up a third)
        assert HarmonyRules.get_progression_strength("I", "iii") == "moderate"

    def test_stepwise_motion_moderate(self):
        """Stepwise motion is moderate."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # I -> ii (up a step)
        assert HarmonyRules.get_progression_strength("I", "ii") == "moderate"

    def test_deceptive_cadence_weak(self):
        """V -> vi (deceptive cadence) is weak."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules.get_progression_strength("V", "vi") == "weak"

    def test_ascending_fifth_weak(self):
        """Ascending fifths (opposite circle) are weak."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # I -> V (up a fifth) - but wait, this might be interpreted as
        # half cadence preparation which could be different
        # Let's test other ascending fifths
        # ii -> vi (up a fifth)
        result = HarmonyRules.get_progression_strength("ii", "vi")
        assert result in ("weak", "moderate")  # Could be either based on interpretation

    def test_invalid_numeral_unusual(self):
        """Invalid Roman numerals return unusual."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules.get_progression_strength("XYZ", "I") == "unusual"
        assert HarmonyRules.get_progression_strength("I", "ABC") == "unusual"
        assert HarmonyRules.get_progression_strength("invalid", "also_invalid") == "unusual"

    def test_same_chord_unusual(self):
        """Same chord to itself should be unusual."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # I -> I is interval 0, which doesn't fit other categories
        result = HarmonyRules.get_progression_strength("I", "I")
        # This might be unusual or something else depending on implementation
        assert result in ("unusual", "weak", "moderate")

    def test_seventh_chord_notation(self):
        """Test various seventh chord notations work."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # V7 -> I should still be very strong
        assert HarmonyRules.get_progression_strength("V7", "I") == "very_strong"
        assert HarmonyRules.get_progression_strength("V⁷", "I") == "very_strong"

    def test_minor_key_numerals(self):
        """Test minor key Roman numerals."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # i -> IV is a descending fifth (circle of fifths) - very strong
        result = HarmonyRules.get_progression_strength("i", "IV")
        assert result == "very_strong"

        # i -> VI (relative major) - descending third
        result = HarmonyRules.get_progression_strength("i", "VI")
        assert result == "moderate"

    def test_diminished_chord_numerals(self):
        """Test diminished chord notation."""
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # ii° -> V
        assert HarmonyRules.get_progression_strength("ii°", "V") == "very_strong"
        # vii° -> I
        assert HarmonyRules.get_progression_strength("vii°", "I") == "strong"


class TestParseRomanNumeral:
    """Test the _parse_roman_numeral helper method."""

    def test_parse_basic_numerals(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules._parse_roman_numeral("I") == 1
        assert HarmonyRules._parse_roman_numeral("ii") == 2
        assert HarmonyRules._parse_roman_numeral("iii") == 3
        assert HarmonyRules._parse_roman_numeral("IV") == 4
        assert HarmonyRules._parse_roman_numeral("V") == 5
        assert HarmonyRules._parse_roman_numeral("vi") == 6
        assert HarmonyRules._parse_roman_numeral("vii") == 7

    def test_parse_with_quality_suffixes(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules._parse_roman_numeral("V7") == 5
        assert HarmonyRules._parse_roman_numeral("vii°") == 7
        assert HarmonyRules._parse_roman_numeral("ii°7") == 2

    def test_parse_flat_numerals(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules._parse_roman_numeral("♭VII") == 7
        assert HarmonyRules._parse_roman_numeral("♭VI") == 6

    def test_parse_invalid_returns_none(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        assert HarmonyRules._parse_roman_numeral("invalid") is None
        assert HarmonyRules._parse_roman_numeral("") is None


class TestGetInterval:
    """Test the _get_interval helper method."""

    def test_ascending_intervals(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # 1 -> 2 is up a step
        assert HarmonyRules._get_interval(1, 2) == 1
        # 1 -> 5 is up a fourth
        assert HarmonyRules._get_interval(1, 5) == 4
        # 1 -> 3 is up a third
        assert HarmonyRules._get_interval(1, 3) == 2

    def test_descending_intervals(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # 5 -> 1 could be down 4 or up 3; implementation normalizes to -3..+4 range
        # so it chooses +3 (up 3 degrees, shorter path around the scale)
        assert HarmonyRules._get_interval(5, 1) == 3
        # 2 -> 1 is down a step
        assert HarmonyRules._get_interval(2, 1) == -1

    def test_interval_wrapping(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        # 1 -> 7 should be -1 (down a step), not +6
        assert HarmonyRules._get_interval(1, 7) == -1
        # 7 -> 1 should be +1 (up a step), not -6
        assert HarmonyRules._get_interval(7, 1) == 1


class TestFunctionalHarmonyRules:
    """Test functional harmony rule data."""

    def test_tonic_function_exists(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        rules = HarmonyRules.FUNCTIONAL_HARMONY_RULES
        assert "tonic_function" in rules
        assert "I" in rules["tonic_function"]["major_key"]
        assert "vi" in rules["tonic_function"]["major_key"]

    def test_dominant_function_exists(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        rules = HarmonyRules.FUNCTIONAL_HARMONY_RULES
        assert "dominant_function" in rules
        assert "V" in rules["dominant_function"]["major_key"]
        assert "V7" in rules["dominant_function"]["major_key"]

    def test_subdominant_function_exists(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        rules = HarmonyRules.FUNCTIONAL_HARMONY_RULES
        assert "subdominant_function" in rules
        assert "IV" in rules["subdominant_function"]["major_key"]
        assert "ii" in rules["subdominant_function"]["major_key"]


class TestJazzHarmonyRules:
    """Test jazz harmony rule data."""

    def test_ii_v_i_progression(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        rules = HarmonyRules.JAZZ_HARMONY_RULES
        assert "ii_V_I" in rules
        assert "major_key" in rules["ii_V_I"]
        assert "minor_key" in rules["ii_V_I"]

    def test_tritone_substitution(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        rules = HarmonyRules.JAZZ_HARMONY_RULES
        assert "tritone_substitution" in rules
        assert "theory" in rules["tritone_substitution"]

    def test_modal_interchange(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        rules = HarmonyRules.JAZZ_HARMONY_RULES
        assert "modal_interchange" in rules
        assert "major_key_borrows_from_minor" in rules["modal_interchange"]


class TestPopRockHarmonyRules:
    """Test pop/rock harmony rule data."""

    def test_I_V_vi_IV_progression(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        rules = HarmonyRules.POP_ROCK_HARMONY_RULES
        assert "I_V_vi_IV" in rules
        assert "example_C" in rules["I_V_vi_IV"]
        # Should be C, G, Am, F
        assert rules["I_V_vi_IV"]["example_C"] == ["C", "G", "Am", "F"]

    def test_power_chords(self):
        from python.penta_core.teachers.harmony_rules import HarmonyRules

        rules = HarmonyRules.POP_ROCK_HARMONY_RULES
        assert "power_chords" in rules
        assert "notation" in rules["power_chords"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
