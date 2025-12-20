"""
Tests for music_brain.arrangement.templates module.

Tests cover:
- SectionType enum
- SectionTemplate dataclass
- ArrangementTemplate dataclass
- Section creation functions (create_intro, create_verse, etc.)
- Genre-specific templates (pop, rock, EDM, lofi, indie)
- Template lookup functionality
"""

import pytest


class TestSectionTypeEnum:
    """Tests for SectionType enum."""

    def test_all_section_types_defined(self):
        """Verify all expected section types exist."""
        from music_brain.arrangement.templates import SectionType

        expected_types = [
            "INTRO",
            "VERSE",
            "PRECHORUS",
            "CHORUS",
            "BRIDGE",
            "BREAKDOWN",
            "BUILDUP",
            "DROP",
            "SOLO",
            "OUTRO",
        ]

        for section in expected_types:
            assert hasattr(SectionType, section), f"Missing section: {section}"

    def test_section_type_values(self):
        """Test section type enum values."""
        from music_brain.arrangement.templates import SectionType

        assert SectionType.INTRO.value == "intro"
        assert SectionType.VERSE.value == "verse"
        assert SectionType.CHORUS.value == "chorus"
        assert SectionType.BRIDGE.value == "bridge"
        assert SectionType.DROP.value == "drop"


class TestSectionTemplate:
    """Tests for SectionTemplate dataclass."""

    def test_section_template_creation(self):
        """Test creating a SectionTemplate instance."""
        from music_brain.arrangement.templates import SectionTemplate, SectionType

        template = SectionTemplate(
            section_type=SectionType.VERSE,
            length_bars=8,
            energy_level=0.5,
        )

        assert template.section_type == SectionType.VERSE
        assert template.length_bars == 8
        assert template.energy_level == 0.5

    def test_section_template_defaults(self):
        """Test default values are set correctly."""
        from music_brain.arrangement.templates import SectionTemplate, SectionType

        template = SectionTemplate(section_type=SectionType.CHORUS)

        assert template.length_bars == 8
        assert template.energy_level == 0.5
        assert template.dynamic_range == 0.3
        assert template.instruments == []
        assert template.vocal_type is None
        assert template.note_density == 0.5
        assert template.rhythmic_complexity == 0.5
        assert template.harmonic_movement == "static"
        assert template.mix_focus == "balanced"
        assert template.reverb_amount == 0.3

    def test_section_template_to_dict(self):
        """Test serialization to dictionary."""
        from music_brain.arrangement.templates import SectionTemplate, SectionType

        template = SectionTemplate(
            section_type=SectionType.BRIDGE,
            length_bars=4,
            energy_level=0.6,
            instruments=["drums", "bass"],
            vocal_type="harmony",
        )

        result = template.to_dict()

        assert result["section_type"] == "bridge"
        assert result["length_bars"] == 4
        assert result["energy_level"] == 0.6
        assert result["instruments"] == ["drums", "bass"]
        assert result["vocal_type"] == "harmony"


class TestArrangementTemplate:
    """Tests for ArrangementTemplate dataclass."""

    def test_arrangement_template_creation(self):
        """Test creating an ArrangementTemplate instance."""
        from music_brain.arrangement.templates import (
            ArrangementTemplate,
            SectionTemplate,
            SectionType,
        )

        sections = [
            SectionTemplate(SectionType.INTRO, length_bars=4),
            SectionTemplate(SectionType.VERSE, length_bars=8),
        ]

        template = ArrangementTemplate(
            name="Test Template",
            genre="pop",
            sections=sections,
            tempo_bpm=120.0,
        )

        assert template.name == "Test Template"
        assert template.genre == "pop"
        assert len(template.sections) == 2
        assert template.tempo_bpm == 120.0

    def test_total_bars_calculated(self):
        """Test total bars is auto-calculated from sections."""
        from music_brain.arrangement.templates import (
            ArrangementTemplate,
            SectionTemplate,
            SectionType,
        )

        sections = [
            SectionTemplate(SectionType.INTRO, length_bars=4),
            SectionTemplate(SectionType.VERSE, length_bars=8),
            SectionTemplate(SectionType.CHORUS, length_bars=8),
        ]

        template = ArrangementTemplate(
            name="Test",
            genre="pop",
            sections=sections,
        )

        assert template.total_bars == 20  # 4 + 8 + 8

    def test_explicit_total_bars(self):
        """Test explicit total_bars is preserved."""
        from music_brain.arrangement.templates import (
            ArrangementTemplate,
            SectionTemplate,
            SectionType,
        )

        sections = [SectionTemplate(SectionType.INTRO, length_bars=4)]

        template = ArrangementTemplate(
            name="Test",
            genre="pop",
            sections=sections,
            total_bars=100,  # Explicit override
        )

        assert template.total_bars == 100

    def test_arrangement_template_to_dict(self):
        """Test serialization to dictionary."""
        from music_brain.arrangement.templates import (
            ArrangementTemplate,
            SectionTemplate,
            SectionType,
        )

        sections = [SectionTemplate(SectionType.INTRO, length_bars=4)]

        template = ArrangementTemplate(
            name="Test",
            genre="rock",
            sections=sections,
            tempo_bpm=140.0,
            time_signature=(4, 4),
        )

        result = template.to_dict()

        assert result["name"] == "Test"
        assert result["genre"] == "rock"
        assert len(result["sections"]) == 1
        assert result["tempo_bpm"] == 140.0
        assert result["time_signature"] == (4, 4)


class TestCreateIntro:
    """Tests for create_intro function."""

    def test_create_intro_default(self):
        """Test default intro creation."""
        from music_brain.arrangement.templates import create_intro, SectionType

        intro = create_intro()

        assert intro.section_type == SectionType.INTRO
        assert intro.length_bars == 4
        assert intro.energy_level == 0.3
        assert intro.vocal_type is None

    def test_create_intro_custom_length(self):
        """Test custom length intro."""
        from music_brain.arrangement.templates import create_intro

        intro = create_intro(length_bars=8)

        assert intro.length_bars == 8

    def test_intro_has_ambient_instruments(self):
        """Intro should have atmospheric instruments."""
        from music_brain.arrangement.templates import create_intro

        intro = create_intro()

        assert "pad" in intro.instruments
        assert intro.reverb_amount > 0.3


class TestCreateVerse:
    """Tests for create_verse function."""

    def test_create_verse_default(self):
        """Test default verse creation."""
        from music_brain.arrangement.templates import create_verse, SectionType

        verse = create_verse()

        assert verse.section_type == SectionType.VERSE
        assert verse.length_bars == 8
        assert verse.vocal_type == "lead"

    def test_create_verse_without_vocals(self):
        """Test verse without vocals."""
        from music_brain.arrangement.templates import create_verse

        verse = create_verse(with_vocals=False)

        assert verse.vocal_type is None

    def test_create_verse_custom_energy(self):
        """Test verse with custom energy level."""
        from music_brain.arrangement.templates import create_verse

        verse = create_verse(energy_level=0.7)

        assert verse.energy_level == 0.7

    def test_verse_has_core_instruments(self):
        """Verse should have core band instruments."""
        from music_brain.arrangement.templates import create_verse

        verse = create_verse()

        assert "drums" in verse.instruments
        assert "bass" in verse.instruments


class TestCreatePrechorus:
    """Tests for create_prechorus function."""

    def test_create_prechorus(self):
        """Test pre-chorus creation."""
        from music_brain.arrangement.templates import create_prechorus, SectionType

        prechorus = create_prechorus()

        assert prechorus.section_type == SectionType.PRECHORUS
        assert prechorus.length_bars == 4

    def test_prechorus_builds_energy(self):
        """Pre-chorus should have building energy."""
        from music_brain.arrangement.templates import create_prechorus, create_verse

        verse = create_verse()
        prechorus = create_prechorus()

        assert prechorus.energy_level > verse.energy_level


class TestCreateChorus:
    """Tests for create_chorus function."""

    def test_create_chorus_default(self):
        """Test default chorus creation."""
        from music_brain.arrangement.templates import create_chorus, SectionType

        chorus = create_chorus()

        assert chorus.section_type == SectionType.CHORUS
        assert chorus.length_bars == 8
        assert chorus.energy_level == 0.8

    def test_chorus_has_doubled_vocals(self):
        """Chorus typically has doubled vocals."""
        from music_brain.arrangement.templates import create_chorus

        chorus = create_chorus()

        assert chorus.vocal_type == "double"

    def test_chorus_custom_energy(self):
        """Test custom energy chorus."""
        from music_brain.arrangement.templates import create_chorus

        chorus = create_chorus(energy_level=0.95)

        assert chorus.energy_level == 0.95


class TestCreateBridge:
    """Tests for create_bridge function."""

    def test_create_bridge(self):
        """Test bridge creation."""
        from music_brain.arrangement.templates import create_bridge, SectionType

        bridge = create_bridge()

        assert bridge.section_type == SectionType.BRIDGE
        assert bridge.harmonic_movement == "modulating"
        assert bridge.vocal_type == "harmony"

    def test_bridge_has_contrast(self):
        """Bridge should provide musical contrast."""
        from music_brain.arrangement.templates import create_bridge

        bridge = create_bridge()

        # Bridge typically has higher dynamic range and reverb
        assert bridge.dynamic_range >= 0.5
        assert bridge.reverb_amount >= 0.5


class TestCreateBreakdown:
    """Tests for create_breakdown function."""

    def test_create_breakdown(self):
        """Test breakdown creation."""
        from music_brain.arrangement.templates import create_breakdown, SectionType

        breakdown = create_breakdown()

        assert breakdown.section_type == SectionType.BREAKDOWN
        assert breakdown.energy_level <= 0.4
        assert breakdown.note_density <= 0.3

    def test_breakdown_sparse_instrumentation(self):
        """Breakdown should have minimal instrumentation."""
        from music_brain.arrangement.templates import create_breakdown

        breakdown = create_breakdown()

        assert len(breakdown.instruments) <= 3


class TestCreateBuildup:
    """Tests for create_buildup function."""

    def test_create_buildup(self):
        """Test buildup creation."""
        from music_brain.arrangement.templates import create_buildup, SectionType

        buildup = create_buildup()

        assert buildup.section_type == SectionType.BUILDUP
        assert buildup.energy_level >= 0.6
        assert buildup.dynamic_range >= 0.7

    def test_buildup_high_complexity(self):
        """Buildup should have high rhythmic complexity."""
        from music_brain.arrangement.templates import create_buildup

        buildup = create_buildup()

        assert buildup.rhythmic_complexity >= 0.7
        assert buildup.note_density >= 0.7


class TestCreateDrop:
    """Tests for create_drop function."""

    def test_create_drop(self):
        """Test drop creation."""
        from music_brain.arrangement.templates import create_drop, SectionType

        drop = create_drop()

        assert drop.section_type == SectionType.DROP
        assert drop.energy_level == 1.0  # Maximum energy

    def test_drop_high_density(self):
        """Drop should have high note density."""
        from music_brain.arrangement.templates import create_drop

        drop = create_drop()

        assert drop.note_density >= 0.8


class TestCreateOutro:
    """Tests for create_outro function."""

    def test_create_outro(self):
        """Test outro creation."""
        from music_brain.arrangement.templates import create_outro, SectionType

        outro = create_outro()

        assert outro.section_type == SectionType.OUTRO
        assert outro.energy_level <= 0.3
        assert outro.vocal_type is None

    def test_outro_atmospheric(self):
        """Outro should be atmospheric."""
        from music_brain.arrangement.templates import create_outro

        outro = create_outro()

        assert outro.reverb_amount >= 0.7
        assert "ambient" in outro.instruments or "pad" in outro.instruments


class TestPopStructure:
    """Tests for get_pop_structure function."""

    def test_pop_structure_exists(self):
        """Pop template should be retrievable."""
        from music_brain.arrangement.templates import get_pop_structure

        template = get_pop_structure()

        assert template.name == "Pop Standard"
        assert template.genre == "pop"

    def test_pop_structure_sections(self):
        """Pop structure should have typical sections."""
        from music_brain.arrangement.templates import get_pop_structure, SectionType

        template = get_pop_structure()

        section_types = [s.section_type for s in template.sections]

        assert SectionType.INTRO in section_types
        assert SectionType.VERSE in section_types
        assert SectionType.CHORUS in section_types
        assert SectionType.OUTRO in section_types

    def test_pop_structure_tempo(self):
        """Pop should have typical pop tempo."""
        from music_brain.arrangement.templates import get_pop_structure

        template = get_pop_structure()

        assert 110 <= template.tempo_bpm <= 130


class TestRockStructure:
    """Tests for get_rock_structure function."""

    def test_rock_structure_exists(self):
        """Rock template should be retrievable."""
        from music_brain.arrangement.templates import get_rock_structure

        template = get_rock_structure()

        assert template.name == "Rock Standard"
        assert template.genre == "rock"

    def test_rock_has_solo(self):
        """Rock should include a solo section."""
        from music_brain.arrangement.templates import get_rock_structure, SectionType

        template = get_rock_structure()

        section_types = [s.section_type for s in template.sections]

        assert SectionType.SOLO in section_types

    def test_rock_higher_tempo(self):
        """Rock typically has higher tempo."""
        from music_brain.arrangement.templates import get_rock_structure

        template = get_rock_structure()

        assert template.tempo_bpm >= 130


class TestEDMStructure:
    """Tests for get_edm_structure function."""

    def test_edm_structure_exists(self):
        """EDM template should be retrievable."""
        from music_brain.arrangement.templates import get_edm_structure

        template = get_edm_structure()

        assert template.name == "EDM Standard"
        assert template.genre == "edm"

    def test_edm_has_drop_buildup(self):
        """EDM should have drop and buildup sections."""
        from music_brain.arrangement.templates import get_edm_structure, SectionType

        template = get_edm_structure()

        section_types = [s.section_type for s in template.sections]

        assert SectionType.DROP in section_types
        assert SectionType.BUILDUP in section_types
        assert SectionType.BREAKDOWN in section_types

    def test_edm_128_bpm(self):
        """EDM should be around 128 BPM."""
        from music_brain.arrangement.templates import get_edm_structure

        template = get_edm_structure()

        assert 125 <= template.tempo_bpm <= 130


class TestLofiStructure:
    """Tests for get_lofi_structure function."""

    def test_lofi_structure_exists(self):
        """Lo-fi template should be retrievable."""
        from music_brain.arrangement.templates import get_lofi_structure

        template = get_lofi_structure()

        assert template.name == "Lo-Fi Standard"
        assert template.genre == "lofi"

    def test_lofi_low_energy(self):
        """Lo-fi should have generally low energy."""
        from music_brain.arrangement.templates import get_lofi_structure

        template = get_lofi_structure()

        avg_energy = sum(s.energy_level for s in template.sections) / len(template.sections)

        assert avg_energy < 0.5

    def test_lofi_slow_tempo(self):
        """Lo-fi should have slow tempo."""
        from music_brain.arrangement.templates import get_lofi_structure

        template = get_lofi_structure()

        assert template.tempo_bpm <= 85


class TestIndieStructure:
    """Tests for get_indie_structure function."""

    def test_indie_structure_exists(self):
        """Indie template should be retrievable."""
        from music_brain.arrangement.templates import get_indie_structure

        template = get_indie_structure()

        assert template.name == "Indie Standard"
        assert template.genre == "indie"


class TestGetGenreTemplate:
    """Tests for get_genre_template function."""

    def test_get_pop_template(self):
        """Get pop template by name."""
        from music_brain.arrangement.templates import get_genre_template

        template = get_genre_template("pop")

        assert template.genre == "pop"

    def test_get_rock_template(self):
        """Get rock template by name."""
        from music_brain.arrangement.templates import get_genre_template

        template = get_genre_template("rock")

        assert template.genre == "rock"

    def test_case_insensitivity(self):
        """Genre lookup should be case-insensitive."""
        from music_brain.arrangement.templates import get_genre_template

        template1 = get_genre_template("EDM")
        template2 = get_genre_template("edm")

        assert template1.genre == template2.genre

    def test_alternative_names(self):
        """Alternative genre names should work."""
        from music_brain.arrangement.templates import get_genre_template

        # "electronic" should map to EDM
        template = get_genre_template("electronic")
        assert template.genre == "edm"

        # "lo-fi" should work
        template = get_genre_template("lo-fi")
        assert template.genre == "lofi"

        # "alternative" should map to indie
        template = get_genre_template("alternative")
        assert template.genre == "indie"

    def test_unknown_genre_raises(self):
        """Unknown genres should raise ValueError."""
        from music_brain.arrangement.templates import get_genre_template

        with pytest.raises(ValueError, match="not found"):
            get_genre_template("unknown_genre")


class TestListAvailableGenres:
    """Tests for list_available_genres function."""

    def test_returns_list(self):
        """Should return a list of strings."""
        from music_brain.arrangement.templates import list_available_genres

        genres = list_available_genres()

        assert isinstance(genres, list)
        assert all(isinstance(g, str) for g in genres)

    def test_includes_main_genres(self):
        """Should include main genre templates."""
        from music_brain.arrangement.templates import list_available_genres

        genres = list_available_genres()

        assert "pop" in genres
        assert "rock" in genres
        assert "edm" in genres
        assert "lofi" in genres
        assert "indie" in genres
