"""
Section Templates - Define standard song sections and genre-specific structures.

Provides templates for:
- Verse, Chorus, Bridge, Pre-Chorus, etc.
- Genre-specific arrangements
- Section characteristics (energy, density, instrumentation)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class SectionType(Enum):
    """Standard song section types."""
    INTRO = "intro"
    VERSE = "verse"
    PRECHORUS = "pre-chorus"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    BREAKDOWN = "breakdown"
    BUILDUP = "buildup"
    DROP = "drop"
    SOLO = "solo"
    OUTRO = "outro"


@dataclass
class SectionTemplate:
    """Template defining characteristics of a song section."""
    section_type: SectionType
    length_bars: int = 8
    
    # Energy and intensity
    energy_level: float = 0.5  # 0.0 (calm) to 1.0 (intense)
    dynamic_range: float = 0.3  # How much dynamics vary
    
    # Instrumentation
    instruments: List[str] = field(default_factory=list)
    vocal_type: Optional[str] = None  # "lead", "harmony", "double", None
    
    # Musical characteristics
    note_density: float = 0.5  # Notes per beat
    rhythmic_complexity: float = 0.5  # Simple to complex
    harmonic_movement: str = "static"  # "static", "moving", "modulating"
    
    # Production
    mix_focus: str = "balanced"  # "low", "mid", "high", "balanced"
    reverb_amount: float = 0.3  # Wet/dry mix
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "section_type": self.section_type.value,
            "length_bars": self.length_bars,
            "energy_level": self.energy_level,
            "dynamic_range": self.dynamic_range,
            "instruments": self.instruments,
            "vocal_type": self.vocal_type,
            "note_density": self.note_density,
            "rhythmic_complexity": self.rhythmic_complexity,
            "harmonic_movement": self.harmonic_movement,
            "mix_focus": self.mix_focus,
            "reverb_amount": self.reverb_amount,
        }


@dataclass
class ArrangementTemplate:
    """Complete arrangement template with section sequence."""
    name: str
    genre: str
    sections: List[SectionTemplate]
    total_bars: int = 0
    tempo_bpm: float = 120.0
    time_signature: tuple = (4, 4)
    
    def __post_init__(self):
        """Calculate total bars if not specified."""
        if self.total_bars == 0:
            self.total_bars = sum(s.length_bars for s in self.sections)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "genre": self.genre,
            "sections": [s.to_dict() for s in self.sections],
            "total_bars": self.total_bars,
            "tempo_bpm": self.tempo_bpm,
            "time_signature": self.time_signature,
        }


# =================================================================
# STANDARD SECTION DEFINITIONS
# =================================================================

def create_intro(length_bars: int = 4) -> SectionTemplate:
    """Create standard intro section."""
    return SectionTemplate(
        section_type=SectionType.INTRO,
        length_bars=length_bars,
        energy_level=0.3,
        dynamic_range=0.2,
        instruments=["drums", "bass", "pad"],
        vocal_type=None,
        note_density=0.3,
        rhythmic_complexity=0.3,
        harmonic_movement="static",
        mix_focus="balanced",
        reverb_amount=0.5,
    )


def create_verse(
    length_bars: int = 8,
    energy_level: float = 0.4,
    with_vocals: bool = True,
) -> SectionTemplate:
    """Create standard verse section."""
    return SectionTemplate(
        section_type=SectionType.VERSE,
        length_bars=length_bars,
        energy_level=energy_level,
        dynamic_range=0.3,
        instruments=["drums", "bass", "guitar", "keys"],
        vocal_type="lead" if with_vocals else None,
        note_density=0.5,
        rhythmic_complexity=0.4,
        harmonic_movement="moving",
        mix_focus="mid",
        reverb_amount=0.3,
    )


def create_prechorus(length_bars: int = 4) -> SectionTemplate:
    """Create pre-chorus section (builds to chorus)."""
    return SectionTemplate(
        section_type=SectionType.PRECHORUS,
        length_bars=length_bars,
        energy_level=0.6,
        dynamic_range=0.4,
        instruments=["drums", "bass", "guitar", "keys"],
        vocal_type="lead",
        note_density=0.6,
        rhythmic_complexity=0.5,
        harmonic_movement="moving",
        mix_focus="mid",
        reverb_amount=0.2,
    )


def create_chorus(
    length_bars: int = 8,
    energy_level: float = 0.8,
) -> SectionTemplate:
    """Create standard chorus section."""
    return SectionTemplate(
        section_type=SectionType.CHORUS,
        length_bars=length_bars,
        energy_level=energy_level,
        dynamic_range=0.5,
        instruments=["drums", "bass", "guitar", "keys", "synth"],
        vocal_type="double",
        note_density=0.7,
        rhythmic_complexity=0.5,
        harmonic_movement="static",
        mix_focus="balanced",
        reverb_amount=0.4,
    )


def create_bridge(length_bars: int = 8) -> SectionTemplate:
    """Create bridge section (contrast/development)."""
    return SectionTemplate(
        section_type=SectionType.BRIDGE,
        length_bars=length_bars,
        energy_level=0.6,
        dynamic_range=0.6,
        instruments=["drums", "bass", "keys"],
        vocal_type="harmony",
        note_density=0.5,
        rhythmic_complexity=0.6,
        harmonic_movement="modulating",
        mix_focus="high",
        reverb_amount=0.6,
    )


def create_breakdown(length_bars: int = 4) -> SectionTemplate:
    """Create breakdown section (sparse, dramatic)."""
    return SectionTemplate(
        section_type=SectionType.BREAKDOWN,
        length_bars=length_bars,
        energy_level=0.3,
        dynamic_range=0.7,
        instruments=["drums", "synth"],
        vocal_type=None,
        note_density=0.2,
        rhythmic_complexity=0.7,
        harmonic_movement="static",
        mix_focus="low",
        reverb_amount=0.7,
    )


def create_buildup(length_bars: int = 4) -> SectionTemplate:
    """Create buildup section (tension before drop)."""
    return SectionTemplate(
        section_type=SectionType.BUILDUP,
        length_bars=length_bars,
        energy_level=0.7,
        dynamic_range=0.8,
        instruments=["drums", "synth", "fx"],
        vocal_type=None,
        note_density=0.8,
        rhythmic_complexity=0.8,
        harmonic_movement="static",
        mix_focus="high",
        reverb_amount=0.3,
    )


def create_drop(length_bars: int = 8) -> SectionTemplate:
    """Create drop section (EDM climax)."""
    return SectionTemplate(
        section_type=SectionType.DROP,
        length_bars=length_bars,
        energy_level=1.0,
        dynamic_range=0.6,
        instruments=["drums", "bass", "synth", "fx"],
        vocal_type=None,
        note_density=0.9,
        rhythmic_complexity=0.6,
        harmonic_movement="static",
        mix_focus="balanced",
        reverb_amount=0.3,
    )


def create_outro(length_bars: int = 4) -> SectionTemplate:
    """Create outro section."""
    return SectionTemplate(
        section_type=SectionType.OUTRO,
        length_bars=length_bars,
        energy_level=0.2,
        dynamic_range=0.3,
        instruments=["pad", "ambient"],
        vocal_type=None,
        note_density=0.2,
        rhythmic_complexity=0.2,
        harmonic_movement="static",
        mix_focus="high",
        reverb_amount=0.8,
    )


# =================================================================
# GENRE-SPECIFIC TEMPLATES
# =================================================================

def get_pop_structure() -> ArrangementTemplate:
    """Standard pop song structure."""
    return ArrangementTemplate(
        name="Pop Standard",
        genre="pop",
        tempo_bpm=120,
        sections=[
            create_intro(4),
            create_verse(8, energy_level=0.4),
            create_chorus(8),
            create_verse(8, energy_level=0.5),
            create_chorus(8),
            create_bridge(8),
            create_chorus(8, energy_level=0.9),
            create_outro(4),
        ],
    )


def get_rock_structure() -> ArrangementTemplate:
    """Standard rock song structure."""
    return ArrangementTemplate(
        name="Rock Standard",
        genre="rock",
        tempo_bpm=140,
        sections=[
            create_intro(4),
            create_verse(8, energy_level=0.5),
            create_prechorus(4),
            create_chorus(8),
            create_verse(8, energy_level=0.6),
            create_prechorus(4),
            create_chorus(8),
            SectionTemplate(
                section_type=SectionType.SOLO,
                length_bars=8,
                energy_level=0.9,
                instruments=["guitar", "drums", "bass"],
            ),
            create_chorus(8, energy_level=0.9),
            create_outro(4),
        ],
    )


def get_edm_structure() -> ArrangementTemplate:
    """Standard EDM/electronic structure."""
    return ArrangementTemplate(
        name="EDM Standard",
        genre="edm",
        tempo_bpm=128,
        sections=[
            create_intro(8),
            create_verse(8, energy_level=0.4, with_vocals=True),
            create_buildup(4),
            create_drop(8),
            create_breakdown(4),
            create_verse(8, energy_level=0.5, with_vocals=True),
            create_buildup(4),
            create_drop(8),
            create_outro(8),
        ],
    )


def get_lofi_structure() -> ArrangementTemplate:
    """Lo-fi hip-hop structure."""
    return ArrangementTemplate(
        name="Lo-Fi Standard",
        genre="lofi",
        tempo_bpm=75,
        sections=[
            create_intro(4),
            create_verse(8, energy_level=0.3, with_vocals=False),
            create_chorus(8, energy_level=0.4),
            create_verse(8, energy_level=0.35, with_vocals=False),
            create_chorus(8, energy_level=0.45),
            create_bridge(8),
            create_outro(8),
        ],
    )


def get_indie_structure() -> ArrangementTemplate:
    """Indie/alternative structure."""
    return ArrangementTemplate(
        name="Indie Standard",
        genre="indie",
        tempo_bpm=110,
        sections=[
            create_verse(8, energy_level=0.3),
            create_verse(8, energy_level=0.4),
            create_chorus(8, energy_level=0.6),
            create_verse(8, energy_level=0.5),
            create_chorus(8, energy_level=0.7),
            create_bridge(8),
            create_chorus(8, energy_level=0.8),
            create_outro(8),
        ],
    )


# =================================================================
# TEMPLATE LOOKUP
# =================================================================

GENRE_TEMPLATES = {
    "pop": get_pop_structure,
    "rock": get_rock_structure,
    "edm": get_edm_structure,
    "electronic": get_edm_structure,
    "lofi": get_lofi_structure,
    "lo-fi": get_lofi_structure,
    "indie": get_indie_structure,
    "alternative": get_indie_structure,
}


def get_genre_template(genre: str) -> ArrangementTemplate:
    """
    Get arrangement template for a genre.
    
    Args:
        genre: Genre name (pop, rock, edm, lofi, indie, etc.)
    
    Returns:
        ArrangementTemplate for the genre
    
    Raises:
        ValueError: If genre not found
    """
    genre_lower = genre.lower()
    
    if genre_lower not in GENRE_TEMPLATES:
        available = ", ".join(GENRE_TEMPLATES.keys())
        raise ValueError(
            f"Genre '{genre}' not found. Available: {available}"
        )
    
    return GENRE_TEMPLATES[genre_lower]()


def list_available_genres() -> List[str]:
    """Get list of available genre templates."""
    return list(GENRE_TEMPLATES.keys())
