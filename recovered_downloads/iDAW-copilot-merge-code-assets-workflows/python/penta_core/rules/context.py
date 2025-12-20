"""
Musical context classification for style-dependent rule application.
"""

from enum import Enum, auto
from typing import Set


class MusicalContext(Enum):
    """
    Musical style contexts that affect rule interpretation.
    
    Attributes:
        RENAISSANCE: Palestrina, modal counterpoint (1450-1600)
        BAROQUE: Bach, Handel, figured bass (1600-1750)
        CLASSICAL: Mozart, Haydn, functional harmony (1750-1820)
        ROMANTIC: Chopin, Wagner, extended harmony (1820-1900)
        JAZZ: Bebop, modal jazz, altered dominants (1920-present)
        CONTEMPORARY: Post-tonal, quartal harmony, polychords (1900-present)
    """
    RENAISSANCE = auto()
    BAROQUE = auto()
    CLASSICAL = auto()
    ROMANTIC = auto()
    JAZZ = auto()
    CONTEMPORARY = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    def __repr__(self) -> str:
        return f"MusicalContext.{self.name}"


# Context groupings for multi-era rule applicability
CONTEXT_GROUPS = {
    "common_practice": {
        MusicalContext.BAROQUE,
        MusicalContext.CLASSICAL,
        MusicalContext.ROMANTIC,
    },
    "early_music": {
        MusicalContext.RENAISSANCE,
        MusicalContext.BAROQUE,
    },
    "tonal": {
        MusicalContext.BAROQUE,
        MusicalContext.CLASSICAL,
        MusicalContext.ROMANTIC,
        MusicalContext.JAZZ,
    },
    "modern": {
        MusicalContext.JAZZ,
        MusicalContext.CONTEMPORARY,
    },
    "all": set(MusicalContext),
}


def get_context_group(group_name: str) -> Set[MusicalContext]:
    """
    Get a predefined group of musical contexts.
    
    Args:
        group_name: Name of the context group (e.g., "common_practice", "jazz")
    
    Returns:
        Set of MusicalContext values
    
    Raises:
        KeyError: If group_name is not recognized
    """
    return CONTEXT_GROUPS[group_name]
