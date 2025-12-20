"""
Species counterpoint classification (Fux's pedagogical framework).
"""

from enum import Enum, auto


class Species(Enum):
    """
    Fux's five species of counterpoint.
    
    Attributes:
        FIRST: Note-against-note, consonances only
        SECOND: Two notes against one, passing tones allowed
        THIRD: Four notes against one, more melodic freedom
        FOURTH: Syncopated suspensions, dissonance preparation/resolution
        FIFTH: Florid counterpoint, combining all prior species
    """
    FIRST = auto()
    SECOND = auto()
    THIRD = auto()
    FOURTH = auto()
    FIFTH = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    def __repr__(self) -> str:
        return f"Species.{self.name}"
