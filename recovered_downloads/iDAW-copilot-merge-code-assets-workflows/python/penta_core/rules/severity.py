"""
Rule severity classification for music theory violations.
"""

from enum import Enum, auto


class RuleSeverity(Enum):
    """
    Severity levels for music theory rule violations.
    
    Attributes:
        STRICT: Forbidden in all contexts (e.g., parallel fifths in strict counterpoint)
        GUIDELINE: Strong preference, breakable with justification (e.g., voice crossing)
        STYLISTIC: Context-dependent preference (e.g., jazz voice leading differs from classical)
        MODERN: Classical prohibition, acceptable in contemporary music
    """
    STRICT = auto()
    GUIDELINE = auto()
    STYLISTIC = auto()
    MODERN = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    def __repr__(self) -> str:
        return f"RuleSeverity.{self.name}"
