"""
Base classes for music theory rules and violations.
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any
from .severity import RuleSeverity
from .context import MusicalContext


@dataclass
class Rule:
    """
    Base class for music theory rules.
    
    Attributes:
        name: Short identifier (e.g., "no_parallel_fifths")
        description: Human-readable explanation
        severity: RuleSeverity or Dict[MusicalContext, RuleSeverity] for context-dependent severity
        contexts: Musical periods where this rule applies
        reason: Theoretical justification (acoustics, voice independence, etc.)
        exceptions: Known cases where the rule can be broken
        category: Organizational grouping (e.g., "parallel_motion", "chord_construction")
    """
    name: str
    description: str
    severity: RuleSeverity | Dict[MusicalContext, RuleSeverity]
    contexts: Set[MusicalContext]
    reason: str
    exceptions: List[str] = field(default_factory=list)
    category: str = ""
    
    def applies_to_context(self, context: MusicalContext) -> bool:
        """Check if this rule applies in the given musical context."""
        return context in self.contexts
    
    def get_severity_for_context(self, context: MusicalContext) -> RuleSeverity:
        """
        Get severity level for a specific musical context.
        
        Allows same rule to have different severity in different styles.
        E.g., parallel fifths are STRICT in classical, FLEXIBLE in jazz.
        
        Args:
            context: Musical context to check
        
        Returns:
            RuleSeverity for that context, or default severity if not context-dependent
        
        Example:
            >>> rule = Rule(name="parallel_fifths", severity={
            ...     MusicalContext.CLASSICAL: RuleSeverity.STRICT,
            ...     MusicalContext.JAZZ: RuleSeverity.STYLISTIC,
            ... })
            >>> rule.get_severity_for_context(MusicalContext.CLASSICAL)
            RuleSeverity.STRICT
        """
        if isinstance(self.severity, dict):
            return self.severity.get(context, RuleSeverity.GUIDELINE)
        return self.severity
    
    def is_strict(self) -> bool:
        """Check if this is a strict (non-breakable) rule in any context."""
        if isinstance(self.severity, dict):
            return RuleSeverity.STRICT in self.severity.values()
        return self.severity == RuleSeverity.STRICT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary format (backward compatibility)."""
        severity_repr = (
            {str(k): str(v) for k, v in self.severity.items()}
            if isinstance(self.severity, dict)
            else str(self.severity)
        )
        return {
            "name": self.name,
            "description": self.description,
            "severity": severity_repr,
            "context": [str(c) for c in self.contexts],
            "reason": self.reason,
            "exceptions": self.exceptions,
        }


@dataclass
class RuleViolation:
    """
    Detected violation of a music theory rule.
    
    Attributes:
        rule: The rule that was violated
        location: Musical location (measure number, beat, voice, etc.)
        pitches: MIDI note numbers involved in the violation
        explanation: Specific description of how the rule was broken
        severity_override: Optional context-specific severity adjustment
    """
    rule: Rule
    location: str
    pitches: List[int]
    explanation: str
    severity_override: Optional[RuleSeverity] = None
    
    @property
    def effective_severity(self) -> RuleSeverity:
        """Get the severity, accounting for any override."""
        return self.severity_override if self.severity_override else self.rule.severity
    
    def __str__(self) -> str:
        return (
            f"{self.rule.name} at {self.location}: "
            f"{self.explanation} (severity: {self.effective_severity})"
        )


@dataclass
class RuleBreakSuggestion:
    """
    Pedagogical suggestion for deliberately breaking a rule.
    
    Attributes:
        rule: The rule to break
        context: Musical context where the break is effective
        musical_example: MIDI note sequence demonstrating the break
        explanation: Why this break works musically
        difficulty: 1-5 scale of how advanced this technique is
    """
    rule: Rule
    context: MusicalContext
    musical_example: List[int]  # MIDI notes
    explanation: str
    difficulty: int = 3  # 1=beginner, 5=advanced
    
    def __str__(self) -> str:
        return (
            f"Break '{self.rule.name}' in {self.context} context: "
            f"{self.explanation} (difficulty: {self.difficulty}/5)"
        )
