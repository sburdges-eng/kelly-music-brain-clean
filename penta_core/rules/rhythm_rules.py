"""
Rhythm and meter rules for timing, phrasing, and metric organization.
"""

from typing import List, Dict, Set
from .base import Rule
from .severity import RuleSeverity
from .context import MusicalContext, CONTEXT_GROUPS


class RhythmRules:
    """
    Rules governing rhythm, meter, and temporal organization.
    
    Categories:
        - Metric hierarchy (downbeat emphasis, hypermeter)
        - Syncopation (suspension preparation/resolution)
        - Phrase structure (4/8/16 bar periods)
        - Polyrhythm (3:2, 4:3 ratios)
    """
    
    @staticmethod
    def get_all_rules() -> List[Rule]:
        """Get all rhythm and meter rules."""
        return [
            # Metric Hierarchy
            Rule(
                name="downbeat_emphasis",
                description="Strongest notes should fall on metric strong beats (1 in 4/4, 1&4 in 6/8)",
                severity=RuleSeverity.GUIDELINE,
                contexts=CONTEXT_GROUPS["common_practice"],
                reason="Metric hierarchy reinforces listener's sense of pulse and phrase structure",
                exceptions=[
                    "Syncopation for rhythmic interest",
                    "Hemiola (3:2 metric displacement)",
                    "Jazz swing patterns displace downbeats",
                ],
                category="metric_hierarchy",
            ),
            
            Rule(
                name="avoid_metric_ambiguity",
                description="Avoid patterns that obscure the meter (unless intentional polymeter)",
                severity=RuleSeverity.STYLISTIC,
                contexts={MusicalContext.CLASSICAL, MusicalContext.ROMANTIC},
                reason="Clear meter aids listener comprehension and performer coordination",
                exceptions=[
                    "Brahms hemiolas (intentional 3:2 metric shifts)",
                    "Contemporary polymetric passages",
                    "Cadential metric displacement",
                ],
                category="metric_hierarchy",
            ),
            
            # Syncopation
            Rule(
                name="prepare_suspensions",
                description="Suspended notes must be consonant when first sounded (preparation)",
                severity=RuleSeverity.STRICT,
                contexts=CONTEXT_GROUPS["common_practice"],
                reason="Unprepared dissonances create harsh acoustic roughness",
                exceptions=[
                    "4-3 suspension over I chord (prepared by consonant 3rd)",
                    "Jazz altered dominants (b9, #9, #11) don't require preparation",
                ],
                category="syncopation",
            ),
            
            Rule(
                name="resolve_suspensions_downward",
                description="Suspensions resolve by stepwise descent (4-3, 9-8, 7-6)",
                severity=RuleSeverity.STRICT,
                contexts={MusicalContext.BAROQUE, MusicalContext.CLASSICAL},
                reason="Downward resolution follows natural acoustic tendency of dissonance",
                exceptions=[
                    "Retardations (upward suspensions, less common)",
                    "Contemporary extended harmonies may freeze suspensions",
                ],
                category="syncopation",
            ),
            
            # Phrase Structure
            Rule(
                name="balanced_phrase_lengths",
                description="Phrases should be symmetrical (4+4, 8+8 bars) in periodic structures",
                severity=RuleSeverity.GUIDELINE,
                contexts={MusicalContext.CLASSICAL, MusicalContext.ROMANTIC},
                reason="Symmetry creates satisfying expectation and resolution for listeners",
                exceptions=[
                    "Phrase extensions (5+4 for dramatic effect)",
                    "Elisions (overlapping phrases)",
                    "Contemporary through-composed forms",
                ],
                category="phrase_structure",
            ),
            
            Rule(
                name="antecedent_consequent",
                description="Antecedent phrase ends with half cadence, consequent with authentic cadence",
                severity=RuleSeverity.GUIDELINE,
                contexts=CONTEXT_GROUPS["common_practice"],
                reason="Question-answer structure creates narrative arc and closure",
                exceptions=[
                    "Modulating periods (consequent ends in new key)",
                    "Contemporary non-cadential phrase endings",
                ],
                category="phrase_structure",
            ),
            
            # Polyrhythm
            Rule(
                name="simple_polyrhythm_ratios",
                description="Polyrhythms should use simple ratios (3:2, 4:3) for clarity",
                severity=RuleSeverity.STYLISTIC,
                contexts={MusicalContext.ROMANTIC, MusicalContext.CONTEMPORARY},
                reason="Complex ratios (7:5) are difficult to perceive and perform accurately",
                exceptions=[
                    "Contemporary music exploring irrational rhythms",
                    "African/Latin traditions with complex cross-rhythms",
                ],
                category="polyrhythm",
            ),
            
            # Tempo and Rubato
            Rule(
                name="gradual_tempo_changes",
                description="Rubato and tempo changes should be gradual, not abrupt",
                severity=RuleSeverity.GUIDELINE,
                contexts={MusicalContext.ROMANTIC},
                reason="Sudden tempo shifts disrupt musical flow and listener orientation",
                exceptions=[
                    "Fermatas (deliberate pauses)",
                    "Subito tempo markings (sudden tempo for dramatic effect)",
                ],
                category="tempo",
            ),
        ]
    
    @staticmethod
    def get_rules_by_category(category: str) -> List[Rule]:
        """Get all rules in a specific category."""
        return [r for r in RhythmRules.get_all_rules() if r.category == category]
    
    @staticmethod
    def get_rules_by_context(context: MusicalContext) -> List[Rule]:
        """Get all rules applicable to a musical context."""
        return [r for r in RhythmRules.get_all_rules() if r.applies_to_context(context)]
    
    @staticmethod
    def get_rules_by_severity(severity: RuleSeverity) -> List[Rule]:
        """Get all rules at a specific severity level."""
        return [r for r in RhythmRules.get_all_rules() if r.severity == severity]
