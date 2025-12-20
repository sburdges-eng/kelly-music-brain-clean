"""
Rule-breaking teacher for teaching harmonic concepts through counterexamples.

This module provides an interactive teacher that demonstrates musical rules
by intentionally breaking them, helping students understand the "why" behind
music theory conventions.
"""

from typing import List, Dict, Any, Optional, Tuple
import random

# Import comprehensive rulebooks
from .voice_leading_rules import VoiceLeadingRules, RuleSeverity
from .harmony_rules import HarmonyRules, ChordQuality
from .counterpoint_rules import CounterpointRules, Species


class RuleBreakingTeacher:
    """
    Interactive music theory teacher that demonstrates concepts by breaking rules.
    
    This teacher introduces harmonic rules (parallel fifths, voice leading,
    spacing, etc.) by showing both correct and incorrect examples, helping
    students develop critical listening and theoretical understanding.
    """
    
    def __init__(self):
        """Initialize the rule-breaking teacher with comprehensive rulebook integration."""
        # Load all comprehensive rules
        self.voice_leading_rules = VoiceLeadingRules.get_all_rules()
        self.harmony_rules = HarmonyRules.get_all_rules()
        self.counterpoint_rules = CounterpointRules.get_all_rules()
        
        # Legacy simple rules for backward compatibility
        self.rules = {
            "parallel_fifths": {
                "description": "Avoid parallel perfect fifths between voices",
                "reason": "Reduces harmonic independence and creates hollow sound",
                "severity": "high"
            },
            "parallel_octaves": {
                "description": "Avoid parallel perfect octaves between voices",
                "reason": "Reduces voice independence and texture",
                "severity": "high"
            },
            "voice_crossing": {
                "description": "Avoid voices crossing their normal ranges",
                "reason": "Can muddy the texture and confuse harmonic clarity",
                "severity": "medium"
            },
            "voice_overlap": {
                "description": "Avoid a voice moving beyond the previous note of an adjacent voice",
                "reason": "Can create confusion in voice leading clarity",
                "severity": "medium"
            },
            "spacing": {
                "description": "Keep upper three voices within an octave of each other",
                "reason": "Maintains balanced sonority and clarity",
                "severity": "low"
            },
            "range": {
                "description": "Keep voices within their comfortable singing ranges",
                "reason": "Ensures performability and natural sound",
                "severity": "medium"
            },
            "doubled_leading_tone": {
                "description": "Don't double the leading tone",
                "reason": "Creates awkward resolution with both notes wanting to rise",
                "severity": "medium"
            },
            "augmented_second": {
                "description": "Avoid melodic augmented seconds",
                "reason": "Difficult to sing and sounds unnatural in tonal music",
                "severity": "medium"
            }
        }
        
        self.current_rule = None
        self.violation_count = 0
    
    def get_comprehensive_rules(self, context: str = "all") -> Dict[str, Any]:
        """
        Get comprehensive rules from all rulebooks.
        
        Args:
            context: "classical", "jazz", "contemporary", or "all"
            
        Returns:
            Dictionary containing all rules organized by category
        """
        return {
            "voice_leading": VoiceLeadingRules.get_rules_by_context(context),
            "harmony": self.harmony_rules,
            "counterpoint": self.counterpoint_rules,
        }
    
    def get_rules_by_severity(self, min_severity: RuleSeverity) -> Dict[str, Any]:
        """
        Get rules filtered by severity level.
        
        Args:
            min_severity: Minimum severity to include
            
        Returns:
            Dictionary of rules meeting severity threshold
        """
        return {
            "voice_leading": VoiceLeadingRules.get_rules_by_severity(min_severity),
            "harmony": self.harmony_rules,  # Add severity filtering for harmony
            "counterpoint": self.counterpoint_rules,
        }
    
    def get_species_counterpoint_rules(self, species: Species) -> Dict[str, Dict]:
        """
        Get rules for specific species counterpoint.
        
        Args:
            species: Species enum (FIRST, SECOND, THIRD, FOURTH, FIFTH)
            
        Returns:
            Dictionary of rules for that species
        """
        return CounterpointRules.get_species_rules(species)
        
    def teach_rule(self, rule_name: str) -> Dict[str, Any]:
        """
        Teach a specific rule by showing violations and corrections.
        
        Args:
            rule_name: Name of the rule to teach (e.g., "parallel_fifths")
            
        Returns:
            Dictionary containing rule info, examples, and explanations
        """
        if rule_name not in self.rules:
            return {
                "error": f"Unknown rule: {rule_name}",
                "available_rules": list(self.rules.keys())
            }
        
        self.current_rule = rule_name
        rule = self.rules[rule_name]
        
        return {
            "rule": rule_name,
            "description": rule["description"],
            "reason": rule["reason"],
            "severity": rule["severity"],
            "lesson": self._generate_lesson(rule_name)
        }
    
    def _generate_lesson(self, rule_name: str) -> Dict[str, Any]:
        """
        Generate a lesson demonstrating rule violations and corrections.
        
        Args:
            rule_name: The rule to create a lesson for
            
        Returns:
            Dictionary with incorrect and correct examples
        """
        # This would generate actual musical examples in a full implementation
        # For now, return conceptual examples
        
        lessons = {
            "parallel_fifths": {
                "violation_example": "C-G moving to D-A (parallel perfect fifths)",
                "correct_example": "C-G moving to D-F (contrary motion avoids parallels)",
                "listening_tip": "Listen for the hollow, organum-like quality in the violation"
            },
            "parallel_octaves": {
                "violation_example": "C4-C5 moving to D4-D5 (parallel octaves)",
                "correct_example": "C4-C5 moving to D4-C5 (oblique motion)",
                "listening_tip": "Parallel octaves make the texture sound thinner"
            },
            "voice_crossing": {
                "violation_example": "Alto moves below tenor line",
                "correct_example": "Alto stays within its range above tenor",
                "listening_tip": "Voice crossing creates momentary confusion about which voice is which"
            }
        }
        
        return lessons.get(rule_name, {
            "violation_example": "Example violation to be implemented",
            "correct_example": "Example correction to be implemented",
            "listening_tip": "Listen for the difference in sound quality"
        })
    
    def break_rule(self, rule_name: str, progression: List[List[int]]) -> Tuple[bool, str]:
        """
        Intentionally break a rule in a chord progression for demonstration.
        
        Args:
            rule_name: The rule to violate
            progression: Chord progression as list of note lists (MIDI numbers)
            
        Returns:
            Tuple of (was_violated, explanation)
        """
        self.violation_count += 1
        
        # This would analyze and modify the progression to demonstrate the violation
        # For now, return a conceptual explanation
        
        return (True, f"Demonstration {self.violation_count}: Breaking '{rule_name}' rule")
    
    def quiz(self, num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate a quiz with examples that may or may not violate rules.
        
        Args:
            num_questions: Number of quiz questions to generate
            
        Returns:
            List of quiz questions with examples and correct answers
        """
        questions = []
        rule_names = list(self.rules.keys())
        
        for i in range(num_questions):
            rule = random.choice(rule_names)
            has_violation = random.choice([True, False])
            
            questions.append({
                "question_num": i + 1,
                "rule_tested": rule,
                "has_violation": has_violation,
                "example": f"Example progression {i+1} (to be generated)",
                "hint": self.rules[rule]["description"]
            })
        
        return questions
    
    def get_all_rules(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about all available rules.
        
        Returns:
            Dictionary of all rules with their descriptions and metadata
        """
        return self.rules.copy()
    
    def suggest_practice_progression(self, difficulty: str = "beginner") -> Dict[str, Any]:
        """
        Suggest a practice progression for learning voice leading rules.
        
        Args:
            difficulty: "beginner", "intermediate", or "advanced"
            
        Returns:
            Suggested practice sequence and exercises
        """
        progressions = {
            "beginner": [
                "parallel_fifths",
                "parallel_octaves",
                "range"
            ],
            "intermediate": [
                "voice_crossing",
                "spacing",
                "doubled_leading_tone"
            ],
            "advanced": [
                "voice_overlap",
                "augmented_second"
            ]
        }
        
        sequence = progressions.get(difficulty, progressions["beginner"])
        
        return {
            "difficulty": difficulty,
            "rule_sequence": sequence,
            "estimated_time": f"{len(sequence) * 15} minutes",
            "description": f"Practice sequence for {difficulty} level"
        }
