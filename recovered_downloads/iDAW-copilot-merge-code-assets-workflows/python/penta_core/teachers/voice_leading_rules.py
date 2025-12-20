"""
Voice Leading Rulebook - Traditional and Contemporary Rules

This module provides comprehensive voice leading rules from classical
counterpoint through jazz and contemporary harmony.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum


class RuleSeverity(Enum):
    """Severity levels for rule violations."""
    STRICT = "strict"      # Never break in traditional music
    HIGH = "high"          # Strongly discouraged
    MEDIUM = "medium"      # Context-dependent
    LOW = "low"            # Stylistic preference
    GUIDELINE = "guideline"  # Suggestion, not rule


class VoiceLeadingRules:
    """
    Comprehensive voice leading rulebook based on:
    - Fux's Gradus ad Parnassum (Species Counterpoint)
    - Bach Chorales analysis
    - Piston's Harmony textbook
    - Tymoczko's "A Geometry of Music"
    - Jazz voice leading practices
    """
    
    # Classical Four-Part Harmony Rules
    PARALLEL_MOTION_RULES = {
        "parallel_perfect_fifths": {
            "name": "No Parallel Perfect Fifths",
            "description": "Avoid moving perfect fifths in parallel motion between any two voices",
            "severity": RuleSeverity.STRICT,
            "context": "classical",
            "reason": "Reduces harmonic independence and creates hollow, archaic sound",
            "exception": "Acceptable in: (1) outer voices if separated by more than an octave, "
                        "(2) modern/jazz contexts for color, (3) intentional archaism",
            "detection": "Check intervals between all voice pairs across chord changes",
            "example_violation": {
                "soprano": [67, 69],  # G4 → A4
                "alto": [60, 62],      # C4 → D4 (P5 → P5)
            },
            "example_correct": {
                "soprano": [67, 69],  # G4 → A4
                "alto": [60, 59],      # C4 → B3 (P5 → m7, contrary motion)
            }
        },
        
        "parallel_perfect_octaves": {
            "name": "No Parallel Perfect Octaves",
            "description": "Avoid moving perfect octaves in parallel motion",
            "severity": RuleSeverity.STRICT,
            "context": "classical",
            "reason": "Reduces to three-part texture, wastes a voice",
            "exception": "Acceptable when: (1) deliberately thickening a line, "
                        "(2) orchestration requires doubling",
            "detection": "Check for P8 → P8 motion between voices",
            "example_violation": {
                "soprano": [72, 74],  # C5 → D5
                "tenor": [60, 62],    # C4 → D4 (P8 → P8)
            }
        },
        
        "parallel_perfect_unisons": {
            "name": "No Parallel Unisons",
            "description": "Avoid moving in parallel unisons",
            "severity": RuleSeverity.HIGH,
            "context": "classical",
            "reason": "Complete loss of independence, reduces texture",
            "exception": "Intentional doubling for orchestral weight",
        },
        
        "hidden_fifths": {
            "name": "Avoid Hidden (Direct) Fifths",
            "description": "Avoid approaching P5 by similar motion in outer voices with leap in soprano",
            "severity": RuleSeverity.MEDIUM,
            "context": "classical",
            "reason": "Suggests parallel fifths to the ear",
            "exception": "Acceptable if: (1) soprano moves by step, (2) in inner voices, "
                        "(3) approaching tonic harmony",
            "detection": "Check outer voices for similar motion to P5 with soprano leap",
        },
        
        "hidden_octaves": {
            "name": "Avoid Hidden (Direct) Octaves",
            "description": "Avoid approaching P8 by similar motion in outer voices with leap in soprano",
            "severity": RuleSeverity.MEDIUM,
            "context": "classical",
            "reason": "Suggests parallel octaves, weak bass approach",
            "exception": "Acceptable when soprano moves by step or in inner voices",
        },
    }
    
    MELODIC_RULES = {
        "augmented_second": {
            "name": "Avoid Augmented Seconds",
            "description": "Avoid melodic intervals of augmented second",
            "severity": RuleSeverity.HIGH,
            "context": "classical",
            "reason": "Difficult to sing, sounds unnatural in diatonic context",
            "exception": "Acceptable in: (1) harmonic minor color, (2) chromatic music, "
                        "(3) intentional exotic flavor",
            "interval_cents": 300,  # 3 semitones
            "example": "F# → E♭ in soprano (augmented 6th chord resolution)",
        },
        
        "augmented_fourth_leap": {
            "name": "Avoid Augmented Fourth Leaps",
            "description": "Avoid melodic leaps of augmented fourth (tritone)",
            "severity": RuleSeverity.MEDIUM,
            "context": "classical",
            "reason": "Tritone is unstable, suggests harmonic rather than melodic motion",
            "exception": "Acceptable when: (1) outlined by chordal motion, "
                        "(2) immediately resolved, (3) in chromatic contexts",
            "interval_cents": 600,
        },
        
        "diminished_fifth_leap": {
            "name": "Avoid Diminished Fifth Leaps",
            "description": "Avoid melodic leaps of diminished fifth",
            "severity": RuleSeverity.MEDIUM,
            "context": "classical",
            "reason": "Tritone inversion, requires resolution",
            "exception": "Acceptable in chromatic or modern contexts",
        },
        
        "large_leaps": {
            "name": "Resolve Large Leaps",
            "description": "Leaps larger than a sixth should be followed by stepwise motion in opposite direction",
            "severity": RuleSeverity.MEDIUM,
            "context": "classical",
            "reason": "Maintains melodic coherence and singability",
            "threshold_semitones": 9,  # Major 6th
            "exception": "Arpeggiating a chord is acceptable",
        },
        
        "repeated_leaps_same_direction": {
            "name": "Avoid Consecutive Leaps in Same Direction",
            "description": "Avoid two or more leaps in the same direction without stepwise compensation",
            "severity": RuleSeverity.LOW,
            "context": "classical",
            "reason": "Creates disjunct, difficult-to-sing lines",
            "exception": "Arpeggiating chords (triadic motion)",
        },
        
        "leading_tone_resolution": {
            "name": "Resolve Leading Tone Up",
            "description": "Leading tone should resolve upward by step to tonic",
            "severity": RuleSeverity.HIGH,
            "context": "classical",
            "reason": "Strong tendency tone resolution",
            "exception": "Acceptable when: (1) in inner voice, (2) V7 → vi (deceptive), "
                        "(3) chromatic passing motion",
        },
        
        "seventh_resolution": {
            "name": "Resolve Chord Sevenths Down",
            "description": "Chord sevenths should resolve downward by step",
            "severity": RuleSeverity.HIGH,
            "context": "classical",
            "reason": "Strong dissonance requires downward resolution",
            "exception": "Jazz voicings may sustain or move sevenths freely",
        },
    }
    
    VOICE_CROSSING_RULES = {
        "voice_crossing": {
            "name": "Avoid Voice Crossing",
            "description": "Voices should not cross their normal ranges (SATB order)",
            "severity": RuleSeverity.MEDIUM,
            "context": "classical",
            "reason": "Creates confusion about which voice is which, muddies texture",
            "exception": "Acceptable for: (1) brief crossing in inner voices, "
                        "(2) special voice leading solutions, (3) keyboard textures",
            "ranges": {
                "soprano": (60, 81),    # C4-A5
                "alto": (55, 74),       # G3-D5
                "tenor": (48, 69),      # C3-A4
                "bass": (40, 64),       # E2-E4
            }
        },
        
        "voice_overlap": {
            "name": "Avoid Voice Overlap",
            "description": "A voice should not move beyond the previous note of an adjacent voice",
            "severity": RuleSeverity.MEDIUM,
            "context": "classical",
            "reason": "Can create momentary confusion about voice identity",
            "exception": "More acceptable in keyboard music than vocal",
        },
        
        "spacing_upper_voices": {
            "name": "Keep Upper Three Voices Close",
            "description": "Upper three voices should be within an octave of each other",
            "severity": RuleSeverity.LOW,
            "context": "classical",
            "reason": "Maintains balanced sonority and clear harmonic structure",
            "exception": "Wide spacing acceptable for: (1) dramatic effect, "
                        "(2) registral clarity, (3) keyboard limitations",
            "max_interval_semitones": 12,
        },
        
        "bass_spacing": {
            "name": "Bass Can Be Widely Spaced",
            "description": "Bass may be separated from tenor by more than an octave",
            "severity": RuleSeverity.GUIDELINE,
            "context": "classical",
            "reason": "Bass provides foundation; wide spacing is acoustically clear",
        },
    }
    
    DOUBLING_RULES = {
        "double_leading_tone": {
            "name": "Don't Double Leading Tone",
            "description": "Avoid doubling the leading tone (7̂) in any chord",
            "severity": RuleSeverity.HIGH,
            "context": "classical",
            "reason": "Both voices want to resolve up, creating parallel octaves or poor voice leading",
            "exception": "Acceptable in: (1) non-dominant contexts, (2) modal harmony",
        },
        
        "double_chord_seventh": {
            "name": "Don't Double Chord Sevenths",
            "description": "Avoid doubling the seventh of a chord",
            "severity": RuleSeverity.HIGH,
            "context": "classical",
            "reason": "Dissonant note shouldn't be emphasized; resolution becomes problematic",
        },
        
        "double_altered_notes": {
            "name": "Don't Double Chromatic Alterations",
            "description": "Avoid doubling chromatically altered notes",
            "severity": RuleSeverity.HIGH,
            "context": "classical",
            "reason": "Altered notes are tendency tones with specific resolutions",
        },
        
        "preferred_doublings": {
            "name": "Prefer Root Doubling",
            "description": "In order of preference: double root > fifth > third",
            "severity": RuleSeverity.GUIDELINE,
            "context": "classical",
            "reason": "Root provides stability, fifth is neutral, third defines quality (less doubling needed)",
            "order": ["root", "fifth", "third"],
            "exception": "Diminished chords: prefer to double third (stable tone)",
        },
        
        "soprano_doubling": {
            "name": "Double Soprano Note When Possible",
            "description": "Often effective to double the soprano note for melodic emphasis",
            "severity": RuleSeverity.GUIDELINE,
            "context": "classical",
            "reason": "Reinforces melodic line, creates cohesive outer-voice structure",
        },
    }
    
    JAZZ_VOICE_LEADING_RULES = {
        "guide_tone_lines": {
            "name": "Maintain Guide Tone Lines",
            "description": "3rds and 7ths should move smoothly in contrary or oblique motion",
            "severity": RuleSeverity.HIGH,
            "context": "jazz",
            "reason": "Defines harmonic progression while maintaining voice leading",
            "technique": "3rd and 7th of each chord form smooth stepwise or common-tone connections",
        },
        
        "avoid_note_motion": {
            "name": "Resolve Avoid Notes",
            "description": "Avoid notes (♯11, ♭9 in certain contexts) should resolve or be treated as passing",
            "severity": RuleSeverity.MEDIUM,
            "context": "jazz",
            "reason": "Creates tension that requires resolution",
            "avoid_notes": {
                "major_chord": ["4"],  # Perfect 4th against major chord
                "minor_chord": ["♭6"],
                "dominant": ["♭9"] if "in_melody" else []
            }
        },
        
        "drop_voicings": {
            "name": "Use Drop-2 and Drop-3 Voicings",
            "description": "Drop voices to create playable and resonant voicings",
            "severity": RuleSeverity.GUIDELINE,
            "context": "jazz",
            "reason": "Creates idiomatic guitar/piano voicings with good voice leading",
            "technique": "Drop second or third voice from top by an octave",
        },
        
        "upper_structure_triads": {
            "name": "Use Upper Structure Triads",
            "description": "Place triads above bass note to create extensions",
            "severity": RuleSeverity.GUIDELINE,
            "context": "jazz",
            "reason": "Efficient way to voice complex extensions",
            "example": "D major triad over C7 = C13(♯11)",
        },
    }
    
    CONTEMPORARY_RULES = {
        "parallel_motion_color": {
            "name": "Parallel Motion for Color",
            "description": "Parallel fifths, fourths, and triads used for harmonic color",
            "severity": RuleSeverity.GUIDELINE,
            "context": "contemporary",
            "reason": "Creates modal, impressionistic, or post-tonal effects",
            "styles": ["impressionist", "modal", "minimalist", "film_music"],
        },
        
        "quartal_harmony": {
            "name": "Quartal/Quintal Voicings",
            "description": "Voices built from stacked fourths or fifths",
            "severity": RuleSeverity.GUIDELINE,
            "context": "contemporary",
            "reason": "Creates open, modern, ambiguous harmony",
            "technique": "Stack perfect fourths instead of thirds",
        },
        
        "cluster_voicings": {
            "name": "Tone Clusters",
            "description": "Adjacent semitones or tones for textural effect",
            "severity": RuleSeverity.GUIDELINE,
            "context": "contemporary",
            "reason": "Creates dense, coloristic textures",
            "composers": ["Cowell", "Bartók", "Ligeti"],
        },
    }
    
    @classmethod
    def get_all_rules(cls) -> Dict[str, Dict]:
        """Get all rules organized by category."""
        return {
            "parallel_motion": cls.PARALLEL_MOTION_RULES,
            "melodic": cls.MELODIC_RULES,
            "voice_crossing": cls.VOICE_CROSSING_RULES,
            "doubling": cls.DOUBLING_RULES,
            "jazz": cls.JAZZ_VOICE_LEADING_RULES,
            "contemporary": cls.CONTEMPORARY_RULES,
        }
    
    @classmethod
    def get_rules_by_context(cls, context: str) -> Dict[str, Dict]:
        """
        Get rules filtered by musical context.
        
        Args:
            context: "classical", "jazz", "contemporary", or "all"
            
        Returns:
            Dictionary of relevant rules
        """
        all_rules = cls.get_all_rules()
        
        if context == "all":
            return all_rules
        
        filtered = {}
        for category, rules in all_rules.items():
            filtered[category] = {
                name: rule for name, rule in rules.items()
                if rule.get("context", "classical") == context or context == "all"
            }
        
        return filtered
    
    @classmethod
    def get_rules_by_severity(cls, min_severity: RuleSeverity) -> Dict[str, Dict]:
        """
        Get rules filtered by minimum severity level.
        
        Args:
            min_severity: Minimum severity to include
            
        Returns:
            Dictionary of rules meeting severity threshold
        """
        severity_order = {
            RuleSeverity.GUIDELINE: 0,
            RuleSeverity.LOW: 1,
            RuleSeverity.MEDIUM: 2,
            RuleSeverity.HIGH: 3,
            RuleSeverity.STRICT: 4,
        }
        
        threshold = severity_order[min_severity]
        all_rules = cls.get_all_rules()
        
        filtered = {}
        for category, rules in all_rules.items():
            filtered[category] = {
                name: rule for name, rule in rules.items()
                if severity_order.get(rule.get("severity", RuleSeverity.GUIDELINE), 0) >= threshold
            }
        
        return filtered
