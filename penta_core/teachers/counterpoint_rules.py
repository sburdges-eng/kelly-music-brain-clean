"""
Counterpoint Rulebook - Species Counterpoint Rules

This module provides comprehensive rules for counterpoint based on
Fux's Gradus ad Parnassum and traditional species counterpoint.
"""

from typing import Dict, List, Optional
from enum import Enum


class Species(Enum):
    """The five species of counterpoint."""
    FIRST = 1    # Note against note
    SECOND = 2   # Two notes against one
    THIRD = 3    # Four notes against one
    FOURTH = 4   # Syncopation
    FIFTH = 5    # Florid (free combination)


class CounterpointRules:
    """
    Comprehensive counterpoint rulebook based on:
    - Fux's Gradus ad Parnassum (1725)
    - Knud Jeppesen's The Style of Palestrina
    - Kent Kennan's Counterpoint
    """
    
    FIRST_SPECIES_RULES = {
        "note_against_note": {
            "name": "Note Against Note",
            "description": "Each note in counterpoint matches duration of cantus firmus note",
            "ratio": "1:1",
            "requirement": "strict"
        },
        
        "begin_and_end_perfect": {
            "name": "Begin and End on Perfect Consonance",
            "description": "First and last intervals must be perfect unison, octave, or fifth",
            "first_interval": ["P1", "P8"],
            "last_interval": ["P1", "P8"],
            "reason": "Provides stability at boundaries"
        },
        
        "contrary_motion_to_perfect": {
            "name": "Approach Perfect Consonances by Contrary Motion",
            "description": "Perfect consonances (P5, P8) should be approached by contrary motion",
            "exception": "Unison may be approached by oblique motion",
            "reason": "Avoids parallel perfect intervals"
        },
        
        "prefer_contrary_motion": {
            "name": "Prefer Contrary Motion",
            "description": "Contrary motion is strongest; use it frequently",
            "hierarchy": ["contrary", "oblique", "similar", "parallel"],
            "guideline": "Use contrary motion at least 50% of the time"
        },
        
        "consonance_only": {
            "name": "Use Only Consonances",
            "description": "All intervals must be consonant",
            "perfect_consonances": ["P1", "P5", "P8"],
            "imperfect_consonances": ["m3", "M3", "m6", "M6"],
            "dissonances": ["m2", "M2", "P4", "A4", "d5", "m7", "M7"],
            "note": "Perfect fourth is dissonance when lowest voice is involved"
        },
        
        "no_parallel_perfects": {
            "name": "No Parallel Perfect Intervals",
            "description": "Avoid parallel unisons, fifths, or octaves",
            "forbidden": ["P1→P1", "P5→P5", "P8→P8"],
            "reason": "Destroys independence of voices"
        },
        
        "avoid_similar_to_perfect": {
            "name": "Avoid Similar Motion to Perfect Consonance",
            "description": "Don't approach P5 or P8 by similar motion",
            "exception": "Acceptable if upper voice moves by step",
            "reason": "Creates hidden parallel fifths or octaves"
        },
        
        "melodic_considerations": {
            "name": "Melodic Smoothness",
            "description": "Counterpoint should have smooth, singable melody",
            "prefer_steps": "Stepwise motion preferred",
            "allowed_leaps": ["m3", "M3", "P4", "P5", "m6", "P8"],
            "forbidden_leaps": ["A4", "d5", "M6", "m7", "M7"],
            "leap_resolution": "Leap of 4th or larger should reverse direction"
        },
        
        "climax": {
            "name": "Single Melodic Climax",
            "description": "Counterpoint should have one clear high point",
            "placement": "Typically in middle or latter 2/3 of line",
            "avoid": "Multiple peaks, too early or too late climax"
        },
        
        "no_repetition": {
            "name": "Avoid Immediate Repetition",
            "description": "Don't repeat same pitch consecutively",
            "reason": "Creates static, uninteresting line"
        },
    }
    
    SECOND_SPECIES_RULES = {
        "two_against_one": {
            "name": "Two Notes Against One",
            "description": "Two counterpoint notes for each cantus firmus note",
            "ratio": "2:1",
            "typical_meter": "Half notes against whole notes"
        },
        
        "first_note_consonant": {
            "name": "Downbeat Must Be Consonant",
            "description": "First (downbeat) note must be consonant with cantus",
            "requirement": "strict",
            "reason": "Downbeat provides harmonic stability"
        },
        
        "passing_tone": {
            "name": "Upbeat May Be Dissonant as Passing Tone",
            "description": "Second note may be dissonant if it passes stepwise between consonances",
            "motion": "Must move stepwise in same direction",
            "example": "C→D→E with cantus on C then E (D is passing tone)",
            "allowed_dissonances": ["m2", "M2", "P4", "m7", "M7"]
        },
        
        "neighbor_tone": {
            "name": "Neighbor Tone (Auxiliary)",
            "description": "Dissonant upbeat that steps away and returns",
            "motion": "Step away, step back to same note",
            "less_common": "Use sparingly compared to passing tones"
        },
        
        "consonant_skips": {
            "name": "Skips Must Be Between Consonances",
            "description": "Any melodic skip must be between two consonant notes",
            "reason": "Dissonant skips break the melodic line"
        },
        
        "cadence": {
            "name": "Final Measure Requirements",
            "description": "Last measure has specific requirements",
            "penultimate": "Step to leading tone or scale degree 2",
            "final": "Resolve to tonic",
            "rhythm": "Often whole note at end (return to 1:1 ratio)"
        },
    }
    
    THIRD_SPECIES_RULES = {
        "four_against_one": {
            "name": "Four Notes Against One",
            "description": "Four counterpoint notes for each cantus firmus note",
            "ratio": "4:1",
            "typical_meter": "Quarter notes against whole notes"
        },
        
        "first_and_third_beats": {
            "name": "Beats 1 and 3 Should Be Consonant",
            "description": "Primary beats (1 and 3) preferably consonant",
            "requirement": "strong preference",
            "beats_2_and_4": "May be dissonant as passing or neighbor tones"
        },
        
        "cambiata": {
            "name": "Cambiata Figure",
            "description": "Special melodic figure allowing dissonant leap",
            "pattern": "Descend step, leap down third (dissonant), step down",
            "example": "D→C→A→G with cantus on G (C is dissonant cambiata)",
            "rarity": "Use occasionally for variety"
        },
        
        "double_neighbor": {
            "name": "Double Neighbor (Neighboring Group)",
            "description": "Upper and lower neighbor around one note",
            "pattern": "Step up, step down, step down, step up (or inverse)",
            "example": "C→D→C→B→C",
            "usage": "Embellish important notes"
        },
        
        "nota_cambiata": {
            "name": "Nota Cambiata",
            "description": "Escape tone with skip to consonance",
            "pattern": "Step in one direction, skip (3rd) in opposite, step continues",
            "example": "C→D→B→C (D steps up from C, skips to B, resolves to C)"
        },
        
        "rhythmic_vitality": {
            "name": "Maintain Rhythmic Interest",
            "description": "Mix of stepwise motion and skips; avoid monotony",
            "variety": "Include some quarter rests for breathing",
            "balance": "Not too much running motion"
        },
    }
    
    FOURTH_SPECIES_RULES = {
        "syncopation": {
            "name": "Syncopated Rhythm",
            "description": "Notes tied across beats, creating suspensions",
            "pattern": "Note begins on weak beat, tied to strong beat",
            "effect": "Creates harmonic tension through suspension"
        },
        
        "suspension_preparation": {
            "name": "Suspension Must Be Prepared",
            "description": "Suspended note must be consonant when first sounded",
            "steps": [
                "1. Preparation: consonance on weak beat",
                "2. Suspension: tied over barline (now dissonant)",
                "3. Resolution: stepwise down to consonance"
            ],
            "requirement": "strict"
        },
        
        "resolution_downward": {
            "name": "Suspensions Resolve Down by Step",
            "description": "Dissonant suspended note resolves down by step",
            "exception": "2-3 suspension in upper voice can resolve up",
            "reason": "Downward resolution is most natural"
        },
        
        "common_suspensions": {
            "name": "Standard Suspension Figures",
            "description": "Common intervallic patterns",
            "upper_voice": {
                "9-8": "Ninth resolves to octave",
                "7-6": "Seventh resolves to sixth",
                "4-3": "Fourth resolves to third",
                "2-3": "Second resolves to third (upward resolution)",
            },
            "lower_voice": {
                "2-3": "Second resolves to third (in bass)",
            }
        },
        
        "breaking_syncopation": {
            "name": "Breaking the Syncopation",
            "description": "Occasionally break the tied rhythm for variety",
            "when": "Use to avoid monotony, typically near cadence",
            "return": "Return to syncopation after break"
        },
        
        "consonant_fourth": {
            "name": "Fourth Can Be Consonant in Suspension",
            "description": "Suspended fourth is acceptable dissonance",
            "context": "When prepared and resolved properly",
            "note": "This is the main place P4 appears as legitimate dissonance"
        },
    }
    
    FIFTH_SPECIES_RULES = {
        "florid_counterpoint": {
            "name": "Florid (Free) Counterpoint",
            "description": "Combines all previous species freely",
            "includes": [
                "Whole notes (first species)",
                "Half notes (second species)",
                "Quarter notes (third species)",
                "Suspensions (fourth species)",
            ],
            "freedom": "Mix rhythms, techniques for musical expression"
        },
        
        "variety": {
            "name": "Rhythmic and Melodic Variety",
            "description": "Use diverse rhythms and melodic patterns",
            "avoid": [
                "Too much of one rhythm",
                "Excessive repetition",
                "Monotonous patterns"
            ],
            "balance": "Mix consonance and controlled dissonance"
        },
        
        "musical_shape": {
            "name": "Create Musical Shape and Direction",
            "description": "Line should have clear phrasing and direction",
            "elements": [
                "Clear climax",
                "Logical phrase structure",
                "Variety in motion types",
                "Satisfying cadence"
            ]
        },
        
        "embellishment": {
            "name": "Use Embellishments Tastefully",
            "description": "Employ passing tones, neighbors, suspensions musically",
            "types": [
                "Passing tones",
                "Neighbor tones",
                "Suspensions",
                "Escape tones",
                "Cambiata figures",
                "Anticipations"
            ],
            "moderation": "Don't overuse any single figure"
        },
    }
    
    GENERAL_COUNTERPOINT_RULES = {
        "independence": {
            "name": "Maintain Voice Independence",
            "description": "Each voice should be satisfying as independent melody",
            "characteristics": [
                "Distinct rhythmic profile",
                "Independent contour",
                "Unique melodic shape",
                "Own sense of direction"
            ]
        },
        
        "interval_quality": {
            "name": "Interval Classifications",
            "description": "Understanding consonance and dissonance",
            "perfect_consonances": {
                "intervals": ["P1", "P5", "P8"],
                "character": "Stable but static",
                "use": "Structurally important points"
            },
            "imperfect_consonances": {
                "intervals": ["m3", "M3", "m6", "M6"],
                "character": "Stable and sonorous",
                "use": "Freely throughout"
            },
            "dissonances": {
                "intervals": ["m2", "M2", "P4*", "A4", "d5", "m7", "M7"],
                "character": "Tense, requires resolution",
                "use": "Carefully controlled (passing, neighbor, suspension)",
                "note": "*P4 dissonant when involving lowest voice"
            }
        },
        
        "range": {
            "name": "Voice Ranges",
            "description": "Keep voices within singable ranges",
            "typical_ranges": {
                "soprano": "C4 to A5",
                "alto": "G3 to D5",
                "tenor": "C3 to A4",
                "bass": "E2 to E4"
            },
            "note": "Adjust for instrumental counterpoint"
        },
        
        "tritone_treatment": {
            "name": "Tritone Treatment",
            "description": "Handle augmented fourths and diminished fifths carefully",
            "melodic": "Avoid as melodic leap",
            "harmonic": "If present, should resolve (A4→m6, d5→M3)",
            "resolution": "Tendency to expand or contract"
        },
        
        "cadence_types": {
            "name": "Counterpoint Cadences",
            "description": "Standard cadence patterns",
            "types": {
                "authentic": {
                    "cantus": "2̂→1̂ or 7̂→1̂",
                    "counterpoint": "7̂→8̂ (in major) or 2̂→1̂",
                    "final_interval": "P8 or P1"
                },
                "plagal": {
                    "cantus": "4̂→1̂ or 5̂→1̂",
                    "motion": "Often contrary",
                    "final_interval": "P8 or P1"
                }
            }
        },
        
        "text_setting": {
            "name": "Text Setting (Vocal Counterpoint)",
            "description": "When setting text, respect natural word stress",
            "principles": [
                "Stressed syllables on strong beats",
                "Melismas on important words",
                "Natural speech rhythm",
                "Clear text declamation"
            ]
        },
    }
    
    MODAL_COUNTERPOINT_RULES = {
        "dorian": {
            "name": "Dorian Mode",
            "description": "Minor mode with raised 6th degree",
            "scale": "D-E-F-G-A-B-C-D",
            "character": "Minor with bright sixth",
            "cadence": "Raised 7th (C#) at final cadence in D Dorian"
        },
        
        "phrygian": {
            "name": "Phrygian Mode",
            "description": "Minor mode with lowered 2nd degree",
            "scale": "E-F-G-A-B-C-D-E",
            "character": "Dark minor with half-step above tonic",
            "cadence": "Half-step descent to tonic (Phrygian cadence)"
        },
        
        "lydian": {
            "name": "Lydian Mode",
            "description": "Major mode with raised 4th degree",
            "scale": "F-G-A-B-C-D-E-F",
            "character": "Bright major with tritone above tonic",
            "note": "Avoid melodic tritone F-B"
        },
        
        "mixolydian": {
            "name": "Mixolydian Mode",
            "description": "Major mode with lowered 7th degree",
            "scale": "G-A-B-C-D-E-F-G",
            "character": "Major with flat seventh",
            "cadence": "F-G cadence (lacking leading tone)"
        },
        
        "ionian_aeolian": {
            "name": "Ionian and Aeolian",
            "description": "Equivalent to modern major and natural minor",
            "ionian": "C-D-E-F-G-A-B-C (major)",
            "aeolian": "A-B-C-D-E-F-G-A (natural minor)",
            "note": "Most familiar to modern ears"
        },
    }
    
    @classmethod
    def get_all_rules(cls) -> Dict[str, Dict]:
        """Get all counterpoint rules organized by species and category."""
        return {
            "first_species": cls.FIRST_SPECIES_RULES,
            "second_species": cls.SECOND_SPECIES_RULES,
            "third_species": cls.THIRD_SPECIES_RULES,
            "fourth_species": cls.FOURTH_SPECIES_RULES,
            "fifth_species": cls.FIFTH_SPECIES_RULES,
            "general": cls.GENERAL_COUNTERPOINT_RULES,
            "modal": cls.MODAL_COUNTERPOINT_RULES,
        }
    
    @classmethod
    def get_species_rules(cls, species: Species) -> Dict[str, Dict]:
        """
        Get rules for a specific species.
        
        Args:
            species: Species enum value
            
        Returns:
            Dictionary of rules for that species
        """
        species_map = {
            Species.FIRST: cls.FIRST_SPECIES_RULES,
            Species.SECOND: cls.SECOND_SPECIES_RULES,
            Species.THIRD: cls.THIRD_SPECIES_RULES,
            Species.FOURTH: cls.FOURTH_SPECIES_RULES,
            Species.FIFTH: cls.FIFTH_SPECIES_RULES,
        }
        return species_map.get(species, {})
    
    @classmethod
    def is_consonant(cls, interval: str) -> bool:
        """
        Check if an interval is consonant.
        
        Args:
            interval: Interval name (e.g., "P5", "M3")
            
        Returns:
            True if consonant, False if dissonant
        """
        consonances = ["P1", "m3", "M3", "P5", "m6", "M6", "P8"]
        # P4 is special case - consonant in upper voices, dissonant with bass
        return interval in consonances
