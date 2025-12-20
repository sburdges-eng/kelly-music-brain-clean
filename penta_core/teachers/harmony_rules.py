"""
Harmony Rulebook - Chord Construction and Progression Rules

This module provides comprehensive rules for chord construction,
progression, and harmonic analysis.
"""

from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


class ChordQuality(Enum):
    """Standard chord qualities."""
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    DOMINANT = "dominant7"
    MAJOR7 = "major7"
    MINOR7 = "minor7"
    HALF_DIMINISHED = "half_diminished7"
    DIMINISHED7 = "diminished7"


class HarmonyRules:
    """
    Comprehensive harmony rulebook based on:
    - Rameau's Treatise on Harmony
    - Schenker's theories
    - Riemann's functional harmony
    - Berklee jazz harmony
    - Pop/rock harmony practices
    """
    
    CHORD_CONSTRUCTION_RULES = {
        "triad_spelling": {
            "name": "Triad Spelling",
            "description": "Triads consist of root, third, and fifth",
            "types": {
                "major": [0, 4, 7],        # Root, M3, P5
                "minor": [0, 3, 7],        # Root, m3, P5
                "diminished": [0, 3, 6],   # Root, m3, d5
                "augmented": [0, 4, 8],    # Root, M3, A5
            },
            "note": "Intervals measured in semitones from root"
        },
        
        "seventh_chords": {
            "name": "Seventh Chord Construction",
            "description": "Seventh chords add a seventh above the root",
            "types": {
                "dominant7": [0, 4, 7, 10],           # M3, P5, m7
                "major7": [0, 4, 7, 11],              # M3, P5, M7
                "minor7": [0, 3, 7, 10],              # m3, P5, m7
                "half_diminished7": [0, 3, 6, 10],    # m3, d5, m7 (ø7)
                "diminished7": [0, 3, 6, 9],          # m3, d5, d7 (°7)
                "minor_major7": [0, 3, 7, 11],        # m3, P5, M7
                "augmented_major7": [0, 4, 8, 11],    # M3, A5, M7
            }
        },
        
        "extended_chords": {
            "name": "Extended Chord Construction",
            "description": "Chords extended beyond the 7th",
            "ninths": {
                "major9": [0, 4, 7, 11, 14],      # Add M9
                "minor9": [0, 3, 7, 10, 14],      # Add M9
                "dominant9": [0, 4, 7, 10, 14],   # Add M9
                "dominant♭9": [0, 4, 7, 10, 13],  # Add m9
                "dominant♯9": [0, 4, 7, 10, 15],  # Add augmented 9
            },
            "elevenths": {
                "major11": [0, 4, 7, 11, 14, 17],    # Add P11
                "minor11": [0, 3, 7, 10, 14, 17],    # Add P11
                "dominant11": [0, 4, 7, 10, 14, 17], # Add P11
                "dominant♯11": [0, 4, 7, 10, 14, 18], # Add augmented 11
            },
            "thirteenths": {
                "major13": [0, 4, 7, 11, 14, 17, 21],    # Add M13
                "minor13": [0, 3, 7, 10, 14, 17, 21],    # Add M13
                "dominant13": [0, 4, 7, 10, 14, 17, 21], # Add M13
            },
            "note": "Extensions typically omit the 5th and sometimes 11th"
        },
        
        "altered_dominants": {
            "name": "Altered Dominant Chords",
            "description": "Dominant chords with altered 5th and/or 9th",
            "alterations": {
                "♭5": -1,    # Lower 5th by semitone
                "♯5": +1,    # Raise 5th by semitone
                "♭9": -1,    # Lower 9th by semitone
                "♯9": +1,    # Raise 9th by semitone
                "♯11": +1,   # Raise 11th by semitone (same as ♭5)
                "♭13": -1,   # Lower 13th by semitone (same as ♯5)
            },
            "common": {
                "7alt": [0, 4, 10, 13, 15, 18, 20],  # All alterations
                "7♯9": [0, 4, 7, 10, 15],             # Sharp 9
                "7♭9": [0, 4, 7, 10, 13],             # Flat 9
                "7♯5": [0, 4, 8, 10],                 # Sharp 5
                "7♭5": [0, 4, 6, 10],                 # Flat 5
            }
        },
        
        "sus_chords": {
            "name": "Suspended Chords",
            "description": "Replace third with second or fourth",
            "types": {
                "sus2": [0, 2, 7],      # Root, M2, P5
                "sus4": [0, 5, 7],      # Root, P4, P5
                "7sus4": [0, 5, 7, 10], # Root, P4, P5, m7
                "9sus4": [0, 5, 7, 10, 14], # Add M9
            },
            "resolution": "Typically resolves to major or minor triad"
        },
        
        "add_chords": {
            "name": "Add Chords",
            "description": "Add extensions without implying full extended chord",
            "types": {
                "add9": [0, 4, 7, 14],      # Major triad + M9
                "madd9": [0, 3, 7, 14],     # Minor triad + M9
                "add♯11": [0, 4, 7, 18],    # Major triad + augmented 11
                "6": [0, 4, 7, 9],          # Major triad + M6
                "m6": [0, 3, 7, 9],         # Minor triad + M6
                "6/9": [0, 4, 7, 9, 14],    # Major 6 + M9
            },
            "note": "Add chords skip intermediate extensions"
        },
    }
    
    FUNCTIONAL_HARMONY_RULES = {
        "tonic_function": {
            "name": "Tonic Function",
            "description": "Chords that provide stability and resolution",
            "major_key": [
                "I",    # Primary tonic
                "vi",   # Relative minor (weak tonic)
                "iii",  # Mediant (weak tonic)
            ],
            "minor_key": [
                "i",    # Primary tonic
                "VI",   # Relative major
                "III",  # Mediant
            ],
            "characteristic": "Stable, conclusive, home feeling"
        },
        
        "dominant_function": {
            "name": "Dominant Function",
            "description": "Chords that create tension and lead to tonic",
            "major_key": [
                "V",    # Primary dominant
                "V7",   # Dominant seventh (stronger)
                "vii°", # Leading tone diminished
                "VII",  # Subtonic (modal)
            ],
            "minor_key": [
                "V",    # Requires raised 7th
                "V7",
                "vii°",
                "VII",  # Natural minor
            ],
            "characteristic": "Tension, motion toward tonic",
            "tendency_tones": {
                "leading_tone": "resolves up to tonic",
                "seventh": "resolves down by step"
            }
        },
        
        "subdominant_function": {
            "name": "Subdominant Function",
            "description": "Chords that create motion away from tonic",
            "major_key": [
                "IV",   # Primary subdominant
                "ii",   # Supertonic
                "ii°",  # Diminished supertonic (rare)
            ],
            "minor_key": [
                "iv",   # Primary subdominant
                "ii°",  # Diminished supertonic
                "II",   # Neapolitan (♭II)
            ],
            "characteristic": "Departure from tonic, preparation for dominant"
        },
        
        "pre_dominant_function": {
            "name": "Pre-Dominant Chords",
            "description": "Chords that typically precede dominant",
            "common": ["ii", "IV", "ii°", "ii°7", "IV7", "♭II (Neapolitan)"],
            "note": "Subdominant function chords used before dominant"
        },
    }
    
    PROGRESSION_RULES = {
        "circle_of_fifths": {
            "name": "Circle of Fifths Progressions",
            "description": "Strong progressions moving by descending fifths",
            "strength": "Very strong",
            "examples": [
                "I → IV → vii° → iii → vi → ii → V → I",
                "I → vi → ii → V → I",
                "ii → V → I",
            ],
            "note": "Root movement down P5 (up P4) is strongest progression"
        },
        
        "ascending_thirds": {
            "name": "Ascending Third Progressions",
            "description": "Progressions moving up by thirds",
            "strength": "Moderate",
            "examples": [
                "I → iii → V",
                "vi → I",
            ],
            "note": "Creates smooth bass line"
        },
        
        "descending_thirds": {
            "name": "Descending Third Progressions",
            "description": "Progressions moving down by thirds",
            "strength": "Moderate to strong",
            "examples": [
                "I → vi → IV → ii",
                "I → vi → ii → V",
            ]
        },
        
        "stepwise_bass": {
            "name": "Stepwise Bass Motion",
            "description": "Bass moves by step (ascending or descending)",
            "strength": "Moderate",
            "examples": [
                "I → ii → iii → IV",
                "I → vii° → vi",
            ],
            "note": "Creates conjunct bass line"
        },
        
        "deceptive_cadence": {
            "name": "Deceptive Cadence",
            "description": "V → vi instead of expected V → I",
            "strength": "Weak (delays resolution)",
            "effect": "Surprise, continuation rather than conclusion",
            "voice_leading": "Keep common tones, bass moves up by step"
        },
        
        "plagal_cadence": {
            "name": "Plagal Cadence (Amen)",
            "description": "IV → I progression",
            "strength": "Moderate conclusive strength",
            "context": "Church music, hymns, supplementary cadence after authentic",
            "voice_leading": "Smooth common-tone connection"
        },
        
        "authentic_cadence": {
            "name": "Authentic Cadence",
            "description": "V → I progression",
            "types": {
                "perfect_authentic": "V → I with both in root position, tonic in soprano",
                "imperfect_authentic": "V → I with inversion or non-tonic soprano",
            },
            "strength": "Strongest conclusion",
            "voice_leading": "Leading tone → tonic, seventh → third of I"
        },
        
        "half_cadence": {
            "name": "Half Cadence",
            "description": "Phrase ending on V chord",
            "strength": "Inconclusive (creates expectation)",
            "common_approaches": ["I → V", "ii → V", "IV → V"],
            "effect": "Question, continuation needed"
        },
    }
    
    JAZZ_HARMONY_RULES = {
        "ii_V_I": {
            "name": "ii-V-I Progression",
            "description": "Fundamental jazz progression",
            "major_key": {
                "chords": ["ii⁷", "V⁷", "Imaj⁷"],
                "example_C": ["Dm⁷", "G⁷", "Cmaj⁷"],
                "extensions": ["Dm⁹", "G¹³", "Cmaj⁹"],
            },
            "minor_key": {
                "chords": ["ii°⁷", "V⁷", "i⁷"],
                "example_C": ["Dø⁷", "G⁷♭⁹", "Cm⁷"],
                "extensions": ["Dø⁹", "G⁷alt", "Cm⁹"],
            },
            "voice_leading": "Guide tones (3rd & 7th) move smoothly"
        },
        
        "tritone_substitution": {
            "name": "Tritone Substitution",
            "description": "Replace V⁷ with ♭II⁷ (tritone away)",
            "theory": "Share the same tritone (3rd and 7th)",
            "example": {
                "original": "G⁷ (in C major)",
                "substitute": "D♭⁷",
                "shared_tritone": "B and F"
            },
            "voice_leading": "Creates chromatic bass motion (♭II → I)"
        },
        
        "secondary_dominants": {
            "name": "Secondary Dominants",
            "description": "Dominant of non-tonic chord",
            "notation": "V⁷/x where x is target chord",
            "examples": {
                "V⁷/V": "Dominant of dominant (e.g., D⁷ → G⁷ in C)",
                "V⁷/ii": "Dominant of ii (e.g., A⁷ → Dm⁷ in C)",
                "V⁷/vi": "Dominant of vi (e.g., E⁷ → Am⁷ in C)",
            },
            "effect": "Temporary tonicization of target chord"
        },
        
        "modal_interchange": {
            "name": "Modal Interchange (Borrowed Chords)",
            "description": "Borrow chords from parallel minor/major",
            "major_key_borrows_from_minor": [
                "♭III", "♭VI", "♭VII",  # Flat mediant, submediant, subtonic
                "iv", "i", "ii°",        # Minor subdominant, tonic, supertonic
            ],
            "minor_key_borrows_from_major": [
                "IV", "I", "ii",         # Major subdominant, tonic, supertonic
            ],
            "example_C_major": {
                "♭VII": "B♭maj⁷",
                "iv": "Fm⁷",
                "♭VI": "A♭maj⁷",
            },
            "effect": "Color, darkness/brightness shift"
        },
        
        "extended_dominants": {
            "name": "Extended Dominant Chain",
            "description": "Series of secondary dominants",
            "example": "VII⁷ → III⁷ → VI⁷ → II⁷ → V⁷ → I",
            "example_C": "B⁷ → E⁷ → A⁷ → D⁷ → G⁷ → C",
            "note": "Each chord is dominant of the next"
        },
        
        "diminished_passing": {
            "name": "Diminished Seventh Passing Chords",
            "description": "Diminished 7 chords between diatonic chords",
            "examples": [
                "I → #I°⁷ → ii",
                "ii → #ii°⁷ → iii",
                "I → ♭III°⁷ → ii",
            ],
            "function": "Chromatic passing motion, voice leading"
        },
    }
    
    POP_ROCK_HARMONY_RULES = {
        "I_V_vi_IV": {
            "name": "I-V-vi-IV (Pop-Punk) Progression",
            "description": "Extremely common pop progression",
            "example_C": ["C", "G", "Am", "F"],
            "songs": "Countless hits use this progression",
            "variations": [
                "I-V-IV-V",
                "I-IV-vi-V",
                "vi-IV-I-V",  # Relative minor start
            ]
        },
        
        "I_♭VII_IV": {
            "name": "Mixolydian ♭VII Progression",
            "description": "Major with flatted seventh from Mixolydian mode",
            "example_C": ["C", "B♭", "F"],
            "effect": "Rock/modal sound",
            "note": "Borrowed from parallel Mixolydian"
        },
        
        "power_chords": {
            "name": "Power Chords (Root-Fifth)",
            "description": "Two-note chords (root + P5), no third",
            "notation": "C5 = C + G",
            "context": "Rock, metal with distortion",
            "reason": "No third avoids dissonant intermodulation from distortion"
        },
        
        "pedal_point": {
            "name": "Pedal Point Bass",
            "description": "Sustained or repeated bass note while harmony changes",
            "types": {
                "tonic_pedal": "Bass holds tonic while chords change above",
                "dominant_pedal": "Bass holds dominant",
            },
            "effect": "Drone, tension, or stability depending on context"
        },
    }
    
    @classmethod
    def get_all_rules(cls) -> Dict[str, Dict]:
        """Get all harmony rules organized by category."""
        return {
            "chord_construction": cls.CHORD_CONSTRUCTION_RULES,
            "functional_harmony": cls.FUNCTIONAL_HARMONY_RULES,
            "progressions": cls.PROGRESSION_RULES,
            "jazz_harmony": cls.JAZZ_HARMONY_RULES,
            "pop_rock_harmony": cls.POP_ROCK_HARMONY_RULES,
        }
    
    @classmethod
    def get_chord_intervals(cls, quality: str) -> Optional[List[int]]:
        """
        Get the interval structure for a chord quality.
        
        Args:
            quality: Chord quality name
            
        Returns:
            List of semitones from root, or None if not found
        """
        # Search through all chord types
        for category in [cls.CHORD_CONSTRUCTION_RULES]:
            for rule_name, rule_data in category.items():
                if "types" in rule_data and quality in rule_data["types"]:
                    return rule_data["types"][quality]
        return None
    
    @classmethod
    def get_progression_strength(cls, from_chord: str, to_chord: str, key: str = "C") -> str:
        """
        Evaluate the strength of a harmonic progression.
        
        Args:
            from_chord: Roman numeral of source chord
            to_chord: Roman numeral of target chord
            key: Key context (default C major)
            
        Returns:
            Strength description: "very_strong", "strong", "moderate", "weak", "unusual"
        """
        # This would implement progression strength analysis
        # For now, return placeholder
        return "moderate"
