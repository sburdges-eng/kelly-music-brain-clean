# Music Theory Teachers Module

Comprehensive music theory education tools with interactive learning and rule-based teaching.

## Overview

The `penta_core.teachers` module provides:

1. **Rule-Breaking Teacher** - Learn through counterexamples and violations
2. **Voice Leading Rulebook** - 100+ rules from classical to contemporary
3. **Harmony Rulebook** - Chord construction, functional harmony, progressions
4. **Counterpoint Rulebook** - Species counterpoint based on Fux

## Quick Start

```python
from penta_core.teachers import (
    RuleBreakingTeacher,
    VoiceLeadingRules,
    HarmonyRules,
    CounterpointRules,
    RuleSeverity,
    Species
)

# Initialize teacher
teacher = RuleBreakingTeacher()

# Teach a specific rule
lesson = teacher.teach_rule("parallel_fifths")
print(lesson['description'])
print(lesson['reason'])

# Get rules by musical context
classical_rules = VoiceLeadingRules.get_rules_by_context("classical")
jazz_rules = VoiceLeadingRules.get_rules_by_context("jazz")

# Filter by severity
strict_rules = VoiceLeadingRules.get_rules_by_severity(RuleSeverity.STRICT)

# Get species counterpoint rules
first_species = CounterpointRules.get_species_rules(Species.FIRST)
```

## Rule Categories

### Voice Leading Rules

**Parallel Motion Rules** (7 rules)
- No parallel perfect fifths/octaves (STRICT)
- Hidden fifths and octaves (MEDIUM)
- Direct fifths and octaves (MEDIUM)

**Melodic Rules** (7 rules)
- Avoid augmented seconds (HIGH)
- Resolve large leaps (MEDIUM)
- Leading tone resolution (HIGH)
- Chord seventh resolution (HIGH)

**Voice Crossing Rules** (4 rules)
- Avoid voice crossing (MEDIUM)
- Avoid voice overlap (MEDIUM)
- Upper voice spacing within octave (LOW)

**Doubling Rules** (5 rules)
- Don't double leading tone (HIGH)
- Don't double chord sevenths (HIGH)
- Prefer root doubling (GUIDELINE)

**Jazz Voice Leading** (4 rules)
- Maintain guide tone lines (HIGH)
- Resolve avoid notes (MEDIUM)
- Drop voicings (GUIDELINE)
- Upper structure triads (GUIDELINE)

**Contemporary Rules** (3 rules)
- Parallel motion for color (GUIDELINE)
- Quartal/quintal harmony (GUIDELINE)
- Tone clusters (GUIDELINE)

### Harmony Rules

**Chord Construction**
- Triad spelling (major, minor, diminished, augmented)
- Seventh chords (7 types)
- Extended chords (9ths, 11ths, 13ths)
- Altered dominants (♭9, ♯9, ♭5, ♯5, ♯11)
- Suspended chords (sus2, sus4, 7sus4)
- Add chords (add9, add♯11, 6, 6/9)

**Functional Harmony**
- Tonic function (I, vi, iii)
- Dominant function (V, V7, vii°)
- Subdominant function (IV, ii)
- Pre-dominant function

**Progressions**
- Circle of fifths (very strong)
- Ascending/descending thirds (moderate)
- Stepwise bass motion
- Cadences (authentic, plagal, half, deceptive)

**Jazz Harmony**
- ii-V-I progressions (major and minor)
- Tritone substitution
- Secondary dominants
- Modal interchange
- Extended dominant chains
- Diminished passing chords

**Pop/Rock Harmony**
- I-V-vi-IV progression
- Mixolydian ♭VII
- Power chords
- Pedal point

### Counterpoint Rules

**First Species** (Note against note)
- Perfect consonances at boundaries
- Contrary motion to perfects
- No parallel perfects
- Smooth melodic motion
- Single climax

**Second Species** (Two against one)
- Downbeat consonant
- Passing tone dissonances
- Neighbor tones
- Consonant skips

**Third Species** (Four against one)
- Beats 1 and 3 consonant
- Cambiata figures
- Double neighbor
- Rhythmic vitality

**Fourth Species** (Syncopation)
- Suspension preparation
- Resolution downward by step
- Common suspension figures (9-8, 7-6, 4-3)
- Breaking syncopation for variety

**Fifth Species** (Florid)
- Combines all previous species
- Rhythmic and melodic variety
- Musical shape and direction
- Tasteful embellishment

**General Rules**
- Maintain voice independence
- Interval classifications (consonance/dissonance)
- Voice ranges
- Tritone treatment
- Cadence types

**Modal Counterpoint**
- Dorian, Phrygian, Lydian, Mixolydian modes
- Modal characteristics
- Mode-specific cadences

## Severity Levels

```python
class RuleSeverity(Enum):
    STRICT = "strict"          # Never break in traditional music
    HIGH = "high"              # Strongly discouraged
    MEDIUM = "medium"          # Context-dependent
    LOW = "low"                # Stylistic preference
    GUIDELINE = "guideline"    # Suggestion, not rule
```

## Usage Examples

### Teaching a Rule with Violations

```python
teacher = RuleBreakingTeacher()

# Teach parallel fifths
lesson = teacher.teach_rule("parallel_fifths")

print(lesson['description'])
# "Avoid parallel perfect fifths between voices"

print(lesson['reason'])
# "Reduces harmonic independence and creates hollow sound"

print(lesson['lesson']['violation_example'])
# "C-G moving to D-A (parallel perfect fifths)"

print(lesson['lesson']['correct_example'])
# "C-G moving to D-F (contrary motion avoids parallels)"
```

### Filtering Rules by Context

```python
# Get only classical rules
classical = VoiceLeadingRules.get_rules_by_context("classical")

# Get only jazz rules
jazz = VoiceLeadingRules.get_rules_by_context("jazz")

# Get only contemporary rules
contemporary = VoiceLeadingRules.get_rules_by_context("contemporary")

# Get all rules
all_rules = VoiceLeadingRules.get_rules_by_context("all")
```

### Working with Chord Construction

```python
# Get interval structure for a chord
intervals = HarmonyRules.get_chord_intervals("dominant7")
# [0, 4, 7, 10]  # Root, M3, P5, m7 in semitones

# Access all seventh chord types
harmony = HarmonyRules.get_all_rules()
seventh_chords = harmony["chord_construction"]["seventh_chords"]

for chord_type, intervals in seventh_chords["types"].items():
    print(f"{chord_type}: {intervals}")
```

### Species Counterpoint Learning

```python
# Get first species rules
first_species = CounterpointRules.get_species_rules(Species.FIRST)

# Check if an interval is consonant
is_consonant = CounterpointRules.is_consonant("P5")  # True
is_consonant = CounterpointRules.is_consonant("A4")  # False

# Get all counterpoint rules
all_cp_rules = CounterpointRules.get_all_rules()
```

### Quiz Generation

```python
teacher = RuleBreakingTeacher()

# Generate a 5-question quiz
quiz = teacher.quiz(num_questions=5)

for question in quiz:
    print(f"Question {question['question_num']}")
    print(f"Rule: {question['rule_tested']}")
    print(f"Has violation: {question['has_violation']}")
    print(f"Hint: {question['hint']}")
```

### Practice Progression

```python
# Get recommended practice sequence
beginner = teacher.suggest_practice_progression("beginner")
# Returns: parallel_fifths, parallel_octaves, range

intermediate = teacher.suggest_practice_progression("intermediate")
# Returns: voice_crossing, spacing, doubled_leading_tone

advanced = teacher.suggest_practice_progression("advanced")
# Returns: voice_overlap, augmented_second
```

## Rule Data Structure

Each rule contains:

```python
{
    "name": str,                    # Human-readable name
    "description": str,             # What the rule says
    "severity": RuleSeverity,       # How important it is
    "context": str,                 # "classical", "jazz", or "contemporary"
    "reason": str,                  # Why the rule exists
    "exception": str,               # When it's okay to break
    "detection": str,               # How to check for violations (optional)
    "example_violation": dict,      # Example breaking the rule (optional)
    "example_correct": dict,        # Correct version (optional)
}
```

## Academic Sources

This module is based on authoritative sources:

### Voice Leading
- J.S. Bach Chorale analysis
- Walter Piston's *Harmony*
- Dmitri Tymoczko's *A Geometry of Music*
- Mark Levine's *The Jazz Theory Book*

### Harmony
- Jean-Philippe Rameau's *Treatise on Harmony*
- Heinrich Schenker's theories
- Hugo Riemann's functional harmony
- Berklee jazz harmony curriculum

### Counterpoint
- Johann Joseph Fux's *Gradus ad Parnassum* (1725)
- Knud Jeppesen's *The Style of Palestrina*
- Kent Kennan's *Counterpoint*

## Integration with Penta Core

The teacher module integrates with Penta Core's analysis engines:

```python
from penta_core import PentaCore
from penta_core.teachers import RuleBreakingTeacher

# Analyze music
penta = PentaCore(sample_rate=48000)
state = penta.get_state()

# Check for rule violations
teacher = RuleBreakingTeacher()
# (Voice leading analysis to be implemented in Week 12)
```

## Running the Example

```bash
cd /workspaces/penta-core
python examples/teacher_example.py
```

This will demonstrate:
1. Voice leading rules by context
2. Filtering by severity
3. Jazz-specific rules
4. Chord construction
5. Functional harmony
6. Species counterpoint
7. Teaching specific rules
8. Chord interval lookup
9. Quiz generation
10. Practice progressions

## Future Enhancements

- [ ] C++ integration for real-time voice leading analysis
- [ ] MIDI file analysis for rule violations
- [ ] Interactive web interface for teaching
- [ ] Audio examples for each rule
- [ ] Machine learning models trained on rulebooks
- [ ] Automatic composition with rule enforcement
- [ ] Real-time feedback in DAW plugin

## License

MIT License - See LICENSE file
