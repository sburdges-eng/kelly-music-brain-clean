#!/usr/bin/env python3
"""
Example demonstrating the Rule-Breaking Teacher and comprehensive rulebooks.

This example shows how to:
1. Access comprehensive voice leading, harmony, and counterpoint rules
2. Filter rules by musical context (classical, jazz, contemporary)
3. Filter rules by severity level
4. Use the teacher to demonstrate rule violations
5. Access specific species counterpoint rules
"""

from penta_core.teachers import (
    RuleBreakingTeacher,
    VoiceLeadingRules,
    HarmonyRules,
    CounterpointRules,
    RuleSeverity,
    Species
)


def main():
    print("=" * 70)
    print("Penta Core - Music Theory Rule-Breaking Teacher")
    print("=" * 70)
    print()
    
    # Initialize the teacher
    teacher = RuleBreakingTeacher()
    
    # Example 1: Access voice leading rules by context
    print("1. VOICE LEADING RULES - CLASSICAL CONTEXT")
    print("-" * 70)
    classical_rules = VoiceLeadingRules.get_rules_by_context("classical")
    
    for category, rules in classical_rules.items():
        if rules:  # Only show categories with rules
            print(f"\n{category.replace('_', ' ').title()}:")
            for rule_name, rule_data in list(rules.items())[:2]:  # Show first 2
                print(f"  • {rule_data['name']}")
                print(f"    {rule_data['description']}")
                print(f"    Severity: {rule_data['severity'].value}")
    
    print()
    
    # Example 2: Filter rules by severity
    print("\n2. STRICT RULES ONLY")
    print("-" * 70)
    strict_rules = VoiceLeadingRules.get_rules_by_severity(RuleSeverity.STRICT)
    
    for category, rules in strict_rules.items():
        if rules:
            print(f"\n{category.replace('_', ' ').title()}:")
            for rule_name, rule_data in rules.items():
                print(f"  • {rule_data['name']}")
                print(f"    Reason: {rule_data['reason']}")
    
    print()
    
    # Example 3: Jazz-specific rules
    print("\n3. JAZZ VOICE LEADING RULES")
    print("-" * 70)
    jazz_rules = VoiceLeadingRules.get_rules_by_context("jazz")
    
    if "jazz" in jazz_rules and jazz_rules["jazz"]:
        for rule_name, rule_data in jazz_rules["jazz"].items():
            print(f"\n• {rule_data['name']}")
            print(f"  {rule_data['description']}")
            if "technique" in rule_data:
                print(f"  Technique: {rule_data['technique']}")
    
    print()
    
    # Example 4: Harmony rules - chord construction
    print("\n4. CHORD CONSTRUCTION RULES")
    print("-" * 70)
    harmony_rules = HarmonyRules.get_all_rules()
    
    if "chord_construction" in harmony_rules:
        seventh_chords = harmony_rules["chord_construction"]["seventh_chords"]
        print(f"\n{seventh_chords['name']}")
        print(f"{seventh_chords['description']}\n")
        
        for chord_type, intervals in list(seventh_chords["types"].items())[:3]:
            print(f"  {chord_type}: {intervals}")
    
    print()
    
    # Example 5: Common chord progressions
    print("\n5. FUNCTIONAL HARMONY - PROGRESSIONS")
    print("-" * 70)
    
    if "progressions" in harmony_rules:
        circle_of_fifths = harmony_rules["progressions"]["circle_of_fifths"]
        print(f"\n{circle_of_fifths['name']}")
        print(f"Strength: {circle_of_fifths['strength']}")
        print("\nExamples:")
        for example in circle_of_fifths["examples"][:2]:
            print(f"  • {example}")
    
    print()
    
    # Example 6: Species counterpoint rules
    print("\n6. SPECIES COUNTERPOINT RULES")
    print("-" * 70)
    
    for species in [Species.FIRST, Species.SECOND, Species.FOURTH]:
        print(f"\n{species.name.title()} Species:")
        rules = teacher.get_species_counterpoint_rules(species)
        
        # Show first 2 rules for each species
        for rule_name, rule_data in list(rules.items())[:2]:
            print(f"  • {rule_data['name']}")
            print(f"    {rule_data['description']}")
    
    print()
    
    # Example 7: Use teacher to teach a specific rule
    print("\n7. TEACHING A SPECIFIC RULE")
    print("-" * 70)
    
    lesson = teacher.teach_rule("parallel_fifths")
    print(f"\nRule: {lesson['rule']}")
    print(f"Description: {lesson['description']}")
    print(f"Reason: {lesson['reason']}")
    print(f"Severity: {lesson['severity']}")
    
    if "lesson" in lesson:
        print("\nLesson Examples:")
        for key, value in lesson["lesson"].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print()
    
    # Example 8: Get interval information for chord construction
    print("\n8. CHORD INTERVAL LOOKUP")
    print("-" * 70)
    
    chord_types = ["major", "minor", "dominant7", "half_diminished7"]
    for chord_type in chord_types:
        intervals = HarmonyRules.get_chord_intervals(chord_type)
        if intervals:
            print(f"{chord_type}: {intervals} semitones from root")
    
    print()
    
    # Example 9: Quiz generation
    print("\n9. GENERATED QUIZ")
    print("-" * 70)
    
    quiz = teacher.quiz(num_questions=3)
    print("\nQuiz Questions:\n")
    
    for question in quiz:
        print(f"Question {question['question_num']}:")
        print(f"  Rule tested: {question['rule_tested']}")
        print(f"  Hint: {question['hint']}")
        print(f"  Has violation: {question['has_violation']}")
        print()
    
    # Example 10: Practice progression suggestion
    print("\n10. SUGGESTED PRACTICE PROGRESSION")
    print("-" * 70)
    
    for difficulty in ["beginner", "intermediate", "advanced"]:
        progression = teacher.suggest_practice_progression(difficulty)
        print(f"\n{difficulty.upper()} Level:")
        print(f"  Rules: {', '.join(progression['rule_sequence'])}")
        print(f"  Estimated time: {progression['estimated_time']}")
    
    print()
    print("=" * 70)
    print("For more information, see the comprehensive rulebooks:")
    print("  - voice_leading_rules.py (172 rules)")
    print("  - harmony_rules.py (Chord construction, progressions, jazz)")
    print("  - counterpoint_rules.py (Species counterpoint, Fux)")
    print("=" * 70)


if __name__ == "__main__":
    main()
