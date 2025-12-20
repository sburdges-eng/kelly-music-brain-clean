"""
Demonstration of Penta Core's advanced music theory rules system.

Features:
- Emotional expression mapping
- Genre-specific timing pockets
- Context-dependent rule severity
- Comprehensive music theory rules
"""

from penta_core.rules import (
    # Core types
    RuleSeverity,
    MusicalContext,
    Emotion,
    # Functions
    get_techniques_for_emotion,
    get_emotions_for_technique,
    get_genre_pocket,
    # Rules
    RhythmRules,
)


def demo_emotional_expression():
    """Demonstrate emotion-to-technique mapping."""
    print("=" * 70)
    print("EMOTIONAL EXPRESSION MAPPING")
    print("=" * 70)
    
    # Get techniques for specific emotions
    print("\n1. Techniques for expressing grief:")
    grief_techniques = get_techniques_for_emotion(Emotion.GRIEF)
    for technique in grief_techniques:
        print(f"   • {technique}")
    
    print("\n2. Techniques for expressing power:")
    power_techniques = get_techniques_for_emotion(Emotion.POWER)
    for technique in power_techniques:
        print(f"   • {technique}")
    
    # Get emotional effects of a technique
    print("\n3. Emotional effects of parallel fifths:")
    emotions = get_emotions_for_technique("parallel_fifths")
    for mapping in emotions:
        print(f"   • {mapping.emotion.name}: intensity {mapping.intensity}/10")
        print(f"     {mapping.explanation}")
    
    print()


def demo_timing_pockets():
    """Demonstrate genre-specific timing characteristics."""
    print("=" * 70)
    print("GENRE TIMING POCKETS")
    print("=" * 70)
    
    # J Dilla's signature pocket
    print("\n1. J Dilla's timing pocket:")
    dilla_pocket = get_genre_pocket("dilla")
    if dilla_pocket:
        print(f"   Swing ratio: {dilla_pocket.swing_ratio:.2f}")
        print(f"   Kick offset: {dilla_pocket.kick_offset_ms:+.1f}ms")
        print(f"   Snare offset: {dilla_pocket.snare_offset_ms:+.1f}ms")
        print(f"   Hi-hat offset: {dilla_pocket.hihat_offset_ms:+.1f}ms")
        print(f"   Humanization: ±{dilla_pocket.humanization_variance:.1f}ms")
        print(f"   Overall feel: {'dragging' if dilla_pocket.push_pull_tendency > 0 else 'rushing'} "
              f"({dilla_pocket.push_pull_tendency:+.1f}ms)")
    
    # Compare different genres
    print("\n2. Comparing genre pockets:")
    genres = ["bebop", "techno", "reggae", "motown"]
    print(f"   {'Genre':<12} {'Swing':<8} {'Kick':<10} {'Feel'}")
    print(f"   {'-'*50}")
    for genre in genres:
        pocket = get_genre_pocket(genre)
        if pocket:
            feel = "dragging" if pocket.push_pull_tendency > 5 else \
                   "rushing" if pocket.push_pull_tendency < -5 else "locked"
            print(f"   {genre:<12} {pocket.swing_ratio:<8.2f} "
                  f"{pocket.kick_offset_ms:+5.1f}ms   {feel}")
    
    print()


def demo_context_dependent_severity():
    """Demonstrate how rules change severity by musical context."""
    print("=" * 70)
    print("CONTEXT-DEPENDENT RULE SEVERITY")
    print("=" * 70)
    
    print("\n1. Creating a context-dependent rule (parallel fifths):")
    
    from penta_core.rules import Rule
    
    parallel_fifths = Rule(
        name="parallel_fifths",
        description="Avoid consecutive perfect fifths in similar motion",
        severity={
            MusicalContext.CLASSICAL: RuleSeverity.STRICT,
            MusicalContext.JAZZ: RuleSeverity.STYLISTIC,
            MusicalContext.CONTEMPORARY: RuleSeverity.MODERN,
        },
        contexts={MusicalContext.CLASSICAL, MusicalContext.JAZZ, MusicalContext.CONTEMPORARY},
        reason="Voice independence vs. harmonic color",
        category="parallel_motion",
    )
    
    # Show how severity changes by context
    contexts_to_test = [
        MusicalContext.CLASSICAL,
        MusicalContext.JAZZ,
        MusicalContext.CONTEMPORARY,
    ]
    
    print("\n   Same rule, different contexts:")
    for context in contexts_to_test:
        severity = parallel_fifths.get_severity_for_context(context)
        print(f"   • {context.name:<15} → {severity.name}")
    
    print("\n   Interpretation:")
    print("   • CLASSICAL: Forbidden (breaks voice independence)")
    print("   • JAZZ: Acceptable (used for harmonic color)")
    print("   • CONTEMPORARY: Encouraged (parallel harmony, quartal voicings)")
    
    print()


def demo_rhythm_rules():
    """Demonstrate rhythm and meter rules."""
    print("=" * 70)
    print("RHYTHM AND METER RULES")
    print("=" * 70)
    
    # Get all rhythm rules
    all_rules = RhythmRules.get_all_rules()
    print(f"\n   Total rhythm rules: {len(all_rules)}")
    
    # Show rules by category
    categories = ["metric_hierarchy", "syncopation", "phrase_structure"]
    for category in categories:
        rules = RhythmRules.get_rules_by_category(category)
        print(f"\n   {category.replace('_', ' ').title()}: {len(rules)} rules")
        for rule in rules[:2]:  # Show first 2
            print(f"   • {rule.name}")
            print(f"     {rule.description}")
    
    # Show context-specific rules
    print(f"\n   Classical-era rhythm rules:")
    classical_rules = RhythmRules.get_rules_by_context(MusicalContext.CLASSICAL)
    for rule in classical_rules[:3]:
        print(f"   • {rule.name} (severity: {rule.severity.name})")
    
    print()


def demo_integration():
    """Show how all systems work together."""
    print("=" * 70)
    print("INTEGRATED EXAMPLE: Creating a 'Grief' piece")
    print("=" * 70)
    
    # 1. Get emotional techniques
    print("\n1. Choose techniques for grief:")
    techniques = get_techniques_for_emotion(Emotion.GRIEF)
    print(f"   Selected: {', '.join(techniques[:3])}")
    
    # 2. Select appropriate timing
    print("\n2. Choose timing pocket:")
    pocket = get_genre_pocket("motown")
    if pocket:
        print(f"   Motown pocket: {pocket.swing_ratio:.2f} swing, "
              f"{pocket.push_pull_tendency:+.1f}ms tendency")
    
    # 3. Apply context-appropriate rules
    print("\n3. Apply romantic-era rules (for emotional depth):")
    romantic_rules = RhythmRules.get_rules_by_context(MusicalContext.ROMANTIC)
    print(f"   Active rhythm rules: {len(romantic_rules)}")
    
    # 4. Demonstrate timing application
    print("\n4. Simulated MIDI timing (C4, downbeats at 0, 500, 1000ms):")
    from penta_core.rules import apply_pocket_to_midi
    
    midi_kicks = [(60, 0), (60, 500), (60, 1000)]
    # For demo, add velocity
    midi_with_vel = [(p, t, 100) for p, t in midi_kicks]
    
    if pocket:
        timed_kicks = apply_pocket_to_midi(midi_with_vel, pocket, "kick")
        print("   Original → With pocket:")
        for orig, modified in zip(midi_with_vel, timed_kicks):
            print(f"   {orig[1]:>6.1f}ms → {modified[1]:>6.1f}ms "
                  f"(offset: {modified[1] - orig[1]:+.1f}ms)")
    
    print()


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "PENTA CORE - ADVANCED RULES DEMO" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    demo_emotional_expression()
    demo_timing_pockets()
    demo_context_dependent_severity()
    demo_rhythm_rules()
    demo_integration()
    
    print("=" * 70)
    print("NEXT STEPS:")
    print("  - Explore voice_leading, harmony, counterpoint rules")
    print("  - Create custom timing pockets for your genre")
    print("  - Map your own emotion → technique relationships")
    print("  - Integrate with C++ real-time analysis engine")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
