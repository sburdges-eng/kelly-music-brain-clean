"""
DAiW CLI - Command line interface for Music Brain toolkit

Usage:
    daiw extract <midi_file>                    Extract groove from MIDI
    daiw apply --genre <genre> <midi_file>      Apply groove template
    daiw humanize <midi_file> [options]         Apply drum humanization (Drunken Drummer)
    daiw analyze --chords <midi_file>           Analyze chord progression
    daiw diagnose <progression>                 Diagnose harmonic issues
    daiw reharm <progression> [--style <style>] Generate reharmonizations
    daiw teach <topic>                          Interactive teaching mode

    daiw intent new [--title <title>]           Create new intent template
    daiw intent process <file>                  Generate elements from intent
    daiw intent suggest <emotion>               Suggest rules to break
    daiw intent list                            List all rule-breaking options
    daiw intent validate <file>                 Validate intent file

    daiw learn instruments                      List all supported instruments
    daiw learn sources <instrument>             Show learning sources for an instrument
    daiw learn plan <instrument>                Generate a learning plan
    daiw learn prompt <instrument> <topic>      Generate AI teaching prompt
    daiw learn curriculum <instrument>          Show curriculum structure

Humanization Styles:
    tight   - Minimal drift, confident (complexity=0.1, vulnerability=0.2)
    natural - Human feel, balanced (complexity=0.4, vulnerability=0.5)
    loose   - Relaxed, laid back (complexity=0.6, vulnerability=0.6)
    drunk   - Maximum chaos, fragile (complexity=0.9, vulnerability=0.8)
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Lazy imports to speed up CLI startup
def get_groove_module():
    from music_brain.groove import extract_groove, apply_groove
    return extract_groove, apply_groove


def get_humanize_module():
    from music_brain.groove import (
        humanize_midi_file, GrooveSettings, settings_from_intent, quick_humanize,
        list_presets, settings_from_preset, get_preset
    )
    return (humanize_midi_file, GrooveSettings, settings_from_intent, quick_humanize,
            list_presets, settings_from_preset, get_preset)

def get_structure_module():
    from music_brain.structure import analyze_chords, detect_sections
    return analyze_chords, detect_sections

def get_session_module():
    from music_brain.session import RuleBreakingTeacher
    return RuleBreakingTeacher


def get_intent_module():
    from music_brain.session.intent_schema import (
        CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints, 
        SystemDirective, suggest_rule_break, validate_intent, list_all_rules
    )
    from music_brain.session.intent_processor import IntentProcessor, process_intent
    return (CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
            SystemDirective, suggest_rule_break, validate_intent, list_all_rules,
            IntentProcessor, process_intent)


def get_audio_module():
    from music_brain.audio import (
        analyze_feel, ChordDetector, analyze_frequency_bands,
        analyze_reference, compare_frequency_profiles, suggest_eq_adjustments
    )
    return (analyze_feel, ChordDetector, analyze_frequency_bands,
            analyze_reference, compare_frequency_profiles, suggest_eq_adjustments)


def get_arrangement_module():
    from music_brain.arrangement import generate_arrangement, ArrangementGenerator
    return generate_arrangement, ArrangementGenerator


def get_learning_module():
    from music_brain.learning import (
        DifficultyLevel, SkillCategory, Curriculum, LearningPath, CurriculumBuilder,
        ResourceFetcher, KNOWN_SOURCES, get_recommended_sources, generate_learning_plan,
        InstrumentFamily, Instrument, INSTRUMENTS, get_instrument, get_instruments_by_family,
        get_beginner_instruments, TeachingStyle, StudentProfile, AdaptiveTeacher,
        PedagogyEngine, generate_ai_teaching_prompt,
    )
    return {
        'DifficultyLevel': DifficultyLevel,
        'SkillCategory': SkillCategory,
        'Curriculum': Curriculum,
        'LearningPath': LearningPath,
        'CurriculumBuilder': CurriculumBuilder,
        'ResourceFetcher': ResourceFetcher,
        'KNOWN_SOURCES': KNOWN_SOURCES,
        'get_recommended_sources': get_recommended_sources,
        'generate_learning_plan': generate_learning_plan,
        'InstrumentFamily': InstrumentFamily,
        'Instrument': Instrument,
        'INSTRUMENTS': INSTRUMENTS,
        'get_instrument': get_instrument,
        'get_instruments_by_family': get_instruments_by_family,
        'get_beginner_instruments': get_beginner_instruments,
        'TeachingStyle': TeachingStyle,
        'StudentProfile': StudentProfile,
        'AdaptiveTeacher': AdaptiveTeacher,
        'PedagogyEngine': PedagogyEngine,
        'generate_ai_teaching_prompt': generate_ai_teaching_prompt,
    }


def cmd_extract(args):
    """Extract groove from MIDI file."""
    extract_groove, _ = get_groove_module()
    
    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return 1
    
    print(f"Extracting groove from: {midi_path}")
    groove = extract_groove(str(midi_path))
    
    output_path = midi_path.stem + "_groove.json"
    if args.output:
        output_path = args.output
    
    with open(output_path, 'w') as f:
        json.dump(groove.to_dict(), f, indent=2)
    
    print(f"Groove saved to: {output_path}")
    print(f"  Timing deviation: {groove.timing_stats['mean_deviation_ms']:.1f}ms avg")
    print(f"  Velocity range: {groove.velocity_stats['min']}-{groove.velocity_stats['max']}")
    print(f"  Swing factor: {groove.swing_factor:.2f}")
    return 0


def cmd_apply(args):
    """Apply groove template to MIDI file."""
    _, apply_groove = get_groove_module()

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return 1

    print(f"Applying {args.genre} groove to: {midi_path}")

    output_path = args.output or f"{midi_path.stem}_grooved.mid"
    apply_groove(str(midi_path), genre=args.genre, output=output_path, intensity=args.intensity)

    print(f"Output saved to: {output_path}")
    return 0


def cmd_humanize(args):
    """Apply drum humanization to MIDI file."""
    (humanize_midi_file, GrooveSettings, settings_from_intent, quick_humanize,
     list_presets, settings_from_preset, get_preset) = get_humanize_module()

    # Handle list-presets subcommand
    if hasattr(args, 'list_presets') and args.list_presets:
        presets = list_presets()
        print("\n=== Available Humanization Presets ===\n")
        for preset_name in sorted(presets):
            preset_data = get_preset(preset_name)
            desc = preset_data.get("description", "No description")
            groove = preset_data.get("groove_settings", {})
            c = groove.get("complexity", 0.5)
            v = groove.get("vulnerability", 0.5)
            print(f"  {preset_name}")
            print(f"    {desc}")
            print(f"    complexity={c:.2f}, vulnerability={v:.2f}")
            print()
        return 0

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return 1

    # Determine humanization approach
    if args.preset:
        # Load from preset
        try:
            settings = settings_from_preset(args.preset)
            preset_data = get_preset(args.preset)
            complexity = settings.complexity
            vulnerability = settings.vulnerability
            print(f"Humanizing drums with '{args.preset}' preset: {midi_path}")
            print(f"  ({preset_data.get('description', '')})")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif args.style:
        # Quick style preset
        print(f"Humanizing drums with '{args.style}' style: {midi_path}")
        complexity_map = {
            "tight": (0.1, 0.2),
            "natural": (0.4, 0.5),
            "loose": (0.6, 0.6),
            "drunk": (0.9, 0.8),
        }
        complexity, vulnerability = complexity_map.get(args.style, (0.4, 0.5))
        settings = GrooveSettings(complexity=complexity, vulnerability=vulnerability)
    else:
        # Manual complexity/vulnerability
        complexity = args.complexity
        vulnerability = args.vulnerability
        print(f"Humanizing drums (complexity={complexity:.2f}, vulnerability={vulnerability:.2f}): {midi_path}")
        settings = GrooveSettings(complexity=complexity, vulnerability=vulnerability)

    # Apply optional overrides
    if args.no_ghost_notes:
        settings.enable_ghost_notes = False

    output_path = args.output or f"{midi_path.stem}_humanized.mid"

    result_path = humanize_midi_file(
        input_path=str(midi_path),
        output_path=output_path,
        complexity=complexity,
        vulnerability=vulnerability,
        drum_channel=args.channel,
        settings=settings,
        seed=args.seed,
    )

    print(f"\n=== Drum Humanization Applied ===")
    print(f"  Complexity:    {complexity:.2f} (timing chaos)")
    print(f"  Vulnerability: {vulnerability:.2f} (dynamic fragility)")
    print(f"  Ghost notes:   {'enabled' if settings.enable_ghost_notes else 'disabled'}")
    print(f"  Output:        {result_path}")

    return 0


def cmd_analyze(args):
    """Analyze chord progression in MIDI file."""
    analyze_chords, detect_sections = get_structure_module()
    
    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return 1
    
    if args.chords:
        print(f"Analyzing chords in: {midi_path}")
        progression = analyze_chords(str(midi_path))
        
        print("\n=== Chord Analysis ===")
        print(f"Key: {progression.key}")
        print(f"Progression: {' - '.join(progression.chords)}")
        print(f"Roman numerals: {' - '.join(progression.roman_numerals)}")
        
        if progression.borrowed_chords:
            print(f"\nBorrowed chords detected:")
            for chord, source in progression.borrowed_chords.items():
                print(f"  {chord} ← borrowed from {source}")
    
    if args.sections:
        print(f"\nDetecting sections in: {midi_path}")
        sections = detect_sections(str(midi_path))
        
        print("\n=== Section Analysis ===")
        for section in sections:
            print(f"  {section.name}: bars {section.start_bar}-{section.end_bar} (energy: {section.energy:.2f})")
    
    return 0


def cmd_diagnose(args):
    """Diagnose issues in a chord progression string."""
    from music_brain.structure.progression import diagnose_progression
    
    progression = args.progression
    print(f"Diagnosing: {progression}")
    
    diagnosis = diagnose_progression(progression)
    
    print("\n=== Harmonic Diagnosis ===")
    print(f"Key estimate: {diagnosis['key']}")
    print(f"Mode: {diagnosis['mode']}")
    
    if diagnosis['issues']:
        print("\nPotential issues:")
        for issue in diagnosis['issues']:
            print(f"  ⚠ {issue}")
    else:
        print("\n✓ No obvious issues detected")
    
    if diagnosis['suggestions']:
        print("\nSuggestions:")
        for suggestion in diagnosis['suggestions']:
            print(f"  → {suggestion}")
    
    return 0


def cmd_reharm(args):
    """Generate reharmonization suggestions."""
    from music_brain.structure.progression import generate_reharmonizations
    
    progression = args.progression
    style = args.style or "jazz"
    
    print(f"Reharmonizing: {progression}")
    print(f"Style: {style}")
    
    suggestions = generate_reharmonizations(progression, style=style, count=args.count)
    
    print("\n=== Reharmonization Suggestions ===")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {' - '.join(suggestion['chords'])}")
        print(f"   Technique: {suggestion['technique']}")
        print(f"   Mood shift: {suggestion['mood']}")
    
    return 0


def cmd_teach(args):
    """Interactive teaching mode."""
    RuleBreakingTeacher = get_session_module()
    
    topic = args.topic.lower().replace("-", "_").replace(" ", "_")
    
    valid_topics = [
        "rulebreaking", "rule_breaking", "borrowed", "borrowed_chords",
        "modal_mixture", "substitutions", "rhythm", "production"
    ]
    
    if topic not in valid_topics:
        print(f"Unknown topic: {args.topic}")
        print(f"Available topics: {', '.join(valid_topics)}")
        return 1
    
    teacher = RuleBreakingTeacher()
    
    if args.quick:
        # Quick single lesson
        teacher.quick_lesson(topic)
    else:
        # Interactive mode
        teacher.interactive_session(topic)
    
    return 0


def cmd_intent(args):
    """Handle intent-based song generation."""
    (CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints,
     SystemDirective, suggest_rule_break, validate_intent, list_all_rules,
     IntentProcessor, process_intent) = get_intent_module()
    
    if args.subcommand == 'new':
        # Create new intent from template
        print("Creating new song intent...")
        
        intent = CompleteSongIntent(
            title=args.title or "Untitled Song",
            song_root=SongRoot(
                core_event="[What happened?]",
                core_resistance="[What holds you back?]",
                core_longing="[What do you want to feel?]",
                core_stakes="Personal",
                core_transformation="[How should you feel at the end?]",
            ),
            song_intent=SongIntent(
                mood_primary="[Primary emotion]",
                mood_secondary_tension=0.5,
                imagery_texture="[Visual/tactile quality]",
                vulnerability_scale="Medium",
                narrative_arc="Climb-to-Climax",
            ),
            technical_constraints=TechnicalConstraints(
                technical_genre="[Genre]",
                technical_tempo_range=(80, 120),
                technical_key="F",
                technical_mode="major",
                technical_groove_feel="Organic/Breathing",
                technical_rule_to_break="",
                rule_breaking_justification="",
            ),
            system_directive=SystemDirective(
                output_target="Chord progression",
                output_feedback_loop="Harmony",
            ),
        )
        
        output = args.output or "song_intent.json"
        intent.save(output)
        print(f"Template saved to: {output}")
        print("\nEdit the file to fill in your intent, then run:")
        print(f"  daiw intent process {output}")
    
    elif args.subcommand == 'process':
        # Process intent to generate elements
        if not args.file:
            print("Error: Please specify an intent file")
            return 1
        
        intent_path = Path(args.file)
        if not intent_path.exists():
            print(f"Error: File not found: {intent_path}")
            return 1
        
        print(f"Processing intent: {intent_path}")
        intent = CompleteSongIntent.load(str(intent_path))
        
        # Validate first
        issues = validate_intent(intent)
        if issues and not args.force:
            print("\n⚠️  Intent validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nFix issues or use --force to proceed anyway")
            return 1
        
        # Process
        result = process_intent(intent)
        
        # Display results
        print("\n" + "=" * 60)
        print("🎵 GENERATED ELEMENTS")
        print("=" * 60)
        
        # Harmony
        harmony = result['harmony']
        print(f"\n📌 HARMONY ({harmony.rule_broken})")
        print(f"   Progression: {' - '.join(harmony.chords)}")
        print(f"   Roman: {' - '.join(harmony.roman_numerals)}")
        print(f"   Effect: {harmony.rule_effect}")
        
        # Groove
        groove = result['groove']
        print(f"\n📌 GROOVE ({groove.rule_broken})")
        print(f"   Pattern: {groove.pattern_name}")
        print(f"   Tempo: {groove.tempo_bpm} BPM")
        print(f"   Effect: {groove.rule_effect}")
        
        # Arrangement
        arr = result['arrangement']
        print(f"\n📌 ARRANGEMENT ({arr.rule_broken})")
        for section in arr.sections:
            print(f"   {section['name']}: {section['bars']} bars @ {section['energy']:.0%} energy")
        
        # Production
        prod = result['production']
        print(f"\n📌 PRODUCTION ({prod.rule_broken})")
        print(f"   Vocal: {prod.vocal_treatment}")
        for note in prod.eq_notes[:2]:
            print(f"   EQ: {note}")
        
        print("\n" + "=" * 60)
        
        # Save output if requested
        if args.output:
            import json
            output_data = {
                "intent_summary": result['intent_summary'],
                "harmony": {
                    "chords": harmony.chords,
                    "roman_numerals": harmony.roman_numerals,
                    "rule_broken": harmony.rule_broken,
                    "effect": harmony.rule_effect,
                },
                "groove": {
                    "pattern": groove.pattern_name,
                    "tempo": groove.tempo_bpm,
                    "swing": groove.swing_factor,
                },
                "arrangement": {
                    "sections": arr.sections,
                    "dynamic_arc": arr.dynamic_arc,
                },
                "production": {
                    "vocal_treatment": prod.vocal_treatment,
                    "eq_notes": prod.eq_notes,
                    "dynamics_notes": prod.dynamics_notes,
                },
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nOutput saved to: {args.output}")
    
    elif args.subcommand == 'suggest':
        # Suggest rules to break based on emotion
        emotion = args.emotion
        suggestions = suggest_rule_break(emotion)
        
        print(f"\n🎯 Suggested rules to break for '{emotion}':\n")
        
        if not suggestions:
            print(f"  No specific suggestions for '{emotion}'")
            print("  Try: grief, anger, nostalgia, defiance, dissociation")
        else:
            for i, sug in enumerate(suggestions, 1):
                print(f"{i}. {sug['rule']}")
                print(f"   What: {sug['description']}")
                print(f"   Effect: {sug['effect']}")
                print(f"   Use when: {sug['use_when']}")
                print()
    
    elif args.subcommand == 'list':
        # List all available rules
        rules = list_all_rules()
        
        print("\n📋 Available Rule-Breaking Options:\n")
        for category, rule_list in rules.items():
            print(f"  {category}:")
            for rule in rule_list:
                print(f"    - {rule}")
            print()
    
    elif args.subcommand == 'validate':
        # Validate an intent file
        if not args.file:
            print("Error: Please specify an intent file")
            return 1
        
        intent_path = Path(args.file)
        if not intent_path.exists():
            print(f"Error: File not found: {intent_path}")
            return 1
        
        intent = CompleteSongIntent.load(str(intent_path))
        issues = validate_intent(intent)
        
        if issues:
            print("\n⚠️  Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        else:
            print("✅ Intent is valid!")
            return 0
    
    return 0


def cmd_audio(args):
    """Audio analysis commands."""
    if args.subcommand == 'analyze':
        return cmd_audio_analyze(args)
    elif args.subcommand == 'detect-chords':
        return cmd_audio_detect_chords(args)
    elif args.subcommand == 'frequency':
        return cmd_audio_frequency(args)
    elif args.subcommand == 'reference':
        return cmd_audio_reference(args)
    else:
        print("Error: Unknown audio subcommand")
        return 1


def cmd_audio_analyze(args):
    """Analyze audio file feel characteristics."""
    (analyze_feel, _, _, _, _, _) = get_audio_module()
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return 1
    
    print(f"Analyzing: {audio_path}")
    try:
        features = analyze_feel(str(audio_path))
        
        print(f"\nTempo: {features.tempo_bpm:.1f} BPM (confidence: {features.tempo_confidence:.2f})")
        print(f"Duration: {features.duration_seconds:.1f}s")
        print(f"Beats detected: {len(features.beat_positions)}")
        print(f"\nEnergy:")
        print(f"  RMS mean: {features.rms_mean:.4f}")
        print(f"  Dynamic range: {features.dynamic_range_db:.1f} dB")
        print(f"\nSpectral:")
        print(f"  Brightness (centroid): {features.spectral_centroid_mean:.0f} Hz")
        print(f"  Rolloff: {features.spectral_rolloff_mean:.0f} Hz")
        print(f"\nFeel:")
        print(f"  Swing estimate: {features.swing_estimate:.2f} (0=straight, 1=swung)")
        print(f"  Groove regularity: {features.groove_regularity:.2f} (0=loose, 1=tight)")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(features.to_dict(), f, indent=2)
            print(f"\nFull analysis saved to: {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return 1


def cmd_audio_detect_chords(args):
    """Detect chords from audio file."""
    (_, ChordDetector, _, _, _, _) = get_audio_module()
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return 1
    
    print(f"Detecting chords in: {audio_path}")
    try:
        detector = ChordDetector(
            window_size=args.window_size,
            min_confidence=args.min_confidence,
        )
        result = detector.detect_progression(str(audio_path))
        
        print(f"\nDetected {len(result.chords)} chords:")
        print(f"Estimated key: {result.estimated_key or 'Unknown'}")
        print(f"Chord sequence: {'-'.join(result.chord_sequence)}")
        print(f"\nChord timeline:")
        for chord in result.chords:
            print(f"  {chord.start_time:.1f}s - {chord.end_time:.1f}s: "
                  f"{chord.chord_name} (confidence: {chord.confidence:.2f})")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nChord analysis saved to: {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error detecting chords: {e}")
        return 1


def cmd_audio_frequency(args):
    """Analyze 8-band frequency profile."""
    (_, _, analyze_frequency_bands, _, _, _) = get_audio_module()
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return 1
    
    print(f"Analyzing frequency profile: {audio_path}")
    try:
        profile = analyze_frequency_bands(str(audio_path))
        
        print(f"\n8-Band Frequency Analysis:")
        print(f"  Sub-bass (20-60Hz):    {profile.sub_bass:.2f}")
        print(f"  Bass (60-250Hz):       {profile.bass:.2f}")
        print(f"  Low-mids (250-500Hz):  {profile.low_mids:.2f}")
        print(f"  Mids (500-2kHz):       {profile.mids:.2f}")
        print(f"  Upper-mids (2-4kHz):   {profile.upper_mids:.2f}")
        print(f"  Presence (4-6kHz):     {profile.presence:.2f}")
        print(f"  Brilliance (6-12kHz):  {profile.brilliance:.2f}")
        print(f"  Air (12-20kHz):        {profile.air:.2f}")
        
        print(f"\nCharacteristics:")
        print(f"  Brightness: {profile.brightness:.2f}")
        print(f"  Warmth: {profile.warmth:.2f}")
        print(f"  Clarity: {profile.clarity:.2f}")
        
        print(f"\nProduction Notes:")
        for note in profile.get_production_notes():
            print(f"  - {note}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            print(f"\nFrequency analysis saved to: {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error analyzing frequency: {e}")
        return 1


def cmd_audio_reference(args):
    """Analyze reference track DNA."""
    (_, _, _, analyze_reference, _, _) = get_audio_module()
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return 1
    
    print(f"Analyzing reference track: {audio_path}")
    try:
        profile = analyze_reference(audio_path)
        
        if profile is None:
            print("Error: Could not analyze reference track")
            return 1
        
        print(f"\nReference DNA:")
        print(f"  Tempo: {profile.tempo_bpm:.1f} BPM")
        key_str = f"{profile.key_root} {profile.key_mode}" if profile.key_root else "Unknown"
        print(f"  Key: {key_str}")
        print(f"  Brightness: {profile.brightness:.2f} (0=dark, 1=bright)")
        print(f"  Energy: {profile.energy:.2f} (0=calm, 1=intense)")
        print(f"  Warmth: {profile.warmth:.2f} (0=thin, 1=warm)")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "tempo_bpm": profile.tempo_bpm,
                    "key_root": profile.key_root,
                    "key_mode": profile.key_mode,
                    "brightness": profile.brightness,
                    "energy": profile.energy,
                    "warmth": profile.warmth,
                }, f, indent=2)
            print(f"\nReference DNA saved to: {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error analyzing reference: {e}")
        return 1


def cmd_arrange(args):
    """Arrangement generation commands."""
    if args.subcommand == 'generate':
        return cmd_arrange_generate(args)
    elif args.subcommand == 'templates':
        return cmd_arrange_templates(args)
    else:
        print("Error: Unknown arrange subcommand")
        return 1


def cmd_arrange_generate(args):
    """Generate song arrangement."""
    generate_arrangement, ArrangementGenerator = get_arrangement_module()
    
    print(f"Generating {args.genre} arrangement...")
    
    # Parse chord progression if provided
    chords = None
    if args.chords:
        chords = args.chords.split('-')
    
    try:
        arrangement = generate_arrangement(
            genre=args.genre,
            emotion=args.emotion,
            chords=chords,
            intensity=args.intensity,
        )
        
        print(f"\nGenerated Arrangement:")
        print(f"  Genre: {arrangement.template.genre}")
        print(f"  Tempo: {arrangement.template.tempo_bpm} BPM")
        print(f"  Sections: {len(arrangement.template.sections)}")
        print(f"  Total bars: {arrangement.template.total_bars}")
        print(f"  Narrative arc: {arrangement.energy_arc.narrative_arc.value}")
        
        print(f"\nProduction Notes:")
        for note in arrangement.get_production_notes():
            print(f"  {note}")
        
        # Export if requested
        if args.output:
            generator = ArrangementGenerator()
            
            # Export arrangement markers
            markers_path = args.output.replace('.mid', '_markers.mid')
            generator.export_arrangement_markers(arrangement, markers_path)
            print(f"\nArrangement markers exported to: {markers_path}")
            
            # Export bass track if requested
            if args.bass:
                bass_path = args.output.replace('.mid', '_bass.mid')
                generator.export_bass_track(arrangement, bass_path)
                print(f"Bass track exported to: {bass_path}")
            
            # Export arrangement data
            json_path = args.output.replace('.mid', '.json')
            with open(json_path, 'w') as f:
                json.dump(arrangement.to_dict(), f, indent=2)
            print(f"Arrangement data saved to: {json_path}")
        
        return 0
    except Exception as e:
        print(f"Error generating arrangement: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_arrange_templates(args):
    """List available arrangement templates."""
    from music_brain.arrangement.templates import list_available_genres
    
    genres = list_available_genres()
    print("Available arrangement templates:")
    for genre in genres:
        print(f"  - {genre}")
    return 0


def cmd_learn(args):
    """Handle learning module commands."""
    learning = get_learning_module()

    if args.subcommand == 'instruments':
        # List all instruments
        instruments = learning['INSTRUMENTS']

        print("\n=== Supported Instruments ===\n")

        # Group by family
        by_family = {}
        for inst in instruments.values():
            family = inst.family.name
            if family not in by_family:
                by_family[family] = []
            by_family[family].append(inst)

        for family, insts in sorted(by_family.items()):
            print(f"{family}:")
            for inst in sorted(insts, key=lambda x: x.name):
                beginner = "✓" if inst.beginner_friendly else " "
                days = inst.days_to_first_song
                print(f"  [{beginner}] {inst.name:<20} ({days} days to first song)")
            print()

        print("Legend: [✓] = Beginner-friendly")
        print("\nUse 'daiw learn sources <instrument>' to see learning resources")

    elif args.subcommand == 'sources':
        # Show learning sources for an instrument
        if not args.instrument:
            print("Error: Please specify an instrument")
            return 1

        instrument = args.instrument.lower()
        get_recommended_sources = learning['get_recommended_sources']
        get_instrument = learning['get_instrument']

        # Get instrument info
        inst = get_instrument(instrument)
        if inst:
            print(f"\n=== Learning Resources for {inst.name} ===\n")
            print(f"Beginner-friendly: {'Yes' if inst.beginner_friendly else 'No'}")
            print(f"Days to first song: {inst.days_to_first_song}")
            print(f"Months to intermediate: {inst.months_to_intermediate}")
            print(f"Primary genres: {', '.join(inst.primary_genres)}")
        else:
            print(f"\n=== Learning Resources for {instrument.title()} ===\n")

        # Get sources
        sources = get_recommended_sources(instrument, difficulty=args.level or 1)

        if not sources:
            print(f"\nNo specific sources found for '{instrument}'.")
            print("Try one of: guitar, piano, drums, bass, voice, violin, flute")
            return 1

        print(f"\nRecommended sources (Level {args.level or 1}):\n")
        for i, source in enumerate(sources, 1):
            quality = "★" * min(source.get('quality_score', 5), 10)
            print(f"{i}. {source['name']}")
            print(f"   URL: {source['base_url']}")
            print(f"   Quality: {quality}")
            print(f"   Difficulty: {source.get('difficulty_range', (1, 10))}")
            print(f"   {source.get('description', '')}")
            print()

    elif args.subcommand == 'plan':
        # Generate a learning plan
        if not args.instrument:
            print("Error: Please specify an instrument")
            return 1

        generate_learning_plan = learning['generate_learning_plan']
        get_instrument = learning['get_instrument']

        instrument = args.instrument.lower()
        current = args.current or 1
        target = args.target or 5
        hours = args.hours or 5.0

        inst = get_instrument(instrument)
        if inst:
            print(f"\n=== Learning Plan for {inst.name} ===\n")
        else:
            print(f"\n=== Learning Plan for {instrument.title()} ===\n")

        plan = generate_learning_plan(
            instrument=instrument,
            current_level=current,
            target_level=target,
            weekly_hours=hours,
        )

        print(f"Current Level: {current} → Target Level: {target}")
        print(f"Weekly Practice: {hours} hours\n")

        for phase in plan['phases']:
            level_name = phase['level_name']
            weeks = phase['estimated_weeks']
            print(f"LEVEL {phase['level']}: {level_name} (~{weeks} weeks)")
            print(f"  Focus: {', '.join(phase['focus_areas'])}")

            if phase['recommended_sources']:
                sources = phase['recommended_sources'][:2]
                source_names = [s['name'] for s in sources]
                print(f"  Resources: {', '.join(source_names)}")
            print()

        # Show instrument-specific tips if available
        if inst:
            print("=== First Skills to Master ===")
            for skill in inst.first_skills[:5]:
                print(f"  • {skill}")

            print("\n=== Common Challenges ===")
            for challenge in inst.common_challenges[:3]:
                print(f"  • {challenge}")

            print("\n=== Practice Tips ===")
            for tip in inst.practice_tips[:3]:
                print(f"  • {tip}")

    elif args.subcommand == 'prompt':
        # Generate AI teaching prompt
        if not args.instrument or not args.topic:
            print("Error: Please specify both instrument and topic")
            return 1

        generate_ai_teaching_prompt = learning['generate_ai_teaching_prompt']
        StudentProfile = learning['StudentProfile']

        instrument = args.instrument
        topic = args.topic
        action = args.action or 'explain'
        difficulty = args.level or 5

        # Create a default student profile if needed
        student = None
        if args.personalize:
            student = StudentProfile(
                id="cli_student",
                name="Student",
                age=args.age or 25,
                experience_level=difficulty,
            )

        prompt = generate_ai_teaching_prompt(
            action=action,
            instrument=instrument,
            topic=topic,
            student=student,
            difficulty=difficulty,
        )

        print("\n=== AI Teaching Prompt ===\n")
        print(prompt)
        print("\n" + "=" * 50)
        print("\nUse this prompt with any AI assistant to get teaching content.")

        # Optionally save to file
        if args.output:
            with open(args.output, 'w') as f:
                f.write(prompt)
            print(f"\nPrompt saved to: {args.output}")

    elif args.subcommand == 'curriculum':
        # Show curriculum structure for an instrument
        if not args.instrument:
            print("Error: Please specify an instrument")
            return 1

        get_instrument = learning['get_instrument']

        instrument = args.instrument.lower()
        inst = get_instrument(instrument)

        if not inst:
            print(f"Unknown instrument: {instrument}")
            return 1

        print(f"\n=== Curriculum Structure for {inst.name} ===\n")
        print(f"Beginner-friendly: {'Yes' if inst.beginner_friendly else 'No'}")
        print(f"Days to first song: {inst.days_to_first_song}")
        print(f"Months to intermediate: {inst.months_to_intermediate}")
        print(f"\nFirst Skills:")
        for skill in inst.first_skills:
            print(f"  • {skill}")
        print(f"\nCommon Challenges:")
        for challenge in inst.common_challenges:
            print(f"  • {challenge}")

    else:
        print("Error: Unknown learn subcommand")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='daiw',
        description='DAiW - Digital Audio intelligent Workstation CLI'
    )
    parser.add_argument('--version', action='version', version='%(prog)s 0.2.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract groove from MIDI')
    extract_parser.add_argument('midi_file', help='MIDI file to extract from')
    extract_parser.add_argument('-o', '--output', help='Output JSON file')
    
    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply groove template')
    apply_parser.add_argument('midi_file', help='MIDI file to process')
    apply_parser.add_argument('-g', '--genre', default='funk',
                              choices=['funk', 'jazz', 'rock', 'hiphop', 'edm', 'latin'],
                              help='Genre groove template')
    apply_parser.add_argument('-o', '--output', help='Output MIDI file')
    apply_parser.add_argument('-i', '--intensity', type=float, default=0.5,
                              help='Groove intensity 0.0-1.0')

    # Humanize command (Drunken Drummer)
    humanize_parser = subparsers.add_parser('humanize', help='Apply drum humanization (Drunken Drummer)')
    humanize_parser.add_argument('midi_file', nargs='?', help='MIDI file to humanize')
    humanize_parser.add_argument('-o', '--output', help='Output MIDI file')
    humanize_parser.add_argument('-p', '--preset',
                                 help='Emotional preset (use --list-presets to see options)')
    humanize_parser.add_argument('-s', '--style',
                                 choices=['tight', 'natural', 'loose', 'drunk'],
                                 help='Quick style preset')
    humanize_parser.add_argument('-c', '--complexity', type=float, default=0.5,
                                 help='Timing chaos 0.0-1.0 (ignored if --style/--preset used)')
    humanize_parser.add_argument('-v', '--vulnerability', type=float, default=0.5,
                                 help='Dynamic fragility 0.0-1.0 (ignored if --style/--preset used)')
    humanize_parser.add_argument('--channel', type=int, default=9,
                                 help='MIDI drum channel (default: 9, i.e. channel 10)')
    humanize_parser.add_argument('--no-ghost-notes', action='store_true',
                                 help='Disable ghost note generation')
    humanize_parser.add_argument('--seed', type=int,
                                 help='Random seed for reproducibility')
    humanize_parser.add_argument('--list-presets', action='store_true',
                                 help='List available emotional presets')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze MIDI file')
    analyze_parser.add_argument('midi_file', help='MIDI file to analyze')
    analyze_parser.add_argument('-c', '--chords', action='store_true', help='Analyze chords')
    analyze_parser.add_argument('-s', '--sections', action='store_true', help='Detect sections')
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Diagnose chord progression')
    diagnose_parser.add_argument('progression', help='Chord progression (e.g., "F-C-Am-Dm")')
    
    # Reharm command
    reharm_parser = subparsers.add_parser('reharm', help='Generate reharmonizations')
    reharm_parser.add_argument('progression', help='Chord progression to reharmonize')
    reharm_parser.add_argument('-s', '--style', default='jazz',
                               choices=['jazz', 'pop', 'rnb', 'classical', 'experimental'],
                               help='Reharmonization style')
    reharm_parser.add_argument('-n', '--count', type=int, default=3,
                               help='Number of suggestions')
    
    # Teach command
    teach_parser = subparsers.add_parser('teach', help='Interactive teaching mode')
    teach_parser.add_argument('topic', help='Topic to learn (rulebreaking, borrowed, etc.)')
    teach_parser.add_argument('-q', '--quick', action='store_true', help='Quick single lesson')
    
    # Intent command with subcommands
    intent_parser = subparsers.add_parser('intent', help='Intent-based song generation')
    intent_subparsers = intent_parser.add_subparsers(dest='subcommand', help='Intent commands')
    
    # intent new
    intent_new = intent_subparsers.add_parser('new', help='Create new intent template')
    intent_new.add_argument('-t', '--title', help='Song title')
    intent_new.add_argument('-o', '--output', help='Output file (default: song_intent.json)')
    
    # intent process
    intent_process = intent_subparsers.add_parser('process', help='Process intent to generate elements')
    intent_process.add_argument('file', help='Intent JSON file')
    intent_process.add_argument('-o', '--output', help='Save output to JSON')
    intent_process.add_argument('-f', '--force', action='store_true', help='Proceed despite validation issues')
    
    # intent suggest
    intent_suggest = intent_subparsers.add_parser('suggest', help='Suggest rules to break')
    intent_suggest.add_argument('emotion', help='Target emotion (grief, anger, nostalgia, etc.)')
    
    # intent list
    intent_subparsers.add_parser('list', help='List all rule-breaking options')
    
    # intent validate
    intent_validate = intent_subparsers.add_parser('validate', help='Validate intent file')
    intent_validate.add_argument('file', help='Intent JSON file')
    
    # Audio analysis commands
    audio_parser = subparsers.add_parser('audio', help='Audio analysis tools')
    audio_subparsers = audio_parser.add_subparsers(dest='subcommand', help='Audio commands')
    
    # audio analyze
    audio_analyze = audio_subparsers.add_parser('analyze', help='Analyze audio feel and characteristics')
    audio_analyze.add_argument('audio_file', help='Audio file (WAV, MP3, FLAC, etc.)')
    audio_analyze.add_argument('-o', '--output', help='Save analysis to JSON file')
    
    # audio detect-chords
    audio_chords = audio_subparsers.add_parser('detect-chords', help='Detect chord progression from audio')
    audio_chords.add_argument('audio_file', help='Audio file')
    audio_chords.add_argument('-o', '--output', help='Save chord analysis to JSON')
    audio_chords.add_argument('-w', '--window-size', type=float, default=0.5,
                              help='Chord detection window size in seconds (default: 0.5)')
    audio_chords.add_argument('-c', '--min-confidence', type=float, default=0.3,
                              help='Minimum confidence threshold (default: 0.3)')
    
    # audio frequency
    audio_freq = audio_subparsers.add_parser('frequency', help='Analyze 8-band frequency profile')
    audio_freq.add_argument('audio_file', help='Audio file')
    audio_freq.add_argument('-o', '--output', help='Save frequency analysis to JSON')
    
    # audio reference
    audio_ref = audio_subparsers.add_parser('reference', help='Extract reference track DNA')
    audio_ref.add_argument('audio_file', help='Audio file')
    audio_ref.add_argument('-o', '--output', help='Save reference DNA to JSON')
    
    # Arrangement commands
    arrange_parser = subparsers.add_parser('arrange', help='Song arrangement generation')
    arrange_subparsers = arrange_parser.add_subparsers(dest='subcommand', help='Arrangement commands')
    
    # arrange generate
    arrange_gen = arrange_subparsers.add_parser('generate', help='Generate song arrangement')
    arrange_gen.add_argument('-g', '--genre', default='pop',
                             help='Genre (pop, rock, edm, lofi, indie)')
    arrange_gen.add_argument('-e', '--emotion', default='neutral',
                             help='Primary emotion for energy mapping')
    arrange_gen.add_argument('-c', '--chords', help='Chord progression (e.g., "C-G-Am-F")')
    arrange_gen.add_argument('-i', '--intensity', type=float, default=0.6,
                             help='Base intensity level 0.0-1.0 (default: 0.6)')
    arrange_gen.add_argument('-o', '--output', help='Output file base name (generates .json, _markers.mid)')
    arrange_gen.add_argument('-b', '--bass', action='store_true', help='Also generate bass track')
    
    # arrange templates
    arrange_subparsers.add_parser('templates', help='List available arrangement templates')
    
    # Learning commands
    learn_parser = subparsers.add_parser('learn', help='AI-powered instrument education')
    learn_subparsers = learn_parser.add_subparsers(dest='subcommand', help='Learning commands')
    
    # learn instruments
    learn_subparsers.add_parser('instruments', help='List all supported instruments')
    
    # learn sources
    learn_sources = learn_subparsers.add_parser('sources', help='Show learning sources for an instrument')
    learn_sources.add_argument('instrument', help='Instrument name (e.g., piano, guitar)')
    learn_sources.add_argument('-l', '--level', type=int, help='Difficulty level (1-10)')
    
    # learn plan
    learn_plan = learn_subparsers.add_parser('plan', help='Generate a learning plan')
    learn_plan.add_argument('instrument', help='Instrument name')
    learn_plan.add_argument('-c', '--current', type=int, help='Current level (1-10, default: 1)')
    learn_plan.add_argument('-t', '--target', type=int, help='Target level (1-10, default: 5)')
    learn_plan.add_argument('-w', '--hours', type=float, help='Weekly practice hours (default: 5.0)')
    
    # learn prompt
    learn_prompt = learn_subparsers.add_parser('prompt', help='Generate AI teaching prompt')
    learn_prompt.add_argument('instrument', help='Instrument name')
    learn_prompt.add_argument('topic', help='Topic to learn')
    learn_prompt.add_argument('-a', '--action', default='explain', help='Action (explain, demonstrate, practice)')
    learn_prompt.add_argument('-l', '--level', type=int, default=5, help='Difficulty level (1-10)')
    learn_prompt.add_argument('-p', '--personalize', action='store_true', help='Personalize for student profile')
    learn_prompt.add_argument('--age', type=int, help='Student age (for personalization)')
    learn_prompt.add_argument('-o', '--output', help='Save prompt to file')
    
    # learn curriculum
    learn_curriculum = learn_subparsers.add_parser('curriculum', help='Show curriculum structure')
    learn_curriculum.add_argument('instrument', help='Instrument name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        'extract': cmd_extract,
        'apply': cmd_apply,
        'humanize': cmd_humanize,
        'analyze': cmd_analyze,
        'diagnose': cmd_diagnose,
        'reharm': cmd_reharm,
        'teach': cmd_teach,
        'intent': cmd_intent,
        'audio': cmd_audio,
        'arrange': cmd_arrange,
        'learn': cmd_learn,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
