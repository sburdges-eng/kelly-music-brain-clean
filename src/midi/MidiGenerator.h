#pragma once

#include "common/Types.h"
#include "engine/IntentPipeline.h"
#include "midi/ChordGenerator.h"
#include "midi/GrooveEngine.h"
#include "engines/MelodyEngine.h"
#include "engines/BassEngine.h"
#include "engines/PadEngine.h"
#include "engines/StringEngine.h"
#include "engines/CounterMelodyEngine.h"
#include "engines/FillEngine.h"
#include "engines/DynamicsEngine.h"
#include "engines/TensionEngine.h"
#include "engines/RhythmEngine.h"
#include "engines/DrumGrooveEngine.h"
#include "engines/TransitionEngine.h"
#include "engines/ArrangementEngine.h"
#include "engines/VariationEngine.h"
#include <vector>
#include <random>

namespace kelly {

/**
 * Complete MIDI Generator - Orchestrates all MIDI engines to generate full arrangements.
 *
 * This is the central orchestrator that takes an IntentResult and generates complete MIDI
 * using all available engines:
 *
 * - Chord progressions (ChordGenerator)
 * - Melody lines (MelodyEngine)
 * - Bass lines (BassEngine)
 * - Pad textures (PadEngine)
 * - String arrangements (StringEngine)
 * - Counter melodies (CounterMelodyEngine)
 * - Rhythmic patterns (RhythmEngine)
 * - Fills and transitions (FillEngine)
 * - Tension and dynamics (TensionEngine, DynamicsEngine)
 * - Groove and humanization (GrooveEngine)
 *
 * All generation is driven by the IntentResult's emotion, rule breaks, and musical parameters.
 */
class MidiGenerator {
public:
    MidiGenerator();

    /**
     * Generate complete MIDI arrangement from emotional intent.
     * This is the main entry point that orchestrates all engines.
     *
     * @param intent The processed emotional intent (contains emotion, rule breaks, musical params)
     * @param bars Number of bars to generate (default 8, can be 4-32)
     * @param complexity 0.0 (simple) to 1.0 (complex) - affects chord extensions, melody density
     * @param humanize 0.0 (quantized) to 1.0 (loose) - affects timing and velocity
     * @param feel -1.0 (pull) to 1.0 (push) - affects timing feel
     * @param dynamics 0.0 to 1.0 - affects velocity range
     * @return Complete MIDI with all layers (chords, melody, bass, pads, strings, etc.)
     */
    GeneratedMidi generate(
        const IntentResult& intent,
        int bars = 8,
        float complexity = 0.5f,
        float humanize = 0.4f,
        float feel = 0.0f,
        float dynamics = 0.75f
    );

private:
    // Core engines
    ChordGenerator chordGen_;
    GrooveEngine grooveEngine_;

    // Melodic/harmonic engines
    MelodyEngine melodyEngine_;
    BassEngine bassEngine_;
    PadEngine padEngine_;
    StringEngine stringEngine_;
    CounterMelodyEngine counterMelodyEngine_;

    // Rhythmic engines
    RhythmEngine rhythmEngine_;
    FillEngine fillEngine_;
    DrumGrooveEngine drumGrooveEngine_;

    // Structural engines
    TransitionEngine transitionEngine_;
    ArrangementEngine arrangementEngine_;

    // Processing engines
    VariationEngine variationEngine_;

    // Expression engines
    DynamicsEngine dynamicsEngine_;
    TensionEngine tensionEngine_;

    // Random number generator
    std::mt19937 rng_;

    // ========================================================================
    // Generation methods (orchestrate individual engines)
    // ========================================================================

    /**
     * Generate chord progression from intent.
     */
    std::vector<Chord> generateChords(const IntentResult& intent, int bars);

    /**
     * Generate melody using MelodyEngine.
     */
    std::vector<MidiNote> generateMelody(
        const std::vector<Chord>& chords,
        const IntentResult& intent,
        float complexity,
        float dynamics
    );

    /**
     * Generate bass using BassEngine.
     */
    std::vector<MidiNote> generateBass(
        const std::vector<Chord>& chords,
        const IntentResult& intent,
        float complexity,
        float dynamics
    );

    /**
     * Generate pad layer using PadEngine.
     */
    std::vector<MidiNote> generatePads(
        const std::vector<Chord>& chords,
        const IntentResult& intent,
        float complexity
    );

    /**
     * Generate string arrangement using StringEngine.
     */
    std::vector<MidiNote> generateStrings(
        const std::vector<Chord>& chords,
        const IntentResult& intent,
        float complexity
    );

    /**
     * Generate counter melody using CounterMelodyEngine.
     */
    std::vector<MidiNote> generateCounterMelody(
        const std::vector<MidiNote>& primaryMelody,
        const IntentResult& intent,
        float complexity
    );

    /**
     * Generate rhythmic patterns using RhythmEngine.
     */
    std::vector<MidiNote> generateRhythm(
        const IntentResult& intent,
        int bars,
        float complexity,
        int tempoBpm
    );

    /**
     * Generate drum groove patterns using DrumGrooveEngine.
     */
    std::vector<MidiNote> generateDrumGroove(
        const IntentResult& intent,
        int bars,
        float complexity,
        int tempoBpm
    );

    /**
     * Generate transitions using TransitionEngine.
     */
    std::vector<MidiNote> generateTransitions(
        const IntentResult& intent,
        int bars,
        int tempoBpm
    );

    /**
     * Generate fills using FillEngine.
     */
    std::vector<MidiNote> generateFills(
        const IntentResult& intent,
        int bars,
        float complexity
    );

    /**
     * Generate arrangement structure using ArrangementEngine.
     * This creates section metadata (Intro, Verse, Chorus, etc.) that can
     * inform the generation process.
     */
    ArrangementOutput generateArrangement(
        const IntentResult& intent,
        int bars
    );

    // ========================================================================
    // Processing and application methods
    // ========================================================================

    /**
     * Apply groove and humanization to all MIDI layers.
     */
    void applyGrooveAndHumanize(
        GeneratedMidi& midi,
        float humanize,
        const EmotionNode& emotion,
        float feel
    );

    /**
     * Apply dynamics to all MIDI layers based on emotion and rule breaks.
     */
    void applyDynamics(
        GeneratedMidi& midi,
        const IntentResult& intent,
        float dynamics
    );

    /**
     * Apply rule breaks to generated MIDI.
     * This is where intentional "violations" of musical rules are applied
     * for emotional authenticity.
     */
    void applyRuleBreaks(
        GeneratedMidi& midi,
        const IntentResult& intent
    );

    /**
     * Apply tension curve from TensionEngine.
     */
    void applyTension(
        GeneratedMidi& midi,
        const IntentResult& intent,
        int bars
    );

    /**
     * Apply variations using VariationEngine (post-processing).
     */
    void applyVariations(
        GeneratedMidi& midi,
        const IntentResult& intent,
        float complexity
    );

    // ========================================================================
    // Helper methods
    // ========================================================================

    /**
     * Convert chord progression to string format for engines that need it.
     */
    std::vector<std::string> chordsToStrings(const std::vector<Chord>& chords);

    /**
     * Convert ticks to beats (for engines that use ticks).
     */
    double ticksToBeats(int ticks, int tempoBpm);

    /**
     * Convert beats to ticks.
     */
    int beatsToTicks(double beats, int tempoBpm);

    /**
     * Convert MelodyNote to MidiNote.
     */
    MidiNote melodyNoteToMidi(const MelodyNote& note, int tempoBpm);

    /**
     * Convert BassNote to MidiNote.
     */
    MidiNote bassNoteToMidi(const BassNote& note, int tempoBpm);

    /**
     * Convert PadNote to MidiNote.
     */
    MidiNote padNoteToMidi(const PadNote& note, int tempoBpm);

    /**
     * Convert StringNote to MidiNote.
     */
    MidiNote stringNoteToMidi(const StringNote& note, int tempoBpm);

    /**
     * Convert CounterMelodyNote to MidiNote.
     */
    MidiNote counterMelodyNoteToMidi(const CounterMelodyNote& note, int tempoBpm);

    /**
     * Convert DrumHit to MidiNote (for rhythm/fills).
     */
    MidiNote drumHitToMidi(const DrumHit& hit, int tempoBpm);

    /**
     * Convert GrooveHit to MidiNote.
     */
    MidiNote grooveHitToMidi(const GrooveHit& hit, int tempoBpm);

    /**
     * Convert TransitionNote to MidiNote.
     */
    MidiNote transitionNoteToMidi(const TransitionNote& note, int tempoBpm);

    /**
     * Get emotion name string from EmotionNode.
     */
    std::string getEmotionName(const EmotionNode& emotion);

    /**
     * Determine which layers to generate based on complexity and emotion.
     */
    struct LayerFlags {
        bool melody = true;
        bool bass = true;
        bool pads = false;
        bool strings = false;
        bool counterMelody = false;
        bool rhythm = false;
        bool fills = false;
        bool drumGroove = false;
        bool transitions = false;
        bool variations = false;
    };

    LayerFlags determineLayers(
        const IntentResult& intent,
        float complexity,
        int bars,
        const std::optional<ArrangementOutput>& arrangement = std::nullopt
    );
};

} // namespace kelly
