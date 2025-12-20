#pragma once
/*
 * CoreTheoryEngine.h - Fundamental Music Theory (Math & Rules)
 * ============================================================
 *
 * The mathematical foundation of music theory:
 * - Interval mathematics and frequency ratios
 * - Scale generation from first principles
 * - Tuning systems (12-TET, Just Intonation, Pythagorean, etc.)
 * - Circle of Fifths relationships
 * - Why notes behave the way they do
 *
 * CONNECTIONS:
 * - Used by: HarmonyEngine (chord construction), MelodyEngine (scale selection)
 * - Uses: Types.h (Interval, Scale, TuningSystem types)
 * - UI: ConceptBrowser displays interval relationships
 */

#include "../Types.h"
#include <cmath>
#include <array>
#include <unordered_map>

namespace midikompanion::theory {

class CoreTheoryEngine {
public:
    CoreTheoryEngine();
    ~CoreTheoryEngine() = default;

    //==========================================================================
    // Interval Mathematics
    //==========================================================================

    /**
     * Calculate interval from semitones
     *
     * @param semitones Distance in half steps (0-127)
     * @param tuning Tuning system to use (default: 12-TET)
     * @return Complete interval with all explanatory data
     *
     * Example:
     *   auto perfectFifth = calculateInterval(7);
     *   // Returns: {semitones: 7, quality: Perfect, name: "Perfect Fifth",
     *   //           ratio: 1.498, consonance: 0.95, explanation: "3:2 ratio..."}
     */
    Interval calculateInterval(int semitones,
                               TuningSystem tuning = TuningSystem::TwelveTET) const;

    /**
     * Calculate interval between two MIDI notes
     */
    Interval calculateInterval(int midiNote1, int midiNote2,
                               TuningSystem tuning = TuningSystem::TwelveTET) const;

    /**
     * Get frequency ratio for interval
     *
     * @param semitones Interval size
     * @param tuning Tuning system (affects exact ratio)
     * @return Frequency ratio (e.g., 1.5 for Perfect Fifth in 12-TET)
     */
    float getFrequencyRatio(int semitones, TuningSystem tuning) const;

    /**
     * Calculate consonance score (0.0 = dissonant, 1.0 = perfectly consonant)
     *
     * Based on:
     * - Frequency ratio simplicity (3:2 more consonant than 45:32)
     * - Position in harmonic series
     * - Psychoacoustic beating patterns
     */
    float calculateConsonance(int semitones, TuningSystem tuning) const;

    /**
     * Get interval name from semitones
     * Returns: "Perfect Unison", "Minor Second", "Major Third", etc.
     */
    std::string getIntervalName(int semitones) const;

    /**
     * Get interval quality
     */
    IntervalQuality getIntervalQuality(int semitones) const;

    /**
     * Explain WHY this interval sounds the way it does
     *
     * Returns multi-level explanation:
     * - Acoustic: frequency ratio, waveform alignment
     * - Mathematical: formula, calculation
     * - Perceptual: how the brain processes it
     * - Historical: usage across eras
     */
    std::string explainInterval(int semitones, ExplanationDepth depth) const;

    //==========================================================================
    // Scale Generation
    //==========================================================================

    /**
     * Generate scale from interval pattern
     *
     * @param rootNote MIDI root note (0-127)
     * @param pattern Interval pattern (e.g., {2,2,1,2,2,2,1} for Major)
     * @param tuning Tuning system
     * @return Complete scale with all notes and metadata
     *
     * Example:
     *   auto cMajor = generateScale(60, {2,2,1,2,2,2,1}); // C Major
     *   auto dDorian = generateScale(62, {2,1,2,2,2,1,2}); // D Dorian
     */
    Scale generateScale(int rootNote,
                       const std::vector<int>& pattern,
                       TuningSystem tuning = TuningSystem::TwelveTET) const;

    /**
     * Generate common scale types
     */
    Scale generateMajorScale(int rootNote) const;
    Scale generateNaturalMinorScale(int rootNote) const;
    Scale generateHarmonicMinorScale(int rootNote) const;
    Scale generateMelodicMinorScale(int rootNote) const;
    Scale generateMode(int rootNote, int modeNumber) const; // 1=Ionian, 2=Dorian, etc.

    /**
     * Get pentatonic scale
     */
    Scale generatePentatonicScale(int rootNote, bool major = true) const;

    /**
     * Get blues scale
     */
    Scale generateBluesScale(int rootNote) const;

    /**
     * Get all notes in a scale (as MIDI note numbers)
     */
    std::vector<int> getScaleNotes(const Scale& scale, int octaveRange = 1) const;

    /**
     * Check if a note is in a scale
     */
    bool isNoteInScale(int midiNote, const Scale& scale) const;

    /**
     * Get scale degree of a note (1-7, or -1 if not in scale)
     */
    int getScaleDegree(int midiNote, const Scale& scale) const;

    //==========================================================================
    // Circle of Fifths
    //==========================================================================

    /**
     * Get Circle of Fifths data
     *
     * Returns complete circle with:
     * - All 12 major keys
     * - All 12 relative minor keys
     * - Sharp/flat counts
     * - Enharmonic equivalents
     */
    struct CircleOfFifths {
        std::array<std::string, 12> majorKeys;
        std::array<std::string, 12> minorKeys;
        std::array<int, 12> sharpsOrFlats; // Positive = sharps, negative = flats

        // Position on circle (0-11, where 0 = C)
        int getPosition(const std::string& key) const;
    };

    CircleOfFifths getCircleOfFifths() const;

    /**
     * Calculate modulation distance (how many steps around circle)
     *
     * @param fromKey Starting key
     * @param toKey Destination key
     * @return Distance (0-6, where 0 = same key, 6 = tritone)
     *
     * Example:
     *   modulationDistance("C", "G") = 1  // One step clockwise
     *   modulationDistance("C", "F#") = 6 // Opposite side (tritone)
     */
    int modulationDistance(const std::string& fromKey, const std::string& toKey) const;

    /**
     * Suggest smooth modulations from current key
     *
     * @param currentKey Current key
     * @param maxDistance Maximum steps around circle (default: 2)
     * @return List of nearby keys, sorted by closeness
     *
     * Example:
     *   suggestModulations("C", 2)
     *   // Returns: ["G", "F", "Am", "Em", "Dm"]
     */
    std::vector<std::string> suggestModulations(const std::string& currentKey,
                                                int maxDistance = 2) const;

    /**
     * Get relative minor/major
     */
    std::string getRelativeMinor(const std::string& majorKey) const;
    std::string getRelativeMajor(const std::string& minorKey) const;

    /**
     * Get parallel minor/major (same root, different mode)
     */
    std::string getParallelMinor(const std::string& majorKey) const;
    std::string getParallelMajor(const std::string& minorKey) const;

    //==========================================================================
    // Tuning Systems
    //==========================================================================

    /**
     * Get frequency for MIDI note in specific tuning
     *
     * @param midiNote MIDI note number (0-127)
     * @param tuning Tuning system
     * @param concertA Reference pitch (default: 440 Hz)
     * @return Frequency in Hz
     */
    double getFrequency(int midiNote,
                       TuningSystem tuning = TuningSystem::TwelveTET,
                       double concertA = CONCERT_A) const;

    /**
     * Compare tuning systems
     * Shows frequency differences between tuning systems
     */
    struct TuningComparison {
        int midiNote;
        double frequency12TET;
        double frequencyJust;
        double frequencyPythagorean;
        double centsDifference; // Cents difference from 12-TET
    };

    TuningComparison compareTuningSystems(int midiNote) const;

    /**
     * Explain tuning system
     *
     * Returns:
     * - Historical context
     * - Mathematical basis
     * - Practical implications
     * - When to use this tuning
     */
    std::string explainTuningSystem(TuningSystem tuning) const;

    //==========================================================================
    // Pattern Recognition
    //==========================================================================

    /**
     * Analyze melodic pattern (ascending, descending, arpeggiated, etc.)
     */
    struct MelodicPattern {
        std::vector<int> intervalSequence; // [2, 2, -1, 4] (steps)
        std::string shapeName;             // "Ascending Arpeggio"
        std::string emotionalEffect;       // "Triumphant"
        std::vector<std::string> famousExamples;
    };

    MelodicPattern analyzeMelodicPattern(const std::vector<int>& notes) const;

    /**
     * Detect sequence (repeating pattern at different pitch levels)
     */
    struct Sequence {
        std::vector<int> pattern;
        int transpositionInterval;
        int repetitions;
        std::string sequenceType; // "Ascending", "Descending", "Rosalia"
    };

    std::optional<Sequence> detectSequence(const std::vector<int>& notes) const;

    //==========================================================================
    // Utilities
    //==========================================================================

    /**
     * Convert MIDI note to note name
     *
     * @param midiNote MIDI note number (0-127)
     * @param preferSharps Use sharps vs flats
     * @return Note name (e.g., "C4", "F#5", "Bb3")
     */
    std::string midiToNoteName(int midiNote, bool preferSharps = true) const;

    /**
     * Convert note name to MIDI number
     *
     * @param noteName Note name (e.g., "C4", "F#5", "Bb3")
     * @return MIDI note number, or -1 if invalid
     */
    int noteNameToMidi(const std::string& noteName) const;

    /**
     * Get enharmonic equivalent
     *
     * @param noteName Original note (e.g., "C#")
     * @return Enharmonic (e.g., "Db")
     */
    std::string getEnharmonic(const std::string& noteName) const;

private:
    //==========================================================================
    // Internal Data
    //==========================================================================

    // Interval names lookup table
    static const std::array<std::string, 13> intervalNames_;
    static const std::array<IntervalQuality, 13> intervalQualities_;

    // Circle of Fifths data
    mutable CircleOfFifths circleOfFifths_;
    bool circleInitialized_ = false;

    // Pattern database (common melodic motifs)
    struct PatternDatabase {
        std::string pattern;
        std::string name;
        std::vector<std::string> examples;
    };
    std::vector<PatternDatabase> patternDatabase_;

    //==========================================================================
    // Internal Helpers
    //==========================================================================

    void initializeCircleOfFifths() const;
    void initializePatternDatabase();

    // Tuning system calculations
    double calculateFrequency12TET(int midiNote, double concertA) const;
    double calculateFrequencyJust(int midiNote, double concertA) const;
    double calculateFrequencyPythagorean(int midiNote, double concertA) const;

    // Consonance calculation (based on harmonic series overlap)
    float calculateHarmonicOverlap(int semitones) const;

    // Note name parsing
    struct ParsedNote {
        int pitchClass;  // 0-11 (C=0, C#=1, etc.)
        int octave;      // MIDI octave
        bool sharp;      // True for sharp, false for flat/natural
    };
    std::optional<ParsedNote> parseNoteName(const std::string& noteName) const;
};

} // namespace midikompanion::theory
