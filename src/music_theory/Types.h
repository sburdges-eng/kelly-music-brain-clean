#pragma once
/*
 * Music Theory Types - Common data structures for the Music Theory Engine
 * =====================================================================
 *
 * Shared types used across all 7 pillars of the music theory system.
 */

#include <string>
#include <vector>
#include <array>
#include <optional>
#include <map>

namespace midikompanion::theory {

//==============================================================================
// Core Musical Constants
//==============================================================================

static constexpr double TWELVE_TET_RATIO = 1.059463094359; // 2^(1/12)
static constexpr double CONCERT_A = 440.0;                  // Hz
static constexpr double PHI = 1.618033988749;               // Golden ratio

//==============================================================================
// Tuning Systems
//==============================================================================

enum class TuningSystem {
    TwelveTET,      // Equal temperament (modern standard)
    JustIntonation, // Pure intervals (3:2, 5:4, etc.)
    Pythagorean,    // Ancient Greek (stacked perfect fifths)
    Meantone,       // Renaissance compromise
    Werckmeister,   // Baroque well-temperament
    Custom          // User-defined
};

//==============================================================================
// Interval Types
//==============================================================================

enum class IntervalQuality {
    Perfect,
    Major,
    Minor,
    Augmented,
    Diminished,
    DoublyAugmented,
    DoublyDiminished
};

struct Interval {
    int semitones;                      // 0-127 (MIDI range)
    float cents;                        // Microtonal precision (0-100 cents per semitone)
    IntervalQuality quality;
    int simpleInterval;                 // 1-8 (reduces compound intervals)
    int octaveDisplacement;             // How many octaves up/down

    // Explanatory data
    std::string intervalName;           // "Perfect Fifth"
    float frequencyRatio;               // 3:2 for Perfect Fifth
    float harmonicSeriesPosition;       // Position in overtone series
    std::string acousticExplanation;
    std::string perceptualEffect;
    std::string historicalUsage;
    std::vector<std::string> whenToAvoid;
    float consonanceScore;              // 0.0 (dissonant) to 1.0 (consonant)
};

//==============================================================================
// Scale Types
//==============================================================================

enum class ScaleType {
    Major,
    NaturalMinor,
    HarmonicMinor,
    MelodicMinor,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Aeolian,
    Locrian,
    Pentatonic,
    Blues,
    WholeTone,
    Diminished,
    Chromatic,
    Custom
};

struct Scale {
    int rootNote;                       // MIDI note (0-127)
    ScaleType type;
    std::vector<int> degrees;           // Scale degrees (semitones from root)
    std::string name;                   // "C Major", "D Dorian"
    std::string characteristicSound;    // "Bright and happy"
    std::string culturalContext;        // "Western classical foundation"
    std::vector<std::string> avoidNotes;
    std::vector<std::string> emphasizeNotes;
};

//==============================================================================
// Chord Types
//==============================================================================

enum class ChordQuality {
    Major,
    Minor,
    Diminished,
    Augmented,
    Dominant7,
    Major7,
    Minor7,
    HalfDiminished7,
    FullyDiminished7,
    Sus2,
    Sus4,
    Add9,
    Custom
};

struct Chord {
    int rootNote;                       // MIDI note (0-127)
    ChordQuality quality;
    std::vector<int> notes;             // All chord tones (MIDI notes)
    std::string symbol;                 // "Cmaj7", "Dm7"
    std::string romanNumeral;           // "I", "V7", "vi"
    std::string function;               // "Tonic", "Dominant", "Subdominant"
    float tension;                      // 0.0 (stable) to 1.0 (max tension)
};

//==============================================================================
// Chord Progression Types
//==============================================================================

struct ChordProgression {
    std::vector<Chord> chords;
    std::string key;
    std::vector<std::string> romanNumerals;
    std::string functionalAnalysis;     // "T-D-T-S"
    std::array<float, 100> tensionCurve; // Tension over time
    std::string harmonyExplanation;
    std::vector<std::string> famousExamples;
};

//==============================================================================
// Voice Leading Types
//==============================================================================

enum class VoiceLeadingStyle {
    Bach,           // SATB counterpoint
    Jazz,           // Drop-2, drop-3 voicings
    Pop,            // Close position, doubling
    Modern,         // Open voicings, quartal harmony
    Strict          // Species counterpoint
};

struct VoiceLeadingSuggestion {
    std::vector<std::vector<int>> voices; // Each voice's notes over time
    std::vector<std::string> rules;       // Rules that were followed
    std::vector<std::string> exceptions;  // Rules that were broken (with reasons)
    float smoothnessScore;                // 0.0 (jumpy) to 1.0 (smooth)
};

//==============================================================================
// Rhythm Types
//==============================================================================

struct TimeSignature {
    int numerator;                      // Beats per measure
    int denominator;                    // Note value that gets the beat
    std::string feel;                   // "Duple", "Triple", "Compound"
    std::string bodyResponse;           // "Walking rhythm", "Waltz"
};

struct RhythmicPattern {
    std::vector<float> onsetTimes;      // In beats
    std::vector<float> durations;       // In beats
    std::string grooveName;             // "Swing 8ths", "Straight 16ths"
    float swingRatio;                   // 1.0 = straight, 2.0 = triplet
    std::string perceptualGroove;
};

struct GrooveAnalysis {
    std::vector<float> actualOnsets;
    std::vector<float> quantizedOnsets;
    std::vector<float> microTimingShifts; // The "feel"
    std::string grooveQuality;
    std::string pocketDescription;      // "Behind the beat", "On the beat"
    float pocketWidth;                  // Acceptable timing variation (ms)
};

//==============================================================================
// Knowledge Graph Types
//==============================================================================

enum class ExplanationDepth {
    Simple,         // "Distance from 1st to 5th note"
    Intermediate,   // "7 semitones, sounds strong and stable"
    Advanced,       // "3:2 frequency ratio, waveforms align"
    Expert          // "Pythagorean tuning vs Just intonation differences"
};

enum class ExplanationType {
    Intuitive,      // "Think of it like..."
    Acoustic,       // "The frequency ratio is..."
    Mathematical,   // "The formula is..."
    Historical,     // "In Bach's time..."
    Practical       // "In this song, you can hear..."
};

struct KnowledgeNode {
    std::string conceptName;                // "Perfect Fifth"
    std::string category;               // "Interval"
    std::vector<std::string> prerequisites;
    std::vector<std::string> relatedConcepts;
    std::vector<std::string> applications;

    // Multiple explanation styles
    std::map<ExplanationType, std::string> explanations;

    // Examples
    struct MusicalExample {
        std::string song;
        float timestamp;
        std::string description;
    };
    std::vector<MusicalExample> examples;
};

//==============================================================================
// Exercise Types
//==============================================================================

enum class ExerciseType {
    IntervalRecognition,
    ChordQualityRecognition,
    ProgressionAnalysis,
    RhythmDictation,
    SightReading,
    EarTraining,
    VoiceLeading,
    Improvisation
};

enum class DifficultyLevel {
    Beginner,       // Level 1-2
    Intermediate,   // Level 3-5
    Advanced,       // Level 6-8
    Expert,         // Level 9-10
    Microtonal      // Beyond standard theory
};

struct Exercise {
    ExerciseType type;
    DifficultyLevel level;
    std::string conceptName;                // What this exercises
    std::string instruction;
    std::string focusArea;              // "Listen for the bright quality"
    std::string readingStrategy;        // "Scan ahead 2 measures"

    // Exercise data (type-specific)
    std::vector<int> notes;             // For pitch exercises
    std::vector<float> onsets;          // For rhythm exercises
    std::string correctAnswer;
    std::vector<std::string> hints;
};

//==============================================================================
// User Profile Types
//==============================================================================

struct UserProfile {
    std::string name;
    DifficultyLevel currentLevel;
    std::map<std::string, float> conceptMastery; // concept → 0.0-1.0
    std::vector<std::string> completedExercises;
    std::vector<std::string> currentLearningPath;

    // Preferences
    ExplanationType preferredExplanationStyle;
    float audioVisualBalance;           // 0.0 = audio learner, 1.0 = visual

    // Statistics
    int totalExercisesCompleted;
    float averageSuccessRate;
    std::vector<std::string> strugglingConcepts;
};

struct Feedback {
    bool correct;
    std::string message;
    std::vector<std::string> hints;
    std::vector<std::string> suggestedReview; // Concepts to review
    bool showPartialSolution;
    bool encourageRetry;

    // Error analysis
    std::string errorType;              // "Conceptual", "Execution", "Careless"
    std::string guidance;               // How to improve
};

//==============================================================================
// Acoustics Types
//==============================================================================

struct HarmonicSeries {
    float fundamentalFrequency;
    std::vector<float> overtones;       // 2f, 3f, 4f, ...
    std::vector<float> amplitudes;      // Loudness of each overtone
    std::string timbre;                 // "Bright", "Dark", "Rich"
    std::string consonanceExplanation;
};

struct BeatingAnalysis {
    float frequency1;
    float frequency2;
    float beatFrequency;                // |f1 - f2|
    std::string tuningGuidance;
};

//==============================================================================
// Historical Types
//==============================================================================

enum class MusicalEra {
    Medieval,       // 500-1400
    Renaissance,    // 1400-1600
    Baroque,        // 1600-1750
    Classical,      // 1750-1820
    Romantic,       // 1820-1900
    Modern,         // 1900-1960
    Contemporary    // 1960-present
};

struct EraCharacteristics {
    MusicalEra era;
    std::string years;
    std::vector<std::string> harmonicDevices;
    std::vector<std::string> melodicDevices;
    std::vector<std::string> formTypes;
    std::vector<std::string> keyComposers;
    std::string culturalContext;
};

//==============================================================================
// Performance Practice Types
//==============================================================================

struct PerformanceGuide {
    MusicalEra era;
    std::string articulationStyle;
    std::vector<std::string> ornaments;
    std::string dynamicApproach;
    std::string tempoFlexibility;
};

} // namespace midikompanion::theory
