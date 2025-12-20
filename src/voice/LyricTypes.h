#pragma once

#include <string>
#include <vector>
#include <array>

namespace kelly {

//==============================================================================
// Core Lyric Data Structures
//==============================================================================

/**
 * Phoneme - Represents a single speech sound (IPA symbol)
 */
struct Phoneme {
    std::string ipa;                    // IPA symbol (e.g., "/aɪ/", "/p/")
    std::string type;                   // "vowel", "consonant", "diphthong"
    std::array<float, 4> formants;      // F1, F2, F3, F4 in Hz
    std::array<float, 4> bandwidths;    // B1, B2, B3, B4 in Hz
    float duration_ms;                  // Typical duration in milliseconds
    bool voiced;                        // Whether the phoneme is voiced

    Phoneme()
        : formants{0.0f, 0.0f, 0.0f, 0.0f}
        , bandwidths{0.0f, 0.0f, 0.0f, 0.0f}
        , duration_ms(100.0f)
        , voiced(true)
    {}
};

/**
 * Syllable - Represents a syllable within a word
 */
struct Syllable {
    std::string text;                   // Text representation
    std::vector<Phoneme> phonemes;      // Phonemes in this syllable
    int stress;                         // Stress level: 0=unstressed, 1=secondary, 2=primary
    float duration_ms;                  // Duration in milliseconds

    Syllable() : stress(0), duration_ms(200.0f) {}
};

/**
 * LyricLine - Represents a single line of lyrics
 */
struct LyricLine {
    std::string text;                   // Full text of the line
    std::vector<Syllable> syllables;    // Syllables in this line
    std::vector<int> stressPattern;     // Stress pattern (0/1/2 for each syllable)
    std::string meter;                  // Meter type (e.g., "iambic", "trochaic")
    int targetSyllables;                // Target number of syllables
    int lineNumber;                     // Line number in verse/section

    LyricLine() : targetSyllables(8), lineNumber(0) {}
};

/**
 * RhymeScheme - Represents a rhyme pattern
 */
struct RhymeScheme {
    std::string name;                   // Scheme name (e.g., "ABAB", "AABB")
    std::vector<int> pattern;           // Pattern indices (0,1,0,1 for ABAB)

    RhymeScheme() {}
    RhymeScheme(const std::string& n, const std::vector<int>& p)
        : name(n), pattern(p)
    {}
};

/**
 * LyricStructure - Represents the overall structure of lyrics (verse, chorus, etc.)
 * Note: Renamed to LyricSectionType to avoid conflict with ArrangementEngine::SectionType
 */
enum class LyricSectionType {
    Verse,
    Chorus,
    Bridge,
    Intro,
    Outro,
    PreChorus,
    PostChorus
};

struct LyricSection {
    LyricSectionType type;
    std::vector<LyricLine> lines;
    int sectionNumber;

    LyricSection() : type(LyricSectionType::Verse), sectionNumber(0) {}
};

struct LyricStructure {
    std::vector<LyricSection> sections;
    std::string pattern;                // Structure pattern (e.g., "V-C-V-C-B-C")
    RhymeScheme rhymeScheme;

    LyricStructure() {}
};

//==============================================================================
// Vocal Expression Types
//==============================================================================

/**
 * VocalExpression - Parameters for vocal expression and dynamics
 */
struct VocalExpression {
    float dynamics;                     // Overall dynamics (0.0 to 1.0)
    float articulation;                 // Articulation type: 0.0=legato, 1.0=staccato
    float vibratoDepth;                 // Vibrato depth variation (0.0 to 1.0)
    float vibratoRate;                  // Vibrato rate variation (Hz)
    float breathiness;                  // Breathiness variation (0.0 to 1.0)
    float brightness;                   // Brightness modulation (0.0 to 1.0)
    float crescendo;                    // Crescendo amount (0.0=none, 1.0=full)
    float diminuendo;                   // Diminuendo amount (0.0=none, 1.0=full)

    VocalExpression()
        : dynamics(0.7f)
        , articulation(0.3f)
        , vibratoDepth(0.3f)
        , vibratoRate(5.0f)
        , breathiness(0.2f)
        , brightness(0.5f)
        , crescendo(0.0f)
        , diminuendo(0.0f)
    {}
};

/**
 * VoiceType - Parameters for different voice types
 */
enum class VoiceType {
    Male,
    Female,
    Child,
    Neutral
};

struct VoiceTypeParams {
    VoiceType type;
    float formantShift;                 // Formant frequency multiplier (1.0=normal, >1.0=higher)
    float pitchRangeMin;                // Minimum MIDI pitch
    float pitchRangeMax;                // Maximum MIDI pitch
    float formantShiftF1;               // F1 specific shift
    float formantShiftF2;               // F2 specific shift
    float formantShiftF3;               // F3 specific shift

    VoiceTypeParams()
        : type(VoiceType::Neutral)
        , formantShift(1.0f)
        , pitchRangeMin(48.0f)  // C3
        , pitchRangeMax(84.0f)  // C6
        , formantShiftF1(1.0f)
        , formantShiftF2(1.0f)
        , formantShiftF3(1.0f)
    {}
};

} // namespace kelly
