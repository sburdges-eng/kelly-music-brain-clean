#pragma once

#include "voice/LyricTypes.h"
#include <string>
#include <vector>

namespace kelly {

/**
 * ProsodyAnalyzer - Analyzes and validates prosody (rhythm, meter, stress patterns).
 *
 * This class handles:
 * - Syllable stress detection (0=unstressed, 1=secondary, 2=primary)
 * - Meter pattern matching (iambic, trochaic, anapestic, dactylic)
 * - Natural speech rhythm modeling
 * - Line length validation
 */
class ProsodyAnalyzer {
public:
    enum class MeterType {
        Iambic,      // unstressed-stressed (da-DUM)
        Trochaic,    // stressed-unstressed (DUM-da)
        Anapestic,   // unstressed-unstressed-stressed (da-da-DUM)
        Dactylic,    // stressed-unstressed-unstressed (DUM-da-da)
        Mixed,       // Variable/natural speech rhythm
        None         // No specific meter
    };

    struct MeterPattern {
        MeterType type;
        std::vector<int> pattern;  // 0=unstressed, 1=secondary, 2=primary
        std::string name;

        MeterPattern() : type(MeterType::None) {}
    };

    ProsodyAnalyzer();
    ~ProsodyAnalyzer() = default;

    /**
     * Detect stress pattern in a word.
     * @param word Input word
     * @return Vector of stress levels (0, 1, or 2) for each syllable
     */
    std::vector<int> detectStress(const std::string& word);

    /**
     * Detect stress pattern for multiple words.
     * @param words Vector of words
     * @return Vector of stress levels for each syllable across all words
     */
    std::vector<int> detectStressPattern(const std::vector<std::string>& words);

    /**
     * Match meter pattern against stress pattern.
     * @param stressPattern Stress pattern to match
     * @param meterType Type of meter to match against
     * @return Match score (0.0 = no match, 1.0 = perfect match)
     */
    float matchMeter(const std::vector<int>& stressPattern, MeterType meterType) const;

    /**
     * Detect meter type from stress pattern.
     * @param stressPattern Stress pattern to analyze
     * @return Detected meter type
     */
    MeterType detectMeter(const std::vector<int>& stressPattern) const;

    /**
     * Get meter pattern for a given meter type.
     * @param meterType Type of meter
     * @param numSyllables Number of syllables
     * @return Meter pattern
     */
    MeterPattern getMeterPattern(MeterType meterType, int numSyllables) const;

    /**
     * Validate line length against target.
     * @param line Line to validate
     * @param targetSyllables Target number of syllables
     * @return true if length matches (within tolerance)
     */
    bool validateLineLength(const LyricLine& line, int targetSyllables) const;

    /**
     * Count syllables in a word.
     * @param word Input word
     * @return Number of syllables
     */
    int countSyllables(const std::string& word) const;

    /**
     * Count syllables in multiple words.
     * @param words Vector of words
     * @return Total number of syllables
     */
    int countSyllables(const std::vector<std::string>& words) const;

    /**
     * Adjust word choices to match meter pattern.
     * @param words Candidate words
     * @param targetMeter Target meter pattern
     * @return Words that best match the meter
     */
    std::vector<std::string> selectWordsForMeter(
        const std::vector<std::string>& words,
        const MeterPattern& targetMeter
    ) const;

    /**
     * Calculate natural speech rhythm score.
     * @param stressPattern Stress pattern to evaluate
     * @return Rhythm score (higher = more natural)
     */
    float calculateRhythmScore(const std::vector<int>& stressPattern) const;

private:
    /**
     * Simple vowel detection (letter-based).
     */
    bool isVowelLetter(char c) const;

    /**
     * Normalize word (lowercase, remove punctuation).
     */
    std::string normalizeWord(const std::string& word) const;

    /**
     * Get stress level for a word (simplified dictionary lookup).
     */
    int getWordStressLevel(const std::string& word, int syllableIndex) const;
};

} // namespace kelly
