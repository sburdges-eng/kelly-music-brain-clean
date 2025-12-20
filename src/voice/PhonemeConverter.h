#pragma once

#include "voice/LyricTypes.h"
#include "voice/CMUDictionary.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace kelly {

/**
 * PhonemeConverter - Converts text to phonemes (IPA symbols).
 *
 * This class handles:
 * - Text-to-phoneme conversion (G2P - Grapheme-to-Phoneme)
 * - Word-to-syllable splitting
 * - Stress pattern detection
 * - Phoneme-to-formant mapping
 */
class PhonemeConverter {
public:
    PhonemeConverter();
    ~PhonemeConverter() = default;

    /**
     * Convert text string to phoneme sequence.
     * @param text Input text
     * @return Vector of phonemes
     */
    std::vector<Phoneme> textToPhonemes(const std::string& text);

    /**
     * Convert word to phoneme sequence.
     * @param word Input word (single word, no spaces)
     * @return Vector of phoneme IPA symbols
     */
    std::vector<std::string> wordToPhonemes(const std::string& word);

    /**
     * Split word into syllables.
     * @param word Input word
     * @return Vector of syllable strings
     */
    std::vector<std::string> splitIntoSyllables(const std::string& word);

    /**
     * Detect stress pattern in a word.
     * @param word Input word
     * @return Vector of stress levels (0=unstressed, 1=secondary, 2=primary) for each syllable
     */
    std::vector<int> detectStress(const std::string& word);

    /**
     * Get phoneme data from IPA symbol.
     * @param ipa IPA symbol (e.g., "/aɪ/")
     * @return Phoneme struct, or empty phoneme if not found
     */
    Phoneme getPhonemeFromIPA(const std::string& ipa);

    /**
     * Get formant data for a phoneme (for vocoder synthesis).
     * @param phoneme Phoneme struct
     * @return Formant frequencies [F1, F2, F3, F4] and bandwidths [B1, B2, B3, B4]
     */
    std::pair<std::array<float, 4>, std::array<float, 4>> getFormants(const Phoneme& phoneme) const;

    /**
     * Interpolate between two phonemes for smooth transitions.
     * @param p1 First phoneme
     * @param p2 Second phoneme
     * @param t Interpolation factor (0.0 = p1, 1.0 = p2)
     * @return Interpolated formant data
     */
    std::pair<std::array<float, 4>, std::array<float, 4>> interpolatePhonemes(
        const Phoneme& p1,
        const Phoneme& p2,
        float t
    ) const;

    /**
     * Load phoneme database from file.
     * @param filePath Path to phonemes.json
     * @return true if loaded successfully
     */
    bool loadPhonemeDatabase(const std::string& filePath);

    /**
     * Enable/disable CMU Dictionary usage
     * @param enable true to use CMU Dictionary for lookups
     */
    void setUseCMUDictionary(bool enable) { useCMUDictionary_ = enable; }

    /**
     * Check if CMU Dictionary is enabled
     */
    bool isUsingCMUDictionary() const { return useCMUDictionary_; }

    /**
     * Count syllables in a word.
     * @param word Input word
     * @return Number of syllables
     */
    int countSyllables(const std::string& word);

private:
    // Phoneme database: IPA symbol -> Phoneme data
    std::map<std::string, Phoneme> phonemeDatabase_;

    // Common word dictionary: word -> phonemes (for accurate G2P)
    std::map<std::string, std::vector<std::string>> wordDictionary_;

    // CMU Dictionary for improved G2P accuracy
    std::unique_ptr<CMUDictionary> cmuDictionary_;
    bool useCMUDictionary_ = true;

    /**
     * Initialize default phoneme database (built-in data).
     */
    void initializeDefaultPhonemes();

    /**
     * Initialize common word dictionary.
     */
    void initializeWordDictionary();

    /**
     * Convert grapheme sequence to phonemes using rules.
     * @param word Input word
     * @return Vector of IPA phoneme symbols
     */
    std::vector<std::string> graphemeToPhoneme(const std::string& word);

    /**
     * Simple vowel detection.
     * @param c Character to check
     * @return true if character is a vowel letter
     */
    bool isVowelLetter(char c) const;

    /**
     * Simple consonant detection.
     * @param c Character to check
     * @return true if character is a consonant letter
     */
    bool isConsonantLetter(char c) const;

    /**
     * Normalize word (convert to lowercase, remove punctuation).
     * @param word Input word
     * @return Normalized word
     */
    std::string normalizeWord(const std::string& word) const;
};

} // namespace kelly
