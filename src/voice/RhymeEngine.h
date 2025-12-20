#pragma once

#include "voice/LyricTypes.h"
#include "voice/PhonemeConverter.h"
#include <string>
#include <vector>
#include <map>

namespace kelly {

/**
 * RhymeEngine - Detects and generates rhymes for lyrics.
 *
 * This class handles:
 * - Phonetic rhyme detection (perfect, slant, internal)
 * - End-rhyme generation following rhyme scheme
 * - Flow/rhythm pattern matching
 * - Rhyme database building from phonemes
 */
class RhymeEngine {
public:
    enum class RhymeType {
        Perfect,    // Perfect rhyme (e.g., "cat" / "bat")
        Slant,      // Slant/near rhyme (e.g., "cat" / "cut")
        Internal,   // Internal rhyme within line
        None        // No rhyme
    };

    struct RhymeMatch {
        std::string word1;
        std::string word2;
        RhymeType type;
        float score;  // 0.0 to 1.0 (1.0 = perfect match)

        RhymeMatch() : type(RhymeType::None), score(0.0f) {}
    };

    RhymeEngine();
    ~RhymeEngine() = default;

    /**
     * Check if two words rhyme.
     * @param word1 First word
     * @param word2 Second word
     * @return RhymeMatch with type and score
     */
    RhymeMatch checkRhyme(const std::string& word1, const std::string& word2);

    /**
     * Find rhyming words from a vocabulary list.
     * @param targetWord Word to find rhymes for
     * @param vocabulary List of candidate words
     * @param maxResults Maximum number of results
     * @return Vector of rhyming words with scores
     */
    std::vector<RhymeMatch> findRhymes(
        const std::string& targetWord,
        const std::vector<std::string>& vocabulary,
        int maxResults = 10
    );

    /**
     * Generate rhyming words for a rhyme scheme.
     * @param scheme Rhyme scheme pattern (e.g., [0,1,0,1] for ABAB)
     * @param vocabulary Available words
     * @param existingWords Words already used (indexed by rhyme group)
     * @return Map of rhyme group index to words
     */
    std::map<int, std::vector<std::string>> generateRhymeWords(
        const std::vector<int>& scheme,
        const std::vector<std::string>& vocabulary,
        const std::map<int, std::string>& existingWords = {}
    );

    /**
     * Extract end phonemes (rhyming part) from a word.
     * @param word Input word
     * @param numPhonemes Number of phonemes to extract (default: 2-3)
     * @return Vector of IPA phoneme symbols (end of word)
     */
    std::vector<std::string> extractEndPhonemes(
        const std::string& word,
        int numPhonemes = 3
    );

    /**
     * Compare phoneme sequences for rhyme.
     * @param phonemes1 First phoneme sequence
     * @param phonemes2 Second phoneme sequence
     * @return Rhyme score (0.0 to 1.0)
     */
    float comparePhonemeSequences(
        const std::vector<std::string>& phonemes1,
        const std::vector<std::string>& phonemes2
    ) const;

    /**
     * Check for internal rhyme within a line.
     * @param words Words in the line
     * @return Vector of RhymeMatch pairs found within the line
     */
    std::vector<RhymeMatch> detectInternalRhymes(const std::vector<std::string>& words);

    /**
     * Build rhyme database from vocabulary.
     * @param vocabulary List of words
     */
    void buildRhymeDatabase(const std::vector<std::string>& vocabulary);

private:
    PhonemeConverter phonemeConverter_;

    // Rhyme database: end phonemes -> words
    std::map<std::string, std::vector<std::string>> rhymeDatabase_;

    /**
     * Get phoneme sequence for a word.
     */
    std::vector<std::string> getPhonemes(const std::string& word);

    /**
     * Normalize phoneme sequence for comparison (focus on vowels and final consonants).
     */
    std::vector<std::string> normalizeForRhyme(const std::vector<std::string>& phonemes) const;

    /**
     * Check if phoneme is a vowel.
     */
    bool isVowelPhoneme(const std::string& ipa) const;

    /**
     * Calculate similarity score between two phonemes.
     */
    float phonemeSimilarity(const std::string& ipa1, const std::string& ipa2) const;
};

} // namespace kelly
