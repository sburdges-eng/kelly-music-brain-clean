#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace kelly {

/**
 * CMU Pronouncing Dictionary Integration
 *
 * Provides access to the CMU Pronouncing Dictionary for accurate
 * grapheme-to-phoneme (G2P) conversion. The dictionary contains
 * over 134,000 words with their pronunciations in ARPABET notation.
 *
 * ARPABET phoneme symbols are converted to IPA for use with the
 * vocoder system.
 */
class CMUDictionary {
public:
    CMUDictionary();
    ~CMUDictionary() = default;

    /**
     * Load CMU dictionary from file
     * @param filePath Path to cmudict-0.7b or similar format file
     * @return true if loaded successfully
     */
    bool loadFromFile(const std::string& filePath);

    /**
     * Load CMU dictionary from embedded data (fallback)
     * Loads a subset of common words
     */
    void loadEmbeddedDictionary();

    /**
     * Look up pronunciation for a word
     * @param word Word to look up (case-insensitive)
     * @return Vector of ARPABET phonemes, or empty if not found
     */
    std::vector<std::string> lookup(const std::string& word) const;

    /**
     * Convert ARPABET phoneme to IPA symbol
     * @param arpabet ARPABET phoneme (e.g., "AA", "IY", "K")
     * @return IPA symbol (e.g., "/ɑ/", "/i/", "/k/")
     */
    static std::string arpabetToIPA(const std::string& arpabet);

    /**
     * Convert ARPABET sequence to IPA sequence
     * @param arpabetPhonemes Vector of ARPABET phonemes
     * @return Vector of IPA symbols
     */
    static std::vector<std::string> arpabetToIPA(const std::vector<std::string>& arpabetPhonemes);

    /**
     * Get pronunciation with stress markers removed
     * @param word Word to look up
     * @return ARPABET phonemes without stress numbers (0, 1, 2)
     */
    std::vector<std::string> lookupWithoutStress(const std::string& word) const;

    /**
     * Get pronunciation with stress information
     * @param word Word to look up
     * @return Vector of pairs: (phoneme, stress_level) where stress is 0=unstressed, 1=primary, 2=secondary
     */
    std::vector<std::pair<std::string, int>> lookupWithStress(const std::string& word) const;

    /**
     * Check if word exists in dictionary
     * @param word Word to check
     * @return true if word is in dictionary
     */
    bool contains(const std::string& word) const;

    /**
     * Get number of entries in dictionary
     */
    size_t size() const { return dictionary_.size(); }

    /**
     * Normalize word for lookup (uppercase, remove punctuation)
     */
    static std::string normalizeWord(const std::string& word);

private:
    // Dictionary: normalized word -> ARPABET phonemes
    std::unordered_map<std::string, std::vector<std::string>> dictionary_;

    // ARPABET to IPA mapping
    static std::map<std::string, std::string> createARPABETToIPAMap();

    // Extract stress from ARPABET phoneme (returns phoneme without number and stress level)
    static std::pair<std::string, int> extractStress(const std::string& arpabet);
};

} // namespace kelly
