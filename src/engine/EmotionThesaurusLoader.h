#pragma once

#include "engine/EmotionThesaurus.h"
#include <juce_core/juce_core.h>
#include <string>
#include <vector>

namespace kelly {

/**
 * Loads the 216-node emotion thesaurus from JSON files.
 *
 * Features:
 * - Multiple fallback paths for finding JSON files
 * - Embedded default JSON data as last resort
 * - Handles both "category" and "name" fields in JSON
 *
 * Expected JSON structure:
 * {
 *   "name": "SAD" or "category": "sad",
 *   "sub_emotions": {
 *     "GRIEF": {
 *       "sub_sub_emotions": {
 *         "bereaved": {
 *           "intensity_tiers": {
 *             "1_subtle": ["touched", "moved"],
 *             "2_mild": ["bereaved", "mourning"],
 *             ...
 *           }
 *         }
 *       }
 *     }
 *   }
 * }
 */
class EmotionThesaurusLoader {
public:
    /**
     * Load all emotion JSON files with fallback path searching.
     * Tries multiple locations and falls back to embedded defaults if needed.
     *
     * @param thesaurus The thesaurus to populate
     * @return Number of emotions loaded
     */
    static int loadWithFallbacks(EmotionThesaurus& thesaurus);

    /**
     * Load all emotion JSON files from a specific directory.
     *
     * @param dataDirectory Path to directory containing emotion JSON files
     * @param thesaurus The thesaurus to populate
     * @return Number of emotions loaded
     */
    static int loadFromJsonFiles(const juce::File& dataDirectory, EmotionThesaurus& thesaurus);

    /**
     * Load a single emotion JSON file.
     *
     * @param jsonFile Path to JSON file
     * @param thesaurus The thesaurus to populate
     * @return Number of emotions loaded from this file
     */
    static int loadFromJsonFile(const juce::File& jsonFile, EmotionThesaurus& thesaurus);

    /**
     * Load a single emotion JSON file with explicit ID counter.
     */
    static int loadFromJsonFile(const juce::File& jsonFile, EmotionThesaurus& thesaurus, int& nextId);

    /**
     * Load from embedded default JSON strings (fallback when files not found).
     */
    static int loadFromEmbeddedDefaults(EmotionThesaurus& thesaurus);

private:
    /**
     * Try to find a JSON file using multiple fallback paths.
     * Uses PathResolver for centralized path resolution.
     * @param baseFilename Base filename (e.g., "sad.json")
     * @return File if found, empty File otherwise
     */
    static juce::File findJsonFile(const std::string& baseFilename);

    /**
     * Get filename mapping (handles joy.json -> happy.json, etc.)
     */
    static std::vector<std::string> getEmotionFilenames();

    /**
     * Get alternative filenames for a given emotion.
     */
    static std::vector<std::string> getAlternativeFilenames(const std::string& baseFilename);

    static EmotionCategory categoryFromString(const std::string& categoryStr);
    static float valenceFromString(const std::string& valenceStr);
    static float intensityFromTier(const std::string& tierStr);
    static float arousalFromIntensity(float intensity, EmotionCategory category);

    /**
     * Extract category name from JSON (handles both "name" and "category" fields).
     */
    static std::string extractCategoryName(const juce::var& root);

    static void processSubEmotion(
        const juce::var& subData,
        const std::string& categoryName,
        const std::string& subEmotionName,
        EmotionThesaurus& thesaurus,
        int& nextId
    );

    static void processSubSubEmotion(
        const juce::var& subSubData,
        const std::string& categoryName,
        const std::string& subEmotionName,
        const std::string& subSubEmotionName,
        EmotionThesaurus& thesaurus,
        int& nextId
    );

    static void processIntensityTier(
        const juce::var& tierData,
        const std::string& categoryName,
        const std::string& subEmotionName,
        const std::string& subSubEmotionName,
        const std::string& tierName,
        EmotionThesaurus& thesaurus,
        int& nextId
    );
};

} // namespace kelly
