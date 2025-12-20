#pragma once

#include "voice/LyricTypes.h"
#include "voice/ProsodyAnalyzer.h"
#include "voice/RhymeEngine.h"
#include "voice/PhonemeConverter.h"
#include "common/Types.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace kelly {

// Forward declarations
struct GeneratedMidi;

/**
 * LyricGenerator - Generates structured lyrics based on emotion and context.
 *
 * This class handles:
 * - Semantic concept expansion from emotions
 * - Lyric structure generation (verse/chorus/bridge)
 * - Word selection based on VAD (Valence, Arousal, Dominance) values
 * - Integration with rhyme schemes and meter
 */
class LyricGenerator {
public:
    struct LyricResult {
        LyricStructure structure;
        std::vector<LyricLine> lines;

        LyricResult() {}
    };

    LyricGenerator();
    ~LyricGenerator() = default;

    /**
     * Generate lyrics based on emotion, wound, and MIDI context.
     * @param emotion The emotion to express
     * @param wound The original wound description for context
     * @param midiContext Optional MIDI context for timing/rhythm
     * @return Generated lyric structure with lines
     */
    LyricResult generateLyrics(
        const EmotionNode& emotion,
        const Wound& wound,
        const GeneratedMidi* midiContext = nullptr
    );

    /**
     * Set lyric style (e.g., "poetic", "conversational", "metaphorical")
     */
    void setLyricStyle(const std::string& style) { lyricStyle_ = style; }

    /**
     * Set structure type (e.g., "verse_chorus", "ballad", "pop")
     */
    void setStructureType(const std::string& type) { structureType_ = type; }

    /**
     * Set rhyme scheme (e.g., "ABAB", "AABB")
     */
    void setRhymeScheme(const std::string& scheme) { rhymeSchemeName_ = scheme; }

    /**
     * Set target line length (number of syllables)
     */
    void setLineLength(int syllables) { targetLineLength_ = syllables; }

    /**
     * Load lyric templates from file
     * @param filePath Path to lyric_templates.json
     * @return true if loaded successfully
     */
    bool loadTemplates(const std::string& filePath);

    /**
     * Load emotion vocabulary/thesaurus (for semantic expansion)
     * @param emotionDataPath Path to emotion data directory
     * @return true if loaded successfully
     */
    bool loadEmotionVocabulary(const std::string& emotionDataPath);

private:
    std::string lyricStyle_ = "poetic";
    std::string structureType_ = "verse_chorus";
    std::string rhymeSchemeName_ = "ABAB";
    int targetLineLength_ = 8;

    // Component analyzers
    ProsodyAnalyzer prosodyAnalyzer_;
    RhymeEngine rhymeEngine_;
    PhonemeConverter phonemeConverter_;

    // Semantic expansion
    std::map<std::string, std::vector<std::string>> emotionVocabulary_;

    /**
     * Expand emotion into vocabulary words
     * @param emotion The emotion node
     * @return Vector of relevant words/phrases
     */
    std::vector<std::string> expandEmotionToVocabulary(const EmotionNode& emotion);

    /**
     * Extract keywords from wound description
     * @param wound The wound struct
     * @return Vector of keywords
     */
    std::vector<std::string> extractWoundKeywords(const Wound& wound);

    /**
     * Generate word list based on VAD values
     * @param valence Valence value (-1.0 to 1.0)
     * @param arousal Arousal value (0.0 to 1.0)
     * @param dominance Dominance value (0.0 to 1.0)
     * @return Vector of words matching VAD profile
     */
    std::vector<std::string> generateWordsFromVAD(
        float valence,
        float arousal,
        float dominance
    );

    /**
     * Generate lyric structure from template
     * @param templateName Name of structure template
     * @return LyricStructure
     */
    LyricStructure generateStructure(const std::string& templateName);

    /**
     * Generate lines for a section
     * @param sectionType Type of section (verse, chorus, etc.)
     * @param numLines Number of lines to generate
     * @param vocabulary Word pool to use
     * @param rhymeScheme Rhyme scheme to follow
     * @return Vector of lyric lines
     */
    std::vector<LyricLine> generateLines(
        LyricSectionType sectionType,
        int numLines,
        const std::vector<std::string>& vocabulary,
        const RhymeScheme& rhymeScheme
    );

    /**
     * Generate a single lyric line
     * @param targetSyllables Target number of syllables
     * @param vocabulary Word pool to use
     * @param stressPattern Target stress pattern
     * @return Generated lyric line
     */
    LyricLine generateLine(
        int targetSyllables,
        const std::vector<std::string>& vocabulary,
        const std::vector<int>& stressPattern = {}
    );

    /**
     * Select words that fit syllable and stress requirements
     * @param vocabulary Word pool
     * @param targetSyllables Target syllable count
     * @param stressPattern Target stress pattern
     * @return Selected words
     */
    std::vector<std::string> selectWords(
        const std::vector<std::string>& vocabulary,
        int targetSyllables,
        const std::vector<int>& stressPattern
    );

    /**
     * Get rhyme scheme from name
     * @param schemeName Name of scheme (e.g., "ABAB")
     * @return RhymeScheme struct
     */
    RhymeScheme getRhymeScheme(const std::string& schemeName);

    // Helper: Get words for valence range
    std::vector<std::string> getWordsForValence(float valence);

    // Helper: Get words for arousal range
    std::vector<std::string> getWordsForArousal(float arousal);

    // Helper: Get words for dominance range
    std::vector<std::string> getWordsForDominance(float dominance);
};

} // namespace kelly
