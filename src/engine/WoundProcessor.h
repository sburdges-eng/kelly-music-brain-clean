#pragma once

#include "common/Types.h"
#include "engine/EmotionThesaurus.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

namespace kelly {

/**
 * Wound Processor - Processes emotional wounds into structured emotion data.
 * 
 * Analyzes wound descriptions using keyword matching and emotion thesaurus
 * to identify the most appropriate emotional state.
 * 
 * Features:
 * - Keyword matching with phrase support and word boundary detection
 * - Intensity calculation from description text (analyzes intensifiers)
 * - Multi-emotion detection and confidence scoring
 * - Context-aware emotion mapping
 */
class WoundProcessor {
public:
    explicit WoundProcessor(EmotionThesaurus& thesaurus) : thesaurus_(thesaurus) {}
    ~WoundProcessor() = default;
    
    /**
     * Process a wound description into an emotion node.
     * Uses keyword matching, intensity calculation, and thesaurus lookup to find the best match.
     * @param wound The wound to process
     * @return The identified emotion node with calculated intensity
     */
    EmotionNode processWound(const Wound& wound);
    
    /**
     * Find emotion by keywords in text.
     * @param text The text to analyze
     * @param intensity The base intensity level (will be adjusted by description analysis)
     * @return The best matching emotion node
     */
    EmotionNode findEmotionByKeywords(const std::string& text, float intensity = 0.5f);
    
    /**
     * Calculate intensity from description text.
     * Analyzes intensifiers, modifiers, and emotional language.
     * @param description The wound description
     * @param baseIntensity The base intensity from the wound struct
     * @return Calculated intensity (0.0 to 1.0)
     */
    float calculateIntensity(const std::string& description, float baseIntensity = 0.5f);

private:
    EmotionThesaurus& thesaurus_;
    
    struct EmotionClue {
        std::vector<std::string> keywords;      // Single words
        std::vector<std::string> phrases;       // Multi-word phrases (checked first)
        std::string emotionName;
        float confidence;
        float intensityBoost;                   // Intensity modifier when this emotion is detected
    };
    
    struct MatchResult {
        std::string emotionName;
        float confidence;
        float intensityModifier;
        bool isPhraseMatch;                     // True if matched via phrase
    };
    
    /**
     * Build comprehensive emotion clues with keywords and phrases.
     */
    std::vector<EmotionClue> buildEmotionClues();
    
    /**
     * Find all emotion matches in text with confidence scores.
     * @param text Lowercase text to search
     * @return Vector of match results sorted by confidence
     */
    std::vector<MatchResult> findMatches(const std::string& text);
    
    /**
     * Check if a keyword matches in text with word boundary awareness.
     * @param text The text to search
     * @param keyword The keyword to find
     * @return True if keyword found as whole word or phrase
     */
    bool matchesKeyword(const std::string& text, const std::string& keyword) const;
    
    /**
     * Extract intensifier modifiers from text.
     * @param text The description text
     * @return Intensity modifier (-1.0 to 1.0, where positive increases intensity)
     */
    float extractIntensifierModifier(const std::string& text) const;
    
    /**
     * Normalize text: lowercase and remove punctuation for matching.
     */
    std::string normalizeText(const std::string& text) const;
};

} // namespace kelly

