#pragma once

#include "common/Types.h"  // For RuleBreak and RuleBreakType definitions (matching IntentPipeline)
#include <vector>
#include <algorithm>
#include <cmath>

namespace kelly {

/**
 * Rule Break Engine - Determines which musical rules to break for emotional authenticity.
 * 
 * This engine analyzes emotional states and generates intentional music theory violations
 * that serve emotional expression. The core principle is that authentic emotion sometimes
 * requires breaking conventional rules:
 * 
 * - Negative emotions (sadness, grief, despair) → Dissonance, unresolved tensions, voice leading violations
 * - High arousal (anger, fear, anxiety) → Rhythmic disruption, syncopation, irregular meters
 * - Extreme intensity → Dynamic extremes, structural disruption, texture collisions
 * - Low arousal + negative valence (depression, emptiness) → Silence, fragmentation, rests
 * - Specific emotion categories → Targeted rule breaks (e.g., grief breaks voice leading)
 * 
 * Each rule break includes:
 * - Type (Harmony, Rhythm, Dynamics, Melody, Form) - matching Types.h definition
 * - Severity (0.0-1.0, scaled by emotion intensity)
 * - Description and emotional justification (reason field)
 * 
 * Note: Uses Types.h's RuleBreak definition to match IntentPipeline expectations.
 * Voice leading violations map to Harmony, texture to Form, range to Melody.
 */
class RuleBreakEngine {
public:
    RuleBreakEngine() = default;
    ~RuleBreakEngine() = default;
    
    /**
     * Generate rule breaks based on emotion intensity and type.
     * This is the primary method for determining which musical rules to break
     * for authentic emotional expression.
     * 
     * @param emotion The emotion node with VAD (valence, arousal, intensity) coordinates
     * @return Vector of rule breaks with all parameters populated
     */
    std::vector<RuleBreak> generateRuleBreaks(const EmotionNode& emotion);
    
    /**
     * Generate rule breaks for a journey between two emotions.
     * Creates rule breaks that serve the emotional transition, not just the destination.
     * 
     * @param emotionA Starting emotion
     * @param emotionB Target emotion
     * @return Vector of rule breaks for the transition
     */
    std::vector<RuleBreak> generateJourneyRuleBreaks(
        const EmotionNode& emotionA,
        const EmotionNode& emotionB
    );

private:
    /**
     * Calculate rule break severity based on emotion intensity.
     * Severity scales with intensity, with extreme emotions (>0.8) getting amplified breaks.
     */
    float calculateSeverity(const EmotionNode& emotion, float baseSeverity = 0.5f) const;
    
    /**
     * Generate harmony rule breaks (dissonance, clusters, unresolved tensions).
     * Triggered by negative valence emotions.
     */
    void addHarmonyRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const;
    
    /**
     * Generate rhythm rule breaks (syncopation, irregular meters, displaced accents).
     * Triggered by high arousal emotions.
     */
    void addRhythmRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const;
    
    /**
     * Generate dynamics rule breaks (extreme contrasts, sudden changes).
     * Triggered by high intensity emotions.
     */
    void addDynamicsRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const;
    
    /**
     * Generate voice leading rule breaks (parallel motion, unresolved tensions).
     * Specifically for grief, sadness, and other emotions that require broken resolution.
     * Maps to Harmony rule breaks in Types.h.
     */
    void addVoiceLeadingRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const;
    
    /**
     * Generate texture rule breaks (layering, collisions, cross-hand voicing).
     * For anger, anxiety, and other complex emotional states.
     * Maps to Form rule breaks in Types.h.
     */
    void addTextureRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const;
    
    /**
     * Generate structure rule breaks (silence, fragmentation, rest).
     * For depression, emptiness, and low-energy negative states.
     * Uses Form rule breaks in Types.h.
     */
    void addStructureRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const;
    
    /**
     * Generate range rule breaks (extreme register, wide leaps).
     * For intense emotions requiring dramatic expression.
     * Maps to Melody rule breaks in Types.h.
     */
    void addRangeRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const;
    
    /**
     * Generate category-specific rule breaks based on emotion category.
     * Each emotion category has characteristic rule-breaking patterns.
     */
    void addCategorySpecificRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const;
};

} // namespace kelly

