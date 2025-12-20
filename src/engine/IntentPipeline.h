#pragma once

#include "common/Types.h"
#include "engine/EmotionThesaurus.h"
#include "engine/WoundProcessor.h"
#include "engine/RuleBreakEngine.h"
#include <memory>

namespace kelly {

/**
 * IntentPipeline - Core three-phase emotion processing engine.
 * 
 * This is the main API for the emotion engine, transforming emotional wounds
 * into musical parameters through a structured three-phase pipeline:
 * 
 * Phase 1: Wound → Emotion
 *   - Analyzes wound description and intensity
 *   - Maps to 216-node emotion thesaurus using WoundProcessor
 *   - Produces EmotionNode with valence, arousal, intensity coordinates
 * 
 * Phase 2: Emotion → Rule Breaks
 *   - Generates intentional music theory violations via RuleBreakEngine
 *   - Determines which rules to break and why (emotional justification)
 *   - Creates RuleBreak objects for harmony, rhythm, dynamics, melody, form
 * 
 * Phase 3: Compile Musical Parameters
 *   - Synthesizes emotion + rule breaks into concrete musical parameters
 *   - Sets mode, tempo, dynamic range, dissonance tolerance, etc.
 *   - Produces IntentResult ready for MIDI generation
 * 
 * Usage:
 *   IntentPipeline pipeline;
 *   Wound wound{"I feel lost and alone", 0.8f, "user_input"};
 *   IntentResult result = pipeline.process(wound);
 *   // Use result.mode, result.tempo, result.ruleBreaks, etc.
 */
class IntentPipeline {
public:
    IntentPipeline();
    
    /**
     * Process a complete intent from wound to musical parameters.
     * 
     * This is the main entry point for the emotion engine.
     * Executes all three phases: Wound → Emotion → Musical Params
     * 
     * @param wound The emotional wound to process
     * @return IntentResult containing emotion, rule breaks, and musical parameters
     */
    IntentResult process(const Wound& wound);
    
    /**
     * Process Side A (current state) and Side B (desired state)
     * to create a musical journey between emotions.
     * 
     * Creates a blended emotion that transitions from current to desired,
     * generating rule breaks that serve the emotional journey.
     * 
     * @param current The current emotional state (Side A)
     * @param desired The desired emotional state (Side B)
     * @return IntentResult representing the journey between states
     */
    IntentResult processJourney(const SideA& current, const SideB& desired);
    
    /**
     * Get direct access to the thesaurus for UI emotion selection.
     * 
     * Allows external code to query the emotion thesaurus for:
     * - Finding emotions by name, ID, or category
     * - Getting all 216 emotion nodes
     * - Suggesting musical parameters for emotions
     */
    const EmotionThesaurus& thesaurus() const { return thesaurus_; }
    EmotionThesaurus& thesaurus() { return thesaurus_; }
    
private:
    EmotionThesaurus thesaurus_;
    WoundProcessor woundProcessor_;
    RuleBreakEngine ruleBreakEngine_;
    
    // Phase 3: Compile musical parameters
    IntentResult compileMusicalParams(
        const Wound& wound,
        const EmotionNode& emotion,
        const std::vector<RuleBreak>& ruleBreaks
    );
};

} // namespace kelly
