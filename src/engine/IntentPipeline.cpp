#include "engine/IntentPipeline.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <cmath>

namespace kelly {

IntentPipeline::IntentPipeline() : woundProcessor_(thesaurus_) {
}

IntentResult IntentPipeline::process(const Wound& wound) {
    // =====================================================================
    // PHASE 1: Wound → Emotion
    // =====================================================================
    // Analyze wound description and map to emotion thesaurus
    // Uses WoundProcessor for keyword matching and emotion lookup
    EmotionNode emotion = woundProcessor_.processWound(wound);

    // =====================================================================
    // PHASE 2: Emotion → Rule Breaks
    // =====================================================================
    // Generate intentional music theory violations based on emotion
    // Uses RuleBreakEngine to determine which rules to break and why
    std::vector<RuleBreak> ruleBreaks = ruleBreakEngine_.generateRuleBreaks(emotion);

    // =====================================================================
    // PHASE 3: Compile Musical Parameters
    // =====================================================================
    // Synthesize emotion + rule breaks into concrete musical parameters
    return compileMusicalParams(wound, emotion, ruleBreaks);
}

IntentResult IntentPipeline::processJourney(const SideA& current, const SideB& desired) {
    // =====================================================================
    // PHASE 1: Process both sides to emotions
    // =====================================================================
    // If emotionId is provided, use it directly; otherwise process description
    Wound sideAWound;
    sideAWound.description = current.description;
    sideAWound.intensity = current.intensity;
    sideAWound.urgency = current.intensity;
    sideAWound.source = "sideA";
    sideAWound.expression = current.description;

    Wound sideBWound;
    sideBWound.description = desired.description;
    sideBWound.intensity = desired.intensity;
    sideBWound.urgency = desired.intensity;
    sideBWound.source = "sideB";
    sideBWound.expression = desired.description;

    EmotionNode emotionA = current.emotionId
        ? thesaurus_.findById(*current.emotionId).value_or(
            woundProcessor_.processWound(sideAWound))
        : woundProcessor_.processWound(sideAWound);

    EmotionNode emotionB = desired.emotionId
        ? thesaurus_.findById(*desired.emotionId).value_or(
            woundProcessor_.processWound(sideBWound))
        : woundProcessor_.processWound(sideBWound);

    // =====================================================================
    // Create blended journey emotion
    // =====================================================================
    // Blend emotions: 40% from current (A), 60% toward desired (B)
    // This creates musical tension that resolves as the journey progresses
    float blendedValence = emotionA.valence * 0.4f + emotionB.valence * 0.6f;
    float blendedArousal = emotionA.arousal * 0.4f + emotionB.arousal * 0.6f;
    float blendedIntensity = std::max(emotionA.intensity, emotionB.intensity);

    // Find nearest emotion node in thesaurus to the blended coordinates
    EmotionNode journeyEmotion = thesaurus_.findNearest(
        blendedValence, blendedArousal, blendedIntensity);

    // Create wound representing the journey
    Wound journeyWound;
    journeyWound.description = "Journey from " + emotionA.name + " toward " + emotionB.name;
    journeyWound.intensity = blendedIntensity;
    journeyWound.urgency = blendedIntensity;
    journeyWound.source = "cassette_journey";
    journeyWound.expression = journeyWound.description;

    // =====================================================================
    // PHASE 2: Generate journey-specific rule breaks
    // =====================================================================
    // Rule breaks that serve the transition between emotions
    std::vector<RuleBreak> ruleBreaks = ruleBreakEngine_.generateJourneyRuleBreaks(emotionA, emotionB);

    // =====================================================================
    // PHASE 3: Compile musical parameters for the journey
    // =====================================================================
    return compileMusicalParams(journeyWound, journeyEmotion, ruleBreaks);
}

// =========================================================================
// PHASE 3: Compile Musical Parameters
// =========================================================================
// Synthesizes emotion coordinates and rule breaks into concrete musical
// parameters that can be used by MIDI generation engines.

IntentResult IntentPipeline::compileMusicalParams(
    const Wound& wound,
    const EmotionNode& emotion,
    const std::vector<RuleBreak>& ruleBreaks
) {
    IntentResult result;

    // Store source data
    result.sourceWound = wound;
    result.emotion = emotion;  // Compatibility field (should match sourceWound.primaryEmotion)
    result.ruleBreaks = ruleBreaks;

    // =====================================================================
    // Base parameters from emotion (via EmotionThesaurus)
    // =====================================================================
    // These are the "default" musical characteristics for this emotion
    result.mode = thesaurus_.suggestMode(emotion);
    result.tempo = thesaurus_.suggestTempoModifier(emotion);
    result.dynamicRange = thesaurus_.suggestDynamicRange(emotion);

    // =====================================================================
    // Default safe values
    // =====================================================================
    // Conservative defaults that can be overridden by rule breaks
    result.allowDissonance = false;
    result.syncopationLevel = 0.3f;
    result.humanization = 0.4f;

    // =====================================================================
    // Apply rule breaks to override/modify parameters
    // =====================================================================
    // Rule breaks represent intentional violations of music theory rules
    // for emotional authenticity. They modify the base parameters.
    for (const auto& rb : ruleBreaks) {
        switch (rb.type) {
            case RuleBreakType::ModalMixture:
                // Allow dissonant intervals and unresolved tensions
                result.allowDissonance = true;
                // Increase dynamic range when using dissonance
                result.dynamicRange = std::min(1.0f, result.dynamicRange + rb.intensity * 0.2f);
                break;

            case RuleBreakType::CrossRhythm:
                // Increase syncopation and off-beat accents
                result.syncopationLevel = std::max(result.syncopationLevel, rb.intensity);
                // More humanization for complex rhythms
                result.humanization = std::max(result.humanization, rb.intensity * 0.8f);
                break;

            case RuleBreakType::DynamicContrast:
                // Expand dynamic range for dramatic expression
                result.dynamicRange = std::max(result.dynamicRange, rb.intensity);
                // Higher dynamics often benefit from more humanization
                if (rb.intensity > 0.7f) {
                    result.humanization = std::max(result.humanization, 0.6f);
                }
                break;

            case RuleBreakType::RegisterShift:
                // Melodic rule breaks (wide leaps, chromaticism) affect
                // generation algorithms rather than direct parameters
                // The melody engine will use this rule break during generation
                break;

            case RuleBreakType::HarmonicAmbiguity:
                // Form rule breaks (structural disruption) affect
                // arrangement and song structure, not direct parameters
                // The arrangement engine will use this rule break
                break;
        }
    }

    // =====================================================================
    // Final parameter validation and clamping
    // =====================================================================
    result.tempo = std::clamp(result.tempo, 0.5f, 2.0f);
    result.dynamicRange = std::clamp(result.dynamicRange, 0.0f, 1.0f);
    result.syncopationLevel = std::clamp(result.syncopationLevel, 0.0f, 1.0f);
    result.humanization = std::clamp(result.humanization, 0.0f, 1.0f);

    return result;
}

} // namespace kelly
