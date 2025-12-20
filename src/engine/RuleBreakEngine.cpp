#include "engine/RuleBreakEngine.h"
#include <algorithm>
#include <cmath>

namespace kelly {

// =============================================================================
// Severity Calculation
// =============================================================================

float RuleBreakEngine::calculateSeverity(const EmotionNode& emotion, float baseSeverity) const {
    // Base severity scales with intensity (0.0-1.0)
    // Formula: base * (0.5 + intensity * 0.5) ensures minimum 50% of base severity
    float severity = baseSeverity * (0.5f + emotion.intensity * 0.5f);

    // Extreme emotions (>0.8 intensity) get amplified rule breaks
    // This ensures that overwhelming emotions break rules more dramatically
    if (emotion.intensity > 0.8f) {
        severity = std::min(1.0f, severity * 1.3f);
    }

    // Very extreme emotions (>0.95) push severity to maximum
    if (emotion.intensity > 0.95f) {
        severity = std::min(1.0f, severity * 1.2f);
    }

    return std::clamp(severity, 0.0f, 1.0f);
}

// =============================================================================
// Main Rule Break Generation
// =============================================================================

std::vector<RuleBreak> RuleBreakEngine::generateRuleBreaks(const EmotionNode& emotion) {
    std::vector<RuleBreak> ruleBreaks;

    // Generate rule breaks across all dimensions based on VAD coordinates
    // Order matters: more fundamental rule breaks (harmony, rhythm) come first

    // 1. Harmony rule breaks (most fundamental - affects harmonic structure)
    addHarmonyRuleBreaks(emotion, ruleBreaks);

    // 2. Rhythm rule breaks (affects temporal structure)
    addRhythmRuleBreaks(emotion, ruleBreaks);

    // 3. Dynamics rule breaks (affects expression and volume)
    addDynamicsRuleBreaks(emotion, ruleBreaks);

    // 4. Voice leading rule breaks (affects melodic/harmonic movement)
    addVoiceLeadingRuleBreaks(emotion, ruleBreaks);

    // 5. Texture rule breaks (affects layering and complexity)
    addTextureRuleBreaks(emotion, ruleBreaks);

    // 6. Structure rule breaks (affects form and organization)
    addStructureRuleBreaks(emotion, ruleBreaks);

    // 7. Range rule breaks (affects register and spatial distribution)
    addRangeRuleBreaks(emotion, ruleBreaks);

    // 8. Category-specific rule breaks (emotion-specific patterns)
    addCategorySpecificRuleBreaks(emotion, ruleBreaks);

    return ruleBreaks;
}

// =============================================================================
// Harmony Rule Breaks
// =============================================================================

void RuleBreakEngine::addHarmonyRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const {
    // Negative valence → harmonic tension and dissonance
    // The more negative the emotion, the more we break harmonic rules

    if (emotion.valence < -0.3f) {
        RuleBreak rb;
        rb.type = RuleBreakType::ModalMixture;
        rb.intensity = calculateSeverity(emotion, std::abs(emotion.valence));
        rb.description = "Intentional dissonance and harmonic tension";
        rb.justification = "Negative emotions require harmonic tension to express authentic pain";

        ruleBreaks.push_back(rb);
    }

    // Extreme negative valence → aggressive harmonic disruption
    if (emotion.valence < -0.7f && emotion.intensity > 0.7f) {
        RuleBreak rb;
        rb.type = RuleBreakType::ModalMixture;
        rb.intensity = calculateSeverity(emotion, 0.9f);
        rb.description = "Aggressive harmonic clusters and extreme dissonance";
        rb.justification = "Overwhelming negative emotion demands complete harmonic disruption";

        ruleBreaks.push_back(rb);
    }
}

// =============================================================================
// Rhythm Rule Breaks
// =============================================================================

void RuleBreakEngine::addRhythmRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const {
    // High arousal → rhythmic disruption and complexity
    // Agitated emotions need syncopation and off-beat accents

    if (emotion.arousal > 0.7f) {
        RuleBreak rb;
        rb.type = RuleBreakType::CrossRhythm;
        rb.intensity = calculateSeverity(emotion, emotion.arousal);
        rb.description = "Syncopated rhythms and displaced accents";
        rb.justification = "High arousal demands rhythmic complexity and displacement for authentic expression";

        ruleBreaks.push_back(rb);
    }

    // Very high arousal + high intensity → extreme rhythmic disruption
    if (emotion.arousal > 0.85f && emotion.intensity > 0.8f) {
        RuleBreak rb;
        rb.type = RuleBreakType::CrossRhythm;
        rb.intensity = calculateSeverity(emotion, 0.95f);
        rb.description = "Extreme rhythmic disruption and irregular time signatures";
        rb.justification = "Overwhelming agitation requires complete rhythmic deconstruction";

        ruleBreaks.push_back(rb);
    }

    // Low arousal but high intensity → subtle rhythmic displacement
    // Creates tension through subtle off-kilter rhythms
    if (emotion.arousal < 0.4f && emotion.intensity > 0.7f) {
        RuleBreak rb;
        rb.type = RuleBreakType::CrossRhythm;
        rb.intensity = calculateSeverity(emotion, 0.4f);
        rb.description = "Subtle rhythmic displacement and rubato";
        rb.justification = "Intense but calm emotions benefit from subtle temporal disruption";

        ruleBreaks.push_back(rb);
    }
}

// =============================================================================
// Dynamics Rule Breaks
// =============================================================================

void RuleBreakEngine::addDynamicsRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const {
    // High intensity → extreme dynamic contrasts
    // Intense emotions require dramatic dynamic expression

    if (emotion.intensity > 0.7f) {
        RuleBreak rb;
        rb.type = RuleBreakType::DynamicContrast;
        rb.intensity = calculateSeverity(emotion, emotion.intensity);
        rb.description = "Extreme dynamic range and contrasts";
        rb.justification = "Intense emotions require dramatic dynamic shifts for authentic expression";

        ruleBreaks.push_back(rb);
    }

    // Extreme intensity → maximum dynamic range
    if (emotion.intensity > 0.9f) {
        RuleBreak rb;
        rb.type = RuleBreakType::DynamicContrast;
        rb.intensity = calculateSeverity(emotion, 1.0f);
        rb.description = "Maximum dynamic range from whisper to scream";
        rb.justification = "Overwhelming emotion demands the full spectrum of dynamic expression";

        ruleBreaks.push_back(rb);
    }
}

// =============================================================================
// Voice Leading Rule Breaks
// =============================================================================

void RuleBreakEngine::addVoiceLeadingRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const {
    // Grief and sadness specifically break voice leading conventions
    // This represents the broken, unresolved nature of these emotions
    // Note: Types.h doesn't have Voice_Leading, so we use Harmony to represent these violations

    if (emotion.categoryEnum == EmotionCategory::Sadness && emotion.intensity > 0.7f) {
        RuleBreak rb;
        rb.type = RuleBreakType::ModalMixture;
        rb.intensity = calculateSeverity(emotion, emotion.intensity * 0.8f);
        rb.description = "Parallel motion and unresolved voice leading violations";
        rb.justification = "Grief breaks conventional resolution, allowing parallel motion and unresolved tensions";

        ruleBreaks.push_back(rb);
    }

    // Very negative valence with high intensity → voice leading violations
    // Represents emotional disconnection and brokenness
    if (emotion.valence < -0.6f && emotion.intensity > 0.7f &&
        emotion.categoryEnum != EmotionCategory::Anger) {  // Anger uses texture instead
        RuleBreak rb;
        rb.type = RuleBreakType::ModalMixture;
        rb.intensity = calculateSeverity(emotion, 0.7f);
        rb.description = "Broken voice leading and unconventional harmonic motion";
        rb.justification = "Deep pain disrupts the flow of conventional voice leading";

        ruleBreaks.push_back(rb);
    }
}

// =============================================================================
// Texture Rule Breaks
// =============================================================================

void RuleBreakEngine::addTextureRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const {
    // Anger and aggressive emotions → texture collisions and layering
    // Note: Types.h doesn't have Texture, so we use Form to represent structural/textural complexity
    if (emotion.categoryEnum == EmotionCategory::Anger) {
        RuleBreak rb;
        rb.type = RuleBreakType::HarmonicAmbiguity;
        rb.intensity = calculateSeverity(emotion, emotion.intensity);
        rb.description = "Aggressive layering and textural collision";
        rb.justification = "Anger demands forceful expression through dense, colliding textures";

        ruleBreaks.push_back(rb);
    }

    // High arousal + high intensity → complex texture
    // Anxiety, fear, excitement all benefit from complex layering
    if (emotion.arousal > 0.75f && emotion.intensity > 0.7f) {
        RuleBreak rb;
        rb.type = RuleBreakType::HarmonicAmbiguity;
        rb.intensity = calculateSeverity(emotion, emotion.arousal * 0.8f);
        rb.description = "Complex textural layering and density";
        rb.justification = "High arousal requires complex texture to express internal complexity";

        ruleBreaks.push_back(rb);
    }
}

// =============================================================================
// Structure Rule Breaks
// =============================================================================

void RuleBreakEngine::addStructureRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const {
    // Note: Types.h uses "Form" instead of "Structure" for structural rule breaks
    // Low arousal + negative valence → silence and fragmentation
    // Depression, emptiness, despair expressed through absence

    if (emotion.arousal < 0.3f && emotion.valence < -0.3f) {
        RuleBreak rb;
        rb.type = RuleBreakType::HarmonicAmbiguity;
        rb.intensity = calculateSeverity(emotion, std::abs(emotion.valence));
        rb.description = "Intentional silence, fragmentation, and structural gaps";
        rb.justification = "Emptiness and despair are expressed through absence and fragmentation";

        ruleBreaks.push_back(rb);
    }

    // Extreme intensity → structural disruption
    // Overwhelming emotions break conventional song structure
    if (emotion.intensity > 0.85f) {
        RuleBreak rb;
        rb.type = RuleBreakType::HarmonicAmbiguity;
        rb.intensity = calculateSeverity(emotion, 0.6f);
        rb.description = "Structural disruption and unconventional form";
        rb.justification = "Overwhelming emotion breaks conventional song structure";

        ruleBreaks.push_back(rb);
    }
}

// =============================================================================
// Range Rule Breaks
// =============================================================================

void RuleBreakEngine::addRangeRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const {
    // Extreme emotions → wide register usage and dramatic leaps (as Melody rule breaks)
    // Intense emotions require the full range of expression
    // Note: Types.h doesn't have Range, so we use Melody to represent wide intervallic leaps

    if (emotion.intensity > 0.8f) {
        RuleBreak rb;
        rb.type = RuleBreakType::RegisterShift;
        rb.intensity = calculateSeverity(emotion, emotion.intensity);
        rb.description = "Extreme register usage and wide intervallic leaps";
        rb.justification = "Intense emotion demands the full range of musical space";

        ruleBreaks.push_back(rb);
    }

    // Very negative + high intensity → dramatic range extremes
    if (emotion.valence < -0.7f && emotion.intensity > 0.8f) {
        RuleBreak rb;
        rb.type = RuleBreakType::RegisterShift;
        rb.intensity = calculateSeverity(emotion, 0.95f);
        rb.description = "Maximum range from lowest to highest registers";
        rb.justification = "Extreme negative emotion requires dramatic spatial extremes";

        ruleBreaks.push_back(rb);
    }
}

// =============================================================================
// Category-Specific Rule Breaks
// =============================================================================

void RuleBreakEngine::addCategorySpecificRuleBreaks(const EmotionNode& emotion, std::vector<RuleBreak>& ruleBreaks) const {
    // Each emotion category has characteristic rule-breaking patterns
    // This adds category-specific refinements beyond VAD-based rules

    switch (emotion.categoryEnum) {
        case EmotionCategory::Joy:
            // Joy can have syncopation and dynamic contrasts, but less dissonance
            if (emotion.intensity > 0.8f) {
                RuleBreak rb;
                rb.type = RuleBreakType::DynamicContrast;
                rb.intensity = calculateSeverity(emotion, 0.6f);
                rb.description = "Bright dynamic contrasts";
                rb.justification = "Joyful intensity benefits from bright, energetic dynamics";
                ruleBreaks.push_back(rb);
            }
            break;

        case EmotionCategory::Sadness:
            // Already handled by voice leading rules, but add subtle harmony breaks
            if (emotion.valence < -0.4f && emotion.intensity > 0.6f) {
                RuleBreak rb;
                rb.type = RuleBreakType::ModalMixture;
                rb.intensity = calculateSeverity(emotion, 0.5f);
                rb.description = "Subtle harmonic colorations and extensions";
                rb.justification = "Sadness benefits from subtle harmonic complexity";
                ruleBreaks.push_back(rb);
            }
            break;

        case EmotionCategory::Anger:
            // Already handled by texture, but add aggressive dynamics
            if (emotion.intensity > 0.7f) {
                RuleBreak rb;
                rb.type = RuleBreakType::DynamicContrast;
                rb.intensity = calculateSeverity(emotion, 0.9f);
                rb.description = "Aggressive dynamic attacks";
                rb.justification = "Anger requires sharp, aggressive dynamic attacks";
                ruleBreaks.push_back(rb);
            }
            break;

        case EmotionCategory::Fear:
            // Fear uses rhythm disruption and texture
            if (emotion.arousal > 0.7f) {
                RuleBreak rb;
                rb.type = RuleBreakType::CrossRhythm;
                rb.intensity = calculateSeverity(emotion, emotion.arousal);
                rb.description = "Irregular, anxious rhythms";
                rb.justification = "Fear manifests through irregular, anxious rhythmic patterns";
                ruleBreaks.push_back(rb);
            }
            break;

        case EmotionCategory::Surprise:
            // Surprise uses sudden dynamic and harmonic changes
            if (emotion.intensity > 0.6f) {
                RuleBreak rb;
                rb.type = RuleBreakType::DynamicContrast;
                rb.intensity = calculateSeverity(emotion, 0.7f);
                rb.description = "Sudden dynamic shifts";
                rb.justification = "Surprise requires sudden, unexpected dynamic changes";
                ruleBreaks.push_back(rb);
            }
            break;

        case EmotionCategory::Disgust:
            // Disgust uses texture (represented as Form) and harmony disruption
            if (emotion.valence < -0.5f) {
                RuleBreak rb;
                rb.type = RuleBreakType::HarmonicAmbiguity;
                rb.intensity = calculateSeverity(emotion, 0.7f);
                rb.description = "Textural disruption and structural collision";
                rb.justification = "Disgust requires textural disruption to express revulsion";
                ruleBreaks.push_back(rb);
            }
            break;

        case EmotionCategory::Trust:
            // Trust has minimal rule breaks - it's a stable, positive emotion
            // Only subtle breaks for intensity
            if (emotion.intensity > 0.85f) {
                RuleBreak rb;
                rb.type = RuleBreakType::DynamicContrast;
                rb.intensity = calculateSeverity(emotion, 0.4f);
                rb.description = "Warm dynamic expression";
                rb.justification = "Deep trust allows for warm, expressive dynamics";
                ruleBreaks.push_back(rb);
            }
            break;

        case EmotionCategory::Anticipation:
            // Anticipation uses rhythm and subtle tension
            if (emotion.arousal > 0.6f) {
                RuleBreak rb;
                rb.type = RuleBreakType::CrossRhythm;
                rb.intensity = calculateSeverity(emotion, 0.5f);
                rb.description = "Building rhythmic patterns";
                rb.justification = "Anticipation benefits from building, forward-driving rhythms";
                ruleBreaks.push_back(rb);
            }
            break;
    }
}

// =============================================================================
// Journey Rule Breaks
// =============================================================================

std::vector<RuleBreak> RuleBreakEngine::generateJourneyRuleBreaks(
    const EmotionNode& emotionA,
    const EmotionNode& emotionB)
{
    // Start with rule breaks for the target emotion (where we're going)
    std::vector<RuleBreak> ruleBreaks = generateRuleBreaks(emotionB);

    // Calculate emotional distance in VAD space
    // Euclidean distance in 3D space (valence, arousal, intensity)
    float valenceDiff = emotionB.valence - emotionA.valence;
    float arousalDiff = emotionB.arousal - emotionA.arousal;
    float intensityDiff = emotionB.intensity - emotionA.intensity;
    float emotionalDistance = std::sqrt(
        valenceDiff * valenceDiff +
        arousalDiff * arousalDiff +
        intensityDiff * intensityDiff
    );

    // If emotions are far apart, add journey-specific structural breaks
    // The journey itself becomes part of the expression
    if (emotionalDistance > 1.0f) {
        RuleBreak rb;
        rb.type = RuleBreakType::HarmonicAmbiguity;
        rb.intensity = std::min(1.0f, emotionalDistance / 2.0f);
        rb.description = "Dramatic structural shift for emotional journey";
        rb.justification = "The emotional journey from " + emotionA.name +
                    " to " + emotionB.name + " requires breaking conventional structure";
        ruleBreaks.push_back(rb);
    }

    // If transitioning across valence boundaries (negative ↔ positive)
    // Add harmony breaks to represent the modal/emotional shift
    bool crossesValenceBoundary = (emotionA.valence < 0 && emotionB.valence > 0) ||
                                  (emotionA.valence > 0 && emotionB.valence < 0);

    if (crossesValenceBoundary) {
        RuleBreak rb;
        rb.type = RuleBreakType::ModalMixture;
        rb.intensity = std::min(1.0f, std::abs(valenceDiff) * 0.7f);
        rb.description = "Modal interchange and harmonic exploration for transition";
        rb.justification = "Crossing emotional boundaries requires harmonic exploration";
        ruleBreaks.push_back(rb);
    }

    // If arousal changes dramatically, add rhythm breaks for the transition
    if (std::abs(arousalDiff) > 0.5f) {
        RuleBreak rb;
        rb.type = RuleBreakType::CrossRhythm;
        rb.intensity = std::min(1.0f, std::abs(arousalDiff) * 0.8f);
        rb.description = "Rhythmic transformation for arousal change";
        rb.justification = "Significant arousal change requires rhythmic transformation";
        ruleBreaks.push_back(rb);
    }

    return ruleBreaks;
}

} // namespace kelly
