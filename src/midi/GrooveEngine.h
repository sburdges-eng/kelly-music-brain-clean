#pragma once

#include "common/Types.h"
#include <vector>
#include <random>
#include <map>
#include <string>

namespace kelly {

/**
 * Groove Template - Predefined groove patterns with timing and velocity characteristics.
 * Based on genre-specific patterns (funk, jazz, rock, hiphop, etc.)
 */
struct GrooveTemplateData {
    std::string name;
    std::string description;
    float swingFactor;              // 0.0 = straight, 0.67 = triplet swing
    float pushPull;                 // -1.0 = laid back, +1.0 = pushed ahead
    std::vector<float> timingDeviations;  // Per-16th-note timing offsets (in beats)
    std::vector<int> velocityCurve;         // Per-16th-note velocity values (0-127)
    std::pair<int, int> tempoRange;        // (min, max) BPM
};

/**
 * Groove Engine - Applies humanization and groove patterns to MIDI.
 * 
 * CRITICAL FEATURES:
 * - Humanization: Micro-timing, velocity variations, ghost notes
 * - Timing Feel: Push/pull (ahead/behind the beat)
 * - Groove Templates: Predefined patterns (funk, jazz, rock, hiphop, etc.)
 * - Swing: Variable swing amounts (0.0 = straight, 0.67 = triplet swing)
 */
class GrooveEngine {
public:
    GrooveEngine();
    ~GrooveEngine() = default;
    
    /**
     * Apply groove/humanization to MIDI notes.
     * @param notes Input notes
     * @param grooveType Type of groove to apply
     * @param humanizationLevel 0.0 (tight) to 1.0 (loose)
     * @return Humanized notes
     */
    std::vector<MidiNote> applyGroove(
        const std::vector<MidiNote>& notes,
        GrooveType grooveType,
        float humanizationLevel
    );
    
    /**
     * Apply groove template with full timing and velocity characteristics.
     * @param notes Input notes
     * @param templateName Name of template (e.g., "funk", "jazz", "rock")
     * @param humanizationLevel 0.0 (tight) to 1.0 (loose)
     * @param intensity 0.0 to 1.0, how much to apply the template
     * @return Grooved notes
     */
    std::vector<MidiNote> applyGrooveTemplate(
        const std::vector<MidiNote>& notes,
        const std::string& templateName,
        float humanizationLevel = 0.5f,
        float intensity = 1.0f
    );
    
    /**
     * Apply swing to notes.
     * @param notes Input notes
     * @param swingAmount 0.0 = straight, 0.5 = moderate, 0.67 = triplet swing
     * @param intensity 0.0 to 1.0, how much to apply swing
     * @return Swung notes
     */
    std::vector<MidiNote> applySwing(
        const std::vector<MidiNote>& notes,
        float swingAmount,
        float intensity = 1.0f
    );
    
    /**
     * Apply push/pull timing feel.
     * @param notes Input notes
     * @param feel -1.0 (laid back/pull) to +1.0 (pushed ahead)
     * @param intensity 0.0 to 1.0, how much to apply the feel
     * @return Notes with timing feel applied
     */
    std::vector<MidiNote> applyTimingFeel(
        const std::vector<MidiNote>& notes,
        float feel,
        float intensity = 1.0f
    );
    
    /**
     * Apply humanization with micro-timing and velocity variations.
     * @param notes Input notes
     * @param humanizationLevel 0.0 (tight) to 1.0 (loose)
     * @param microTimingBias Consistent early/late tendency (-1.0 to +1.0)
     * @param velocityVariation 0.0 to 1.0, amount of velocity randomization
     * @return Humanized notes
     */
    std::vector<MidiNote> humanize(
        const std::vector<MidiNote>& notes,
        float humanizationLevel,
        float microTimingBias = 0.0f,
        float velocityVariation = 0.1f
    );
    
    /**
     * Apply emotion-based timing adjustments.
     * Sad emotions drag behind, angry emotions rush ahead.
     */
    std::vector<MidiNote> applyEmotionTiming(
        const std::vector<MidiNote>& notes,
        const EmotionNode& emotion
    );
    
    /**
     * Get available groove template names.
     */
    std::vector<std::string> getTemplateNames() const;
    
    /**
     * Get groove template by name.
     */
    const GrooveTemplateData* getTemplate(const std::string& name) const;

private:
    std::mt19937 rng_;  // Random number generator for humanization
    std::map<std::string, GrooveTemplateData> templates_;
    
    // Initialize predefined groove templates
    void initializeTemplates();
    
    // Get timing deviation for a specific beat position
    float getTimingDeviation(const GrooveTemplateData& grooveTemplate, double beatPosition) const;

    // Get velocity from template for a specific beat position
    int getTemplateVelocity(const GrooveTemplateData& grooveTemplate, double beatPosition, int originalVelocity) const;
    
    // Calculate swing offset for a beat position
    float calculateSwingOffset(double beatPosition, float swingAmount) const;
    
    // Get 16th note position within a beat (0-15)
    int getSixteenthNotePosition(double beatPosition) const;
};

} // namespace kelly

