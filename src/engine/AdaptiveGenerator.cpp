#include "AdaptiveGenerator.h"
#include <algorithm>
#include <cmath>

namespace kelly {

AdaptiveGenerator::AdaptiveGenerator(KellyBrain& brain, PreferenceTracker& preferenceTracker)
    : brain_(brain), preferenceTracker_(preferenceTracker) {
}

GeneratedMidi AdaptiveGenerator::generateMidi(const IntentResult& intent, int bars) {
    IntentResult adaptedIntent = intent;

    if (adaptiveEnabled_.load() && preferenceTracker_.isEnabled()) {
        adaptedIntent = adaptIntent(intent);
    }

    return brain_.generateMidi(adaptedIntent, bars);
}

GeneratedMidi AdaptiveGenerator::generateMidiFromWound(const Wound& wound, int bars) {
    // Process wound normally first
    IntentResult baseIntent = brain_.fromWound(wound);

    // Adapt if enabled
    if (adaptiveEnabled_.load() && preferenceTracker_.isEnabled()) {
        baseIntent = adaptIntent(baseIntent);
    }

    return brain_.generateMidi(baseIntent, bars);
}

IntentResult AdaptiveGenerator::adaptIntent(const IntentResult& baseIntent) {
    IntentResult adapted = baseIntent;

    // Get preferred adjustments (simplified - would need to query PreferenceTracker
    // for actual learned preferences, but for now we just pass through)
    // TODO: Implement actual preference-based adjustment logic

    return adapted;
}

std::map<std::string, float> AdaptiveGenerator::getPreferredAdjustments() const {
    // TODO: Query PreferenceTracker for learned parameter preferences
    // For now, return empty map
    return {};
}

} // namespace kelly
