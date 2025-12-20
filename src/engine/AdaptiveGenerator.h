#pragma once
/*
 * AdaptiveGenerator.h - Preference-Adaptive MIDI Generation Engine
 * ================================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Engine Layer: KellyBrain (high-level API wrapper)
 * - Learning Layer: PreferenceTracker (user preference learning)
 * - Type System: KellyTypes.h (IntentResult, GeneratedMidi, Wound)
 * - Engine Layer: Adapts generation based on learned user preferences
 *
 * Purpose: Wraps KellyBrain to adapt generation based on user preferences.
 *          Part of Phase 4 of the "All-Knowing Interactive Musical Customization System".
 *          Adjusts IntentResult based on learned user preferences before MIDI generation.
 *
 * Features:
 * - Preference-based parameter adjustment
 * - Adaptive MIDI generation
 * - User preference learning integration
 */

#include "MidiKompanionBrain.h"  // High-level API
#include "common/KellyTypes.h"  // IntentResult, GeneratedMidi, Wound
#include "learning/PreferenceTracker.h"  // User preference learning
#include <memory>
#include <map>
#include <string>

namespace kelly {

/**
 * AdaptiveGenerator - Wraps KellyBrain to adapt generation based on user preferences
 *
 * Part of Phase 4 of the "All-Knowing Interactive Musical Customization System".
 * Wraps KellyBrain and adjusts IntentResult based on learned user preferences.
 */
class AdaptiveGenerator {
public:
    AdaptiveGenerator(KellyBrain& brain, PreferenceTracker& preferenceTracker);
    ~AdaptiveGenerator() = default;

    /**
     * Generate MIDI with adaptive parameters
     */
    GeneratedMidi generateMidi(const IntentResult& intent, int bars = 8);

    /**
     * Generate MIDI from wound with adaptation
     */
    GeneratedMidi generateMidiFromWound(const Wound& wound, int bars = 8);

    /**
     * Enable/disable adaptive learning
     */
    void setAdaptiveEnabled(bool enabled) { adaptiveEnabled_.store(enabled); }
    bool isAdaptiveEnabled() const { return adaptiveEnabled_.load(); }

private:
    KellyBrain& brain_;
    PreferenceTracker& preferenceTracker_;
    std::atomic<bool> adaptiveEnabled_{true};

    /**
     * Adapt IntentResult based on learned preferences
     */
    IntentResult adaptIntent(const IntentResult& baseIntent);

    /**
     * Get preferred parameter adjustments from preference tracker
     */
    std::map<std::string, float> getPreferredAdjustments() const;
};

} // namespace kelly
