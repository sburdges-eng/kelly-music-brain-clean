#pragma once
/*
 * PreferenceTracker.h - User Preference Learning System
 * ====================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Bridge Layer: PreferenceBridge (Python UserPreferenceModel integration)
 * - UI Layer: Used by EmotionWorkstation (tracks user adjustments)
 * - Plugin Layer: Used by PluginProcessor (preference learning)
 * - Learning Layer: Core preference tracking and learning system
 *
 * Purpose: Tracks parameter adjustments, emotion selections, and generation feedback.
 *          Can be bridged to Python UserPreferenceModel for full learning capabilities.
 *
 * Features:
 * - Parameter adjustment tracking
 * - Emotion selection tracking
 * - MIDI generation event tracking
 * - Feedback (thumbs up/down) tracking
 * - Context-aware learning
 */

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>

namespace kelly {

/**
 * PreferenceTracker - C++ interface for user preference learning
 *
 * Tracks parameter adjustments, emotion selections, and generation feedback.
 * Can be bridged to Python UserPreferenceModel for full learning capabilities.
 */
class PreferenceTracker {
public:
    PreferenceTracker();
    ~PreferenceTracker() = default;

    /**
     * Record a parameter adjustment
     */
    void recordParameterAdjustment(
        const std::string& parameterName,
        float oldValue,
        float newValue,
        const std::map<std::string, std::string>& context = {}
    );

    /**
     * Record an emotion selection
     */
    void recordEmotionSelection(
        const std::string& emotionName,
        float valence,
        float arousal,
        float intensity,
        const std::map<std::string, std::string>& context = {}
    );

    /**
     * Record a MIDI generation event
     */
    void recordMidiGeneration(
        const std::string& generationId,
        const std::string& intentText,
        const std::map<std::string, float>& parameters,
        const std::string& emotion = "",
        const std::vector<std::string>& ruleBreaks = {}
    );

    /**
     * Record explicit feedback (thumbs up/down)
     */
    void recordMidiFeedback(const std::string& generationId, bool accepted);

    /**
     * Record a parameter modification after generation
     */
    void recordMidiModification(
        const std::string& generationId,
        const std::string& parameterName,
        float oldValue,
        float newValue
    );

    /**
     * Enable/disable preference tracking
     */
    void setEnabled(bool enabled) { enabled_.store(enabled); }
    bool isEnabled() const { return enabled_.load(); }

    /**
     * Clear all preferences
     */
    void clearPreferences();

    /**
     * Set callback for when preferences should be saved (async)
     */
    void setSaveCallback(std::function<void()> callback) { saveCallback_ = callback; }

private:
    std::atomic<bool> enabled_{true};
    std::mutex dataMutex_;

    // In-memory storage (can be bridged to Python for persistence)
    struct ParameterAdjustment {
        std::string parameterName;
        float oldValue;
        float newValue;
        std::string timestamp;
        std::map<std::string, std::string> context;
    };

    struct EmotionSelection {
        std::string emotionName;
        float valence;
        float arousal;
        float intensity;
        std::string timestamp;
        std::map<std::string, std::string> context;
    };

    struct MidiGenerationEvent {
        std::string generationId;
        std::string intentText;
        std::map<std::string, float> parameters;
        std::string emotion;
        std::vector<std::string> ruleBreaks;
        std::string timestamp;
        std::optional<bool> accepted;
        std::vector<ParameterAdjustment> modifications;
    };

    std::vector<ParameterAdjustment> parameterAdjustments_;
    std::vector<EmotionSelection> emotionSelections_;
    std::vector<MidiGenerationEvent> midiGenerations_;

    std::function<void()> saveCallback_;

    void triggerSave();
    std::string getCurrentTimestamp() const;
};

} // namespace kelly
