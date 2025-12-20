#include "PreferenceTracker.h"
#include <chrono>
#include <iomanip>
#include <sstream>

namespace kelly {

PreferenceTracker::PreferenceTracker() {
}

void PreferenceTracker::recordParameterAdjustment(
    const std::string& parameterName,
    float oldValue,
    float newValue,
    const std::map<std::string, std::string>& context
) {
    if (!enabled_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(dataMutex_);

    ParameterAdjustment adj;
    adj.parameterName = parameterName;
    adj.oldValue = oldValue;
    adj.newValue = newValue;
    adj.timestamp = getCurrentTimestamp();
    adj.context = context;

    parameterAdjustments_.push_back(adj);

    // Limit history size (keep last 1000 adjustments)
    if (parameterAdjustments_.size() > 1000) {
        parameterAdjustments_.erase(parameterAdjustments_.begin());
    }

    triggerSave();
}

void PreferenceTracker::recordEmotionSelection(
    const std::string& emotionName,
    float valence,
    float arousal,
    float intensity,
    const std::map<std::string, std::string>& context
) {
    if (!enabled_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(dataMutex_);

    EmotionSelection sel;
    sel.emotionName = emotionName;
    sel.valence = valence;
    sel.arousal = arousal;
    sel.intensity = intensity;
    sel.timestamp = getCurrentTimestamp();
    sel.context = context;

    emotionSelections_.push_back(sel);

    // Limit history size
    if (emotionSelections_.size() > 500) {
        emotionSelections_.erase(emotionSelections_.begin());
    }

    triggerSave();
}

void PreferenceTracker::recordMidiGeneration(
    const std::string& generationId,
    const std::string& intentText,
    const std::map<std::string, float>& parameters,
    const std::string& emotion,
    const std::vector<std::string>& ruleBreaks
) {
    if (!enabled_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(dataMutex_);

    MidiGenerationEvent event;
    event.generationId = generationId;
    event.intentText = intentText;
    event.parameters = parameters;
    event.emotion = emotion;
    event.ruleBreaks = ruleBreaks;
    event.timestamp = getCurrentTimestamp();
    event.accepted = std::nullopt;

    midiGenerations_.push_back(event);

    // Limit history size
    if (midiGenerations_.size() > 500) {
        midiGenerations_.erase(midiGenerations_.begin());
    }

    triggerSave();
}

void PreferenceTracker::recordMidiFeedback(const std::string& generationId, bool accepted) {
    if (!enabled_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(dataMutex_);

    // Find the generation event
    for (auto& event : midiGenerations_) {
        if (event.generationId == generationId) {
            event.accepted = accepted;
            triggerSave();
            return;
        }
    }
}

void PreferenceTracker::recordMidiModification(
    const std::string& generationId,
    const std::string& parameterName,
    float oldValue,
    float newValue
) {
    if (!enabled_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(dataMutex_);

    // Find the generation event and add modification
    for (auto& event : midiGenerations_) {
        if (event.generationId == generationId) {
            ParameterAdjustment mod;
            mod.parameterName = parameterName;
            mod.oldValue = oldValue;
            mod.newValue = newValue;
            mod.timestamp = getCurrentTimestamp();
            event.modifications.push_back(mod);
            triggerSave();
            return;
        }
    }
}

void PreferenceTracker::clearPreferences() {
    std::lock_guard<std::mutex> lock(dataMutex_);
    parameterAdjustments_.clear();
    emotionSelections_.clear();
    midiGenerations_.clear();
    triggerSave();
}

void PreferenceTracker::triggerSave() {
    // Call async save callback if set (for Python bridge)
    if (saveCallback_) {
        saveCallback_();
    }
}

std::string PreferenceTracker::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

} // namespace kelly
