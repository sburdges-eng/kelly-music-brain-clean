#include "ParameterMorphEngine.h"
#include <algorithm>
#include <cmath>

namespace kelly {

ParameterMorphEngine::ParameterMorphEngine() {
    startTimer(16);  // ~60fps for smooth interpolation
}

ParameterMorphEngine::~ParameterMorphEngine() {
    stopTimer();
}

void ParameterMorphEngine::morphParameter(const juce::String& parameterName, float targetValue, int durationMs) {
    std::lock_guard<std::mutex> lock(morphMutex_);

    auto& state = morphStates_[parameterName];
    state.startValue = state.active ? getCurrentValue(parameterName) : targetValue;
    state.targetValue = targetValue;
    state.startTime = juce::Time::currentTimeMillis();
    state.durationMs = durationMs;
    state.active = true;
}

void ParameterMorphEngine::morphParameters(const std::map<juce::String, float>& targets, int durationMs) {
    std::lock_guard<std::mutex> lock(morphMutex_);

    for (const auto& [name, target] : targets) {
        auto& state = morphStates_[name];
        state.startValue = state.active ? getCurrentValue(name) : target;
        state.targetValue = target;
        state.startTime = juce::Time::currentTimeMillis();
        state.durationMs = durationMs;
        state.active = true;
    }
}

float ParameterMorphEngine::getCurrentValue(const juce::String& parameterName) const {
    std::lock_guard<std::mutex> lock(morphMutex_);

    auto it = morphStates_.find(parameterName);
    if (it == morphStates_.end() || !it->second.active) {
        return 0.0f;
    }

    const auto& state = it->second;
    auto now = juce::Time::currentTimeMillis();
    auto elapsed = now - state.startTime;

    if (elapsed >= state.durationMs) {
        return state.targetValue;
    }

    float t = static_cast<float>(elapsed) / static_cast<float>(state.durationMs);
    return interpolate(state.startValue, state.targetValue, t, curveType_);
}

bool ParameterMorphEngine::isMorphing(const juce::String& parameterName) const {
    std::lock_guard<std::mutex> lock(morphMutex_);

    auto it = morphStates_.find(parameterName);
    return it != morphStates_.end() && it->second.active;
}

void ParameterMorphEngine::stopMorph(const juce::String& parameterName) {
    std::lock_guard<std::mutex> lock(morphMutex_);

    auto it = morphStates_.find(parameterName);
    if (it != morphStates_.end()) {
        it->second.active = false;
    }
}

void ParameterMorphEngine::stopAllMorphs() {
    std::lock_guard<std::mutex> lock(morphMutex_);

    for (auto& [name, state] : morphStates_) {
        state.active = false;
    }
}

void ParameterMorphEngine::timerCallback() {
    std::vector<juce::String> completed;
    std::map<juce::String, float> updates;

    {
        std::lock_guard<std::mutex> lock(morphMutex_);

        auto now = juce::Time::currentTimeMillis();

        for (auto& [name, state] : morphStates_) {
            if (!state.active) continue;

            auto elapsed = now - state.startTime;

            if (elapsed >= state.durationMs) {
                // Morph complete
                updates[name] = state.targetValue;
                state.active = false;
                completed.push_back(name);
            } else {
                // Still morphing - get current value
                float t = static_cast<float>(elapsed) / static_cast<float>(state.durationMs);
                float current = interpolate(state.startValue, state.targetValue, t, curveType_);
                updates[name] = current;
            }
        }
    }

    // Notify callbacks (outside lock)
    for (const auto& [name, value] : updates) {
        if (onParameterUpdated) {
            onParameterUpdated(name, value);
        }
    }

    for (const auto& name : completed) {
        if (onMorphComplete) {
            onMorphComplete(name);
        }
    }
}

float ParameterMorphEngine::interpolate(float start, float end, float t, CurveType curve) const {
    t = juce::jlimit(0.0f, 1.0f, t);
    float curvedT = applyCurve(t, curve);
    return start + (end - start) * curvedT;
}

float ParameterMorphEngine::applyCurve(float t, CurveType curve) const {
    switch (curve) {
        case CurveType::Linear:
            return t;

        case CurveType::EaseIn:
            return t * t;

        case CurveType::EaseOut:
            return 1.0f - (1.0f - t) * (1.0f - t);

        case CurveType::EaseInOut:
            return t < 0.5f
                ? 2.0f * t * t
                : 1.0f - std::pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;

        case CurveType::Exponential:
            return t == 0.0f ? 0.0f : std::pow(2.0f, 10.0f * (t - 1.0f));

        default:
            return t;
    }
}

} // namespace kelly
