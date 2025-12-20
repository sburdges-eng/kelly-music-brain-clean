#include "engine/VADCalculator.h"
#include "common/MusicConstants.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace kelly {
using namespace MusicConstants;

VADCalculator::VADCalculator(const EmotionThesaurus* thesaurus)
    : thesaurus_(thesaurus) {
}

VADState VADCalculator::calculateFromEmotionId(int emotionId, float intensityModifier) const {
    // Validate inputs
    if (!thesaurus_) {
        return VADState(0.0f, 0.5f, 0.5f);  // Neutral default
    }

    // Clamp intensity modifier
    intensityModifier = std::clamp(intensityModifier, 0.0f, 2.0f);

    auto emotionOpt = thesaurus_->findById(emotionId);
    if (!emotionOpt) {
        return VADState(0.0f, 0.5f, 0.5f);  // Neutral default
    }

    return calculateFromEmotion(*emotionOpt, intensityModifier);
}

VADState VADCalculator::calculateFromEmotion(const EmotionNode& emotion, float intensityModifier) const {
    VADState state;

    // Valence and arousal come directly from emotion
    state.valence = emotion.valence;
    state.arousal = emotion.arousal;

    // Apply intensity modifier to arousal
    state.arousal *= intensityModifier;
    state.arousal = std::clamp(state.arousal, 0.0f, 1.0f);

    // Calculate dominance from emotion characteristics
    state.dominance = calculateDominanceFromEmotion(emotion);

    // Apply intensity modifier to dominance
    state.dominance *= intensityModifier;
    state.dominance = std::clamp(state.dominance, 0.0f, 1.0f);

    // Set timestamp
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    state.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;

    state.clamp();
    return state;
}

VADState VADCalculator::calculateFromBiometrics(const BiometricInput::BiometricData& biometricData) const {
    VADState state;

    // Use existing BiometricInput logic for valence and arousal
    BiometricInput bioInput;
    auto emotionFromBio = bioInput.processBiometricData(biometricData);

    state.valence = emotionFromBio.valence;
    state.arousal = emotionFromBio.arousal;

    // Calculate dominance from biometrics (primarily HRV)
    state.dominance = calculateDominanceFromBiometrics(biometricData);

    // Set timestamp (use current time if not provided)
    if (biometricData.timestamp > 0.0) {
        state.timestamp = biometricData.timestamp;
    } else {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        state.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;
    }

    // Final validation and normalization
    state.clamp();
    return normalizeVADState(state);
}

VADState VADCalculator::blendStates(const std::vector<VADState>& states, const std::vector<float>& weights) const {
    if (states.empty()) {
        return VADState(0.0f, 0.5f, 0.5f);
    }

    // Normalize weights if they don't sum to 1.0
    std::vector<float> normalizedWeights = weights;
    if (weights.size() != states.size()) {
        normalizedWeights.resize(states.size(), 1.0f / states.size());
    }

    float weightSum = std::accumulate(normalizedWeights.begin(), normalizedWeights.end(), 0.0f);
    if (weightSum > 0.0f) {
        for (auto& w : normalizedWeights) {
            w /= weightSum;
        }
    } else {
        // Equal weights if sum is zero
        std::fill(normalizedWeights.begin(), normalizedWeights.end(), 1.0f / states.size());
    }

    VADState blended;
    for (size_t i = 0; i < states.size() && i < normalizedWeights.size(); ++i) {
        blended.valence += states[i].valence * normalizedWeights[i];
        blended.arousal += states[i].arousal * normalizedWeights[i];
        blended.dominance += states[i].dominance * normalizedWeights[i];
    }

    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    blended.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;

    blended.clamp();
    return blended;
}

VADState VADCalculator::applyContextAdjustments(
    const VADState& state,
    int hourOfDay,
    int dayOfWeek,
    float circadianPhase
) const {
    VADState adjusted = state;

    // Calculate circadian phase if not provided
    if (circadianPhase < 0.0f) {
        circadianPhase = hourOfDay / 24.0f;
    }

    // Apply circadian adjustments
    float arousalMod = getCircadianArousalModifier(hourOfDay);
    float valenceMod = getCircadianValenceModifier(hourOfDay);
    float dominanceMod = getCircadianDominanceModifier(hourOfDay);

    // Apply modifiers (multiplicative, then clamp)
    adjusted.arousal *= (1.0f + arousalMod);
    adjusted.valence += valenceMod;  // Additive for valence (can go negative)
    adjusted.dominance *= (1.0f + dominanceMod);

    // Apply day-of-week modifier (subtle effect)
    float dayMod = getDayOfWeekModifier(dayOfWeek);
    adjusted.valence += dayMod * 0.1f;  // Small adjustment

    adjusted.clamp();
    return adjusted;
}

float VADCalculator::calculateDominanceFromEmotion(const EmotionNode& emotion) {
    // Dominance model:
    // - High arousal + positive valence = high dominance (excitement, joy, confidence)
    // - High arousal + negative valence = low dominance (panic, fear, overwhelm)
    // - Low arousal + positive valence = moderate dominance (contentment, peace)
    // - Low arousal + negative valence = low dominance (sadness, grief, resignation)

    float valenceNorm = (emotion.valence + 1.0f) / 2.0f;  // Normalize to 0-1

    // Base dominance from valence (positive = more control)
    float baseDominance = 0.3f + valenceNorm * 0.4f;  // 0.3-0.7 range

    // Arousal modifies dominance:
    // - High arousal amplifies the valence effect
    // - Low arousal reduces dominance regardless of valence
    float arousalEffect = (emotion.arousal - 0.5f) * 0.3f;  // -0.15 to +0.15

    // For negative valence + high arousal, dominance should be very low
    if (emotion.valence < VALENCE_NEUTRAL && emotion.arousal > AROUSAL_HIGH) {
        baseDominance *= 0.5f;  // Reduce dominance significantly
    }

    // For positive valence + high arousal, boost dominance
    if (emotion.valence > VALENCE_NEUTRAL && emotion.arousal > AROUSAL_HIGH) {
        baseDominance += 0.2f;
    }

    float dominance = baseDominance + arousalEffect;
    return std::clamp(dominance, 0.0f, 1.0f);
}

float VADCalculator::calculateDominanceFromBiometrics(const BiometricInput::BiometricData& data) {
    // HRV (Heart Rate Variability) is the primary indicator of dominance
    // High HRV = autonomic flexibility = sense of control = higher dominance
    // Low HRV = stress response = feeling overwhelmed = lower dominance

    float dominance = 0.5f;  // Default neutral

    // If HRV is available, use it directly (most accurate)
    if (data.heartRateVariability) {
        float hrv = *data.heartRateVariability;
        // HRV typically measured in milliseconds (RMSSD or SDNN)
        // Normal HRV: 20-60ms (varies by person)
        // High HRV (good): >40ms = higher dominance
        // Low HRV (stress): <20ms = lower dominance

        if (hrv >= 40.0f) {
            dominance = 0.6f + std::min((hrv - 40.0f) / 60.0f, 0.4f);  // 0.6-1.0
        } else if (hrv >= 20.0f) {
            dominance = 0.4f + (hrv - 20.0f) / 20.0f * 0.2f;  // 0.4-0.6
        } else {
            dominance = 0.2f + (hrv / 20.0f) * 0.2f;  // 0.2-0.4
        }
        dominance = std::clamp(dominance, 0.0f, 1.0f);
        return dominance;  // HRV is most reliable, return early
    }

    // Fallback: infer HRV from heart rate stability
    // More stable HR = higher HRV = higher dominance
    // More variable HR = lower HRV = lower dominance

    if (data.heartRate) {
        // Use heart rate as proxy for HRV
        // Resting HR (60-80) with low variability = higher dominance
        // Elevated HR (>100) or very low HR (<50) = lower dominance

        float hr = *data.heartRate;
        if (hr >= 60.0f && hr <= 80.0f) {
            dominance = 0.6f;  // Optimal range = higher dominance
        } else if (hr > 100.0f || hr < 50.0f) {
            dominance = 0.3f;  // Stress or bradycardia = lower dominance
        } else {
            // Linear interpolation
            if (hr < 60.0f) {
                dominance = 0.3f + (hr - 50.0f) / 10.0f * 0.3f;  // 50-60: 0.3-0.6
            } else {
                dominance = 0.6f - (hr - 80.0f) / 20.0f * 0.3f;  // 80-100: 0.6-0.3
            }
        }
    }

    // Temperature can also indicate stress (elevated = lower dominance)
    if (data.temperature) {
        float temp = *data.temperature;
        // Normal: 36.5-37.5°C
        // Elevated (>37.5) or low (<36.0) = stress = lower dominance
        if (temp > 37.5f || temp < 36.0f) {
            dominance *= 0.8f;
        }
    }

    // EDA (skin conductance) - higher = stress = lower dominance
    if (data.skinConductance) {
        float eda = *data.skinConductance;
        // Normal: 1-5 microsiemens
        // High (>10) = stress = lower dominance
        if (eda > 10.0f) {
            dominance *= 0.7f;
        } else if (eda < 3.0f) {
            dominance *= 1.1f;  // Low EDA = calm = slightly higher dominance
        }
        dominance = std::clamp(dominance, 0.0f, 1.0f);
    }

    return std::clamp(dominance, 0.0f, 1.0f);
}

float VADCalculator::calculateDominanceFromVA(float valence, float arousal) const {
    float valenceNorm = (valence + 1.0f) / 2.0f;
    float baseDominance = 0.3f + valenceNorm * 0.4f;
    float arousalEffect = (arousal - AROUSAL_MODERATE) * 0.3f;

    if (valence < VALENCE_NEUTRAL && arousal > AROUSAL_HIGH) {
        baseDominance *= 0.5f;
    }
    if (valence > VALENCE_NEUTRAL && arousal > AROUSAL_HIGH) {
        baseDominance += 0.2f;
    }

    return std::clamp(baseDominance + arousalEffect, 0.0f, 1.0f);
}

float VADCalculator::getCircadianArousalModifier(int hourOfDay) const {
    // Circadian rhythm for arousal:
    // - Low in early morning (4-6am): -0.3
    // - Rising through morning (6-10am): -0.1 to +0.1
    // - Peak in afternoon (2-4pm): +0.2
    // - Declining evening (6-10pm): 0.0 to -0.1
    // - Low at night (10pm-4am): -0.2

    if (hourOfDay >= 4 && hourOfDay < 6) {
        return -0.3f;  // Very low in early morning
    } else if (hourOfDay >= 6 && hourOfDay < 10) {
        return -0.1f + (hourOfDay - 6) / 4.0f * 0.2f;  // Rising
    } else if (hourOfDay >= 10 && hourOfDay < 14) {
        return 0.1f + (hourOfDay - 10) / 4.0f * 0.1f;  // Peak building
    } else if (hourOfDay >= 14 && hourOfDay < 18) {
        return 0.2f - (hourOfDay - 14) / 4.0f * 0.2f;  // Peak declining
    } else if (hourOfDay >= 18 && hourOfDay < 22) {
        return 0.0f - (hourOfDay - 18) / 4.0f * 0.1f;  // Evening decline
    } else {
        return -0.2f;  // Night (22-4)
    }
}

float VADCalculator::getCircadianValenceModifier(int hourOfDay) const {
    // Circadian rhythm for valence (mood):
    // - Slightly negative in early morning: -0.1
    // - Positive in morning/afternoon: +0.1
    // - Neutral in evening: 0.0
    // - Slightly negative at night: -0.05

    if (hourOfDay >= 4 && hourOfDay < 8) {
        return -0.1f;  // Early morning blues
    } else if (hourOfDay >= 8 && hourOfDay < 16) {
        return 0.1f;  // Positive daytime mood
    } else if (hourOfDay >= 16 && hourOfDay < 20) {
        return 0.05f;  // Slightly positive evening
    } else {
        return -0.05f;  // Night
    }
}

float VADCalculator::getCircadianDominanceModifier(int hourOfDay) const {
    // Circadian rhythm for dominance (sense of control):
    // - Low in early morning: -0.2
    // - Rising through day: -0.1 to +0.1
    // - Peak in afternoon: +0.15
    // - Declining evening: 0.0 to -0.1

    if (hourOfDay >= 4 && hourOfDay < 8) {
        return -0.2f;  // Low control in early morning
    } else if (hourOfDay >= 8 && hourOfDay < 12) {
        return -0.1f + (hourOfDay - 8) / 4.0f * 0.2f;  // Rising
    } else if (hourOfDay >= 12 && hourOfDay < 16) {
        return 0.1f + (hourOfDay - 12) / 4.0f * 0.05f;  // Peak
    } else if (hourOfDay >= 16 && hourOfDay < 20) {
        return 0.15f - (hourOfDay - 16) / 4.0f * 0.15f;  // Declining
    } else {
        return 0.0f;  // Evening/night
    }
}

float VADCalculator::getDayOfWeekModifier(int dayOfWeek) const {
    // Day of week effects (subtle):
    // - Monday: -0.1 (Monday blues)
    // - Friday: +0.1 (TGIF)
    // - Weekend: +0.05 (slight positive)
    // - Others: 0.0

    switch (dayOfWeek) {
        case 1: return -0.1f;  // Monday
        case 5: return 0.1f;   // Friday
        case 0: case 6: return 0.05f;  // Weekend
        default: return 0.0f;
    }
}

VADState VADCalculator::smoothMovingAverage(
    const std::vector<VADState>& states,
    size_t windowSize
) {
    if (states.empty()) {
        return VADState(0.0f, 0.5f, 0.5f);
    }

    size_t actualWindow = std::min(windowSize, states.size());
    size_t startIdx = states.size() - actualWindow;

    VADState smoothed;
    for (size_t i = startIdx; i < states.size(); ++i) {
        smoothed.valence += states[i].valence;
        smoothed.arousal += states[i].arousal;
        smoothed.dominance += states[i].dominance;
    }

    float n = static_cast<float>(actualWindow);
    smoothed.valence /= n;
    smoothed.arousal /= n;
    smoothed.dominance /= n;

    if (!states.empty()) {
        smoothed.timestamp = states.back().timestamp;
    }

    smoothed.clamp();
    return smoothed;
}

VADState VADCalculator::smoothExponential(
    const VADState& current,
    const VADState& previous,
    float alpha
) {
    // Exponential smoothing: S_t = α * X_t + (1 - α) * S_{t-1}
    alpha = std::clamp(alpha, 0.0f, 1.0f);

    VADState smoothed;
    smoothed.valence = alpha * current.valence + (1.0f - alpha) * previous.valence;
    smoothed.arousal = alpha * current.arousal + (1.0f - alpha) * previous.arousal;
    smoothed.dominance = alpha * current.dominance + (1.0f - alpha) * previous.dominance;
    smoothed.timestamp = current.timestamp;

    smoothed.clamp();
    return smoothed;
}

VADState VADCalculator::smoothWeighted(
    const std::vector<VADState>& states,
    const std::vector<float>& weights
) {
    if (states.empty()) {
        return VADState(0.0f, 0.5f, 0.5f);
    }

    std::vector<float> normalizedWeights = weights;
    if (weights.size() != states.size()) {
        // Generate default weights (more recent = higher weight)
        normalizedWeights.resize(states.size());
        float totalWeight = 0.0f;
        for (size_t i = 0; i < states.size(); ++i) {
            normalizedWeights[i] = static_cast<float>(i + 1);  // Linear weighting
            totalWeight += normalizedWeights[i];
        }
        for (auto& w : normalizedWeights) {
            w /= totalWeight;
        }
    } else {
        // Normalize provided weights
        float totalWeight = std::accumulate(normalizedWeights.begin(), normalizedWeights.end(), 0.0f);
        if (totalWeight > 0.0f) {
            for (auto& w : normalizedWeights) {
                w /= totalWeight;
            }
        }
    }

    VADState smoothed;
    for (size_t i = 0; i < states.size() && i < normalizedWeights.size(); ++i) {
        smoothed.valence += states[i].valence * normalizedWeights[i];
        smoothed.arousal += states[i].arousal * normalizedWeights[i];
        smoothed.dominance += states[i].dominance * normalizedWeights[i];
    }

    if (!states.empty()) {
        smoothed.timestamp = states.back().timestamp;
    }

    smoothed.clamp();
    return smoothed;
}

VADState VADCalculator::smoothKalman(
    const VADState& current,
    const VADState& previous,
    float processNoise,
    float measurementNoise
) {
    // Simplified Kalman filter for 1D (applied to each dimension)
    // For each dimension: x_est = x_prev + K * (x_meas - x_prev)
    // where K = P / (P + R), P = process noise, R = measurement noise

    VADState filtered;

    // Kalman gain
    float K = processNoise / (processNoise + measurementNoise);
    K = std::clamp(K, 0.0f, 1.0f);

    // Filter each dimension
    filtered.valence = previous.valence + K * (current.valence - previous.valence);
    filtered.arousal = previous.arousal + K * (current.arousal - previous.arousal);
    filtered.dominance = previous.dominance + K * (current.dominance - previous.dominance);
    filtered.timestamp = current.timestamp;

    filtered.clamp();
    return filtered;
}

bool VADCalculator::isValidEmotionId(int emotionId) const {
    if (!thesaurus_) {
        return false;
    }

    auto emotion = thesaurus_->findById(emotionId);
    return emotion.has_value();
}

bool VADCalculator::isValidVADState(const VADState& state) {
    return state.valence >= -1.0f && state.valence <= 1.0f &&
           state.arousal >= 0.0f && state.arousal <= 1.0f &&
           state.dominance >= 0.0f && state.dominance <= 1.0f;
}

VADState VADCalculator::normalizeVADState(const VADState& state) {
    VADState normalized = state;
    normalized.clamp();
    return normalized;
}

VADState VADCalculator::calculateFromEmotionName(
    const std::string& emotionName,
    float intensityModifier
) const {
    if (!thesaurus_) {
        return VADState(0.0f, 0.5f, 0.5f);
    }

    auto emotionOpt = thesaurus_->findByName(emotionName);
    if (!emotionOpt) {
        return VADState(0.0f, 0.5f, 0.5f);
    }

    return calculateFromEmotion(*emotionOpt, intensityModifier);
}

VADState VADCalculator::interpolate(
    const VADState& state1,
    const VADState& state2,
    float t
) {
    t = std::clamp(t, 0.0f, 1.0f);

    VADState interpolated;
    interpolated.valence = state1.valence + t * (state2.valence - state1.valence);
    interpolated.arousal = state1.arousal + t * (state2.arousal - state1.arousal);
    interpolated.dominance = state1.dominance + t * (state2.dominance - state1.dominance);

    // Interpolate timestamp
    interpolated.timestamp = state1.timestamp + t * (state2.timestamp - state1.timestamp);

    interpolated.clamp();
    return interpolated;
}

VADState VADCalculator::calculateRateOfChange(
    const VADState& state1,
    const VADState& state2,
    float deltaTime
) {
    VADState rate;

    if (deltaTime > 0.0f) {
        rate.valence = (state2.valence - state1.valence) / deltaTime;
        rate.arousal = (state2.arousal - state1.arousal) / deltaTime;
        rate.dominance = (state2.dominance - state1.dominance) / deltaTime;
    } else {
        rate.valence = 0.0f;
        rate.arousal = 0.0f;
        rate.dominance = 0.0f;
    }

    rate.timestamp = state2.timestamp;
    return rate;
}

} // namespace kelly
