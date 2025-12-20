#include "biometric/AdaptiveNormalizer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace kelly {
namespace biometric {

AdaptiveNormalizer::AdaptiveNormalizer() {
    // Initialize with default baseline
    currentBaseline_.heartRate = 70.0f;
    currentBaseline_.heartRateVariability = 50.0f;
    currentBaseline_.skinConductance = 5.0f;
    currentBaseline_.temperature = 36.5f;
}

void AdaptiveNormalizer::addReading(const BiometricInput::BiometricData& data) {
    TimestampedData timestamped;
    timestamped.data = data;
    timestamped.timestamp = std::chrono::system_clock::now();

    history_.push_back(timestamped);

    // Limit history size
    if (history_.size() > MAX_HISTORY_SIZE) {
        history_.pop_front();
    }
}

AdaptiveNormalizer::Baseline AdaptiveNormalizer::calculateBaseline(int days) {
    Baseline baseline;

    auto windowData = getDataInWindow(days);

    if (windowData.empty()) {
        return currentBaseline_;  // Return cached baseline if no data
    }

    // Calculate averages
    float sumHR = 0.0f, sumHRV = 0.0f, sumSC = 0.0f, sumTemp = 0.0f;
    int countHR = 0, countHRV = 0, countSC = 0, countTemp = 0;

    for (const auto& data : windowData) {
        if (data.heartRate) {
            sumHR += *data.heartRate;
            countHR++;
        }
        if (data.heartRateVariability) {
            sumHRV += *data.heartRateVariability;
            countHRV++;
        }
        if (data.skinConductance) {
            sumSC += *data.skinConductance;
            countSC++;
        }
        if (data.temperature) {
            sumTemp += *data.temperature;
            countTemp++;
        }
    }

    if (countHR > 0) baseline.heartRate = sumHR / countHR;
    if (countHRV > 0) baseline.heartRateVariability = sumHRV / countHRV;
    if (countSC > 0) baseline.skinConductance = sumSC / countSC;
    if (countTemp > 0) baseline.temperature = sumTemp / countTemp;

    baseline.sampleCount = static_cast<int>(windowData.size());

    return baseline;
}

AdaptiveNormalizer::NormalizedData AdaptiveNormalizer::normalize(
    const BiometricInput::BiometricData& data,
    const Baseline& baseline)
{
    NormalizedData normalized;

    // Normalize heart rate (relative to baseline ±30 BPM range)
    if (data.heartRate && baseline.heartRate > 0.0f) {
        float deviation = *data.heartRate - baseline.heartRate;
        normalized.heartRate = 0.5f + (deviation / 60.0f);  // ±30 BPM = ±0.5
        normalized.heartRate = std::clamp(normalized.heartRate, 0.0f, 1.0f);
    }

    // Normalize HRV (higher HRV = calmer = higher value)
    if (data.heartRateVariability && baseline.heartRateVariability > 0.0f) {
        float ratio = *data.heartRateVariability / baseline.heartRateVariability;
        normalized.hrv = std::clamp(ratio, 0.0f, 2.0f) / 2.0f;  // 0-2x baseline -> 0-1.0
    }

    // Normalize skin conductance (relative to baseline)
    if (data.skinConductance && baseline.skinConductance > 0.0f) {
        float ratio = *data.skinConductance / baseline.skinConductance;
        normalized.skinConductance = std::clamp(ratio, 0.0f, 2.0f) / 2.0f;
    }

    // Normalize temperature (relative to baseline ±1°C)
    if (data.temperature && baseline.temperature > 0.0f) {
        float deviation = *data.temperature - baseline.temperature;
        normalized.temperature = 0.5f + (deviation / 2.0f);  // ±1°C = ±0.5
        normalized.temperature = std::clamp(normalized.temperature, 0.0f, 1.0f);
    }

    return normalized;
}

void AdaptiveNormalizer::updateBaseline(int days) {
    currentBaseline_ = calculateBaseline(days);
}

void AdaptiveNormalizer::clearHistory() {
    history_.clear();
}

std::vector<BiometricInput::BiometricData> AdaptiveNormalizer::getDataInWindow(int days) {
    std::vector<BiometricInput::BiometricData> result;

    if (history_.empty()) {
        return result;
    }

    auto cutoff = std::chrono::system_clock::now() - std::chrono::hours(24 * days);

    for (const auto& entry : history_) {
        if (entry.timestamp >= cutoff) {
            result.push_back(entry.data);
        }
    }

    return result;
}

} // namespace biometric
} // namespace kelly
