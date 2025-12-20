#include "engine/ResonanceCalculator.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace kelly {

ResonanceCalculator::ResonanceCalculator() {
}

ResonanceMetrics ResonanceCalculator::calculateResonance(
    const VADState& emotionVAD,
    const VADState& biometricVAD
) const {
    ResonanceMetrics metrics;
    
    // Calculate distance in VAD space
    float distance = emotionVAD.distanceTo(biometricVAD);
    float maxDistance = std::sqrt(3.0f);  // Max distance in 3D unit cube
    
    // Convert distance to match score (closer = higher match)
    metrics.emotionBiometricMatch = 1.0f - (distance / maxDistance);
    metrics.emotionBiometricMatch = std::clamp(metrics.emotionBiometricMatch, 0.0f, 1.0f);
    
    // Individual dimension matches
    float valenceMatch = 1.0f - std::abs(emotionVAD.valence - biometricVAD.valence) / 2.0f;
    float arousalMatch = 1.0f - std::abs(emotionVAD.arousal - biometricVAD.arousal);
    float dominanceMatch = 1.0f - std::abs(emotionVAD.dominance - biometricVAD.dominance);
    
    // Overall coherence is weighted average
    metrics.coherence = (valenceMatch * 0.4f + arousalMatch * 0.4f + dominanceMatch * 0.2f);
    
    return metrics;
}

float ResonanceCalculator::calculateBiometricCoherence(
    const BiometricInput::BiometricData& biometricData,
    const std::vector<BiometricInput::BiometricData>& history
) const {
    if (history.empty()) {
        return 0.5f;  // Neutral if no history
    }
    
    // Check if signals are within expected physiological ranges
    float rangeScore = 1.0f;
    
    if (biometricData.heartRate) {
        float hr = *biometricData.heartRate;
        if (hr < EXPECTED_HR_MIN || hr > EXPECTED_HR_MAX) {
            rangeScore *= 0.7f;  // Penalize out-of-range
        }
    }
    
    if (biometricData.skinConductance) {
        float eda = *biometricData.skinConductance;
        if (eda < EXPECTED_EDA_MIN || eda > EXPECTED_EDA_MAX) {
            rangeScore *= 0.8f;
        }
    }
    
    if (biometricData.temperature) {
        float temp = *biometricData.temperature;
        if (temp < EXPECTED_TEMP_MIN || temp > EXPECTED_TEMP_MAX) {
            rangeScore *= 0.6f;
        }
    }
    
    // Calculate variance in recent history
    std::vector<float> hrValues, edaValues, tempValues;
    
    for (const auto& h : history) {
        if (h.heartRate) hrValues.push_back(*h.heartRate);
        if (h.skinConductance) edaValues.push_back(*h.skinConductance);
        if (h.temperature) tempValues.push_back(*h.temperature);
    }
    
    // Low variance = higher coherence (signals are stable)
    float varianceScore = 1.0f;
    
    if (hrValues.size() > 1) {
        float hrVar = calculateVariance(hrValues);
        float hrStdDev = std::sqrt(hrVar);
        // Normal HR variability: 5-15 BPM
        if (hrStdDev > 20.0f) {
            varianceScore *= 0.7f;  // Too variable
        } else if (hrStdDev < 2.0f) {
            varianceScore *= 0.9f;  // Too stable (possibly sensor issue)
        }
    }
    
    if (edaValues.size() > 1) {
        float edaVar = calculateVariance(edaValues);
        float edaStdDev = std::sqrt(edaVar);
        // Normal EDA variability: 1-3 microsiemens
        if (edaStdDev > 5.0f) {
            varianceScore *= 0.8f;
        }
    }
    
    // Cross-signal consistency
    // HR and EDA should correlate (both increase with arousal)
    float crossSignalScore = 1.0f;
    if (hrValues.size() > 1 && edaValues.size() > 1 && hrValues.size() == edaValues.size()) {
        // Simple correlation check
        float hrMean = calculateMean(hrValues);
        float edaMean = calculateMean(edaValues);
        
        float covariance = 0.0f;
        for (size_t i = 0; i < hrValues.size(); ++i) {
            covariance += (hrValues[i] - hrMean) * (edaValues[i] - edaMean);
        }
        covariance /= hrValues.size();
        
        float hrStd = calculateStandardDeviation(hrValues);
        float edaStd = calculateStandardDeviation(edaValues);
        
        if (hrStd > 0.0f && edaStd > 0.0f) {
            float correlation = covariance / (hrStd * edaStd);
            // Positive correlation expected (both increase with stress/arousal)
            if (correlation < -0.3f) {
                crossSignalScore *= 0.7f;  // Negative correlation is suspicious
            }
        }
    }
    
    float coherence = (rangeScore * 0.4f + varianceScore * 0.4f + crossSignalScore * 0.2f);
    return std::clamp(coherence, 0.0f, 1.0f);
}

float ResonanceCalculator::calculateTemporalStability(
    const std::vector<VADState>& vadHistory
) const {
    if (vadHistory.size() < 2) {
        return 0.5f;  // Neutral if insufficient data
    }
    
    // Calculate variance in each dimension
    std::vector<float> valenceValues, arousalValues, dominanceValues;
    
    for (const auto& state : vadHistory) {
        valenceValues.push_back(state.valence);
        arousalValues.push_back(state.arousal);
        dominanceValues.push_back(state.dominance);
    }
    
    float valenceVar = calculateVariance(valenceValues);
    float arousalVar = calculateVariance(arousalValues);
    float dominanceVar = calculateVariance(dominanceValues);
    
    // Normalize variance (max variance in unit range is ~0.25)
    float maxVar = 0.25f;
    float valenceStability = 1.0f - std::min(valenceVar / maxVar, 1.0f);
    float arousalStability = 1.0f - std::min(arousalVar / maxVar, 1.0f);
    float dominanceStability = 1.0f - std::min(dominanceVar / maxVar, 1.0f);
    
    // Overall stability is average
    float stability = (valenceStability + arousalStability + dominanceStability) / 3.0f;
    return std::clamp(stability, 0.0f, 1.0f);
}

float ResonanceCalculator::calculateCorrelation(
    const std::vector<VADState>& states1,
    const std::vector<VADState>& states2,
    int dimension
) const {
    if (states1.size() != states2.size() || states1.size() < 2) {
        return 0.0f;
    }
    
    std::vector<float> values1, values2;
    
    for (const auto& state : states1) {
        switch (dimension) {
            case 0: values1.push_back(state.valence); break;
            case 1: values1.push_back(state.arousal); break;
            case 2: values1.push_back(state.dominance); break;
        }
    }
    
    for (const auto& state : states2) {
        switch (dimension) {
            case 0: values2.push_back(state.valence); break;
            case 1: values2.push_back(state.arousal); break;
            case 2: values2.push_back(state.dominance); break;
        }
    }
    
    float mean1 = calculateMean(values1);
    float mean2 = calculateMean(values2);
    
    float covariance = 0.0f;
    for (size_t i = 0; i < values1.size(); ++i) {
        covariance += (values1[i] - mean1) * (values2[i] - mean2);
    }
    covariance /= values1.size();
    
    float std1 = calculateStandardDeviation(values1);
    float std2 = calculateStandardDeviation(values2);
    
    if (std1 > 0.0f && std2 > 0.0f) {
        return std::clamp(covariance / (std1 * std2), -1.0f, 1.0f);
    }
    
    return 0.0f;
}

std::vector<float> ResonanceCalculator::detectAnomalies(
    const std::vector<VADState>& vadHistory
) const {
    if (vadHistory.size() < 3) {
        return std::vector<float>(vadHistory.size(), 0.0f);
    }
    
    std::vector<float> anomalies;
    
    // Calculate rolling mean and std dev
    const size_t windowSize = std::min(size_t(5), vadHistory.size());
    
    for (size_t i = 0; i < vadHistory.size(); ++i) {
        size_t start = (i >= windowSize) ? i - windowSize : 0;
        size_t end = i + 1;
        
        std::vector<float> valenceWindow, arousalWindow, dominanceWindow;
        for (size_t j = start; j < end; ++j) {
            valenceWindow.push_back(vadHistory[j].valence);
            arousalWindow.push_back(vadHistory[j].arousal);
            dominanceWindow.push_back(vadHistory[j].dominance);
        }
        
        float vMean = calculateMean(valenceWindow);
        float aMean = calculateMean(arousalWindow);
        float dMean = calculateMean(dominanceWindow);
        
        float vStd = calculateStandardDeviation(valenceWindow);
        float aStd = calculateStandardDeviation(arousalWindow);
        float dStd = calculateStandardDeviation(dominanceWindow);
        
        // Calculate z-scores
        float vZ = (vStd > 0.0f) ? std::abs((vadHistory[i].valence - vMean) / vStd) : 0.0f;
        float aZ = (aStd > 0.0f) ? std::abs((vadHistory[i].arousal - aMean) / aStd) : 0.0f;
        float dZ = (dStd > 0.0f) ? std::abs((vadHistory[i].dominance - dMean) / dStd) : 0.0f;
        
        // Anomaly score: max z-score normalized to 0-1
        float maxZ = std::max({vZ, aZ, dZ});
        float anomalyScore = std::min(maxZ / 3.0f, 1.0f);  // Z > 3 is very anomalous
        
        anomalies.push_back(anomalyScore);
    }
    
    return anomalies;
}

float ResonanceCalculator::calculateVariance(const std::vector<float>& values) const {
    if (values.size() < 2) return 0.0f;
    
    float mean = calculateMean(values);
    float variance = 0.0f;
    
    for (float v : values) {
        float diff = v - mean;
        variance += diff * diff;
    }
    
    return variance / values.size();
}

float ResonanceCalculator::calculateMean(const std::vector<float>& values) const {
    if (values.empty()) return 0.0f;
    return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
}

float ResonanceCalculator::calculateStandardDeviation(const std::vector<float>& values) const {
    float variance = calculateVariance(values);
    return std::sqrt(variance);
}

} // namespace kelly
