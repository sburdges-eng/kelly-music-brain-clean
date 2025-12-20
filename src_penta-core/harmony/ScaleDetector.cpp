#include "penta/harmony/ScaleDetector.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace penta::harmony {

// Krumhansl-Schmuckler key profiles
// Based on empirical research of perceived stability of pitch classes in major/minor keys
const std::array<ScaleDetector::ScaleProfile, 7> ScaleDetector::kMajorMinorProfiles = {{
    // Major profile (Ionian) - empirical weights from Krumhansl & Kessler 1982
    {{6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 4.09f, 2.52f, 5.19f, 2.39f, 3.66f, 2.29f, 2.88f}, 0, "Major"},
    
    // Minor profile (Aeolian) - empirical weights
    {{6.33f, 2.68f, 3.52f, 5.38f, 2.60f, 3.53f, 2.54f, 4.75f, 3.98f, 2.69f, 3.34f, 3.17f}, 1, "Minor"},
    
    // Dorian mode
    {{6.35f, 2.23f, 3.48f, 4.38f, 2.33f, 4.09f, 2.52f, 5.19f, 3.66f, 2.39f, 2.29f, 2.88f}, 2, "Dorian"},
    
    // Phrygian mode
    {{6.33f, 3.52f, 2.68f, 5.38f, 2.60f, 3.53f, 2.54f, 4.75f, 2.69f, 3.98f, 3.34f, 3.17f}, 3, "Phrygian"},
    
    // Lydian mode
    {{6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 2.52f, 4.09f, 5.19f, 2.39f, 3.66f, 2.29f, 2.88f}, 4, "Lydian"},
    
    // Mixolydian mode
    {{6.35f, 2.23f, 3.48f, 2.33f, 4.38f, 4.09f, 2.52f, 5.19f, 2.39f, 3.66f, 2.88f, 2.29f}, 5, "Mixolydian"},
    
    // Locrian mode
    {{6.33f, 3.52f, 2.68f, 5.38f, 2.60f, 3.53f, 4.75f, 2.54f, 2.69f, 3.98f, 3.34f, 3.17f}, 6, "Locrian"},
}};

ScaleDetector::ScaleDetector()
    : confidenceThreshold_(0.5f)
    , decayFactor_(0.95f)
{
    pitchClassHistogram_.fill(0.0f);
}

Scale ScaleDetector::analyze(const std::array<bool, 12>& pitchClassSet) noexcept {
    // Convert boolean pitch class set to weighted histogram
    std::array<float, 12> histogram;
    for (int i = 0; i < 12; ++i) {
        histogram[i] = pitchClassSet[i] ? 1.0f : 0.0f;
    }
    
    Scale result;
    findBestScale(histogram, result);
    return result;
}

void ScaleDetector::update(const std::array<float, 12>& pitchClassWeights) noexcept {
    // Apply temporal decay to existing histogram
    for (int i = 0; i < 12; ++i) {
        pitchClassHistogram_[i] *= decayFactor_;
        pitchClassHistogram_[i] += pitchClassWeights[i];
    }
    
    // Find best matching scale
    findBestScale(pitchClassHistogram_, currentScale_);
}

void ScaleDetector::setConfidenceThreshold(float threshold) noexcept {
    confidenceThreshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

void ScaleDetector::setDecayFactor(float factor) noexcept {
    decayFactor_ = std::clamp(factor, 0.0f, 1.0f);
}

float ScaleDetector::correlateWithProfile(
    const std::array<float, 12>& histogram,
    const ScaleProfile& profile,
    uint8_t tonic
) const noexcept {
    // Pearson correlation coefficient between histogram and profile
    
    // Rotate profile to match tonic
    std::array<float, 12> rotatedProfile;
    for (int i = 0; i < 12; ++i) {
        rotatedProfile[i] = profile.weights[(i + tonic) % 12];
    }
    
    // Calculate means
    float histMean = std::accumulate(histogram.begin(), histogram.end(), 0.0f) / 12.0f;
    float profMean = std::accumulate(rotatedProfile.begin(), rotatedProfile.end(), 0.0f) / 12.0f;
    
    // Calculate correlation
    float numerator = 0.0f;
    float histVariance = 0.0f;
    float profVariance = 0.0f;
    
    for (int i = 0; i < 12; ++i) {
        float histDev = histogram[i] - histMean;
        float profDev = rotatedProfile[i] - profMean;
        
        numerator += histDev * profDev;
        histVariance += histDev * histDev;
        profVariance += profDev * profDev;
    }
    
    // Avoid division by zero
    float denominator = std::sqrt(histVariance * profVariance);
    if (denominator < 1e-6f) {
        return 0.0f;
    }
    
    // Return correlation coefficient (range: -1 to 1)
    return numerator / denominator;
}

void ScaleDetector::findBestScale(
    const std::array<float, 12>& histogram,
    Scale& outScale
) noexcept {
    float bestCorrelation = -1.0f;
    uint8_t bestTonic = 0;
    uint8_t bestMode = 0;
    
    // Try all profiles at all tonics
    for (uint8_t tonic = 0; tonic < 12; ++tonic) {
        for (const auto& profile : kMajorMinorProfiles) {
            float correlation = correlateWithProfile(histogram, profile, tonic);
            
            if (correlation > bestCorrelation) {
                bestCorrelation = correlation;
                bestTonic = tonic;
                bestMode = profile.mode;
            }
        }
    }
    
    // Convert correlation (-1 to 1) to confidence (0 to 1)
    // Scale and shift so that correlation of 0 gives confidence of 0.5
    float confidence = (bestCorrelation + 1.0f) * 0.5f;
    
    outScale.tonic = bestTonic;
    outScale.mode = bestMode;
    outScale.confidence = confidence;
    
    // Copy the pitch class histogram to scale degrees
    for (int i = 0; i < 12; ++i) {
        outScale.degrees[i] = histogram[i] > 0.1f;  // Threshold for boolean representation
    }
}

} // namespace penta::harmony
