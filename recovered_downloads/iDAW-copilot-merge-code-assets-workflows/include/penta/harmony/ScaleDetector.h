#pragma once

#include "penta/common/RTTypes.h"
#include <array>

namespace penta::harmony {

/**
 * Real-time scale detection using Krumhansl-Schmuckler algorithm
 * Enhanced with chromatic profile correlation
 */
class ScaleDetector {
public:
    ScaleDetector();
    ~ScaleDetector() = default;
    
    // RT-safe: Analyze pitch class distribution
    Scale analyze(const std::array<bool, 12>& pitchClassSet) noexcept;
    
    // RT-safe: Update with weighted pitch class histogram
    void update(const std::array<float, 12>& pitchClassWeights) noexcept;
    
    // RT-safe: Get current detected scale
    const Scale& getCurrentScale() const noexcept { return currentScale_; }
    
    // Configuration
    void setConfidenceThreshold(float threshold) noexcept;
    void setDecayFactor(float factor) noexcept; // Temporal decay
    
private:
    struct ScaleProfile {
        std::array<float, 12> weights;
        uint8_t mode;
        const char* name;
    };
    
    float correlateWithProfile(
        const std::array<float, 12>& histogram,
        const ScaleProfile& profile,
        uint8_t tonic
    ) const noexcept;
    
    void findBestScale(
        const std::array<float, 12>& histogram,
        Scale& outScale
    ) noexcept;
    
    static const std::array<ScaleProfile, 7> kMajorMinorProfiles;
    
    Scale currentScale_;
    std::array<float, 12> pitchClassHistogram_;
    float confidenceThreshold_;
    float decayFactor_;
};

} // namespace penta::harmony
