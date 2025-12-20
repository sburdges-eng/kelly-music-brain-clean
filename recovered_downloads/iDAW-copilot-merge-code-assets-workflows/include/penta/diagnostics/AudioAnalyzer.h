#pragma once

#include <atomic>
#include <cstddef>

namespace penta::diagnostics {

/**
 * Real-time audio analysis
 * Monitors levels, clipping, and signal quality
 */
class AudioAnalyzer {
public:
    AudioAnalyzer();
    ~AudioAnalyzer() = default;
    
    // RT-safe: Analyze audio buffer
    void analyze(const float* buffer, size_t frames, size_t channels) noexcept;
    
    // RT-safe: Get RMS level (0.0-1.0)
    float getRmsLevel() const noexcept;
    
    // RT-safe: Get peak level (0.0-1.0+)
    float getPeakLevel() const noexcept;
    
    // RT-safe: Check if clipping detected
    bool isClipping() const noexcept { return clipping_.load(); }
    
    // RT-safe: Get dynamic range estimate (dB)
    float getDynamicRange() const noexcept;
    
    // Configuration
    void setClippingThreshold(float threshold) noexcept;
    void setDecayRate(float rate) noexcept; // For peak hold
    
    // Non-RT: Reset statistics
    void reset();
    
private:
    std::atomic<float> rmsLevel_;
    std::atomic<float> peakLevel_;
    std::atomic<bool> clipping_;
    std::atomic<float> minLevel_;
    std::atomic<float> maxLevel_;
    
    float clippingThreshold_;
    float decayRate_;
    
    static constexpr float kEpsilon = 1e-10f;
};

} // namespace penta::diagnostics
