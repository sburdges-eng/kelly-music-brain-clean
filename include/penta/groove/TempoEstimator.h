#pragma once

#include <cstdint>
#include <vector>

namespace penta::groove {

/**
 * Real-time tempo estimation using autocorrelation
 * Tracks tempo changes with adaptive filtering
 */
class TempoEstimator {
public:
    struct Config {
        double sampleRate;
        float minTempo;
        float maxTempo;
        float adaptationRate; // How quickly to adapt to changes
        size_t historySize;     // Number of onsets to consider
        
        Config()
            : sampleRate(48000.0)
            , minTempo(60.0f)
            , maxTempo(180.0f)
            , adaptationRate(0.1f)
            , historySize(32)
        {}
    };
    
    explicit TempoEstimator(const Config& config = Config{});
    ~TempoEstimator() = default;
    
    // RT-safe: Add onset time for tempo calculation
    void addOnset(uint64_t samplePosition) noexcept;
    
    // RT-safe: Get current tempo estimate
    float getCurrentTempo() const noexcept { return currentTempo_; }
    
    // RT-safe: Get confidence of tempo estimate (0.0-1.0)
    float getConfidence() const noexcept { return confidence_; }
    
    // RT-safe: Get samples per beat
    uint64_t getSamplesPerBeat() const noexcept;
    
    // Configuration
    void updateConfig(const Config& config) noexcept;
    void reset() noexcept;
    
private:
    void estimateTempo() noexcept;
    float findBestInterval(const std::vector<float>& intervals) const noexcept;
    float computeCorrelation(const std::vector<float>& intervals, float testInterval) const noexcept;
    
    Config config_;
    
    std::vector<uint64_t> onsetHistory_;
    float currentTempo_;
    float confidence_;
    uint64_t lastOnsetPosition_;
};

} // namespace penta::groove
