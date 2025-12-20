#pragma once

#include "penta/common/RTTypes.h"
#include "penta/groove/OnsetDetector.h"
#include "penta/groove/TempoEstimator.h"
#include "penta/groove/RhythmQuantizer.h"
#include <memory>
#include <vector>

namespace penta::groove {

/**
 * Main groove analysis engine
 * Combines onset detection, tempo estimation, and rhythm quantization
 */
class GrooveEngine {
public:
    struct Config {
        double sampleRate;
        size_t hopSize;
        float minTempo;
        float maxTempo;
        bool enableQuantization;
        float quantizationStrength;
        
        Config()
            : sampleRate(kDefaultSampleRate)
            , hopSize(512)
            , minTempo(60.0f)
            , maxTempo(180.0f)
            , enableQuantization(true)
            , quantizationStrength(0.8f)
        {}
    };
    
    struct GrooveAnalysis {
        float currentTempo;
        float tempoConfidence;
        std::vector<uint64_t> onsetPositions;
        std::vector<float> onsetStrengths;
        uint32_t timeSignatureNum;
        uint32_t timeSignatureDen;
        float swing;  // 0.0 = straight, 1.0 = maximum swing
    };
    
    explicit GrooveEngine(const Config& config = Config{});
    ~GrooveEngine();
    
    // Non-copyable, movable
    GrooveEngine(const GrooveEngine&) = delete;
    GrooveEngine& operator=(const GrooveEngine&) = delete;
    GrooveEngine(GrooveEngine&&) noexcept = default;
    GrooveEngine& operator=(GrooveEngine&&) noexcept = default;
    
    // RT-safe: Process audio buffer for groove analysis
    void processAudio(const float* buffer, size_t frames) noexcept;
    
    // RT-safe: Get current groove analysis
    const GrooveAnalysis& getAnalysis() const noexcept { return analysis_; }
    
    // RT-safe: Quantize timestamp to grid
    uint64_t quantizeToGrid(uint64_t timestamp) const noexcept;
    
    // RT-safe: Get swing-adjusted position
    uint64_t applySwing(uint64_t position) const noexcept;
    
    // Non-RT: Update configuration
    void updateConfig(const Config& config);
    
    // Non-RT: Reset analysis
    void reset();
    
private:
    void updateTempoEstimate() noexcept;
    void detectTimeSignature() noexcept;
    void analyzeSwing() noexcept;
    
    Config config_;
    GrooveAnalysis analysis_;
    
    std::unique_ptr<OnsetDetector> onsetDetector_;
    std::unique_ptr<TempoEstimator> tempoEstimator_;
    std::unique_ptr<RhythmQuantizer> quantizer_;
    
    uint64_t samplePosition_;
    std::vector<uint64_t> onsetHistory_;
};

} // namespace penta::groove
