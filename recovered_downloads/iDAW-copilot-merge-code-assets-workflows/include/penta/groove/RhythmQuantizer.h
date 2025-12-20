#pragma once

#include <cstdint>

namespace penta::groove {

/**
 * Real-time rhythm quantization
 * Snaps timestamps to musical grid with configurable strength
 */
class RhythmQuantizer {
public:
    enum class GridResolution {
        Whole = 1,
        Half = 2,
        Quarter = 4,
        Eighth = 8,
        Sixteenth = 16,
        ThirtySecond = 32
    };
    
    struct Config {
        GridResolution resolution;
        float strength;     // 0.0 = no quantize, 1.0 = full quantize
        bool enableSwing;
        float swingAmount;  // 0.0 = straight, 1.0 = maximum swing
        uint32_t timeSignatureNum;
        uint32_t timeSignatureDen;
        
        Config()
            : resolution(GridResolution::Sixteenth)
            , strength(0.8f)
            , enableSwing(false)
            , swingAmount(0.5f)
            , timeSignatureNum(4)
            , timeSignatureDen(4)
        {}
    };
    
    explicit RhythmQuantizer(const Config& config = Config{});
    ~RhythmQuantizer() = default;
    
    // RT-safe: Quantize sample position to grid
    uint64_t quantize(
        uint64_t samplePosition,
        uint64_t samplesPerBeat,
        uint64_t barStartPosition
    ) const noexcept;
    
    // RT-safe: Apply swing to position
    uint64_t applySwing(
        uint64_t samplePosition,
        uint64_t samplesPerBeat,
        uint64_t barStartPosition
    ) const noexcept;
    
    // RT-safe: Get grid subdivision at position
    uint64_t getGridInterval(uint64_t samplesPerBeat) const noexcept;
    
    // Configuration
    void updateConfig(const Config& config) noexcept;
    
private:
    uint64_t findNearestGridPoint(
        uint64_t position,
        uint64_t gridInterval,
        uint64_t barStart
    ) const noexcept;
    
    Config config_;
};

} // namespace penta::groove
