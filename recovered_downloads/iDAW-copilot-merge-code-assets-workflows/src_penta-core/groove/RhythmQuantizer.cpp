#include "penta/groove/RhythmQuantizer.h"
#include <algorithm>
#include <cmath>

namespace penta::groove {

RhythmQuantizer::RhythmQuantizer(const Config& config)
    : config_(config)
{
    // TODO: Week 10 implementation - rhythm quantization with swing
}

uint64_t RhythmQuantizer::quantize(
    uint64_t samplePosition,
    uint64_t samplesPerBeat,
    uint64_t barStartPosition
) const noexcept {
    // Get grid interval
    uint64_t gridInterval = getGridInterval(samplesPerBeat);
    
    // Find nearest grid point
    uint64_t nearestGrid = findNearestGridPoint(samplePosition, gridInterval, barStartPosition);
    
    // Apply quantization strength
    int64_t diff = static_cast<int64_t>(nearestGrid) - static_cast<int64_t>(samplePosition);
    int64_t quantized = samplePosition + static_cast<int64_t>(diff * config_.strength);
    
    return static_cast<uint64_t>(quantized);
}

uint64_t RhythmQuantizer::applySwing(
    uint64_t samplePosition,
    uint64_t samplesPerBeat,
    uint64_t barStartPosition
) const noexcept {
    if (!config_.enableSwing || config_.swingAmount <= 0.0f) {
        return samplePosition;
    }
    
    // Stub implementation - TODO Week 10
    // Apply swing by delaying every other subdivision
    (void)samplesPerBeat;
    (void)barStartPosition;
    return samplePosition;
}

uint64_t RhythmQuantizer::getGridInterval(uint64_t samplesPerBeat) const noexcept {
    int divisor = static_cast<int>(config_.resolution);
    return samplesPerBeat / divisor;
}

void RhythmQuantizer::updateConfig(const Config& config) noexcept {
    config_ = config;
}

uint64_t RhythmQuantizer::findNearestGridPoint(
    uint64_t position,
    uint64_t gridInterval,
    uint64_t barStart
) const noexcept {
    if (gridInterval == 0) return position;
    
    // Calculate position relative to bar
    int64_t relativePos = static_cast<int64_t>(position - barStart);
    
    // Find nearest grid point
    int64_t gridIndex = (relativePos + gridInterval / 2) / gridInterval;
    int64_t gridPos = gridIndex * gridInterval;
    
    return barStart + static_cast<uint64_t>(gridPos);
}

} // namespace penta::groove
