#include "penta/groove/RhythmQuantizer.h"
#include <algorithm>
#include <cmath>

namespace penta::groove {

RhythmQuantizer::RhythmQuantizer(const Config& config)
    : config_(config)
{
    // Rhythm quantization with swing implemented
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
    
    // Get grid interval based on resolution
    uint64_t gridInterval = getGridInterval(samplesPerBeat);
    if (gridInterval == 0) {
        return samplePosition;
    }
    
    // Calculate position relative to bar
    int64_t relativePos = static_cast<int64_t>(samplePosition - barStartPosition);
    
    // Find which grid point we're at
    int64_t gridIndex = relativePos / gridInterval;
    
    // Apply swing to every other subdivision (8th notes, 16th notes, etc.)
    // Swing delays the upbeat (odd-numbered subdivisions)
    if (gridIndex % 2 == 1) {
        // This is an upbeat - delay it based on swing amount
        // Swing amount: 0.5 = straight, 0.66 = triplet feel, higher = more swing
        float swingFactor = config_.swingAmount;
        
        // Calculate delay: swing pushes upbeat later in the beat
        // At 0.5 (50%), upbeat is exactly halfway
        // At 0.66 (66%), upbeat is at 2/3 position (triplet)
        float delayRatio = (swingFactor - 0.5f) * 2.0f;  // 0.0 to 1.0 range
        int64_t maxDelay = gridInterval / 2;  // Maximum delay is half a grid
        int64_t swingDelay = static_cast<int64_t>(maxDelay * delayRatio);
        
        return samplePosition + swingDelay;
    }
    
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
