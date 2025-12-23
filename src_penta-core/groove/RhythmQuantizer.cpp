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
    if (samplesPerBeat == 0) {
        return samplePosition;
    }
    
    uint64_t gridInterval = getGridInterval(samplesPerBeat);
    if (gridInterval == 0) {
        return samplePosition;
    }
    
    // Snap to nearest grid point first
    uint64_t baseGrid = findNearestGridPoint(samplePosition, gridInterval, barStartPosition);
    
    // Apply swing to the target grid position (upbeats only)
    uint64_t targetGrid = baseGrid;
    const int64_t relativePos = static_cast<int64_t>(baseGrid - barStartPosition);
    const int64_t gridIndex = relativePos / static_cast<int64_t>(gridInterval);
    
    if (config_.enableSwing && (gridIndex % 2 != 0)) {
        float swingFactor = std::clamp(config_.swingAmount, 0.0f, 1.0f);
        float delayRatio = (swingFactor - 0.5f) * 2.0f; // -1.0 .. 1.0 (0.0 = straight)
        int64_t maxDelay = static_cast<int64_t>(gridInterval / 2);
        int64_t swingOffset = static_cast<int64_t>(maxDelay * delayRatio);
        targetGrid = static_cast<uint64_t>(std::max<int64_t>(
            0, static_cast<int64_t>(targetGrid) + swingOffset));
    }
    
    // Apply quantization strength toward the (possibly swung) grid point
    int64_t diff = static_cast<int64_t>(targetGrid) - static_cast<int64_t>(samplePosition);
    int64_t quantized = static_cast<int64_t>(samplePosition) +
                        static_cast<int64_t>(diff * config_.strength);
    
    return static_cast<uint64_t>(std::max<int64_t>(0, quantized));
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
        float swingFactor = std::clamp(config_.swingAmount, 0.0f, 1.0f);
        float delayRatio = (swingFactor - 0.5f) * 2.0f; // -1.0 .. 1.0
        int64_t maxDelay = static_cast<int64_t>(gridInterval / 2);  // Half a subdivision
        int64_t swingDelay = static_cast<int64_t>(maxDelay * delayRatio);
        
        int64_t adjusted = static_cast<int64_t>(samplePosition) + swingDelay;
        return static_cast<uint64_t>(std::max<int64_t>(0, adjusted));
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
