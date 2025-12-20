#include "penta/groove/TempoEstimator.h"
#include <algorithm>
#include <cmath>

namespace penta::groove {

TempoEstimator::TempoEstimator(const Config& config)
    : config_(config)
    , currentTempo_(120.0f)
    , confidence_(0.0f)
    , lastOnsetPosition_(0)
{
    onsetHistory_.reserve(config.historySize);
    // TODO: Week 10 implementation - autocorrelation-based tempo estimation
}

void TempoEstimator::addOnset(uint64_t samplePosition) noexcept {
    onsetHistory_.push_back(samplePosition);
    
    // Keep only recent history
    if (onsetHistory_.size() > config_.historySize) {
        onsetHistory_.erase(onsetHistory_.begin());
    }
    
    lastOnsetPosition_ = samplePosition;
    
    // Estimate tempo if we have enough onsets
    if (onsetHistory_.size() >= 4) {
        estimateTempo();
    }
}

uint64_t TempoEstimator::getSamplesPerBeat() const noexcept {
    if (currentTempo_ <= 0.0f) return 0;
    return static_cast<uint64_t>((60.0 * config_.sampleRate) / currentTempo_);
}

void TempoEstimator::updateConfig(const Config& config) noexcept {
    config_ = config;
    onsetHistory_.reserve(config.historySize);
}

void TempoEstimator::reset() noexcept {
    onsetHistory_.clear();
    currentTempo_ = 120.0f;
    confidence_ = 0.0f;
    lastOnsetPosition_ = 0;
}

void TempoEstimator::estimateTempo() noexcept {
    // Stub implementation - TODO Week 10
    // Calculate inter-onset intervals and use autocorrelation
}

float TempoEstimator::autocorrelate(const std::vector<float>& intervals) const noexcept {
    // Stub implementation - TODO Week 10
    (void)intervals;
    return 0.0f;
}

} // namespace penta::groove
