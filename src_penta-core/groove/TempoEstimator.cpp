#include "penta/groove/TempoEstimator.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace penta::groove {

TempoEstimator::TempoEstimator(const Config& config)
    : config_(config)
    , currentTempo_(120.0f)
    , confidence_(0.0f)
    , lastOnsetPosition_(0)
{
    onsetHistory_.reserve(config.historySize);
    // Autocorrelation-based tempo estimation implemented
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
    if (onsetHistory_.size() < 4) {
        return;  // Need at least 4 onsets
    }
    
    // Calculate inter-onset intervals (IOI) in seconds
    std::vector<float> intervals;
    intervals.reserve(onsetHistory_.size() - 1);
    
    for (size_t i = 1; i < onsetHistory_.size(); ++i) {
        uint64_t ioi = onsetHistory_[i] - onsetHistory_[i - 1];
        float ioiSeconds = static_cast<float>(ioi) / static_cast<float>(config_.sampleRate);
        intervals.push_back(ioiSeconds);
    }
    
    // Search for best-fitting tempo using autocorrelation-style scoring
    float bestInterval = findBestInterval(intervals);
    if (bestInterval <= 0.0f) {
        return;
    }
    
    // Convert interval to BPM and clamp to configured bounds
    float estimatedTempo = 60.0f / bestInterval;
    estimatedTempo = std::clamp(estimatedTempo, config_.minTempo, config_.maxTempo);
    
    // Apply adaptive smoothing to avoid jittery tempo updates
    currentTempo_ = currentTempo_ * (1.0f - config_.adaptationRate) +
                    estimatedTempo * config_.adaptationRate;
    
    // Confidence derived from best correlation score
    confidence_ = std::clamp(computeCorrelation(intervals, bestInterval), 0.0f, 1.0f);
}

float TempoEstimator::findBestInterval(const std::vector<float>& intervals) const noexcept {
    if (intervals.empty()) {
        return 0.0f;
    }
    
    float bestInterval = 0.0f;
    float bestScore = -std::numeric_limits<float>::infinity();
    
    // Sweep a reasonable tempo range to find the strongest periodicity
    for (float testTempo = config_.minTempo; testTempo <= config_.maxTempo; testTempo += 0.5f) {
        float testInterval = 60.0f / testTempo;
        float score = computeCorrelation(intervals, testInterval);
        if (score > bestScore) {
            bestScore = score;
            bestInterval = testInterval;
        }
    }
    
    return bestInterval;
}

float TempoEstimator::computeCorrelation(
    const std::vector<float>& intervals,
    float testInterval
) const noexcept {
    if (intervals.empty() || testInterval <= 0.0f) {
        return 0.0f;
    }
    
    // Allow for slight human timing variations (~12% tolerance)
    float tolerance = testInterval * 0.12f;
    float invTwoSigmaSq = 1.0f / (2.0f * tolerance * tolerance);
    
    float score = 0.0f;
    for (float interval : intervals) {
        float multiple = std::max(1.0f, std::round(interval / testInterval));
        float expected = multiple * testInterval;
        float error = std::abs(interval - expected);
        
        // Gaussian weighting: tighter clustering yields higher score
        score += std::exp(-(error * error) * invTwoSigmaSq);
    }
    
    return score / static_cast<float>(intervals.size());
}

} // namespace penta::groove
