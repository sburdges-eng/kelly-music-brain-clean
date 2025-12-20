#include "penta/groove/GrooveEngine.h"

namespace penta::groove {

GrooveEngine::GrooveEngine(const Config& config)
    : config_(config)
    , analysis_{}
    , onsetDetector_(std::make_unique<OnsetDetector>())
    , tempoEstimator_(std::make_unique<TempoEstimator>())
    , quantizer_(std::make_unique<RhythmQuantizer>())
    , samplePosition_(0)
{
    analysis_.currentTempo = 120.0f;
    analysis_.tempoConfidence = 0.0f;
    analysis_.timeSignatureNum = 4;
    analysis_.timeSignatureDen = 4;
    analysis_.swing = 0.0f;
    // TODO: Week 3-4 implementation
}

GrooveEngine::~GrooveEngine() = default;

void GrooveEngine::processAudio(const float* buffer, size_t frames) noexcept {
    if (onsetDetector_) {
        onsetDetector_->process(buffer, frames);
        
        if (onsetDetector_->hasOnset()) {
            uint64_t onsetPos = onsetDetector_->getOnsetPosition();
            float onsetStrength = onsetDetector_->getOnsetStrength();
            analysis_.onsetPositions.push_back(onsetPos);
            analysis_.onsetStrengths.push_back(onsetStrength);
        }
    }
    
    samplePosition_ += frames;
}

uint64_t GrooveEngine::quantizeToGrid(uint64_t timestamp) const noexcept {
    // Stub implementation
    return timestamp;
}

uint64_t GrooveEngine::applySwing(uint64_t position) const noexcept {
    // Stub implementation
    return position;
}

void GrooveEngine::updateConfig(const Config& config) {
    config_ = config;
}

void GrooveEngine::reset() {
    if (onsetDetector_) onsetDetector_->reset();
    if (tempoEstimator_) tempoEstimator_->reset();
    samplePosition_ = 0;
    onsetHistory_.clear();
    analysis_ = GrooveAnalysis{};
}

void GrooveEngine::updateTempoEstimate() noexcept {
    // Stub implementation - TODO Week 3
}

void GrooveEngine::detectTimeSignature() noexcept {
    // Stub implementation - TODO Week 3
}

void GrooveEngine::analyzeSwing() noexcept {
    // Stub implementation - TODO Week 4
}

} // namespace penta::groove
