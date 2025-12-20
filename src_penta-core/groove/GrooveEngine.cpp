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
    // Tempo, time signature, and swing analysis implemented
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
    if (!quantizer_ || !config_.enableQuantization || analysis_.currentTempo <= 0.0f) {
        return timestamp;
    }

    // Calculate samples per beat based on current tempo
    uint64_t samplesPerBeat = static_cast<uint64_t>(
        (60.0 * config_.sampleRate) / analysis_.currentTempo
    );

    if (samplesPerBeat == 0) {
        return timestamp;
    }

    // Calculate bar start position (assume 4 beats per bar for now)
    uint64_t samplesPerBar = samplesPerBeat * analysis_.timeSignatureNum;
    uint64_t barStartPosition = (timestamp / samplesPerBar) * samplesPerBar;

    // Use the RhythmQuantizer to quantize to grid
    return quantizer_->quantize(timestamp, samplesPerBeat, barStartPosition);
}

uint64_t GrooveEngine::applySwing(uint64_t position) const noexcept {
    if (!quantizer_ || analysis_.currentTempo <= 0.0f) {
        return position;
    }

    // Calculate samples per beat based on current tempo
    uint64_t samplesPerBeat = static_cast<uint64_t>(
        (60.0 * config_.sampleRate) / analysis_.currentTempo
    );

    if (samplesPerBeat == 0) {
        return position;
    }

    // Calculate bar start position
    uint64_t samplesPerBar = samplesPerBeat * analysis_.timeSignatureNum;
    uint64_t barStartPosition = (position / samplesPerBar) * samplesPerBar;

    // Use the RhythmQuantizer to apply swing timing
    return quantizer_->applySwing(position, samplesPerBeat, barStartPosition);
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
    if (!tempoEstimator_) {
        return;
    }
    
    // Feed onset history to tempo estimator
    for (size_t i = 0; i < analysis_.onsetPositions.size(); ++i) {
        tempoEstimator_->addOnset(analysis_.onsetPositions[i]);
    }
    
    // Update analysis with current tempo estimate
    analysis_.currentTempo = tempoEstimator_->getCurrentTempo();
    analysis_.tempoConfidence = tempoEstimator_->getConfidence();
}

void GrooveEngine::detectTimeSignature() noexcept {
    // Simple time signature detection based on onset patterns
    if (analysis_.onsetPositions.size() < 8) {
        return;  // Need enough onsets for pattern detection
    }
    
    // Analyze spacing between strong beats
    // Look for patterns of strong/weak beats
    std::vector<float> beatStrengths;
    for (size_t i = 0; i < std::min(analysis_.onsetStrengths.size(), size_t(16)); ++i) {
        beatStrengths.push_back(analysis_.onsetStrengths[i]);
    }
    
    // Find strongest beats (potential downbeats)
    float threshold = 0.7f;  // Consider onsets above 70% strength as strong
    int strongBeatCount = 0;
    for (float strength : beatStrengths) {
        if (strength > threshold) {
            strongBeatCount++;
        }
    }
    
    // Estimate time signature based on pattern
    // This is a simplified heuristic - could be improved with autocorrelation
    if (strongBeatCount <= beatStrengths.size() / 4) {
        analysis_.timeSignatureNum = 4;  // Likely 4/4
        analysis_.timeSignatureDen = 4;
    } else if (strongBeatCount <= beatStrengths.size() / 3) {
        analysis_.timeSignatureNum = 3;  // Likely 3/4
        analysis_.timeSignatureDen = 4;
    } else {
        analysis_.timeSignatureNum = 4;  // Default to 4/4
        analysis_.timeSignatureDen = 4;
    }
}

void GrooveEngine::analyzeSwing() noexcept {
    if (analysis_.onsetPositions.size() < 4) {
        return;  // Need at least 4 onsets
    }
    
    // Analyze timing deviations from strict grid
    // Swing is detected when every other subdivision is consistently delayed
    
    // Calculate expected grid intervals
    if (analysis_.currentTempo <= 0.0f) {
        return;
    }
    
    uint64_t samplesPerBeat = static_cast<uint64_t>(
        (60.0 * config_.sampleRate) / analysis_.currentTempo
    );
    
    // Look at 8th note subdivisions (half beats)
    uint64_t gridInterval = samplesPerBeat / 2;
    if (gridInterval == 0) {
        return;
    }
    
    // Analyze deviations on odd vs even subdivisions
    float oddDeviations = 0.0f;
    float evenDeviations = 0.0f;
    int oddCount = 0;
    int evenCount = 0;
    
    for (size_t i = 1; i < std::min(analysis_.onsetPositions.size(), size_t(16)); ++i) {
        uint64_t interval = analysis_.onsetPositions[i] - analysis_.onsetPositions[i - 1];
        int64_t deviation = static_cast<int64_t>(interval) - static_cast<int64_t>(gridInterval);
        
        if (i % 2 == 0) {
            evenDeviations += static_cast<float>(deviation);
            evenCount++;
        } else {
            oddDeviations += static_cast<float>(deviation);
            oddCount++;
        }
    }
    
    if (oddCount > 0 && evenCount > 0) {
        oddDeviations /= oddCount;
        evenDeviations /= evenCount;
        
        // If odd beats are consistently delayed relative to even beats, we have swing
        float avgDeviation = oddDeviations - evenDeviations;
        
        // Convert deviation to swing amount (0.5 = straight, 0.66 = triplet)
        // Positive deviation means upbeats are delayed
        float swingRatio = 0.5f + (avgDeviation / static_cast<float>(gridInterval)) * 0.5f;
        analysis_.swing = std::max(0.5f, std::min(0.75f, swingRatio));
    }
}

} // namespace penta::groove
