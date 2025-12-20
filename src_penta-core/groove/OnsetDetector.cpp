#include "penta/groove/OnsetDetector.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace penta::groove {

OnsetDetector::OnsetDetector(const Config& config)
    : config_(config)
    , onsetDetected_(false)
    , onsetStrength_(0.0f)
    , onsetPosition_(0)
    , lastOnsetPosition_(0)
    , sampleCounter_(0)
{
    // Pre-allocate buffers
    fftBuffer_.resize(config_.fftSize);
    spectrum_.resize(config_.fftSize / 2 + 1);
    prevSpectrum_.resize(config_.fftSize / 2 + 1);
    fluxHistory_.resize(100);
    
    // Initialize Hann window for spectral analysis
    window_.resize(config_.fftSize);
    for (size_t i = 0; i < config_.fftSize; ++i) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (config_.fftSize - 1)));
    }
}

OnsetDetector::~OnsetDetector() = default;

void OnsetDetector::process(const float* buffer, size_t frames) noexcept {
    onsetDetected_ = false;
    
    // Process in hop-size chunks
    for (size_t i = 0; i + config_.hopSize <= frames; i += config_.hopSize) {
        computeSpectralFlux(buffer + i, config_.hopSize);
        detectPeaks();
        sampleCounter_ += config_.hopSize;
    }
}

void OnsetDetector::setThreshold(float threshold) noexcept {
    config_.threshold = threshold;
}

void OnsetDetector::reset() noexcept {
    onsetDetected_ = false;
    onsetStrength_ = 0.0f;
    onsetPosition_ = 0;
    lastOnsetPosition_ = 0;
    sampleCounter_ = 0;
    std::fill(prevSpectrum_.begin(), prevSpectrum_.end(), 0.0f);
    std::fill(fluxHistory_.begin(), fluxHistory_.end(), 0.0f);
}

void OnsetDetector::computeSpectralFlux(const float* buffer, size_t frames) noexcept {
    // Simple energy-based onset detection (spectral flux approximation)
    // This is a simplified version that doesn't require FFT library
    // For production, would use actual FFT (e.g., FFTW, pffft, or Accelerate framework)
    
    // Calculate energy in frequency bands using filterbank approach
    const size_t numBands = spectrum_.size();
    std::fill(spectrum_.begin(), spectrum_.end(), 0.0f);
    
    // Divide audio into frequency bands and compute energy
    // This is a simplified filterbank that approximates spectral content
    for (size_t band = 0; band < numBands && band < frames; ++band) {
        float bandEnergy = 0.0f;
        size_t samplesPerBand = std::max(size_t(1), frames / numBands);
        size_t start = band * samplesPerBand;
        size_t end = std::min(start + samplesPerBand, frames);
        
        for (size_t j = start; j < end; ++j) {
            // Apply window relative to the frame position, not absolute index
            size_t framePos = j - start;
            size_t windowIdx = (framePos * window_.size()) / samplesPerBand;
            windowIdx = std::min(windowIdx, window_.size() - 1);
            
            float sample = buffer[j] * window_[windowIdx];
            bandEnergy += sample * sample;
        }
        
        spectrum_[band] = std::sqrt(bandEnergy / samplesPerBand);
    }
    
    // Compute spectral flux: sum of positive differences between frames
    float flux = 0.0f;
    for (size_t i = 0; i < numBands; ++i) {
        float diff = spectrum_[i] - prevSpectrum_[i];
        if (diff > 0.0f) {
            flux += diff;
        }
    }
    
    // Normalize flux
    flux /= numBands;
    
    // Update flux history (rolling buffer)
    std::rotate(fluxHistory_.begin(), fluxHistory_.begin() + 1, fluxHistory_.end());
    fluxHistory_.back() = flux;
    
    // Store current spectrum for next frame
    prevSpectrum_ = spectrum_;
}

void OnsetDetector::detectPeaks() noexcept {
    if (fluxHistory_.size() < 3) {
        return;
    }
    
    // Get current flux value
    float currentFlux = fluxHistory_.back();
    
    // Calculate adaptive threshold from recent history
    float mean = 0.0f;
    size_t historyWindow = std::min(size_t(20), fluxHistory_.size() - 1);
    for (size_t i = fluxHistory_.size() - historyWindow; i < fluxHistory_.size() - 1; ++i) {
        mean += fluxHistory_[i];
    }
    mean /= historyWindow;
    
    float adaptiveThreshold = mean + config_.threshold;
    
    // Check if current flux is a peak above threshold
    if (currentFlux > adaptiveThreshold) {
        // Check previous and next values to confirm it's a local maximum
        float prevFlux = fluxHistory_[fluxHistory_.size() - 2];
        
        if (currentFlux > prevFlux) {
            // Check minimum time between onsets
            uint64_t timeSinceLastOnset = sampleCounter_ - lastOnsetPosition_;
            uint64_t minSamples = static_cast<uint64_t>(
                config_.minTimeBetweenOnsets * config_.sampleRate
            );
            
            if (timeSinceLastOnset >= minSamples) {
                onsetDetected_ = true;
                onsetStrength_ = std::min(1.0f, currentFlux / (adaptiveThreshold + 0.1f));
                onsetPosition_ = sampleCounter_;
                lastOnsetPosition_ = sampleCounter_;
            }
        }
    }
}

} // namespace penta::groove
