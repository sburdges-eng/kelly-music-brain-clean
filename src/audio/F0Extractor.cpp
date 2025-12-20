/**
 * @file F0Extractor.cpp
 * @brief F0 extraction and audio analysis implementation
 */

#include "F0Extractor.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace midikompanion {
namespace audio {

//==============================================================================
// F0Extractor Implementation
//==============================================================================

F0Extractor::F0Extractor() {
    setConfig(F0Config());
}

F0Extractor::F0Extractor(const F0Config& config) {
    setConfig(config);
}

void F0Extractor::prepare(double sampleRate) {
    sampleRate_ = sampleRate;
    reset();
}

void F0Extractor::reset() {
    inputBuffer_.clear();
    pitchHistory_.clear();
    overallConfidence_ = 0.0f;
    
    // Resize buffers
    int maxLag = static_cast<int>(sampleRate_ / config_.minFrequency);
    yinBuffer_.resize(maxLag / 2 + 1);
    frameBuffer_.resize(config_.frameSize);
}

void F0Extractor::setConfig(const F0Config& config) {
    config_ = config;
}

std::vector<PitchResult> F0Extractor::extractPitch(const juce::AudioBuffer<float>& audio) {
    std::vector<PitchResult> results;
    
    if (audio.getNumChannels() == 0 || audio.getNumSamples() == 0) {
        return results;
    }
    
    const float* samples = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();
    
    // Process in frames with hop
    for (int offset = 0; offset + config_.frameSize <= numSamples; offset += config_.hopSize) {
        results.push_back(processFrame(samples + offset, config_.frameSize));
    }
    
    return results;
}

PitchResult F0Extractor::processFrame(const float* samples, int numSamples) {
    PitchResult result;
    
    if (numSamples < config_.frameSize) {
        return result;
    }
    
    // Compute RMS amplitude
    result.amplitude = computeRMS(samples, numSamples);
    
    // Skip if too quiet
    if (result.amplitude < 0.001f) {
        result.voiced = false;
        return result;
    }
    
    // YIN algorithm for pitch detection
    int minLag = static_cast<int>(sampleRate_ / config_.maxFrequency);
    int maxLag = static_cast<int>(sampleRate_ / config_.minFrequency);
    maxLag = juce::jmin(maxLag, numSamples / 2);
    
    // Resize YIN buffer if needed
    if (yinBuffer_.size() < static_cast<size_t>(maxLag + 1)) {
        yinBuffer_.resize(maxLag + 1);
    }
    
    // Step 1: Compute difference function
    yinBuffer_[0] = 1.0f;
    for (int lag = 1; lag <= maxLag; ++lag) {
        yinBuffer_[lag] = computeYinDifference(samples, lag);
    }
    
    // Step 2: Cumulative mean normalized difference
    computeCumulativeMeanNormalizedDifference(yinBuffer_);
    
    // Step 3: Find minimum below threshold
    int minIndex = findYinMinimum(yinBuffer_);
    
    if (minIndex < minLag || minIndex > maxLag) {
        result.voiced = false;
        return result;
    }
    
    // Check threshold
    if (yinBuffer_[minIndex] > config_.yinThreshold) {
        result.voiced = false;
        return result;
    }
    
    // Step 4: Parabolic interpolation for better precision
    float interpolatedIndex = minIndex;
    if (config_.useParabolicInterpolation && minIndex > 1 && minIndex < maxLag) {
        interpolatedIndex = parabolicInterpolation(yinBuffer_, minIndex);
    }
    
    // Calculate frequency
    result.frequency = static_cast<float>(sampleRate_) / interpolatedIndex;
    result.confidence = 1.0f - yinBuffer_[minIndex];
    result.voiced = result.confidence > config_.voicingThreshold;
    
    if (result.voiced) {
        auto [midiNote, cents] = frequencyToMidiWithCents(result.frequency);
        result.midiNote = midiNote;
        result.midiCents = cents;
        
        updatePitchHistory(result.frequency);
    }
    
    // Update overall confidence
    overallConfidence_ = overallConfidence_ * 0.9f + result.confidence * 0.1f;
    
    return result;
}

void F0Extractor::pushSamples(const float* samples, int numSamples) {
    const juce::ScopedLock lock(bufferLock_);
    
    for (int i = 0; i < numSamples; ++i) {
        inputBuffer_.push_back(samples[i]);
    }
    
    // Limit buffer size
    while (inputBuffer_.size() > static_cast<size_t>(config_.frameSize * 4)) {
        inputBuffer_.pop_front();
    }
}

bool F0Extractor::hasFrame() const {
    const juce::ScopedLock lock(bufferLock_);
    return inputBuffer_.size() >= static_cast<size_t>(config_.frameSize);
}

PitchResult F0Extractor::popFrame() {
    const juce::ScopedLock lock(bufferLock_);
    
    if (inputBuffer_.size() < static_cast<size_t>(config_.frameSize)) {
        return PitchResult();
    }
    
    // Copy to frame buffer
    for (int i = 0; i < config_.frameSize; ++i) {
        frameBuffer_[i] = inputBuffer_[i];
    }
    
    // Remove hop size samples
    for (int i = 0; i < config_.hopSize && !inputBuffer_.empty(); ++i) {
        inputBuffer_.pop_front();
    }
    
    return processFrame(frameBuffer_.data(), config_.frameSize);
}

float F0Extractor::getSmoothedPitch() const {
    if (pitchHistory_.empty()) return 0.0f;
    
    // Median filter
    std::vector<float> sorted(pitchHistory_.begin(), pitchHistory_.end());
    std::sort(sorted.begin(), sorted.end());
    
    return sorted[sorted.size() / 2];
}

int F0Extractor::frequencyToMidi(float frequency) {
    if (frequency <= 0.0f) return -1;
    return static_cast<int>(std::round(12.0f * std::log2(frequency / 440.0f) + 69.0f));
}

std::pair<int, float> F0Extractor::frequencyToMidiWithCents(float frequency) {
    if (frequency <= 0.0f) return {-1, 0.0f};
    
    float midiFloat = 12.0f * std::log2(frequency / 440.0f) + 69.0f;
    int midiNote = static_cast<int>(std::round(midiFloat));
    float cents = (midiFloat - midiNote) * 100.0f;
    
    return {midiNote, cents};
}

float F0Extractor::midiToFrequency(int midiNote) {
    return 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
}

float F0Extractor::computeYinDifference(const float* samples, int lag) const {
    float sum = 0.0f;
    int frameSize = config_.frameSize / 2;
    
    for (int i = 0; i < frameSize; ++i) {
        float diff = samples[i] - samples[i + lag];
        sum += diff * diff;
    }
    
    return sum;
}

void F0Extractor::computeCumulativeMeanNormalizedDifference(std::vector<float>& yinBuffer) const {
    float runningSum = 0.0f;
    yinBuffer[0] = 1.0f;
    
    for (size_t i = 1; i < yinBuffer.size(); ++i) {
        runningSum += yinBuffer[i];
        if (runningSum == 0.0f) {
            yinBuffer[i] = 1.0f;
        } else {
            yinBuffer[i] = yinBuffer[i] * i / runningSum;
        }
    }
}

int F0Extractor::findYinMinimum(const std::vector<float>& yinBuffer) const {
    int minLag = static_cast<int>(sampleRate_ / config_.maxFrequency);
    int maxLag = static_cast<int>(yinBuffer.size()) - 1;
    
    // Find first minimum below threshold
    for (int i = minLag; i < maxLag - 1; ++i) {
        if (yinBuffer[i] < config_.yinThreshold) {
            // Check if it's a local minimum
            if (yinBuffer[i] < yinBuffer[i - 1] && yinBuffer[i] <= yinBuffer[i + 1]) {
                return i;
            }
        }
    }
    
    // If no minimum found, return the global minimum
    int minIndex = minLag;
    float minValue = yinBuffer[minLag];
    
    for (int i = minLag + 1; i <= maxLag; ++i) {
        if (yinBuffer[i] < minValue) {
            minValue = yinBuffer[i];
            minIndex = i;
        }
    }
    
    return minIndex;
}

float F0Extractor::parabolicInterpolation(const std::vector<float>& yinBuffer, int minIndex) const {
    if (minIndex <= 0 || minIndex >= static_cast<int>(yinBuffer.size()) - 1) {
        return static_cast<float>(minIndex);
    }
    
    float y0 = yinBuffer[minIndex - 1];
    float y1 = yinBuffer[minIndex];
    float y2 = yinBuffer[minIndex + 1];
    
    // Parabolic interpolation formula
    float d = (y0 - y2) / (2.0f * (y0 - 2.0f * y1 + y2));
    
    return minIndex + d;
}

float F0Extractor::computeRMS(const float* samples, int numSamples) const {
    float sum = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        sum += samples[i] * samples[i];
    }
    return std::sqrt(sum / numSamples);
}

void F0Extractor::updatePitchHistory(float pitch) {
    pitchHistory_.push_back(pitch);
    while (pitchHistory_.size() > PITCH_HISTORY_SIZE) {
        pitchHistory_.pop_front();
    }
}

//==============================================================================
// SpectralAnalyzer Implementation
//==============================================================================

SpectralAnalyzer::SpectralAnalyzer() = default;

void SpectralAnalyzer::prepare(double sampleRate, int fftSize) {
    sampleRate_ = sampleRate;
    fftSize_ = fftSize;
    
    // Create FFT
    int fftOrder = static_cast<int>(std::log2(fftSize));
    fft_ = std::make_unique<juce::dsp::FFT>(fftOrder);
    
    // Resize buffers
    fftInput_.resize(fftSize * 2, 0.0f);
    fftOutput_.resize(fftSize * 2, 0.0f);
    magnitudeSpectrum_.resize(fftSize / 2 + 1, 0.0f);
    previousSpectrum_.resize(fftSize / 2 + 1, 0.0f);
    
    // Create Hann window
    window_.resize(fftSize);
    for (int i = 0; i < fftSize; ++i) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * juce::MathConstants<float>::pi * i / (fftSize - 1)));
    }
    
    computeMelFilterbank();
}

SpectralAnalyzer::SpectralFeatures SpectralAnalyzer::analyze(const juce::AudioBuffer<float>& audio) {
    SpectralFeatures features;
    
    if (!fft_ || audio.getNumChannels() == 0 || audio.getNumSamples() < fftSize_) {
        return features;
    }
    
    const float* samples = audio.getReadPointer(0);
    
    // Apply window and copy to FFT buffer
    for (int i = 0; i < fftSize_; ++i) {
        fftInput_[i] = samples[i] * window_[i];
    }
    
    // Perform FFT
    fft_->performRealOnlyForwardTransform(fftInput_.data(), true);
    
    // Compute magnitude spectrum
    float totalEnergy = 0.0f;
    for (int i = 0; i <= fftSize_ / 2; ++i) {
        float real = fftInput_[i * 2];
        float imag = fftInput_[i * 2 + 1];
        magnitudeSpectrum_[i] = std::sqrt(real * real + imag * imag);
        totalEnergy += magnitudeSpectrum_[i] * magnitudeSpectrum_[i];
    }
    
    // Spectral centroid
    float centroidNum = 0.0f;
    float centroidDen = 0.0f;
    for (int i = 0; i <= fftSize_ / 2; ++i) {
        float freq = static_cast<float>(i) * sampleRate_ / fftSize_;
        centroidNum += freq * magnitudeSpectrum_[i];
        centroidDen += magnitudeSpectrum_[i];
    }
    features.centroid = centroidDen > 0.0f ? centroidNum / centroidDen : 0.0f;
    
    // Spectral spread
    float spreadNum = 0.0f;
    for (int i = 0; i <= fftSize_ / 2; ++i) {
        float freq = static_cast<float>(i) * sampleRate_ / fftSize_;
        float diff = freq - features.centroid;
        spreadNum += diff * diff * magnitudeSpectrum_[i];
    }
    features.spread = centroidDen > 0.0f ? std::sqrt(spreadNum / centroidDen) : 0.0f;
    
    // Spectral flatness
    float geometricMean = 0.0f;
    float arithmeticMean = 0.0f;
    int count = 0;
    for (int i = 1; i <= fftSize_ / 2; ++i) {
        if (magnitudeSpectrum_[i] > 0.0001f) {
            geometricMean += std::log(magnitudeSpectrum_[i]);
            arithmeticMean += magnitudeSpectrum_[i];
            ++count;
        }
    }
    if (count > 0) {
        geometricMean = std::exp(geometricMean / count);
        arithmeticMean /= count;
        features.flatness = arithmeticMean > 0.0f ? geometricMean / arithmeticMean : 0.0f;
    }
    
    // Spectral rolloff (85%)
    float rolloffThreshold = totalEnergy * 0.85f;
    float cumulativeEnergy = 0.0f;
    for (int i = 0; i <= fftSize_ / 2; ++i) {
        cumulativeEnergy += magnitudeSpectrum_[i] * magnitudeSpectrum_[i];
        if (cumulativeEnergy >= rolloffThreshold) {
            features.rolloff = static_cast<float>(i) * sampleRate_ / fftSize_;
            break;
        }
    }
    
    // Spectral flux
    float flux = 0.0f;
    for (int i = 0; i <= fftSize_ / 2; ++i) {
        float diff = magnitudeSpectrum_[i] - previousSpectrum_[i];
        flux += diff * diff;
    }
    features.flux = std::sqrt(flux);
    
    // Store current spectrum for next frame
    previousSpectrum_ = magnitudeSpectrum_;
    
    // MFCC (simplified)
    features.mfcc.resize(numMfcc_, 0.0f);
    if (!melFilterbank_.empty()) {
        std::vector<float> melEnergies(numMelFilters_, 0.0f);
        
        for (int m = 0; m < numMelFilters_; ++m) {
            for (int k = 0; k < static_cast<int>(melFilterbank_[m].size()); ++k) {
                melEnergies[m] += magnitudeSpectrum_[k] * melFilterbank_[m][k];
            }
            melEnergies[m] = std::log(melEnergies[m] + 1e-10f);
        }
        
        // DCT to get MFCCs
        for (int i = 0; i < numMfcc_; ++i) {
            for (int m = 0; m < numMelFilters_; ++m) {
                features.mfcc[i] += melEnergies[m] * 
                    std::cos(juce::MathConstants<float>::pi * i * (m + 0.5f) / numMelFilters_);
            }
        }
    }
    
    return features;
}

void SpectralAnalyzer::computeMelFilterbank() {
    melFilterbank_.resize(numMelFilters_);
    
    int numBins = fftSize_ / 2 + 1;
    float fMin = 0.0f;
    float fMax = static_cast<float>(sampleRate_) / 2.0f;
    
    float melMin = hzToMel(fMin);
    float melMax = hzToMel(fMax);
    
    std::vector<float> melPoints(numMelFilters_ + 2);
    for (int i = 0; i < numMelFilters_ + 2; ++i) {
        melPoints[i] = melMin + i * (melMax - melMin) / (numMelFilters_ + 1);
    }
    
    std::vector<int> binPoints(numMelFilters_ + 2);
    for (int i = 0; i < numMelFilters_ + 2; ++i) {
        binPoints[i] = static_cast<int>(std::floor(
            (fftSize_ + 1) * melToHz(melPoints[i]) / sampleRate_));
    }
    
    for (int m = 0; m < numMelFilters_; ++m) {
        melFilterbank_[m].resize(numBins, 0.0f);
        
        for (int k = binPoints[m]; k < binPoints[m + 1] && k < numBins; ++k) {
            melFilterbank_[m][k] = static_cast<float>(k - binPoints[m]) / 
                                   (binPoints[m + 1] - binPoints[m]);
        }
        
        for (int k = binPoints[m + 1]; k < binPoints[m + 2] && k < numBins; ++k) {
            melFilterbank_[m][k] = static_cast<float>(binPoints[m + 2] - k) / 
                                   (binPoints[m + 2] - binPoints[m + 1]);
        }
    }
}

float SpectralAnalyzer::hzToMel(float hz) const {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float SpectralAnalyzer::melToHz(float mel) const {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

//==============================================================================
// LoudnessAnalyzer Implementation
//==============================================================================

LoudnessAnalyzer::LoudnessAnalyzer() = default;

void LoudnessAnalyzer::prepare(double sampleRate) {
    sampleRate_ = sampleRate;
    loudnessHistory_.clear();
    
    // Initialize K-weighting filter (simplified)
    // Full implementation would use a proper IIR filter
}

LoudnessAnalyzer::LoudnessResult LoudnessAnalyzer::analyze(const juce::AudioBuffer<float>& audio) {
    LoudnessResult result;
    
    if (audio.getNumChannels() == 0 || audio.getNumSamples() == 0) {
        return result;
    }
    
    const float* samples = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();
    
    // RMS
    float sumSquares = 0.0f;
    float peak = 0.0f;
    
    for (int i = 0; i < numSamples; ++i) {
        float sample = samples[i];
        sumSquares += sample * sample;
        float absSample = std::abs(sample);
        if (absSample > peak) peak = absSample;
    }
    
    result.rms = std::sqrt(sumSquares / numSamples);
    result.peak = peak;
    
    // Convert to dB
    result.rmsDb = result.rms > 0.0f ? 20.0f * std::log10(result.rms) : -100.0f;
    result.peakDb = result.peak > 0.0f ? 20.0f * std::log10(result.peak) : -100.0f;
    
    // Simplified LUFS (not ITU-R BS.1770 compliant)
    float meanSquare = sumSquares / numSamples;
    result.lufs = meanSquare > 0.0f ? -0.691f + 10.0f * std::log10(meanSquare) : -100.0f;
    
    // Perceptual loudness (0-1)
    result.loudness = juce::jlimit(0.0f, 1.0f, (result.rmsDb + 60.0f) / 60.0f);
    
    // Update history
    loudnessHistory_.push_back(result.loudness);
    while (loudnessHistory_.size() > 100) {
        loudnessHistory_.pop_front();
    }
    
    // Short-term loudness (average of recent)
    if (!loudnessHistory_.empty()) {
        float sum = 0.0f;
        for (float l : loudnessHistory_) {
            sum += l;
        }
        shortTermLoudness_ = sum / loudnessHistory_.size();
    }
    
    momentaryLoudness_ = result.loudness;
    
    return result;
}

//==============================================================================
// MLFeatureExtractor Implementation
//==============================================================================

MLFeatureExtractor::MLFeatureExtractor() = default;

void MLFeatureExtractor::prepare(double sampleRate) {
    f0Extractor_.prepare(sampleRate);
    spectralAnalyzer_.prepare(sampleRate);
    loudnessAnalyzer_.prepare(sampleRate);
    initializeNormalization();
}

std::vector<float> MLFeatureExtractor::extractFeatures(const juce::AudioBuffer<float>& audio) {
    std::vector<float> features;
    features.reserve(128);
    
    // F0 features (16 dimensions)
    auto pitchResults = f0Extractor_.extractPitch(audio);
    
    float avgPitch = 0.0f;
    float pitchStd = 0.0f;
    float voicingRatio = 0.0f;
    
    if (!pitchResults.empty()) {
        std::vector<float> pitches;
        for (const auto& pr : pitchResults) {
            if (pr.voiced) {
                pitches.push_back(pr.frequency);
            }
        }
        
        voicingRatio = static_cast<float>(pitches.size()) / pitchResults.size();
        
        if (!pitches.empty()) {
            avgPitch = std::accumulate(pitches.begin(), pitches.end(), 0.0f) / pitches.size();
            
            float variance = 0.0f;
            for (float p : pitches) {
                variance += (p - avgPitch) * (p - avgPitch);
            }
            pitchStd = std::sqrt(variance / pitches.size());
        }
    }
    
    // Normalize pitch to reasonable range
    features.push_back(avgPitch / 500.0f);
    features.push_back(pitchStd / 100.0f);
    features.push_back(voicingRatio);
    features.push_back(f0Extractor_.getConfidence());
    
    // Pad to 16 dimensions
    while (features.size() < 16) {
        features.push_back(0.0f);
    }
    
    // Spectral features (48 dimensions)
    auto spectral = spectralAnalyzer_.analyze(audio);
    
    features.push_back(spectral.centroid / 5000.0f);
    features.push_back(spectral.spread / 2000.0f);
    features.push_back(spectral.flatness);
    features.push_back(spectral.rolloff / 10000.0f);
    features.push_back(spectral.flux / 10.0f);
    features.push_back(spectral.hnr);
    
    // Add MFCCs (normalized)
    for (size_t i = 0; i < spectral.mfcc.size() && i < 13; ++i) {
        features.push_back(spectral.mfcc[i] / 50.0f);
    }
    
    // Pad to 32 dimensions for spectral
    while (features.size() < 48) {
        features.push_back(0.0f);
    }
    
    // Magnitude spectrum bins (32 dimensions, downsampled)
    const auto& spectrum = spectralAnalyzer_.getMagnitudeSpectrum();
    int fftSize = spectralAnalyzer_.getFFTSize();
    int binSkip = (fftSize / 2) / 32;
    
    for (int i = 0; i < 32 && i * binSkip < static_cast<int>(spectrum.size()); ++i) {
        features.push_back(spectrum[i * binSkip] / 10.0f);
    }
    
    while (features.size() < 80) {
        features.push_back(0.0f);
    }
    
    // Loudness features (16 dimensions)
    auto loudness = loudnessAnalyzer_.analyze(audio);
    
    features.push_back(loudness.rms);
    features.push_back((loudness.rmsDb + 60.0f) / 60.0f);
    features.push_back(loudness.peak);
    features.push_back((loudness.peakDb + 60.0f) / 60.0f);
    features.push_back((loudness.lufs + 60.0f) / 60.0f);
    features.push_back(loudness.loudness);
    features.push_back(loudnessAnalyzer_.getShortTermLoudness());
    features.push_back(loudnessAnalyzer_.getMomentaryLoudness());
    
    while (features.size() < 96) {
        features.push_back(0.0f);
    }
    
    // Additional derived features (32 dimensions)
    // These would include temporal features, dynamics, etc.
    
    while (features.size() < 128) {
        features.push_back(0.0f);
    }
    
    // Normalize if initialized
    if (normalizationInitialized_) {
        normalizeFeatures(features);
    }
    
    return features;
}

void MLFeatureExtractor::initializeNormalization() {
    // Initialize with reasonable defaults
    // In production, these would be computed from training data
    featureMean_.resize(128, 0.0f);
    featureStd_.resize(128, 1.0f);
    normalizationInitialized_ = true;
}

void MLFeatureExtractor::normalizeFeatures(std::vector<float>& features) const {
    for (size_t i = 0; i < features.size() && i < featureMean_.size(); ++i) {
        features[i] = (features[i] - featureMean_[i]) / (featureStd_[i] + 1e-8f);
    }
}

} // namespace audio
} // namespace midikompanion
