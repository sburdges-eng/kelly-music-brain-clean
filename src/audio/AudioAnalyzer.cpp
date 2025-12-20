/**
 * @file AudioAnalyzer.cpp
 * @brief Audio analysis implementation
 */

#include "AudioAnalyzer.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace midikompanion::audio {

//==============================================================================
// AudioFeatures
//==============================================================================

std::vector<float> AudioFeatures::toMLInput() const {
    std::vector<float> features(128, 0.0f);
    int idx = 0;
    
    // F0 features (4 dims)
    features[idx++] = f0.frequency / 2000.0f;  // Normalize to [0, 1]
    features[idx++] = f0.confidence;
    features[idx++] = f0.voiced ? 1.0f : 0.0f;
    features[idx++] = (f0.midiNote + 12.0f) / 128.0f;  // Normalize MIDI note
    
    // Loudness features (4 dims)
    features[idx++] = loudness.rms;
    features[idx++] = (loudness.rmsDb + 60.0f) / 60.0f;  // Normalize dB
    features[idx++] = loudness.peak;
    features[idx++] = (loudness.lufs + 60.0f) / 60.0f;
    
    // Spectral features (6 dims)
    features[idx++] = spectral.centroid / 10000.0f;
    features[idx++] = spectral.rolloff / 10000.0f;
    features[idx++] = spectral.brightness;
    features[idx++] = spectral.spread / 10000.0f;
    features[idx++] = spectral.flatness;
    features[idx++] = zcr;
    
    // MFCC features (40 dims)
    for (size_t i = 0; i < 40 && i < mfcc.size(); ++i) {
        features[idx++] = std::clamp(mfcc[i] / 100.0f, -1.0f, 1.0f);
    }
    idx = 54;  // Ensure we're at the right position
    
    // Chroma features (12 dims)
    for (size_t i = 0; i < 12 && i < chroma.size(); ++i) {
        features[idx++] = chroma[i];
    }
    idx = 66;
    
    // Spectral bins (remaining dims)
    size_t specBins = std::min<size_t>(62, spectral.spectrum.size());
    for (size_t i = 0; i < specBins; ++i) {
        features[idx++] = std::clamp(spectral.spectrum[i], 0.0f, 1.0f);
    }
    
    return features;
}

//==============================================================================
// F0Extractor
//==============================================================================

F0Extractor::F0Extractor() {
    yinBuffer_.resize(2048);
}

void F0Extractor::configure(double sampleRate, int frameSize) {
    sampleRate_ = sampleRate;
    frameSize_ = frameSize;
    yinBuffer_.resize(frameSize / 2);
}

F0Result F0Extractor::extract(const juce::AudioBuffer<float>& audio) {
    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) {
        return F0Result{};
    }
    
    // Use first channel
    return extract(audio.getReadPointer(0), audio.getNumSamples());
}

F0Result F0Extractor::extract(const float* samples, int numSamples) {
    F0Result result;
    
    if (numSamples < frameSize_) {
        return result;
    }
    
    float frequency = computeYIN(samples, frameSize_);
    
    if (frequency > 0.0f && frequency >= minFreq_ && frequency <= maxFreq_) {
        result.frequency = frequency;
        result.voiced = true;
        result.midiNote = frequencyToMidiNote(frequency);
        result.midiCents = frequencyToMidiCents(frequency, result.midiNote);
        
        // Find minimum in YIN buffer for confidence
        float minVal = 1.0f;
        for (const auto& val : yinBuffer_) {
            if (val < minVal) minVal = val;
        }
        result.confidence = 1.0f - std::min(1.0f, minVal);
    }
    
    return result;
}

void F0Extractor::setFrequencyRange(float minHz, float maxHz) {
    minFreq_ = minHz;
    maxFreq_ = maxHz;
}

float F0Extractor::computeYIN(const float* samples, int numSamples) {
    int halfSize = numSamples / 2;
    
    // Resize buffer if needed
    if (static_cast<int>(yinBuffer_.size()) != halfSize) {
        yinBuffer_.resize(halfSize);
    }
    
    // Step 1 & 2: Compute difference function and cumulative mean normalized difference
    computeCMND(samples, numSamples);
    
    // Step 3: Find pitch period
    int period = findPitchPeriod();
    
    if (period == 0) {
        return 0.0f;
    }
    
    // Step 4: Parabolic interpolation
    float refinedPeriod = parabolicInterpolation(period);
    
    if (refinedPeriod <= 0.0f) {
        return 0.0f;
    }
    
    return static_cast<float>(sampleRate_ / refinedPeriod);
}

void F0Extractor::computeCMND(const float* samples, int numSamples) {
    int halfSize = numSamples / 2;
    
    // Difference function
    yinBuffer_[0] = 1.0f;
    
    float runningSum = 0.0f;
    
    for (int tau = 1; tau < halfSize; ++tau) {
        float sum = 0.0f;
        
        for (int j = 0; j < halfSize; ++j) {
            float delta = samples[j] - samples[j + tau];
            sum += delta * delta;
        }
        
        runningSum += sum;
        yinBuffer_[tau] = sum * tau / runningSum;
    }
}

int F0Extractor::findPitchPeriod() {
    int halfSize = static_cast<int>(yinBuffer_.size());
    
    // Find first dip below threshold
    int minPeriod = static_cast<int>(sampleRate_ / maxFreq_);
    int maxPeriod = static_cast<int>(sampleRate_ / minFreq_);
    
    minPeriod = std::max(1, std::min(minPeriod, halfSize - 1));
    maxPeriod = std::min(maxPeriod, halfSize - 1);
    
    for (int tau = minPeriod; tau < maxPeriod; ++tau) {
        if (yinBuffer_[tau] < confidenceThreshold_) {
            // Find local minimum
            while (tau + 1 < maxPeriod && yinBuffer_[tau + 1] < yinBuffer_[tau]) {
                ++tau;
            }
            return tau;
        }
    }
    
    return 0;
}

float F0Extractor::parabolicInterpolation(int period) {
    if (period <= 0 || period >= static_cast<int>(yinBuffer_.size()) - 1) {
        return static_cast<float>(period);
    }
    
    float y0 = yinBuffer_[period - 1];
    float y1 = yinBuffer_[period];
    float y2 = yinBuffer_[period + 1];
    
    float denominator = 2.0f * (y0 - 2.0f * y1 + y2);
    
    if (std::abs(denominator) < 1e-10f) {
        return static_cast<float>(period);
    }
    
    float correction = (y0 - y2) / denominator;
    
    return period + correction;
}

int F0Extractor::frequencyToMidiNote(float frequency) {
    if (frequency <= 0.0f) return -1;
    return static_cast<int>(std::round(12.0f * std::log2(frequency / 440.0f) + 69.0f));
}

float F0Extractor::frequencyToMidiCents(float frequency, int midiNote) {
    if (frequency <= 0.0f || midiNote < 0) return 0.0f;
    
    float expectedFreq = 440.0f * std::pow(2.0f, (midiNote - 69.0f) / 12.0f);
    return 1200.0f * std::log2(frequency / expectedFreq);
}

//==============================================================================
// SpectralAnalyzer
//==============================================================================

SpectralAnalyzer::SpectralAnalyzer() {
    fftBuffer_.resize(4096);
    magnitudeSpectrum_.resize(1024);
}

void SpectralAnalyzer::configure(double sampleRate, int fftSize) {
    sampleRate_ = sampleRate;
    fftSize_ = fftSize;
    
    // Determine FFT order
    int order = static_cast<int>(std::log2(fftSize));
    fft_ = juce::dsp::FFT(order);
    
    window_ = juce::dsp::WindowingFunction<float>(
        fftSize, juce::dsp::WindowingFunction<float>::hann);
    
    fftBuffer_.resize(fftSize * 2);
    magnitudeSpectrum_.resize(fftSize / 2);
}

SpectralResult SpectralAnalyzer::analyze(const juce::AudioBuffer<float>& audio) {
    if (audio.getNumSamples() == 0) {
        return SpectralResult{};
    }
    return analyze(audio.getReadPointer(0), audio.getNumSamples());
}

SpectralResult SpectralAnalyzer::analyze(const float* samples, int numSamples) {
    SpectralResult result;
    
    int frameSize = std::min(numSamples, fftSize_);
    
    // Copy and window
    std::fill(fftBuffer_.begin(), fftBuffer_.end(), 0.0f);
    std::copy(samples, samples + frameSize, fftBuffer_.begin());
    applyWindow(fftBuffer_.data(), frameSize);
    
    // Perform FFT
    fft_.performRealOnlyForwardTransform(fftBuffer_.data());
    
    // Compute magnitude spectrum
    int specSize = fftSize_ / 2;
    magnitudeSpectrum_.resize(specSize);
    
    for (int i = 0; i < specSize; ++i) {
        float real = fftBuffer_[i * 2];
        float imag = fftBuffer_[i * 2 + 1];
        magnitudeSpectrum_[i] = std::sqrt(real * real + imag * imag) / specSize;
    }
    
    // Compute spectral features
    result.centroid = computeCentroid();
    result.rolloff = computeRolloff();
    result.flatness = computeFlatness();
    
    // Brightness: ratio of energy above 3kHz
    float totalEnergy = 0.0f;
    float highEnergy = 0.0f;
    float binFreq = static_cast<float>(sampleRate_) / fftSize_;
    float brightnessThreshold = 3000.0f;
    
    for (int i = 0; i < specSize; ++i) {
        float energy = magnitudeSpectrum_[i] * magnitudeSpectrum_[i];
        totalEnergy += energy;
        if (i * binFreq > brightnessThreshold) {
            highEnergy += energy;
        }
    }
    
    result.brightness = totalEnergy > 0.0f ? highEnergy / totalEnergy : 0.0f;
    
    // Spectral spread
    float variance = 0.0f;
    for (int i = 0; i < specSize; ++i) {
        float freq = i * binFreq;
        float diff = freq - result.centroid;
        variance += magnitudeSpectrum_[i] * diff * diff;
    }
    result.spread = totalEnergy > 0.0f ? std::sqrt(variance / totalEnergy) : 0.0f;
    
    // Copy spectrum
    result.spectrum = magnitudeSpectrum_;
    
    return result;
}

void SpectralAnalyzer::applyWindow(float* data, int size) {
    window_.multiplyWithWindowingTable(data, size);
}

float SpectralAnalyzer::computeCentroid() {
    float binFreq = static_cast<float>(sampleRate_) / fftSize_;
    
    float weightedSum = 0.0f;
    float totalMag = 0.0f;
    
    for (size_t i = 0; i < magnitudeSpectrum_.size(); ++i) {
        float freq = i * binFreq;
        weightedSum += freq * magnitudeSpectrum_[i];
        totalMag += magnitudeSpectrum_[i];
    }
    
    return totalMag > 0.0f ? weightedSum / totalMag : 0.0f;
}

float SpectralAnalyzer::computeRolloff(float percentile) {
    float binFreq = static_cast<float>(sampleRate_) / fftSize_;
    
    // Calculate total energy
    float totalEnergy = 0.0f;
    for (const auto& mag : magnitudeSpectrum_) {
        totalEnergy += mag * mag;
    }
    
    float threshold = totalEnergy * percentile;
    float cumulativeEnergy = 0.0f;
    
    for (size_t i = 0; i < magnitudeSpectrum_.size(); ++i) {
        cumulativeEnergy += magnitudeSpectrum_[i] * magnitudeSpectrum_[i];
        if (cumulativeEnergy >= threshold) {
            return i * binFreq;
        }
    }
    
    return magnitudeSpectrum_.size() * binFreq;
}

float SpectralAnalyzer::computeFlatness() {
    if (magnitudeSpectrum_.empty()) return 0.0f;
    
    // Geometric mean / arithmetic mean
    float logSum = 0.0f;
    float arithmeticSum = 0.0f;
    int count = 0;
    
    for (const auto& mag : magnitudeSpectrum_) {
        if (mag > 1e-10f) {
            logSum += std::log(mag);
            arithmeticSum += mag;
            ++count;
        }
    }
    
    if (count == 0 || arithmeticSum < 1e-10f) return 0.0f;
    
    float geometricMean = std::exp(logSum / count);
    float arithmeticMean = arithmeticSum / count;
    
    return geometricMean / arithmeticMean;
}

//==============================================================================
// AudioAnalyzer
//==============================================================================

AudioAnalyzer::AudioAnalyzer() {
    analysisBuffer_.resize(4096);
}

void AudioAnalyzer::configure(double sampleRate, int blockSize) {
    sampleRate_ = sampleRate;
    blockSize_ = blockSize;
    hopSize_ = blockSize / 2;
    
    f0Extractor_.configure(sampleRate, 2048);
    spectralAnalyzer_.configure(sampleRate, 2048);
    
    analysisBuffer_.resize(4096);
    analysisBufferPos_ = 0;
    
    // Initialize mel filterbank
    // (Simplified - proper implementation would use triangular filters)
    melFilters_.resize(numMelBins_);
    for (auto& filter : melFilters_) {
        filter.resize(1024, 0.0f);
    }
}

AudioFeatures AudioAnalyzer::analyze(const juce::AudioBuffer<float>& audio) {
    AudioFeatures features;
    
    if (audio.getNumSamples() == 0) {
        return features;
    }
    
    const float* samples = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();
    
    // F0 extraction
    features.f0 = f0Extractor_.extract(samples, numSamples);
    
    // Loudness
    features.loudness = computeLoudness(samples, numSamples);
    
    // Spectral analysis
    features.spectral = spectralAnalyzer_.analyze(samples, numSamples);
    
    // MFCC
    features.mfcc = computeMFCC(features.spectral.spectrum);
    
    // Chroma
    features.chroma = computeChroma(features.spectral.spectrum);
    
    // Zero crossing rate
    features.zcr = computeZCR(samples, numSamples);
    
    latestFeatures_ = features;
    return features;
}

void AudioAnalyzer::processBlock(const juce::AudioBuffer<float>& audio) {
    // Accumulate samples for frame-based analysis
    const float* samples = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();
    
    for (int i = 0; i < numSamples; ++i) {
        analysisBuffer_[analysisBufferPos_++] = samples[i];
        
        if (analysisBufferPos_ >= 2048) {
            // Analyze frame
            juce::AudioBuffer<float> frame(1, 2048);
            frame.copyFrom(0, 0, analysisBuffer_.data(), 2048);
            latestFeatures_ = analyze(frame);
            
            // Shift buffer
            std::copy(analysisBuffer_.begin() + hopSize_,
                     analysisBuffer_.begin() + analysisBufferPos_,
                     analysisBuffer_.begin());
            analysisBufferPos_ -= hopSize_;
        }
    }
}

std::vector<float> AudioAnalyzer::extractMLFeatures(const juce::AudioBuffer<float>& audio) {
    AudioFeatures features = analyze(audio);
    return features.toMLInput();
}

LoudnessResult AudioAnalyzer::computeLoudness(const float* samples, int numSamples) {
    LoudnessResult result;
    
    if (numSamples == 0) return result;
    
    // RMS
    float sumSquares = 0.0f;
    float peak = 0.0f;
    
    for (int i = 0; i < numSamples; ++i) {
        float sample = samples[i];
        sumSquares += sample * sample;
        peak = std::max(peak, std::abs(sample));
    }
    
    result.rms = std::sqrt(sumSquares / numSamples);
    result.peak = peak;
    
    // Convert to dB
    if (result.rms > 0.0f) {
        result.rmsDb = 20.0f * std::log10(result.rms);
    } else {
        result.rmsDb = -100.0f;
    }
    
    // LUFS approximation (simplified)
    result.lufs = result.rmsDb - 0.691f;
    
    return result;
}

std::vector<float> AudioAnalyzer::computeMFCC(const std::vector<float>& spectrum) {
    std::vector<float> mfcc(20, 0.0f);
    
    if (spectrum.empty() || melFilters_.empty()) {
        return mfcc;
    }
    
    // Apply mel filterbank (simplified)
    std::vector<float> melEnergies(numMelBins_, 0.0f);
    
    for (int m = 0; m < numMelBins_; ++m) {
        float energy = 0.0f;
        int startBin = static_cast<int>(spectrum.size() * m / numMelBins_);
        int endBin = static_cast<int>(spectrum.size() * (m + 1) / numMelBins_);
        
        for (int i = startBin; i < endBin && i < static_cast<int>(spectrum.size()); ++i) {
            energy += spectrum[i] * spectrum[i];
        }
        
        melEnergies[m] = std::log(std::max(energy, 1e-10f));
    }
    
    // DCT to get MFCCs
    for (int k = 0; k < 20; ++k) {
        float sum = 0.0f;
        for (int n = 0; n < numMelBins_; ++n) {
            sum += melEnergies[n] * std::cos(M_PI * k * (n + 0.5f) / numMelBins_);
        }
        mfcc[k] = sum;
    }
    
    return mfcc;
}

std::vector<float> AudioAnalyzer::computeChroma(const std::vector<float>& spectrum) {
    std::vector<float> chroma(12, 0.0f);
    
    if (spectrum.empty()) {
        return chroma;
    }
    
    float binFreq = static_cast<float>(sampleRate_) / (spectrum.size() * 2);
    
    for (size_t i = 1; i < spectrum.size(); ++i) {
        float freq = i * binFreq;
        
        if (freq < 20.0f || freq > 5000.0f) continue;
        
        // Convert frequency to pitch class
        float pitchClass = 12.0f * std::log2(freq / 440.0f);
        pitchClass = std::fmod(pitchClass + 9.0f, 12.0f);
        if (pitchClass < 0) pitchClass += 12.0f;
        
        int bin = static_cast<int>(pitchClass) % 12;
        chroma[bin] += spectrum[i];
    }
    
    // Normalize
    float maxVal = *std::max_element(chroma.begin(), chroma.end());
    if (maxVal > 0.0f) {
        for (auto& c : chroma) {
            c /= maxVal;
        }
    }
    
    return chroma;
}

float AudioAnalyzer::computeZCR(const float* samples, int numSamples) {
    if (numSamples < 2) return 0.0f;
    
    int crossings = 0;
    
    for (int i = 1; i < numSamples; ++i) {
        if ((samples[i] >= 0.0f) != (samples[i - 1] >= 0.0f)) {
            ++crossings;
        }
    }
    
    return static_cast<float>(crossings) / (numSamples - 1);
}

} // namespace midikompanion::audio
