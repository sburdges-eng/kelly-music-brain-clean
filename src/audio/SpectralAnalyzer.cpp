#include "audio/SpectralAnalyzer.h"
#include <juce_dsp/juce_dsp.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace midikompanion {
namespace audio {

SpectralAnalyzer::SpectralAnalyzer(int fftSize)
    : fftSize_(fftSize)
    , fft_(std::make_unique<juce::dsp::FFT>(static_cast<int>(std::log2(fftSize))))
{
    // Initialize buffers
    fftBuffer_.resize(fftSize_ * 2);  // Complex FFT output
    magnitudeBuffer_.resize(fftSize_ / 2 + 1);
    phaseBuffer_.resize(fftSize_ / 2 + 1);
    windowBuffer_.resize(fftSize_);
    previousMagnitude_.resize(fftSize_ / 2 + 1, 0.0f);

    // Pre-compute Hanning window
    for (int i = 0; i < fftSize_; ++i) {
        windowBuffer_[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265359f * static_cast<float>(i) / static_cast<float>(fftSize_ - 1)));
    }
}

SpectralAnalyzer::~SpectralAnalyzer() = default;

SpectralAnalyzer::SpectralFeatures SpectralAnalyzer::analyze(const juce::AudioBuffer<float>& audio, double sampleRate) {
    SpectralFeatures features;

    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) {
        return features;
    }

    // Use first channel for mono analysis
    const float* channelData = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();

    // Ensure we have enough samples
    if (numSamples < fftSize_) {
        // Pad with zeros if needed
        std::fill(fftBuffer_.begin(), fftBuffer_.end(), 0.0f);
        std::copy(channelData, channelData + numSamples, fftBuffer_.begin());
    } else {
        // Use first fftSize_ samples
        std::copy(channelData, channelData + fftSize_, fftBuffer_.begin());
        std::fill(fftBuffer_.begin() + fftSize_, fftBuffer_.end(), 0.0f);
    }

    // Apply window
    applyWindow(fftBuffer_.data(), fftSize_);

    // Compute FFT
    computeFFT(fftBuffer_.data(), magnitudeBuffer_, &phaseBuffer_);

    // Calculate features
    features.centroid = calculateCentroid(magnitudeBuffer_, sampleRate);
    features.rolloff = calculateRolloff(magnitudeBuffer_, sampleRate);
    features.bandwidth = calculateBandwidth(magnitudeBuffer_, features.centroid, sampleRate);
    features.flatness = calculateFlatness(magnitudeBuffer_);
    features.harmonicity = calculateHNR(audio, sampleRate);

    // Calculate spectral flux (change from previous frame)
    float flux = 0.0f;
    for (size_t i = 0; i < magnitudeBuffer_.size(); ++i) {
        float diff = magnitudeBuffer_[i] - previousMagnitude_[i];
        if (diff > 0.0f) {
            flux += diff;  // Only positive changes
        }
    }
    features.flux = flux / static_cast<float>(magnitudeBuffer_.size());

    // Update previous magnitude for next frame
    previousMagnitude_ = magnitudeBuffer_;

    return features;
}

std::vector<std::vector<float>> SpectralAnalyzer::computeSTFT(const juce::AudioBuffer<float>& audio,
                                                                 double sampleRate,
                                                                 int frameSize,
                                                                 int hopSize) {
    std::vector<std::vector<float>> stft;

    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) {
        return stft;
    }

    const float* channelData = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();

    // Process in overlapping frames
    for (int start = 0; start + frameSize <= numSamples; start += hopSize) {
        // Copy frame to buffer
        std::fill(fftBuffer_.begin(), fftBuffer_.end(), 0.0f);
        std::copy(channelData + start, channelData + start + frameSize, fftBuffer_.begin());

        // Apply window
        applyWindow(fftBuffer_.data(), frameSize);

        // Compute FFT
        std::vector<float> magnitude(frameSize / 2 + 1);
        computeFFT(fftBuffer_.data(), magnitude);

        stft.push_back(magnitude);
    }

    return stft;
}

std::vector<float> SpectralAnalyzer::extractSpectralEnvelope(const juce::AudioBuffer<float>& audio, double sampleRate) {
    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) {
        return {};
    }

    const float* channelData = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();

    // Use first fftSize_ samples or pad
    std::fill(fftBuffer_.begin(), fftBuffer_.end(), 0.0f);
    int copySize = std::min(numSamples, fftSize_);
    std::copy(channelData, channelData + copySize, fftBuffer_.begin());

    // Apply window
    applyWindow(fftBuffer_.data(), fftSize_);

    // Compute FFT
    computeFFT(fftBuffer_.data(), magnitudeBuffer_);

    // Return magnitude spectrum (spectral envelope)
    return magnitudeBuffer_;
}

float SpectralAnalyzer::calculateHNR(const juce::AudioBuffer<float>& audio, double sampleRate) {
    // Simplified HNR calculation using autocorrelation
    // Full implementation would use cepstral analysis

    if (audio.getNumSamples() < 1024) {
        return 0.0f;
    }

    const float* samples = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();

    // Calculate autocorrelation
    int maxLag = std::min(400, numSamples / 2);
    float maxCorr = 0.0f;
    int bestLag = 0;

    for (int lag = 40; lag < maxLag; ++lag) {
        float corr = 0.0f;
        for (int i = 0; i < numSamples - lag; ++i) {
            corr += samples[i] * samples[i + lag];
        }
        corr /= (numSamples - lag);

        if (corr > maxCorr) {
            maxCorr = corr;
            bestLag = lag;
        }
    }

    // Calculate energy
    float energy = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        energy += samples[i] * samples[i];
    }
    energy /= numSamples;

    // HNR is ratio of periodic (harmonic) energy to total energy
    if (energy > 0.0f && bestLag > 0) {
        float hnr = maxCorr / (energy + 1e-10f);
        return std::clamp(hnr, 0.0f, 1.0f);
    }

    return 0.0f;
}

void SpectralAnalyzer::computeFFT(const float* frame, std::vector<float>& magnitude, std::vector<float>* phase) {
    // Copy real input to FFT buffer
    // JUCE FFT expects interleaved complex format [real, imag, real, imag, ...]
    for (int i = 0; i < fftSize_; ++i) {
        fftBuffer_[i * 2] = frame[i];      // Real part
        fftBuffer_[i * 2 + 1] = 0.0f;      // Imaginary part (zero for real input)
    }

    // Perform FFT (real-to-complex)
    fft_->performRealOnlyForwardTransform(fftBuffer_.data(), false);

    // Extract magnitude and phase from FFT output
    // FFT output is in format: [DC, real1, imag1, real2, imag2, ..., Nyquist]
    int numBins = fftSize_ / 2 + 1;

    // DC component (bin 0)
    magnitude[0] = std::abs(fftBuffer_[0]);
    if (phase) {
        phase->at(0) = 0.0f;  // DC has no phase
    }

    // Positive frequencies (bins 1 to Nyquist-1)
    for (int i = 1; i < numBins - 1; ++i) {
        float real = fftBuffer_[i * 2];
        float imag = fftBuffer_[i * 2 + 1];

        magnitude[i] = std::sqrt(real * real + imag * imag);

        if (phase) {
            phase->at(i) = std::atan2(imag, real);
        }
    }

    // Nyquist frequency (last bin)
    if (numBins > 0) {
        magnitude[numBins - 1] = std::abs(fftBuffer_[1]);  // Nyquist is real-only
        if (phase) {
            phase->at(numBins - 1) = 0.0f;
        }
    }
}

void SpectralAnalyzer::applyWindow(float* samples, int numSamples) {
    int windowSize = std::min(numSamples, static_cast<int>(windowBuffer_.size()));
    for (int i = 0; i < windowSize; ++i) {
        samples[i] *= windowBuffer_[i];
    }
}

float SpectralAnalyzer::calculateCentroid(const std::vector<float>& magnitude, double sampleRate) {
    float weightedSum = 0.0f;
    float magnitudeSum = 0.0f;

    int numBins = static_cast<int>(magnitude.size());
    float binWidth = static_cast<float>(sampleRate) / static_cast<float>(fftSize_);

    for (int i = 0; i < numBins; ++i) {
        float freq = static_cast<float>(i) * binWidth;
        float mag = magnitude[i];

        weightedSum += freq * mag;
        magnitudeSum += mag;
    }

    if (magnitudeSum > 0.0f) {
        return weightedSum / magnitudeSum;
    }

    return 0.0f;
}

float SpectralAnalyzer::calculateRolloff(const std::vector<float>& magnitude, double sampleRate, float percentile) {
    // Calculate total energy
    float totalEnergy = 0.0f;
    for (float mag : magnitude) {
        totalEnergy += mag * mag;
    }

    if (totalEnergy == 0.0f) {
        return 0.0f;
    }

    float threshold = totalEnergy * percentile;
    float cumulativeEnergy = 0.0f;

    int numBins = static_cast<int>(magnitude.size());
    float binWidth = static_cast<float>(sampleRate) / static_cast<float>(fftSize_);

    for (int i = 0; i < numBins; ++i) {
        cumulativeEnergy += magnitude[i] * magnitude[i];
        if (cumulativeEnergy >= threshold) {
            return static_cast<float>(i) * binWidth;
        }
    }

    return static_cast<float>(numBins - 1) * binWidth;
}

float SpectralAnalyzer::calculateBandwidth(const std::vector<float>& magnitude, float centroid, double sampleRate) {
    float weightedSum = 0.0f;
    float magnitudeSum = 0.0f;

    int numBins = static_cast<int>(magnitude.size());
    float binWidth = static_cast<float>(sampleRate) / static_cast<float>(fftSize_);

    for (int i = 0; i < numBins; ++i) {
        float freq = static_cast<float>(i) * binWidth;
        float mag = magnitude[i];
        float diff = freq - centroid;

        weightedSum += diff * diff * mag;
        magnitudeSum += mag;
    }

    if (magnitudeSum > 0.0f) {
        return std::sqrt(weightedSum / magnitudeSum);
    }

    return 0.0f;
}

float SpectralAnalyzer::calculateFlatness(const std::vector<float>& magnitude) {
    // Spectral flatness = geometric mean / arithmetic mean
    // Higher values indicate more noise-like (less tonal)

    float geometricMean = 1.0f;
    float arithmeticMean = 0.0f;
    int count = 0;

    for (float mag : magnitude) {
        if (mag > 1e-10f) {  // Avoid log(0)
            geometricMean *= std::pow(mag, 1.0f / static_cast<float>(magnitude.size()));
            arithmeticMean += mag;
            count++;
        }
    }

    if (count > 0 && arithmeticMean > 0.0f) {
        arithmeticMean /= count;
        if (arithmeticMean > 0.0f) {
            return geometricMean / arithmeticMean;
        }
    }

    return 0.0f;
}

} // namespace audio
} // namespace midikompanion
