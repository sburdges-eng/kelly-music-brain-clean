#pragma once

#include "audio/AudioAnalyzer.h" // Phase 3: Use real audio analysis
#include <algorithm>             // For std::copy
#include <array>
#include <cmath>
#include <juce_audio_basics/juce_audio_basics.h>

namespace kelly {

/**
 * MLFeatureExtractor - Extracts audio features for ML inference.
 *
 * Extracts spectral features, RMS, zero crossings, and other
 * features suitable for emotion-conditioned processing.
 */
class MLFeatureExtractor {
public:
  static constexpr size_t FEATURE_SIZE = 128;

  MLFeatureExtractor();
  ~MLFeatureExtractor() = default;

  /**
   * Extract features from audio buffer.
   * @param buffer Audio buffer
   * @param startSample Start sample index
   * @param numSamples Number of samples to analyze
   * @return Feature vector (128 dimensions)
   */
  std::array<float, FEATURE_SIZE>
  extractFeatures(const juce::AudioBuffer<float> &buffer, int startSample = 0,
                  int numSamples = -1) const {
    std::array<float, FEATURE_SIZE> features{};

    if (buffer.getNumSamples() == 0) {
      return features;
    }

    const int actualNumSamples =
        (numSamples < 0) ? buffer.getNumSamples() : numSamples;
    const int endSample =
        std::min(startSample + actualNumSamples, buffer.getNumSamples());
    const int actualLength = endSample - startSample;

    if (actualLength == 0) {
      return features;
    }

    // Use first channel for mono features
    const float *channelData = buffer.getReadPointer(0);

    // Extract basic features
    int featureIdx = 0;

    // 1. RMS (Root Mean Square) - overall energy
    float rms = 0.0f;
    for (int i = startSample; i < endSample; ++i) {
      rms += channelData[i] * channelData[i];
    }
    rms = std::sqrt(rms / actualLength);
    features[featureIdx++] = rms;

    // 2. Zero Crossing Rate
    int zeroCrossings = 0;
    for (int i = startSample + 1; i < endSample; ++i) {
      if ((channelData[i] >= 0.0f) != (channelData[i - 1] >= 0.0f)) {
        zeroCrossings++;
      }
    }
    features[featureIdx++] = static_cast<float>(zeroCrossings) / actualLength;

    // 3-5. Use real AudioAnalyzer for spectral features (Phase 3: Audio
    // Analysis) Create temporary buffer for analysis
    juce::AudioBuffer<float> tempBuffer(1, actualLength);
    float *tempData = tempBuffer.getWritePointer(0);
    std::copy(channelData + startSample, channelData + endSample, tempData);

    double sampleRate = 44100.0; // Default sample rate
    auto spectralFeatures =
        audioAnalyzer_.getSpectralAnalyzer().analyze(tempBuffer, sampleRate);

    features[featureIdx++] = spectralFeatures.centroid / 10000.0f; // Normalize
    features[featureIdx++] = spectralFeatures.rolloff / 10000.0f;  // Normalize
    features[featureIdx++] = spectralFeatures.flux;

    // 6. MFCC-like features (simplified)
    extractMFCCFeatures(channelData, startSample, actualLength, features,
                        featureIdx);
    featureIdx += 13; // 13 MFCC coefficients

    // 7. Spectral features (FFT-based) - Enhanced with real spectral analysis
    // Use real AudioAnalyzer for spectral envelope (reuse tempBuffer from
    // above)
    auto spectralEnvelope =
        audioAnalyzer_.getSpectralAnalyzer().extractSpectralEnvelope(
            tempBuffer, sampleRate);

    // Use spectral envelope for features (up to 64 bins)
    size_t envelopeSize = std::min(spectralEnvelope.size(), size_t(64));
    for (size_t i = 0; i < envelopeSize; ++i) {
      features[featureIdx + static_cast<int>(i)] = spectralEnvelope[i];
    }
    // Fill remaining with zeros
    for (size_t i = envelopeSize; i < 64; ++i) {
      features[featureIdx + static_cast<int>(i)] = 0.0f;
    }
    featureIdx += 64; // 64 spectral bins

    // 8. Temporal features
    extractTemporalFeatures(channelData, startSample, actualLength, features,
                            featureIdx);
    featureIdx += 20; // 20 temporal features

    // 9. Harmonic features
    extractHarmonicFeatures(channelData, startSample, actualLength, features,
                            featureIdx);
    featureIdx += 20; // 20 harmonic features

    // Fill remaining with zeros if needed
    while (featureIdx < FEATURE_SIZE) {
      features[featureIdx++] = 0.0f;
    }

    return features;
  }

private:
  float computeSpectralCentroid(const float *data, int start,
                                int length) const {
    // Simplified spectral centroid calculation
    float weightedSum = 0.0f;
    float magnitudeSum = 0.0f;

    for (int i = 0; i < length; ++i) {
      float mag = std::abs(data[start + i]);
      weightedSum += i * mag;
      magnitudeSum += mag;
    }

    return (magnitudeSum > 0.0f) ? (weightedSum / magnitudeSum) : 0.0f;
  }

  float computeSpectralRolloff(const float *data, int start, int length) const {
    // Simplified spectral rolloff (85% energy)
    float totalEnergy = 0.0f;
    for (int i = 0; i < length; ++i) {
      totalEnergy += data[start + i] * data[start + i];
    }

    float threshold = totalEnergy * 0.85f;
    float cumulativeEnergy = 0.0f;

    for (int i = 0; i < length; ++i) {
      cumulativeEnergy += data[start + i] * data[start + i];
      if (cumulativeEnergy >= threshold) {
        return static_cast<float>(i) / length;
      }
    }

    return 1.0f;
  }

  float computeSpectralFlux(const float *data, int start, int length) const {
    // Simplified spectral flux (change in spectrum)
    if (length < 2)
      return 0.0f;

    float flux = 0.0f;
    for (int i = 1; i < length; ++i) {
      float diff = data[start + i] - data[start + i - 1];
      flux += (diff > 0.0f) ? diff : 0.0f; // Only positive changes
    }

    return flux / (length - 1);
  }

  void extractMFCCFeatures(const float *data, int start, int length,
                           std::array<float, FEATURE_SIZE> &features,
                           int startIdx) const {
    // Simplified MFCC extraction (13 coefficients)
    // In a full implementation, this would use FFT and mel filterbank
    for (int i = 0; i < 13; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < length; ++j) {
        float freq = static_cast<float>(j) / length;
        sum += data[start + j] *
               std::cos(static_cast<float>(i) * freq * 2.0f * 3.14159f);
      }
      features[startIdx + i] = sum / length;
    }
  }

  void extractSpectralFeatures(const float *data, int start, int length,
                               std::array<float, FEATURE_SIZE> &features,
                               int startIdx) const {
    // Simplified spectral features using windowed FFT
    // In a full implementation, use juce::dsp::FFT
    const int fftSize = 64;
    for (int i = 0; i < fftSize && (start + i) < (start + length); ++i) {
      features[startIdx + i] = std::abs(data[start + i]);
    }
  }

  void extractTemporalFeatures(const float *data, int start, int length,
                               std::array<float, FEATURE_SIZE> &features,
                               int startIdx) const {
    // Temporal features: attack, decay, sustain, release
    if (length < 4)
      return;

    // Attack time
    float maxVal = 0.0f;
    int maxIdx = 0;
    for (int i = 0; i < length; ++i) {
      float absVal = std::abs(data[start + i]);
      if (absVal > maxVal) {
        maxVal = absVal;
        maxIdx = i;
      }
    }
    features[startIdx] = static_cast<float>(maxIdx) / length;

    // Decay rate
    if (maxIdx < length - 1) {
      float decay =
          (maxVal - std::abs(data[start + length - 1])) / (length - maxIdx);
      features[startIdx + 1] = decay;
    }

    // Additional temporal statistics
    float mean = 0.0f, variance = 0.0f;
    for (int i = 0; i < length; ++i) {
      mean += data[start + i];
    }
    mean /= length;

    for (int i = 0; i < length; ++i) {
      float diff = data[start + i] - mean;
      variance += diff * diff;
    }
    variance /= length;

    features[startIdx + 2] = mean;
    features[startIdx + 3] = std::sqrt(variance);

    // Fill remaining with zeros
    for (int i = 4; i < 20; ++i) {
      features[startIdx + i] = 0.0f;
    }
  }

  void extractHarmonicFeatures(const float *data, int start, int length,
                               std::array<float, FEATURE_SIZE> &features,
                               int startIdx) const;

  // Phase 3: Audio Analysis - Use real audio analysis classes
  mutable midikompanion::audio::AudioAnalyzer audioAnalyzer_;
};

} // namespace kelly
