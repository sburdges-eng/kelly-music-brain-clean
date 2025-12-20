#include "ml/MLFeatureExtractor.h"

namespace kelly {

MLFeatureExtractor::MLFeatureExtractor() : audioAnalyzer_() {}

// Phase 3: Audio Analysis - Enhanced harmonic feature extraction using real
// audio analysis
void MLFeatureExtractor::extractHarmonicFeatures(
    const float *data, int start, int length,
    std::array<float, FEATURE_SIZE> &features, int startIdx) const {
  // Create temporary audio buffer for analysis
  juce::AudioBuffer<float> tempBuffer(1, length);
  float *channelData = tempBuffer.getWritePointer(0);
  std::copy(data + start, data + start + length, channelData);

  // Use real AudioAnalyzer for F0 and spectral analysis
  double sampleRate = 44100.0; // Default, could be passed as parameter
  auto analysisResult = audioAnalyzer_.analyze(tempBuffer, sampleRate);

  // Extract F0 and confidence
  features[startIdx] = analysisResult.f0 /
                       1000.0f; // Normalize to 0-1 range (assuming max 1000 Hz)
  features[startIdx + 1] = analysisResult.f0Confidence;

  // Extract spectral features
  features[startIdx + 2] =
      analysisResult.spectral.centroid / 10000.0f; // Normalize
  features[startIdx + 3] =
      analysisResult.spectral.rolloff / 10000.0f; // Normalize
  features[startIdx + 4] = analysisResult.spectral.flux;
  features[startIdx + 5] =
      analysisResult.spectral.bandwidth / 10000.0f; // Normalize
  features[startIdx + 6] = analysisResult.spectral.flatness;
  features[startIdx + 7] = analysisResult.spectral.harmonicity;

  // Extract loudness and RMS
  features[startIdx + 8] =
      (analysisResult.loudness + 60.0f) / 60.0f; // Normalize -60 to 0 dB to 0-1
  features[startIdx + 9] = analysisResult.rms;

  // Extract spectral envelope features (first 10 bins)
  size_t envelopeSize =
      std::min(analysisResult.spectralEnvelope.size(), size_t(10));
  for (size_t i = 0; i < envelopeSize; ++i) {
    features[startIdx + 10 + static_cast<int>(i)] =
        analysisResult.spectralEnvelope[i];
  }

  // Fill remaining with zeros
  for (int i = static_cast<int>(10 + envelopeSize); i < 20; ++i) {
    features[startIdx + i] = 0.0f;
  }
}

} // namespace kelly
