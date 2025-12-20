#include "audio/AudioAnalyzer.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace midikompanion {
namespace audio {

AudioAnalyzer::AudioAnalyzer()
    : f0Extractor_()
    , spectralAnalyzer_(2048)  // 2048-point FFT
{
}

AudioAnalyzer::AnalysisResult AudioAnalyzer::analyze(const juce::AudioBuffer<float>& audio, double sampleRate) {
    AnalysisResult result;

    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) {
        return result;
    }

    // Extract F0
    result.f0 = extractF0(audio, sampleRate);
    result.f0Confidence = f0Extractor_.getConfidence();

    // Extract loudness
    result.loudness = extractLoudness(audio, sampleRate);

    // Extract RMS
    result.rms = extractRMS(audio);

    // Extract spectral features
    result.spectral = extractSpectralFeatures(audio, sampleRate);

    // Extract spectral envelope
    result.spectralEnvelope = spectralAnalyzer_.extractSpectralEnvelope(audio, sampleRate);

    return result;
}

float AudioAnalyzer::extractF0(const juce::AudioBuffer<float>& audio, double sampleRate) {
    return f0Extractor_.extractPitch(audio, sampleRate);
}

float AudioAnalyzer::extractLoudness(const juce::AudioBuffer<float>& audio, double sampleRate) {
    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) {
        return -std::numeric_limits<float>::infinity();  // Silence
    }

    // Use first channel for mono analysis
    const float* channelData = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();

    // Calculate RMS
    float rms = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        rms += channelData[i] * channelData[i];
    }
    rms = std::sqrt(rms / static_cast<float>(numSamples));

    // Convert to dB with K-weighting approximation
    // K-weighting is a simplified perceptual loudness measure
    // Full implementation would use proper K-weighting filter
    float db = 20.0f * std::log10(rms + 1e-10f);

    // Apply K-weighting correction (simplified)
    // K-weighting boosts low frequencies and attenuates high frequencies
    // For now, we'll use a simple correction factor
    // In production, apply proper K-weighting filter before RMS calculation
    float kWeightedDb = db + 2.0f;  // Simplified correction

    return kWeightedDb;
}

float AudioAnalyzer::extractRMS(const juce::AudioBuffer<float>& audio) {
    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) {
        return 0.0f;
    }

    // Use first channel for mono analysis
    const float* channelData = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();

    float sumSquares = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        sumSquares += channelData[i] * channelData[i];
    }

    return std::sqrt(sumSquares / static_cast<float>(numSamples));
}

SpectralAnalyzer::SpectralFeatures AudioAnalyzer::extractSpectralFeatures(const juce::AudioBuffer<float>& audio, double sampleRate) {
    return spectralAnalyzer_.analyze(audio, sampleRate);
}

} // namespace audio
} // namespace midikompanion
