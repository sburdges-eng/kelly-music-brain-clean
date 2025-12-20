#pragma once
/*
 * AudioAnalyzer.h - Comprehensive Audio Analysis
 * ==============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Audio Layer: Orchestrates F0Extractor and SpectralAnalyzer
 * - ML Layer: Used by MLFeatureExtractor for comprehensive feature extraction
 * - DDSP Layer: Used by DDSPProcessor for analysis-driven synthesis
 *
 * Purpose: High-level audio analysis combining F0 extraction, spectral analysis,
 *          and loudness measurement. Provides unified interface for audio feature extraction.
 *
 * Features:
 * - F0 (fundamental frequency) extraction
 * - Loudness measurement (RMS, perceptual)
 * - Spectral analysis (STFT, spectral features)
 * - Harmonic analysis
 * - Real-time processing
 */

#include "audio/F0Extractor.h"
#include "audio/SpectralAnalyzer.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>

namespace midikompanion {
namespace audio {

/**
 * Audio Analyzer - Comprehensive audio analysis.
 *
 * Combines F0 extraction, spectral analysis, and loudness measurement
 * into a unified interface for audio feature extraction.
 */
class AudioAnalyzer {
public:
    AudioAnalyzer();
    ~AudioAnalyzer() = default;

    /**
     * Analyze audio buffer and extract all features.
     *
     * @param audio Audio buffer
     * @param sampleRate Sample rate in Hz
     * @return Analysis results
     */
    struct AnalysisResult {
        float f0 = 0.0f;                    // Fundamental frequency (Hz)
        float f0Confidence = 0.0f;         // F0 detection confidence (0-1)
        float loudness = 0.0f;             // Loudness in dB
        float rms = 0.0f;                  // RMS energy
        SpectralAnalyzer::SpectralFeatures spectral;  // Spectral features
        std::vector<float> spectralEnvelope;  // Spectral envelope
    };

    AnalysisResult analyze(const juce::AudioBuffer<float>& audio, double sampleRate);

    /**
     * Extract F0 (fundamental frequency).
     *
     * @param audio Audio buffer
     * @param sampleRate Sample rate in Hz
     * @return F0 in Hz, or 0.0 if not detected
     */
    float extractF0(const juce::AudioBuffer<float>& audio, double sampleRate);

    /**
     * Extract loudness (perceptual loudness using K-weighting).
     *
     * @param audio Audio buffer
     * @param sampleRate Sample rate in Hz
     * @return Loudness in dB
     */
    float extractLoudness(const juce::AudioBuffer<float>& audio, double sampleRate);

    /**
     * Extract RMS energy.
     *
     * @param audio Audio buffer
     * @return RMS value (0-1)
     */
    float extractRMS(const juce::AudioBuffer<float>& audio);

    /**
     * Extract spectral features.
     *
     * @param audio Audio buffer
     * @param sampleRate Sample rate in Hz
     * @return Spectral features
     */
    SpectralAnalyzer::SpectralFeatures extractSpectralFeatures(const juce::AudioBuffer<float>& audio, double sampleRate);

    /**
     * Get F0 extractor (for configuration).
     *
     * @return Reference to F0 extractor
     */
    F0Extractor& getF0Extractor() { return f0Extractor_; }

    /**
     * Get spectral analyzer (for configuration).
     *
     * @return Reference to spectral analyzer
     */
    SpectralAnalyzer& getSpectralAnalyzer() { return spectralAnalyzer_; }

private:
    F0Extractor f0Extractor_;
    SpectralAnalyzer spectralAnalyzer_;
};

} // namespace audio
} // namespace midikompanion
