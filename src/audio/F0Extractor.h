#pragma once
/*
 * F0Extractor.h - Fundamental Frequency (Pitch) Extraction
 * ========================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Audio Layer: Extracts F0 from juce::AudioBuffer
 * - ML Layer: Used by DDSPProcessor, MLFeatureExtractor for pitch analysis
 * - Voice Layer: Used by VoiceSynthesizer for vocal pitch tracking
 *
 * Purpose: Real-time F0 (fundamental frequency) extraction using YIN algorithm.
 *          Provides accurate pitch detection for audio analysis and DDSP synthesis.
 *
 * Features:
 * - YIN algorithm for robust pitch detection
 * - Real-time processing (<10ms latency)
 * - Confidence scoring
 * - Frame-based analysis
 */

#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>
#include <cmath>

namespace midikompanion {
namespace audio {

/**
 * F0 Extractor - Real-time fundamental frequency extraction.
 *
 * Uses YIN algorithm for robust pitch detection in noisy signals.
 * Suitable for real-time audio processing with <10ms latency target.
 */
class F0Extractor {
public:
    F0Extractor();
    ~F0Extractor() = default;

    /**
     * Extract pitch from audio buffer.
     * Uses YIN algorithm for robust pitch detection.
     *
     * @param audio Audio buffer (mono or stereo - uses first channel)
     * @param sampleRate Sample rate in Hz
     * @return Detected fundamental frequency in Hz, or 0.0 if not detected
     */
    float extractPitch(const juce::AudioBuffer<float>& audio, double sampleRate);

    /**
     * Extract pitch from raw audio samples.
     *
     * @param samples Audio samples (mono)
     * @param numSamples Number of samples
     * @param sampleRate Sample rate in Hz
     * @return Detected fundamental frequency in Hz, or 0.0 if not detected
     */
    float extractPitch(const float* samples, int numSamples, double sampleRate);

    /**
     * Extract pitch trajectory (frame-by-frame).
     * Processes audio in frames and returns F0 for each frame.
     *
     * @param audio Audio buffer
     * @param sampleRate Sample rate in Hz
     * @param frameSize Frame size in samples (default: 2048)
     * @param hopSize Hop size in samples (default: 512)
     * @return Vector of F0 values (one per frame) in Hz
     */
    std::vector<float> extractPitchTrajectory(const juce::AudioBuffer<float>& audio,
                                               double sampleRate,
                                               int frameSize = 2048,
                                               int hopSize = 512);

    /**
     * Get confidence score for last extraction.
     * Higher values (0.0-1.0) indicate more reliable pitch detection.
     *
     * @return Confidence score (0.0 = unreliable, 1.0 = very reliable)
     */
    float getConfidence() const { return confidence_; }

    /**
     * Set minimum frequency for pitch detection.
     *
     * @param minFreq Minimum frequency in Hz (default: 50.0)
     */
    void setMinFrequency(float minFreq) { minFreq_ = minFreq; }

    /**
     * Set maximum frequency for pitch detection.
     *
     * @param maxFreq Maximum frequency in Hz (default: 2000.0)
     */
    void setMaxFrequency(float maxFreq) { maxFreq_ = maxFreq; }

    /**
     * Set threshold for YIN algorithm.
     * Lower values = more sensitive, higher values = more robust to noise.
     *
     * @param threshold YIN threshold (default: 0.1)
     */
    void setThreshold(float threshold) { threshold_ = threshold; }

private:
    /**
     * YIN algorithm implementation.
     * Returns the period (in samples) of the fundamental frequency.
     *
     * @param samples Audio samples
     * @param numSamples Number of samples
     * @param sampleRate Sample rate in Hz
     * @return Period in samples, or 0 if not detected
     */
    int yinAlgorithm(const float* samples, int numSamples, double sampleRate);

    /**
     * Calculate difference function for YIN algorithm.
     *
     * @param samples Audio samples
     * @param numSamples Number of samples
     * @param maxLag Maximum lag to search
     * @param diff Output difference function
     */
    void calculateDifferenceFunction(const float* samples, int numSamples, int maxLag, std::vector<float>& diff);

    /**
     * Calculate cumulative mean normalized difference function.
     *
     * @param diff Difference function
     * @param numSamples Number of samples
     * @param cmndf Output cumulative mean normalized difference function
     */
    void calculateCMNDF(const std::vector<float>& diff, int numSamples, std::vector<float>& cmndf);

    /**
     * Find minimum in CMNDF below threshold.
     *
     * @param cmndf Cumulative mean normalized difference function
     * @param minLag Minimum lag (in samples)
     * @param maxLag Maximum lag (in samples)
     * @param threshold Threshold value
     * @return Lag (in samples) of minimum, or 0 if not found
     */
    int findMinimum(const std::vector<float>& cmndf, int minLag, int maxLag, float threshold);

    /**
     * Parabolic interpolation for sub-sample accuracy.
     *
     * @param cmndf Cumulative mean normalized difference function
     * @param lag Lag of minimum
     * @return Refined lag with sub-sample accuracy
     */
    float parabolicInterpolation(const std::vector<float>& cmndf, int lag);

    float minFreq_ = 50.0f;      // Minimum frequency (Hz)
    float maxFreq_ = 2000.0f;   // Maximum frequency (Hz)
    float threshold_ = 0.1f;     // YIN threshold
    float confidence_ = 0.0f;   // Last confidence score

    // Internal buffers (reused for performance)
    std::vector<float> diffBuffer_;
    std::vector<float> cmndfBuffer_;
};

} // namespace audio
} // namespace midikompanion
