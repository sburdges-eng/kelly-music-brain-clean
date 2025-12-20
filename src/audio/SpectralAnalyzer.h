#pragma once
/*
 * SpectralAnalyzer.h - Spectral Analysis (STFT and Spectral Features)
 * ==================================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Audio Layer: Performs STFT and spectral feature extraction
 * - ML Layer: Used by MLFeatureExtractor for spectral features
 * - DDSP Layer: Used by DDSPProcessor for harmonic analysis
 *
 * Purpose: Real-time spectral analysis using STFT.
 *          Extracts spectral features for ML inference and audio analysis.
 *
 * Features:
 * - STFT (Short-Time Fourier Transform)
 * - Spectral centroid (brightness)
 * - Spectral rolloff
 * - Spectral flux
 * - Harmonic-to-noise ratio
 * - Spectral envelope extraction
 */

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <complex>
#include <cmath>
#include <memory>

namespace midikompanion {
namespace audio {

/**
 * Spectral Analyzer - Real-time spectral analysis using STFT.
 *
 * Provides:
 * - STFT computation
 * - Spectral feature extraction
 * - Harmonic analysis
 * - Real-time processing (<10ms latency)
 */
class SpectralAnalyzer {
public:
    /**
     * Constructor - Initialize with FFT size.
     *
     * @param fftSize FFT size (must be power of 2, default: 2048)
     */
    explicit SpectralAnalyzer(int fftSize = 2048);

    /**
     * Destructor - Must be defined in .cpp file due to unique_ptr<FFT>
     */
    ~SpectralAnalyzer();

    /**
     * Analyze audio buffer and extract spectral features.
     *
     * @param audio Audio buffer
     * @param sampleRate Sample rate in Hz
     * @return Spectral features (centroid, rolloff, flux, etc.)
     */
    struct SpectralFeatures {
        float centroid = 0.0f;      // Spectral centroid (brightness) in Hz
        float rolloff = 0.0f;       // Spectral rolloff frequency in Hz
        float flux = 0.0f;          // Spectral flux (change in spectrum)
        float bandwidth = 0.0f;     // Spectral bandwidth in Hz
        float flatness = 0.0f;      // Spectral flatness (noise measure)
        float harmonicity = 0.0f;   // Harmonic-to-noise ratio (0-1)
    };

    SpectralFeatures analyze(const juce::AudioBuffer<float>& audio, double sampleRate);

    /**
     * Compute STFT of audio buffer.
     *
     * @param audio Audio buffer
     * @param sampleRate Sample rate in Hz
     * @param frameSize Frame size in samples
     * @param hopSize Hop size in samples
     * @return STFT magnitude spectrum (frames × frequency bins)
     */
    std::vector<std::vector<float>> computeSTFT(const juce::AudioBuffer<float>& audio,
                                                 double sampleRate,
                                                 int frameSize = 2048,
                                                 int hopSize = 512);

    /**
     * Extract spectral envelope.
     * Returns the magnitude spectrum envelope.
     *
     * @param audio Audio buffer
     * @param sampleRate Sample rate in Hz
     * @return Spectral envelope (magnitude per frequency bin)
     */
    std::vector<float> extractSpectralEnvelope(const juce::AudioBuffer<float>& audio, double sampleRate);

    /**
     * Calculate harmonic-to-noise ratio.
     * Higher values indicate more harmonic content.
     *
     * @param audio Audio buffer
     * @param sampleRate Sample rate in Hz
     * @return HNR value (0-1, higher = more harmonic)
     */
    float calculateHNR(const juce::AudioBuffer<float>& audio, double sampleRate);

    /**
     * Get FFT size.
     *
     * @return FFT size
     */
    int getFFTSize() const { return fftSize_; }

private:
    /**
     * Compute FFT of a single frame.
     *
     * @param frame Audio frame
     * @param magnitude Output magnitude spectrum
     * @param phase Output phase spectrum (optional)
     */
    void computeFFT(const float* frame, std::vector<float>& magnitude, std::vector<float>* phase = nullptr);

    /**
     * Apply window function (Hanning window).
     *
     * @param samples Input/output samples
     * @param numSamples Number of samples
     */
    void applyWindow(float* samples, int numSamples);

    /**
     * Calculate spectral centroid from magnitude spectrum.
     *
     * @param magnitude Magnitude spectrum
     * @param sampleRate Sample rate in Hz
     * @return Spectral centroid in Hz
     */
    float calculateCentroid(const std::vector<float>& magnitude, double sampleRate);

    /**
     * Calculate spectral rolloff from magnitude spectrum.
     *
     * @param magnitude Magnitude spectrum
     * @param sampleRate Sample rate in Hz
     * @param percentile Percentile for rolloff (default: 0.85 = 85%)
     * @return Spectral rolloff frequency in Hz
     */
    float calculateRolloff(const std::vector<float>& magnitude, double sampleRate, float percentile = 0.85f);

    /**
     * Calculate spectral bandwidth from magnitude spectrum.
     *
     * @param magnitude Magnitude spectrum
     * @param centroid Spectral centroid in Hz
     * @param sampleRate Sample rate in Hz
     * @return Spectral bandwidth in Hz
     */
    float calculateBandwidth(const std::vector<float>& magnitude, float centroid, double sampleRate);

    /**
     * Calculate spectral flatness (noise measure).
     *
     * @param magnitude Magnitude spectrum
     * @return Spectral flatness (0-1, higher = more noise-like)
     */
    float calculateFlatness(const std::vector<float>& magnitude);

    int fftSize_;
    std::unique_ptr<juce::dsp::FFT> fft_;

    // Internal buffers (reused for performance)
    std::vector<float> fftBuffer_;
    std::vector<float> windowBuffer_;
    std::vector<float> magnitudeBuffer_;
    std::vector<float> phaseBuffer_;
    std::vector<float> previousMagnitude_;  // For spectral flux calculation
};

} // namespace audio
} // namespace midikompanion
