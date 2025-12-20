/**
 * @file F0Extractor.h
 * @brief Fundamental frequency (F0) extraction for pitch detection
 * 
 * Implements the YIN algorithm for real-time pitch detection.
 * Designed for voice and monophonic instrument analysis.
 */

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <vector>
#include <deque>

namespace midikompanion {
namespace audio {

/**
 * @brief Pitch detection result
 */
struct PitchResult {
    float frequency = 0.0f;      // Detected frequency in Hz (0 if unvoiced)
    float confidence = 0.0f;     // Detection confidence (0-1)
    float amplitude = 0.0f;      // RMS amplitude
    bool voiced = false;         // Whether the frame is voiced
    int midiNote = -1;           // Nearest MIDI note (-1 if unvoiced)
    float midiCents = 0.0f;      // Cents deviation from nearest MIDI note
};

/**
 * @brief F0 extraction configuration
 */
struct F0Config {
    float minFrequency = 50.0f;     // Minimum detectable frequency (Hz)
    float maxFrequency = 2000.0f;   // Maximum detectable frequency (Hz)
    float yinThreshold = 0.15f;     // YIN aperiodicity threshold (lower = stricter)
    int hopSize = 256;              // Samples between analysis frames
    int frameSize = 2048;           // Analysis window size
    float voicingThreshold = 0.5f;  // Confidence threshold for voicing
    bool useParabolicInterpolation = true;
};

/**
 * @brief Real-time F0 (fundamental frequency) extractor
 * 
 * Uses the YIN algorithm for robust pitch detection.
 * Thread-safe and suitable for real-time audio processing.
 */
class F0Extractor {
public:
    F0Extractor();
    explicit F0Extractor(const F0Config& config);
    ~F0Extractor() = default;
    
    /**
     * @brief Initialize with sample rate
     * @param sampleRate Audio sample rate in Hz
     */
    void prepare(double sampleRate);
    
    /**
     * @brief Reset internal state
     */
    void reset();
    
    /**
     * @brief Set configuration
     */
    void setConfig(const F0Config& config);
    const F0Config& getConfig() const { return config_; }
    
    /**
     * @brief Process audio buffer and extract pitch
     * @param audio Input audio buffer (mono)
     * @return Vector of pitch results (one per hop)
     */
    std::vector<PitchResult> extractPitch(const juce::AudioBuffer<float>& audio);
    
    /**
     * @brief Process a single audio frame
     * @param samples Pointer to audio samples
     * @param numSamples Number of samples
     * @return Pitch result for this frame
     */
    PitchResult processFrame(const float* samples, int numSamples);
    
    /**
     * @brief Push samples for streaming processing
     * @param samples Audio samples to add
     * @param numSamples Number of samples
     */
    void pushSamples(const float* samples, int numSamples);
    
    /**
     * @brief Check if a complete frame is available
     */
    bool hasFrame() const;
    
    /**
     * @brief Get the next available pitch result
     * @return Pitch result if available
     */
    PitchResult popFrame();
    
    /**
     * @brief Get overall confidence based on recent detections
     */
    float getConfidence() const { return overallConfidence_; }
    
    /**
     * @brief Get smoothed pitch (median filtered)
     */
    float getSmoothedPitch() const;
    
    /**
     * @brief Convert frequency to MIDI note number
     */
    static int frequencyToMidi(float frequency);
    
    /**
     * @brief Convert frequency to MIDI note with cents
     */
    static std::pair<int, float> frequencyToMidiWithCents(float frequency);
    
    /**
     * @brief Convert MIDI note to frequency
     */
    static float midiToFrequency(int midiNote);

private:
    F0Config config_;
    double sampleRate_ = 44100.0;
    
    // YIN algorithm buffers
    std::vector<float> yinBuffer_;
    std::vector<float> frameBuffer_;
    
    // Streaming input buffer
    std::deque<float> inputBuffer_;
    mutable juce::CriticalSection bufferLock_;
    
    // Pitch tracking
    std::deque<float> pitchHistory_;
    float overallConfidence_ = 0.0f;
    static constexpr int PITCH_HISTORY_SIZE = 5;
    
    // YIN algorithm implementation
    float computeYinDifference(const float* samples, int lag) const;
    void computeCumulativeMeanNormalizedDifference(std::vector<float>& yinBuffer) const;
    int findYinMinimum(const std::vector<float>& yinBuffer) const;
    float parabolicInterpolation(const std::vector<float>& yinBuffer, int minIndex) const;
    
    // Helper functions
    float computeRMS(const float* samples, int numSamples) const;
    void updatePitchHistory(float pitch);
};

/**
 * @brief Spectral analyzer for audio feature extraction
 */
class SpectralAnalyzer {
public:
    SpectralAnalyzer();
    ~SpectralAnalyzer() = default;
    
    /**
     * @brief Initialize with sample rate and FFT size
     */
    void prepare(double sampleRate, int fftSize = 2048);
    
    /**
     * @brief Analyze audio buffer
     * @param audio Input audio (mono)
     * @return Spectral features
     */
    struct SpectralFeatures {
        float centroid = 0.0f;       // Spectral centroid (brightness)
        float spread = 0.0f;         // Spectral spread
        float flatness = 0.0f;       // Spectral flatness (noisiness)
        float rolloff = 0.0f;        // Spectral rolloff (85%)
        float flux = 0.0f;           // Spectral flux (change)
        float hnr = 0.0f;            // Harmonic-to-noise ratio
        std::vector<float> mfcc;     // Mel-frequency cepstral coefficients
    };
    
    SpectralFeatures analyze(const juce::AudioBuffer<float>& audio);
    
    /**
     * @brief Get magnitude spectrum from last analysis
     */
    const std::vector<float>& getMagnitudeSpectrum() const { return magnitudeSpectrum_; }
    
    /**
     * @brief Get FFT size
     */
    int getFFTSize() const { return fftSize_; }

private:
    double sampleRate_ = 44100.0;
    int fftSize_ = 2048;
    
    std::unique_ptr<juce::dsp::FFT> fft_;
    std::vector<float> fftInput_;
    std::vector<float> fftOutput_;
    std::vector<float> magnitudeSpectrum_;
    std::vector<float> previousSpectrum_;
    std::vector<float> window_;
    
    // Mel filterbank for MFCC
    std::vector<std::vector<float>> melFilterbank_;
    int numMelFilters_ = 26;
    int numMfcc_ = 13;
    
    void computeMelFilterbank();
    float hzToMel(float hz) const;
    float melToHz(float mel) const;
};

/**
 * @brief Loudness analyzer with perceptual weighting
 */
class LoudnessAnalyzer {
public:
    LoudnessAnalyzer();
    ~LoudnessAnalyzer() = default;
    
    /**
     * @brief Initialize with sample rate
     */
    void prepare(double sampleRate);
    
    /**
     * @brief Analyze loudness of audio buffer
     */
    struct LoudnessResult {
        float rms = 0.0f;            // RMS level (linear)
        float rmsDb = -100.0f;       // RMS level (dB)
        float peak = 0.0f;           // Peak level (linear)
        float peakDb = -100.0f;      // Peak level (dB)
        float lufs = -100.0f;        // Integrated LUFS (simplified)
        float loudness = 0.0f;       // Perceptual loudness (0-1)
    };
    
    LoudnessResult analyze(const juce::AudioBuffer<float>& audio);
    
    /**
     * @brief Get short-term loudness
     */
    float getShortTermLoudness() const { return shortTermLoudness_; }
    
    /**
     * @brief Get momentary loudness
     */
    float getMomentaryLoudness() const { return momentaryLoudness_; }

private:
    double sampleRate_ = 44100.0;
    
    // K-weighting filter coefficients
    std::vector<float> kWeightingCoeffs_;
    std::vector<float> filterState_;
    
    // Loudness history
    std::deque<float> loudnessHistory_;
    float shortTermLoudness_ = 0.0f;
    float momentaryLoudness_ = 0.0f;
    
    void applyKWeighting(juce::AudioBuffer<float>& audio);
    float computeMeanSquare(const juce::AudioBuffer<float>& audio) const;
};

/**
 * @brief Combined audio feature extractor for ML pipeline
 */
class MLFeatureExtractor {
public:
    MLFeatureExtractor();
    ~MLFeatureExtractor() = default;
    
    /**
     * @brief Initialize with sample rate
     */
    void prepare(double sampleRate);
    
    /**
     * @brief Extract all features for ML
     * @param audio Input audio buffer
     * @return 128-dimensional feature vector
     */
    std::vector<float> extractFeatures(const juce::AudioBuffer<float>& audio);
    
    /**
     * @brief Get F0 extractor
     */
    F0Extractor& getF0Extractor() { return f0Extractor_; }
    
    /**
     * @brief Get spectral analyzer
     */
    SpectralAnalyzer& getSpectralAnalyzer() { return spectralAnalyzer_; }
    
    /**
     * @brief Get loudness analyzer
     */
    LoudnessAnalyzer& getLoudnessAnalyzer() { return loudnessAnalyzer_; }

private:
    F0Extractor f0Extractor_;
    SpectralAnalyzer spectralAnalyzer_;
    LoudnessAnalyzer loudnessAnalyzer_;
    
    // Feature normalization
    std::vector<float> featureMean_;
    std::vector<float> featureStd_;
    bool normalizationInitialized_ = false;
    
    void initializeNormalization();
    void normalizeFeatures(std::vector<float>& features) const;
};

} // namespace audio
} // namespace midikompanion
