/**
 * @file AudioAnalyzer.h
 * @brief Real-time audio analysis for emotion-to-music generation
 * 
 * Provides:
 * - F0 (fundamental frequency) extraction
 * - Loudness extraction (RMS and perceptual)
 * - Spectral analysis (STFT, centroid, rolloff)
 * - Feature extraction for ML models
 */

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <deque>
#include <memory>

namespace midikompanion::audio {

/**
 * @brief F0 extraction result
 */
struct F0Result {
    float frequency{0.0f};     // Detected F0 in Hz
    float confidence{0.0f};    // Detection confidence [0, 1]
    bool voiced{false};        // Whether frame is voiced
    int midiNote{-1};          // Closest MIDI note
    float midiCents{0.0f};     // Cents deviation from MIDI note
};

/**
 * @brief Loudness result
 */
struct LoudnessResult {
    float rms{0.0f};           // RMS value [0, 1]
    float rmsDb{-100.0f};      // RMS in dB
    float peak{0.0f};          // Peak value
    float lufs{-100.0f};       // Integrated LUFS (if available)
};

/**
 * @brief Spectral analysis result
 */
struct SpectralResult {
    float centroid{0.0f};      // Spectral centroid in Hz
    float rolloff{0.0f};       // Spectral rolloff in Hz
    float brightness{0.0f};    // High frequency ratio [0, 1]
    float spread{0.0f};        // Spectral spread
    float flatness{0.0f};      // Spectral flatness [0, 1]
    std::vector<float> spectrum;  // Magnitude spectrum
};

/**
 * @brief Complete audio features for ML
 */
struct AudioFeatures {
    F0Result f0;
    LoudnessResult loudness;
    SpectralResult spectral;
    
    std::vector<float> mfcc;      // Mel-frequency cepstral coefficients
    std::vector<float> chroma;    // Chroma features
    float zcr{0.0f};              // Zero crossing rate
    
    /**
     * @brief Get 128-dim feature vector for ML models
     */
    std::vector<float> toMLInput() const;
};

/**
 * @brief F0 extractor using YIN algorithm
 */
class F0Extractor {
public:
    F0Extractor();
    ~F0Extractor() = default;
    
    /**
     * @brief Configure the extractor
     * @param sampleRate Audio sample rate
     * @param frameSize Analysis frame size
     */
    void configure(double sampleRate, int frameSize);
    
    /**
     * @brief Extract F0 from audio buffer
     * @param audio Input audio buffer
     * @return F0 result with frequency and confidence
     */
    F0Result extract(const juce::AudioBuffer<float>& audio);
    
    /**
     * @brief Extract F0 from single channel
     */
    F0Result extract(const float* samples, int numSamples);
    
    /**
     * @brief Get confidence threshold
     */
    float getConfidenceThreshold() const { return confidenceThreshold_; }
    void setConfidenceThreshold(float threshold) { confidenceThreshold_ = threshold; }
    
    /**
     * @brief Get frequency range
     */
    float getMinFrequency() const { return minFreq_; }
    float getMaxFrequency() const { return maxFreq_; }
    void setFrequencyRange(float minHz, float maxHz);
    
private:
    /**
     * @brief YIN algorithm implementation
     */
    float computeYIN(const float* samples, int numSamples);
    
    /**
     * @brief Compute cumulative mean normalized difference function
     */
    void computeCMND(const float* samples, int numSamples);
    
    /**
     * @brief Find pitch period in CMND
     */
    int findPitchPeriod();
    
    /**
     * @brief Parabolic interpolation for sub-sample accuracy
     */
    float parabolicInterpolation(int period);
    
    /**
     * @brief Convert frequency to MIDI note
     */
    static int frequencyToMidiNote(float frequency);
    static float frequencyToMidiCents(float frequency, int midiNote);
    
    double sampleRate_{44100.0};
    int frameSize_{2048};
    float confidenceThreshold_{0.85f};
    float minFreq_{50.0f};
    float maxFreq_{2000.0f};
    
    std::vector<float> yinBuffer_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(F0Extractor)
};

/**
 * @brief Spectral analyzer using STFT
 */
class SpectralAnalyzer {
public:
    SpectralAnalyzer();
    ~SpectralAnalyzer() = default;
    
    /**
     * @brief Configure the analyzer
     */
    void configure(double sampleRate, int fftSize);
    
    /**
     * @brief Analyze audio buffer
     */
    SpectralResult analyze(const juce::AudioBuffer<float>& audio);
    SpectralResult analyze(const float* samples, int numSamples);
    
    /**
     * @brief Get magnitude spectrum
     */
    const std::vector<float>& getSpectrum() const { return magnitudeSpectrum_; }
    
private:
    /**
     * @brief Apply window function
     */
    void applyWindow(float* data, int size);
    
    /**
     * @brief Compute spectral centroid
     */
    float computeCentroid();
    
    /**
     * @brief Compute spectral rolloff
     */
    float computeRolloff(float percentile = 0.85f);
    
    /**
     * @brief Compute spectral flatness
     */
    float computeFlatness();
    
    double sampleRate_{44100.0};
    int fftSize_{2048};
    
    juce::dsp::FFT fft_{11}; // 2048 samples
    juce::dsp::WindowingFunction<float> window_{2048, juce::dsp::WindowingFunction<float>::hann};
    
    std::vector<float> fftBuffer_;
    std::vector<float> magnitudeSpectrum_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectralAnalyzer)
};

/**
 * @brief Complete audio analyzer combining F0, loudness, and spectral analysis
 */
class AudioAnalyzer {
public:
    AudioAnalyzer();
    ~AudioAnalyzer() = default;
    
    /**
     * @brief Configure the analyzer
     * @param sampleRate Audio sample rate
     * @param blockSize Processing block size
     */
    void configure(double sampleRate, int blockSize);
    
    /**
     * @brief Analyze a block of audio
     * @param audio Input audio buffer
     * @return Complete audio features
     */
    AudioFeatures analyze(const juce::AudioBuffer<float>& audio);
    
    /**
     * @brief Process audio block incrementally
     * 
     * Call this from processBlock for frame-based analysis
     */
    void processBlock(const juce::AudioBuffer<float>& audio);
    
    /**
     * @brief Get latest analysis results
     */
    AudioFeatures getLatestFeatures() const { return latestFeatures_; }
    
    /**
     * @brief Get 128-dim ML feature vector
     */
    std::vector<float> extractMLFeatures(const juce::AudioBuffer<float>& audio);
    
    /**
     * @brief Get F0 extractor
     */
    F0Extractor& getF0Extractor() { return f0Extractor_; }
    
    /**
     * @brief Get spectral analyzer
     */
    SpectralAnalyzer& getSpectralAnalyzer() { return spectralAnalyzer_; }
    
private:
    /**
     * @brief Compute loudness
     */
    LoudnessResult computeLoudness(const float* samples, int numSamples);
    
    /**
     * @brief Compute MFCC features
     */
    std::vector<float> computeMFCC(const std::vector<float>& spectrum);
    
    /**
     * @brief Compute chroma features
     */
    std::vector<float> computeChroma(const std::vector<float>& spectrum);
    
    /**
     * @brief Compute zero crossing rate
     */
    float computeZCR(const float* samples, int numSamples);
    
    double sampleRate_{44100.0};
    int blockSize_{512};
    int hopSize_{256};
    
    F0Extractor f0Extractor_;
    SpectralAnalyzer spectralAnalyzer_;
    
    // Mel filterbank
    std::vector<std::vector<float>> melFilters_;
    int numMelBins_{40};
    
    // Analysis buffers
    std::vector<float> analysisBuffer_;
    int analysisBufferPos_{0};
    
    AudioFeatures latestFeatures_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioAnalyzer)
};

} // namespace midikompanion::audio
