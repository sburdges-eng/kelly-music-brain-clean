#pragma once

#include "audio/AudioAnalyzer.h" // For real F0 and loudness extraction
#include <array>
#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>

namespace kelly {

/**
 * DDSPProcessor - DDSP (Differentiable Digital Signal Processing) timbre
 * transfer.
 *
 * Provides interface for DDSP-based neural timbre synthesis and transfer.
 * In full implementation, this would use a trained DDSP model for real-time
 * timbre manipulation and emotion-conditioned synthesis.
 */
class DDSPProcessor {
public:
  static constexpr int N_HARMONICS = 64;
  static constexpr int N_NOISE_FILTERS = 65;
  static constexpr int SAMPLE_RATE = 44100;
  static constexpr int HOP_SIZE = SAMPLE_RATE / 50; // 20ms frames

  DDSPProcessor() = default;
  ~DDSPProcessor() = default;

  /**
   * Prepare processor for audio processing.
   * @param sampleRate Audio sample rate
   */
  void prepare(double sampleRate);

  /**
   * Process audio block with DDSP synthesis.
   * @param f0 Fundamental frequency (pitch) in Hz
   * @param loudness Loudness in dB
   * @param output Output audio buffer
   * @param numSamples Number of samples to generate
   */
  void processBlock(const float *f0, const float *loudness, float *output,
                    int numSamples);

  /**
   * Load DDSP model from file.
   * @param modelPath Path to model file (ONNX, TFLite, etc.)
   * @return true if loaded successfully
   */
  bool loadModel(const juce::File &modelPath);

  /**
   * Check if model is loaded.
   */
  bool isModelLoaded() const { return modelLoaded_; }

  /**
   * Set emotion conditioning (valence/arousal).
   * @param valence Valence value (-1.0 to 1.0)
   * @param arousal Arousal value (0.0 to 1.0)
   */
  void setEmotionConditioning(float valence, float arousal) {
    emotionValence_ = valence;
    emotionArousal_ = arousal;
  }

private:
  bool modelLoaded_ = false;
  double sampleRate_ = 44100.0;
  float emotionValence_ = 0.0f;
  float emotionArousal_ = 0.5f;

  // Audio analyzer for real F0 and loudness extraction (Phase 3: Audio
  // Analysis) Mutable to allow const methods to use it (analyzer is stateless
  // for extraction)
  mutable midikompanion::audio::AudioAnalyzer audioAnalyzer_;

  // DDSP synthesis parameters (would come from model in full implementation)
  std::vector<float> harmonicAmplitudes_;
  std::vector<float> noiseMagnitudes_;
  float overallAmplitude_ = 1.0f;

  // Phase accumulation for smooth harmonic synthesis
  std::vector<float> phaseAccumulators_;

  /**
   * Synthesize harmonic signal from f0 and amplitudes.
   * Implements proper additive synthesis with 64 harmonics.
   */
  void synthesizeHarmonics(const float *f0, const float *amplitudes,
                           float *output, int numSamples);

  /**
   * Synthesize filtered noise using 65 noise filter bands.
   */
  void synthesizeNoise(const float *magnitudes, float *output, int numSamples);

  /**
   * Extract f0 (fundamental frequency) from audio.
   * Uses real AudioAnalyzer instead of placeholder.
   */
  float extractF0(const juce::AudioBuffer<float> &audio) const;

  /**
   * Extract loudness from audio.
   * Uses real AudioAnalyzer instead of placeholder.
   */
  float extractLoudness(const juce::AudioBuffer<float> &audio) const;
};

} // namespace kelly
