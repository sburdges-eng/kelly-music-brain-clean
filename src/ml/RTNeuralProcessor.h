#pragma once

#include <juce_core/juce_core.h>
#include <array>
#include <memory>
#include <string>
#include <sstream>
#include <fstream>

#ifdef ENABLE_RTNEURAL
#include <RTNeural/RTNeural.h>
#endif

namespace kelly {

/**
 * RTNeuralProcessor - Real-time neural network inference for emotion processing.
 *
 * Provides runtime model loading and inference using RTNeural library.
 * Supports emotion-conditioned MIDI generation and audio feature processing.
 *
 * Model architecture: 128 (input) → 128 (dense+tanh) → 64 (LSTM) → 64 (dense+tanh) → 64 (output)
 */
class RTNeuralProcessor {
public:
    RTNeuralProcessor() = default;
    ~RTNeuralProcessor() = default;

    /**
     * Load model from JSON file (RTNeural format).
     * @param jsonFile Path to model JSON file
     * @return true if loaded successfully
     */
    bool loadModel(const juce::File& jsonFile);

    /**
     * Process audio samples through model.
     * @param input Input audio samples
     * @param output Output buffer
     * @param numSamples Number of samples to process
     */
    void process(const float* input, float* output, int numSamples) {
        if (!isModelLoaded()) {
            // Passthrough if no model loaded
            std::memcpy(output, input, numSamples * sizeof(float));
            return;
        }

#ifdef ENABLE_RTNEURAL
        // Note: This method processes raw audio samples
        // For emotion inference, use inferEmotion() with extracted features instead
        // This is a simplified implementation that just passes through
        juce::ignoreUnused(input, output, numSamples);
        juce::Logger::writeToLog("Warning: process() called but not implemented. Use inferEmotion() instead.");
        std::memcpy(output, input, numSamples * sizeof(float));
#else
        // Passthrough when RTNeural is not available
        std::memcpy(output, input, numSamples * sizeof(float));
#endif
    }

    /**
     * Batch inference for MIDI features (emotion vector output).
     * @param features Input feature vector (128 dimensions)
     * @return Emotion vector (64 dimensions)
     */
    std::array<float, 64> inferEmotion(const std::array<float, 128>& features);

    /**
     * Check if model is loaded.
     */
    bool isModelLoaded() const {
#ifdef ENABLE_RTNEURAL
        return modelLoaded_;
#else
        return model_.loaded;
#endif
    }

    /**
     * Get model path.
     */
    std::string getModelPath() const {
#ifdef ENABLE_RTNEURAL
        return modelPath_;
#else
        return model_.modelPath;
#endif
    }

private:
#ifdef ENABLE_RTNEURAL
    std::unique_ptr<RTNeural::Model<float>> model_;
    bool modelLoaded_ = false;
    std::string modelPath_;
#else
    struct PlaceholderModel {
        bool loaded = false;
        std::string modelPath;
    };
    PlaceholderModel model_;
#endif
};

} // namespace kelly
