// Example: Loading ONNX models in PluginProcessor
// ================================================================

#include "ml/ONNXInference.h"

void PluginProcessor::loadMLModels() {
    // Get models directory (from Resources or user directory)
    juce::File modelsDir = juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                              .getParentDirectory()
                              .getChildFile("Resources")
                              .getChildFile("models");

    // Load EmotionRecognizer
    emotionRecognizer_ = std::make_unique<midikompanion::ml::ONNXInference>();
    juce::File emotionModel = modelsDir.getChildFile("emotionrecognizer.onnx");
    if (emotionModel.existsAsFile()) {
        if (emotionRecognizer_->loadModel(emotionModel)) {
            DBG("EmotionRecognizer loaded successfully");
        } else {
            DBG("Failed to load EmotionRecognizer: " + emotionRecognizer_->getLastError());
        }
    }

    // Load MelodyTransformer
    melodyTransformer_ = std::make_unique<midikompanion::ml::ONNXInference>();
    juce::File melodyModel = modelsDir.getChildFile("melodytransformer.onnx");
    if (melodyModel.existsAsFile()) {
        if (melodyTransformer_->loadModel(melodyModel)) {
            DBG("MelodyTransformer loaded successfully");
        }
    }

    // Load other models similarly...
}

// Usage in processBlock:
void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    if (emotionRecognizer_ && emotionRecognizer_->isModelLoaded()) {
        // Extract audio features (128-dim)
        std::vector<float> features = extractAudioFeatures(buffer);

        // Run inference
        std::vector<float> emotionEmbedding = emotionRecognizer_->infer(features);

        // Use embedding for music generation...
    }
}
