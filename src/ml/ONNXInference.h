/**
 * @file ONNXInference.h
 * @brief ONNX Runtime wrapper for ML model inference
 * 
 * Provides a unified interface for loading and running ONNX models.
 * Supports the 5-model ML pipeline:
 * - EmotionRecognizer: Audio features → 64-dim embedding
 * - MelodyTransformer: Node embedding → MIDI notes
 * - HarmonyPredictor: Node context → Chord probabilities
 * - DynamicsEngine: Node intensity → Expression parameters
 * - GroovePredictor: Node arousal → Groove parameters
 */

#pragma once

#include <juce_core/juce_core.h>
#include <memory>
#include <string>
#include <vector>
#include <optional>

// Forward declare ONNX Runtime types to avoid header dependency
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
    struct MemoryInfo;
    struct Value;
}

namespace midikompanion {
namespace ml {

/**
 * @brief Model types supported by the ML pipeline
 */
enum class ModelType {
    EmotionRecognizer,   // Audio → 64-dim embedding
    MelodyTransformer,   // 64-dim → 128 note probabilities
    HarmonyPredictor,    // 128-dim → 64 chord probabilities
    DynamicsEngine,      // 32-dim → 16 expression parameters
    GroovePredictor      // 64-dim → 32 groove parameters
};

/**
 * @brief Model specification (input/output sizes)
 */
struct ModelSpec {
    ModelType type;
    size_t inputSize;
    size_t outputSize;
    juce::String modelName;
    
    static ModelSpec forType(ModelType type);
};

/**
 * @brief Inference result with confidence
 */
struct InferenceResult {
    std::vector<float> output;
    float confidence = 0.0f;
    double inferenceTimeMs = 0.0;
    bool success = false;
    juce::String errorMessage;
};

/**
 * @brief ONNX Runtime inference wrapper
 * 
 * Thread-safe ONNX model loading and inference.
 * Supports batch inference for better performance.
 */
class ONNXInference {
public:
    ONNXInference();
    ~ONNXInference();
    
    // Non-copyable
    ONNXInference(const ONNXInference&) = delete;
    ONNXInference& operator=(const ONNXInference&) = delete;
    
    // Movable
    ONNXInference(ONNXInference&&) noexcept;
    ONNXInference& operator=(ONNXInference&&) noexcept;
    
    /**
     * @brief Load ONNX model from file
     * @param modelPath Path to .onnx file
     * @return true if successful
     */
    bool loadModel(const juce::File& modelPath);
    
    /**
     * @brief Load ONNX model from memory
     * @param data Model data
     * @param size Size in bytes
     * @return true if successful
     */
    bool loadModelFromMemory(const void* data, size_t size);
    
    /**
     * @brief Check if a model is loaded
     */
    bool isModelLoaded() const;
    
    /**
     * @brief Unload current model
     */
    void unloadModel();
    
    /**
     * @brief Run inference on input data
     * @param input Input vector (must match expected input size)
     * @return Inference result
     */
    InferenceResult infer(const std::vector<float>& input);
    
    /**
     * @brief Run batch inference
     * @param inputs Vector of input vectors
     * @return Vector of inference results
     */
    std::vector<InferenceResult> inferBatch(const std::vector<std::vector<float>>& inputs);
    
    /**
     * @brief Get expected input size
     */
    size_t getInputSize() const { return inputSize_; }
    
    /**
     * @brief Get expected output size
     */
    size_t getOutputSize() const { return outputSize_; }
    
    /**
     * @brief Get input tensor shape
     */
    std::vector<int64_t> getInputShape() const { return inputShape_; }
    
    /**
     * @brief Get output tensor shape
     */
    std::vector<int64_t> getOutputShape() const { return outputShape_; }
    
    /**
     * @brief Get model name/path
     */
    juce::String getModelName() const { return modelName_; }
    
    /**
     * @brief Get last error message
     */
    juce::String getLastError() const { return lastError_; }
    
    /**
     * @brief Check if ONNX Runtime is available
     * @return true if ONNX Runtime was compiled in
     */
    static bool isONNXAvailable();
    
    /**
     * @brief Get ONNX Runtime version
     */
    static juce::String getONNXVersion();
    
    /**
     * @brief Set number of inference threads
     */
    void setNumThreads(int numThreads);
    
    /**
     * @brief Enable/disable graph optimization
     */
    void setGraphOptimization(bool enable);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    
    size_t inputSize_ = 0;
    size_t outputSize_ = 0;
    std::vector<int64_t> inputShape_;
    std::vector<int64_t> outputShape_;
    juce::String modelName_;
    juce::String lastError_;
    int numThreads_ = 1;
    bool graphOptimization_ = true;
    
    mutable juce::CriticalSection lock_;
};

/**
 * @brief Multi-model processor for the 5-model pipeline
 * 
 * Manages loading and inference for all ML models.
 * Supports hybrid mode: ML-enhanced + rule-based fallback.
 */
class MultiModelProcessor {
public:
    MultiModelProcessor();
    ~MultiModelProcessor() = default;
    
    /**
     * @brief Load all models from directory
     * @param modelsDir Directory containing .onnx files
     * @return Number of models loaded successfully
     */
    int loadModels(const juce::File& modelsDir);
    
    /**
     * @brief Load a specific model
     * @param type Model type
     * @param modelPath Path to .onnx file
     * @return true if successful
     */
    bool loadModel(ModelType type, const juce::File& modelPath);
    
    /**
     * @brief Check if a specific model is loaded
     */
    bool isModelLoaded(ModelType type) const;
    
    /**
     * @brief Check if all models are loaded
     */
    bool areAllModelsLoaded() const;
    
    /**
     * @brief Run emotion recognition on audio features
     * @param audioFeatures 128-dim audio feature vector
     * @return 64-dim emotion embedding
     */
    InferenceResult recognizeEmotion(const std::vector<float>& audioFeatures);
    
    /**
     * @brief Generate melody from emotion embedding
     * @param emotionEmbedding 64-dim emotion vector
     * @return 128-dim note probabilities
     */
    InferenceResult generateMelody(const std::vector<float>& emotionEmbedding);
    
    /**
     * @brief Predict harmonies from context
     * @param context 128-dim context vector
     * @return 64-dim chord probabilities
     */
    InferenceResult predictHarmony(const std::vector<float>& context);
    
    /**
     * @brief Generate dynamics from intensity
     * @param intensity 32-dim intensity vector
     * @return 16-dim expression parameters
     */
    InferenceResult generateDynamics(const std::vector<float>& intensity);
    
    /**
     * @brief Predict groove from arousal
     * @param arousal 64-dim arousal vector
     * @return 32-dim groove parameters
     */
    InferenceResult predictGroove(const std::vector<float>& arousal);
    
    /**
     * @brief Enable/disable ML mode (vs rule-based fallback)
     */
    void setMLEnabled(bool enabled) { mlEnabled_ = enabled; }
    bool isMLEnabled() const { return mlEnabled_; }
    
    /**
     * @brief Get status summary
     */
    juce::String getStatusSummary() const;

private:
    std::map<ModelType, std::unique_ptr<ONNXInference>> models_;
    bool mlEnabled_ = true;
    
    InferenceResult runModel(ModelType type, const std::vector<float>& input);
};

} // namespace ml
} // namespace midikompanion
