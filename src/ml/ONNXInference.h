#pragma once
/*
 * ONNXInference.h - ONNX Runtime Wrapper for Real-Time Inference
 * ==============================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - ML Layer: ONNX Runtime library (cross-platform inference engine)
 * - ML Layer: Used by MultiModelProcessor, NodeMLMapper for ONNX model inference
 * - Audio Layer: Real-time inference wrapper (thread-safe, RT-safe)
 *
 * Purpose: RT-safe wrapper around ONNX Runtime for loading and running ONNX models.
 *          Designed for real-time audio processing with <10ms latency target.
 *
 * Features:
 * - Model loading from file
 * - Thread-safe inference (can be called from audio thread with proper setup)
 * - Input/output size queries
 * - Error handling and validation
 *
 * Interface Contract (must match Agent 2's ONNX model specifications):
 * - Input/Output sizes match model specs (128→64 for EmotionRecognizer, etc.)
 * - Models are optimized for <10ms inference
 * - Models are <5MB total size
 */

#include <juce_core/juce_core.h>
#include <vector>
#include <optional>
#include <memory>
#include <mutex>
#include <string>

// Forward declare ONNX Runtime types to avoid requiring full header inclusion
// This allows compilation even if ONNX Runtime is not available
// Note: Actual types are defined in onnxruntime_cxx_api.h (included in .cpp)

namespace midikompanion {
namespace ml {

/**
 * ONNX Inference Wrapper - RT-safe ONNX Runtime interface.
 *
 * Provides:
 * - Model loading from ONNX files
 * - Thread-safe inference
 * - Input/output size queries
 * - Error handling
 *
 * Thread Safety:
 * - loadModel(): Must be called from non-audio thread (e.g., prepareToPlay)
 * - infer(): Can be called from audio thread if model is pre-loaded and inputs pre-allocated
 * - For RT-safety, pre-allocate input/output buffers and avoid allocations during inference
 */
class ONNXInference {
public:
    ONNXInference();
    ~ONNXInference();

    // Non-copyable, non-movable
    ONNXInference(const ONNXInference&) = delete;
    ONNXInference& operator=(const ONNXInference&) = delete;
    ONNXInference(ONNXInference&&) = delete;
    ONNXInference& operator=(ONNXInference&&) = delete;

    /**
     * Load ONNX model from file.
     * Must be called from non-audio thread (e.g., prepareToPlay).
     *
     * @param modelPath Path to .onnx model file
     * @return true if successful
     */
    bool loadModel(const juce::File& modelPath);

    /**
     * Load ONNX model from file path string.
     *
     * @param modelPath Path to .onnx model file
     * @return true if successful
     */
    bool loadModel(const std::string& modelPath);

    /**
     * Run inference on input data.
     * Thread-safe: Can be called from audio thread if model is pre-loaded.
     * For RT-safety, ensure input/output buffers are pre-allocated.
     *
     * @param input Input vector (size must match getInputSize())
     * @return Output vector (size matches getOutputSize()), or empty on error
     */
    std::vector<float> infer(const std::vector<float>& input);

    /**
     * Run inference with pre-allocated output buffer (RT-safe).
     * Avoids allocations during inference for audio thread safety.
     *
     * @param input Input data (size must match getInputSize())
     * @param output Output buffer (size must match getOutputSize())
     * @return true if successful
     */
    bool infer(const float* input, float* output);

    /**
     * Get input size required by the model.
     *
     * @return Input size (e.g., 128 for EmotionRecognizer)
     */
    size_t getInputSize() const { return inputSize_; }

    /**
     * Get output size produced by the model.
     *
     * @return Output size (e.g., 64 for EmotionRecognizer)
     */
    size_t getOutputSize() const { return outputSize_; }

    /**
     * Check if model is loaded and ready for inference.
     *
     * @return true if model is loaded
     */
    bool isLoaded() const { return isLoaded_; }

    /**
     * Get last error message if operation failed.
     *
     * @return Error message string, empty if no error
     */
    juce::String getLastError() const { return lastError_; }

    /**
     * Get model file path if loaded.
     *
     * @return Model file path, or empty if not loaded
     */
    juce::String getModelPath() const { return modelPath_; }

private:
    void clearError() { lastError_.clear(); }
    void setError(const juce::String& error) { lastError_ = error; }

    // Initialize ONNX Runtime environment (called once)
    void initializeONNX();

    // Validate input/output sizes
    bool validateInputSize(size_t size) const;
    bool validateOutputSize(size_t size) const;

#ifdef ENABLE_ONNX_RUNTIME
    // ONNX Runtime types (forward declared, actual types in .cpp)
    // Using void* to avoid requiring ONNX Runtime headers in header file
    void* sessionPtr_;  // Points to Ort::Session
    void* envPtr_;      // Points to Ort::Env
    void* memoryInfoPtr_;  // Points to Ort::MemoryInfo
#else
    // Stub members when ONNX Runtime is not available
    void* sessionPtr_ = nullptr;
    void* envPtr_ = nullptr;
    void* memoryInfoPtr_ = nullptr;
#endif

    size_t inputSize_ = 0;
    size_t outputSize_ = 0;
    bool isLoaded_ = false;
    juce::String modelPath_;
    juce::String lastError_;

    // Thread safety
    mutable std::mutex mutex_;

    // ONNX Runtime initialization flag (static, initialized once)
    static bool onnxInitialized_;
    static std::mutex initMutex_;
};

} // namespace ml
} // namespace midikompanion
