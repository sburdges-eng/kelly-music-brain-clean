#include "ml/ONNXInference.h"
#include <juce_core/juce_core.h>
#include <cstring>

#ifdef ENABLE_ONNX_RUNTIME
// Include ONNX Runtime headers
#include <onnxruntime_cxx_api.h>
using namespace Ort;
#endif

namespace midikompanion {
namespace ml {

// Static initialization
bool ONNXInference::onnxInitialized_ = false;
std::mutex ONNXInference::initMutex_;

ONNXInference::ONNXInference()
    : inputSize_(0)
    , outputSize_(0)
    , isLoaded_(false)
{
    clearError();
}

ONNXInference::~ONNXInference() {
#ifdef ENABLE_ONNX_RUNTIME
    // Cleanup ONNX Runtime objects
    if (sessionPtr_) {
        delete static_cast<Session*>(sessionPtr_);
        sessionPtr_ = nullptr;
    }
    if (memoryInfoPtr_) {
        delete static_cast<MemoryInfo*>(memoryInfoPtr_);
        memoryInfoPtr_ = nullptr;
    }
    // Note: Env is typically kept alive for the lifetime of the application
    // We'll keep it for now (could be made static if needed)
#endif
}

void ONNXInference::initializeONNX() {
    std::lock_guard<std::mutex> lock(initMutex_);

    if (onnxInitialized_) {
        return;
    }

#ifdef ENABLE_ONNX_RUNTIME
    try {
        // Initialize ONNX Runtime environment (global, initialized once)
        // Note: Ort::Env is thread-safe and can be initialized multiple times safely
        // We use a static flag to avoid redundant initialization
        onnxInitialized_ = true;
    } catch (const std::exception& e) {
        setError("Failed to initialize ONNX Runtime: " + juce::String(e.what()));
        onnxInitialized_ = false;
    }
#else
    // ONNX Runtime not available - stub mode
    onnxInitialized_ = true;  // Allow stub mode to work
#endif
}

bool ONNXInference::loadModel(const juce::File& modelPath) {
    return loadModel(modelPath.getFullPathName().toStdString());
}

bool ONNXInference::loadModel(const std::string& modelPath) {
    clearError();

    if (!juce::File(modelPath).existsAsFile()) {
        setError("Model file does not exist: " + juce::String(modelPath));
        return false;
    }

    initializeONNX();

#ifdef ENABLE_ONNX_RUNTIME
    try {
        std::lock_guard<std::mutex> lock(mutex_);

        using namespace Ort;

        // Create ONNX Runtime environment if not already created
        if (!envPtr_) {
            envPtr_ = new Env(ORT_LOGGING_LEVEL_WARNING, "MidiKompanion");
        }
        Env* env = static_cast<Env*>(envPtr_);

        // Create session options
        SessionOptions sessionOptions;

        // Optimize for inference speed
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // Set execution providers (prefer CPU for now, can add GPU later)
        // For Apple platforms, could use CoreMLExecutionProvider
        // For NVIDIA, could use CUDAExecutionProvider

        // Create session
        sessionPtr_ = new Session(*env, modelPath.c_str(), sessionOptions);

        Session* session = static_cast<Session*>(sessionPtr_);

        // Get input/output information
        size_t numInputNodes = session->GetInputCount();
        size_t numOutputNodes = session->GetOutputCount();

        if (numInputNodes == 0 || numOutputNodes == 0) {
            setError("Invalid model: no input or output nodes");
            return false;
        }

        Session* session = static_cast<Session*>(sessionPtr_);

        // Get input shape
        AllocatorWithDefaultOptions allocator;
        auto inputName = session->GetInputNameAllocated(0, allocator);
        auto inputTypeInfo = session->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputShape = inputTensorInfo.GetShape();

        // Calculate input size (handle dynamic dimensions)
        inputSize_ = 1;
        for (auto dim : inputShape) {
            if (dim > 0) {
                inputSize_ *= static_cast<size_t>(dim);
            } else {
                // Dynamic dimension - use default based on model type
                // For EmotionRecognizer, default to 128
                inputSize_ *= 128;
                break;
            }
        }

        // Get output shape
        auto outputName = session->GetOutputNameAllocated(0, allocator);
        auto outputTypeInfo = session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputShape = outputTensorInfo.GetShape();

        // Calculate output size
        outputSize_ = 1;
        for (auto dim : outputShape) {
            if (dim > 0) {
                outputSize_ *= static_cast<size_t>(dim);
            } else {
                // Dynamic dimension - use default
                // For EmotionRecognizer, default to 64
                outputSize_ *= 64;
                break;
            }
        }

        // Create memory info
        memoryInfoPtr_ = new MemoryInfo(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        modelPath_ = juce::String(modelPath);
        isLoaded_ = true;

        return true;

    } catch (const std::exception& e) {
        setError("Failed to load ONNX model: " + juce::String(e.what()));
        isLoaded_ = false;
        return false;
    }
#else
    // Stub mode: Return success but don't actually load
    // This allows code to compile without ONNX Runtime
    setError("ONNX Runtime not enabled. Set ENABLE_ONNX_RUNTIME=ON in CMake.");
    inputSize_ = 128;  // Default stub sizes
    outputSize_ = 64;
    modelPath_ = juce::String(modelPath);
    isLoaded_ = false;  // Mark as not loaded in stub mode
    return false;
#endif
}

std::vector<float> ONNXInference::infer(const std::vector<float>& input) {
    if (!isLoaded_) {
        setError("Model not loaded");
        return {};
    }

    if (!validateInputSize(input.size())) {
        return {};
    }

    std::vector<float> output(outputSize_);

    if (infer(input.data(), output.data())) {
        return output;
    }

    return {};
}

bool ONNXInference::infer(const float* input, float* output) {
    if (!isLoaded_) {
        setError("Model not loaded");
        return false;
    }

#ifdef ENABLE_ONNX_RUNTIME
    try {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!sessionPtr_ || !memoryInfoPtr_) {
            setError("ONNX session not initialized");
            return false;
        }

        using namespace Ort;

        Session* session = static_cast<Session*>(sessionPtr_);
        MemoryInfo* memoryInfo = static_cast<MemoryInfo*>(memoryInfoPtr_);

        AllocatorWithDefaultOptions allocator;

        // Get input/output names
        auto inputName = session->GetInputNameAllocated(0, allocator);
        auto outputName = session->GetOutputNameAllocated(0, allocator);

        // Create input tensor
        std::vector<int64_t> inputShape = {1, static_cast<int64_t>(inputSize_)};
        Value inputTensor = Value::CreateTensor<float>(
            *memoryInfo,
            const_cast<float*>(input),  // ONNX Runtime doesn't modify, but API requires non-const
            inputSize_,
            inputShape.data(),
            inputShape.size()
        );

        // Run inference
        auto outputTensors = session->Run(
            RunOptions{nullptr},
            &inputName.get(), &inputTensor, 1,
            &outputName.get(), 1
        );

        if (outputTensors.empty()) {
            setError("Inference returned no outputs");
            return false;
        }

        // Extract output data
        float* outputData = outputTensors.front().GetTensorMutableData<float>();
        size_t outputCount = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

        if (outputCount != outputSize_) {
            setError("Output size mismatch: expected " + juce::String(outputSize_) +
                     ", got " + juce::String(outputCount));
            return false;
        }

        // Copy output
        std::memcpy(output, outputData, outputSize_ * sizeof(float));

        return true;

    } catch (const std::exception& e) {
        setError("Inference failed: " + juce::String(e.what()));
        return false;
    }
#else
    // Stub mode: Return random data for testing
    setError("ONNX Runtime not enabled");
    for (size_t i = 0; i < outputSize_; ++i) {
        output[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;  // Random -1 to 1
    }
    return false;  // Return false to indicate stub mode
#endif
}

bool ONNXInference::validateInputSize(size_t size) const {
    if (size != inputSize_) {
        setError("Input size mismatch: expected " + juce::String(inputSize_) +
                 ", got " + juce::String(size));
        return false;
    }
    return true;
}

bool ONNXInference::validateOutputSize(size_t size) const {
    if (size != outputSize_) {
        setError("Output size mismatch: expected " + juce::String(outputSize_) +
                 ", got " + juce::String(size));
        return false;
    }
    return true;
}

} // namespace ml
} // namespace midikompanion
