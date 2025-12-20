/**
 * @file ONNXInference.cpp
 * @brief ONNX Runtime inference implementation
 * 
 * Provides ONNX model loading and inference with fallback
 * for when ONNX Runtime is not available.
 */

#include "ONNXInference.h"
#include <chrono>

// Conditionally include ONNX Runtime
#ifndef KELLY_HAS_ONNX
#define KELLY_HAS_ONNX 0
#endif

#if KELLY_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace midikompanion {
namespace ml {

//==============================================================================
// ModelSpec Implementation
//==============================================================================

ModelSpec ModelSpec::forType(ModelType type) {
    ModelSpec spec;
    spec.type = type;
    
    switch (type) {
        case ModelType::EmotionRecognizer:
            spec.inputSize = 128;
            spec.outputSize = 64;
            spec.modelName = "emotion_recognizer.onnx";
            break;
        case ModelType::MelodyTransformer:
            spec.inputSize = 64;
            spec.outputSize = 128;
            spec.modelName = "melody_transformer.onnx";
            break;
        case ModelType::HarmonyPredictor:
            spec.inputSize = 128;
            spec.outputSize = 64;
            spec.modelName = "harmony_predictor.onnx";
            break;
        case ModelType::DynamicsEngine:
            spec.inputSize = 32;
            spec.outputSize = 16;
            spec.modelName = "dynamics_engine.onnx";
            break;
        case ModelType::GroovePredictor:
            spec.inputSize = 64;
            spec.outputSize = 32;
            spec.modelName = "groove_predictor.onnx";
            break;
    }
    
    return spec;
}

//==============================================================================
// ONNXInference Implementation
//==============================================================================

struct ONNXInference::Impl {
#if KELLY_HAS_ONNX
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "midikompanion"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<std::string> inputNameStrings;
    std::vector<std::string> outputNameStrings;
#endif
    bool modelLoaded = false;
};

ONNXInference::ONNXInference() : pImpl_(std::make_unique<Impl>()) {}

ONNXInference::~ONNXInference() = default;

ONNXInference::ONNXInference(ONNXInference&&) noexcept = default;
ONNXInference& ONNXInference::operator=(ONNXInference&&) noexcept = default;

bool ONNXInference::loadModel(const juce::File& modelPath) {
    const juce::ScopedLock lock(lock_);
    
#if KELLY_HAS_ONNX
    try {
        if (!modelPath.existsAsFile()) {
            lastError_ = "Model file not found: " + modelPath.getFullPathName();
            return false;
        }
        
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(numThreads_);
        
        if (graphOptimization_) {
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        } else {
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        }
        
        // Load the model
        pImpl_->session = std::make_unique<Ort::Session>(
            pImpl_->env, 
            modelPath.getFullPathName().toRawUTF8(), 
            sessionOptions
        );
        
        // Get input info
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t numInputs = pImpl_->session->GetInputCount();
        pImpl_->inputNameStrings.clear();
        pImpl_->inputNames.clear();
        
        for (size_t i = 0; i < numInputs; ++i) {
            auto name = pImpl_->session->GetInputNameAllocated(i, allocator);
            pImpl_->inputNameStrings.push_back(name.get());
        }
        
        for (const auto& name : pImpl_->inputNameStrings) {
            pImpl_->inputNames.push_back(name.c_str());
        }
        
        // Get output info
        size_t numOutputs = pImpl_->session->GetOutputCount();
        pImpl_->outputNameStrings.clear();
        pImpl_->outputNames.clear();
        
        for (size_t i = 0; i < numOutputs; ++i) {
            auto name = pImpl_->session->GetOutputNameAllocated(i, allocator);
            pImpl_->outputNameStrings.push_back(name.get());
        }
        
        for (const auto& name : pImpl_->outputNameStrings) {
            pImpl_->outputNames.push_back(name.c_str());
        }
        
        // Get input shape
        auto inputTypeInfo = pImpl_->session->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShape_ = inputTensorInfo.GetShape();
        
        inputSize_ = 1;
        for (auto dim : inputShape_) {
            if (dim > 0) inputSize_ *= static_cast<size_t>(dim);
        }
        
        // Get output shape
        auto outputTypeInfo = pImpl_->session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputShape_ = outputTensorInfo.GetShape();
        
        outputSize_ = 1;
        for (auto dim : outputShape_) {
            if (dim > 0) outputSize_ *= static_cast<size_t>(dim);
        }
        
        modelName_ = modelPath.getFileName();
        pImpl_->modelLoaded = true;
        
        return true;
    }
    catch (const Ort::Exception& e) {
        lastError_ = juce::String("ONNX error: ") + e.what();
        pImpl_->session.reset();
        pImpl_->modelLoaded = false;
        return false;
    }
    catch (const std::exception& e) {
        lastError_ = juce::String("Error loading model: ") + e.what();
        pImpl_->session.reset();
        pImpl_->modelLoaded = false;
        return false;
    }
#else
    lastError_ = "ONNX Runtime not available";
    return false;
#endif
}

bool ONNXInference::loadModelFromMemory(const void* data, size_t size) {
    const juce::ScopedLock lock(lock_);
    
#if KELLY_HAS_ONNX
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(numThreads_);
        
        if (graphOptimization_) {
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }
        
        pImpl_->session = std::make_unique<Ort::Session>(
            pImpl_->env,
            data,
            size,
            sessionOptions
        );
        
        pImpl_->modelLoaded = true;
        modelName_ = "memory_model";
        
        // Get shapes (same as loadModel)
        Ort::AllocatorWithDefaultOptions allocator;
        
        auto inputTypeInfo = pImpl_->session->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShape_ = inputTensorInfo.GetShape();
        
        inputSize_ = 1;
        for (auto dim : inputShape_) {
            if (dim > 0) inputSize_ *= static_cast<size_t>(dim);
        }
        
        auto outputTypeInfo = pImpl_->session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputShape_ = outputTensorInfo.GetShape();
        
        outputSize_ = 1;
        for (auto dim : outputShape_) {
            if (dim > 0) outputSize_ *= static_cast<size_t>(dim);
        }
        
        return true;
    }
    catch (const Ort::Exception& e) {
        lastError_ = juce::String("ONNX error: ") + e.what();
        return false;
    }
#else
    juce::ignoreUnused(data, size);
    lastError_ = "ONNX Runtime not available";
    return false;
#endif
}

bool ONNXInference::isModelLoaded() const {
    const juce::ScopedLock lock(lock_);
    return pImpl_->modelLoaded;
}

void ONNXInference::unloadModel() {
    const juce::ScopedLock lock(lock_);
    
#if KELLY_HAS_ONNX
    pImpl_->session.reset();
    pImpl_->inputNames.clear();
    pImpl_->outputNames.clear();
    pImpl_->inputNameStrings.clear();
    pImpl_->outputNameStrings.clear();
#endif
    
    pImpl_->modelLoaded = false;
    inputSize_ = 0;
    outputSize_ = 0;
    inputShape_.clear();
    outputShape_.clear();
    modelName_.clear();
}

InferenceResult ONNXInference::infer(const std::vector<float>& input) {
    const juce::ScopedLock lock(lock_);
    
    InferenceResult result;
    
#if KELLY_HAS_ONNX
    if (!pImpl_->modelLoaded || !pImpl_->session) {
        result.success = false;
        result.errorMessage = "No model loaded";
        return result;
    }
    
    if (input.size() != inputSize_) {
        result.success = false;
        result.errorMessage = "Input size mismatch: expected " + 
                              juce::String(inputSize_) + ", got " + 
                              juce::String(input.size());
        return result;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Prepare input shape (handle batch dimension)
        std::vector<int64_t> inputDims = inputShape_;
        if (inputDims.empty() || inputDims[0] == -1) {
            inputDims = {1, static_cast<int64_t>(input.size())};
        }
        
        // Create input tensor
        auto inputTensor = Ort::Value::CreateTensor<float>(
            pImpl_->memoryInfo,
            const_cast<float*>(input.data()),
            input.size(),
            inputDims.data(),
            inputDims.size()
        );
        
        // Run inference
        auto outputTensors = pImpl_->session->Run(
            Ort::RunOptions{nullptr},
            pImpl_->inputNames.data(),
            &inputTensor,
            1,
            pImpl_->outputNames.data(),
            pImpl_->outputNames.size()
        );
        
        // Extract output
        if (!outputTensors.empty() && outputTensors[0].IsTensor()) {
            auto& outputTensor = outputTensors[0];
            float* outputData = outputTensor.GetTensorMutableData<float>();
            size_t outputCount = outputTensor.GetTensorTypeAndShapeInfo().GetElementCount();
            
            result.output.assign(outputData, outputData + outputCount);
            result.success = true;
            
            // Calculate confidence (mean of absolute values)
            float sum = 0.0f;
            for (float val : result.output) {
                sum += std::abs(val);
            }
            result.confidence = result.output.empty() ? 0.0f : sum / result.output.size();
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        result.inferenceTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    }
    catch (const Ort::Exception& e) {
        result.success = false;
        result.errorMessage = juce::String("ONNX inference error: ") + e.what();
    }
    catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = juce::String("Inference error: ") + e.what();
    }
#else
    // Fallback: generate placeholder output
    result.success = true;
    result.output.resize(outputSize_, 0.0f);
    
    // Generate some deterministic output based on input
    for (size_t i = 0; i < result.output.size() && i < input.size(); ++i) {
        result.output[i] = input[i] * 0.5f;
    }
    
    result.confidence = 0.5f;
    result.inferenceTimeMs = 0.1;
#endif
    
    return result;
}

std::vector<InferenceResult> ONNXInference::inferBatch(const std::vector<std::vector<float>>& inputs) {
    std::vector<InferenceResult> results;
    results.reserve(inputs.size());
    
    // For now, just run individual inferences
    // TODO: Implement true batch inference for better performance
    for (const auto& input : inputs) {
        results.push_back(infer(input));
    }
    
    return results;
}

bool ONNXInference::isONNXAvailable() {
#if KELLY_HAS_ONNX
    return true;
#else
    return false;
#endif
}

juce::String ONNXInference::getONNXVersion() {
#if KELLY_HAS_ONNX
    return juce::String(ORT_API_VERSION);
#else
    return "Not available";
#endif
}

void ONNXInference::setNumThreads(int numThreads) {
    numThreads_ = juce::jmax(1, numThreads);
}

void ONNXInference::setGraphOptimization(bool enable) {
    graphOptimization_ = enable;
}

//==============================================================================
// MultiModelProcessor Implementation
//==============================================================================

MultiModelProcessor::MultiModelProcessor() {
    // Pre-create model slots
    models_[ModelType::EmotionRecognizer] = std::make_unique<ONNXInference>();
    models_[ModelType::MelodyTransformer] = std::make_unique<ONNXInference>();
    models_[ModelType::HarmonyPredictor] = std::make_unique<ONNXInference>();
    models_[ModelType::DynamicsEngine] = std::make_unique<ONNXInference>();
    models_[ModelType::GroovePredictor] = std::make_unique<ONNXInference>();
}

int MultiModelProcessor::loadModels(const juce::File& modelsDir) {
    if (!modelsDir.isDirectory()) {
        return 0;
    }
    
    int loaded = 0;
    
    for (auto& [type, model] : models_) {
        auto spec = ModelSpec::forType(type);
        auto modelFile = modelsDir.getChildFile(spec.modelName);
        
        if (modelFile.existsAsFile()) {
            if (model->loadModel(modelFile)) {
                ++loaded;
            }
        }
    }
    
    return loaded;
}

bool MultiModelProcessor::loadModel(ModelType type, const juce::File& modelPath) {
    auto it = models_.find(type);
    if (it == models_.end()) {
        return false;
    }
    
    return it->second->loadModel(modelPath);
}

bool MultiModelProcessor::isModelLoaded(ModelType type) const {
    auto it = models_.find(type);
    if (it == models_.end()) {
        return false;
    }
    
    return it->second->isModelLoaded();
}

bool MultiModelProcessor::areAllModelsLoaded() const {
    for (const auto& [type, model] : models_) {
        if (!model->isModelLoaded()) {
            return false;
        }
    }
    return true;
}

InferenceResult MultiModelProcessor::runModel(ModelType type, const std::vector<float>& input) {
    if (!mlEnabled_) {
        // Return placeholder result when ML is disabled
        InferenceResult result;
        result.success = true;
        auto spec = ModelSpec::forType(type);
        result.output.resize(spec.outputSize, 0.0f);
        result.confidence = 0.0f;
        return result;
    }
    
    auto it = models_.find(type);
    if (it == models_.end() || !it->second->isModelLoaded()) {
        InferenceResult result;
        result.success = false;
        result.errorMessage = "Model not loaded";
        return result;
    }
    
    return it->second->infer(input);
}

InferenceResult MultiModelProcessor::recognizeEmotion(const std::vector<float>& audioFeatures) {
    return runModel(ModelType::EmotionRecognizer, audioFeatures);
}

InferenceResult MultiModelProcessor::generateMelody(const std::vector<float>& emotionEmbedding) {
    return runModel(ModelType::MelodyTransformer, emotionEmbedding);
}

InferenceResult MultiModelProcessor::predictHarmony(const std::vector<float>& context) {
    return runModel(ModelType::HarmonyPredictor, context);
}

InferenceResult MultiModelProcessor::generateDynamics(const std::vector<float>& intensity) {
    return runModel(ModelType::DynamicsEngine, intensity);
}

InferenceResult MultiModelProcessor::predictGroove(const std::vector<float>& arousal) {
    return runModel(ModelType::GroovePredictor, arousal);
}

juce::String MultiModelProcessor::getStatusSummary() const {
    juce::String summary;
    
    summary += "ML Mode: " + juce::String(mlEnabled_ ? "Enabled" : "Disabled") + "\n";
    summary += "ONNX Available: " + juce::String(ONNXInference::isONNXAvailable() ? "Yes" : "No") + "\n\n";
    summary += "Models:\n";
    
    const char* modelNames[] = {
        "EmotionRecognizer",
        "MelodyTransformer", 
        "HarmonyPredictor",
        "DynamicsEngine",
        "GroovePredictor"
    };
    
    ModelType types[] = {
        ModelType::EmotionRecognizer,
        ModelType::MelodyTransformer,
        ModelType::HarmonyPredictor,
        ModelType::DynamicsEngine,
        ModelType::GroovePredictor
    };
    
    for (int i = 0; i < 5; ++i) {
        bool loaded = isModelLoaded(types[i]);
        summary += "  " + juce::String(modelNames[i]) + ": " + 
                   (loaded ? "Loaded" : "Not loaded") + "\n";
    }
    
    return summary;
}

} // namespace ml
} // namespace midikompanion
