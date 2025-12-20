#include "MultiModelProcessor.h"
#include <cmath>
#include <fstream>

namespace Kelly {
namespace ML {

// ============================================================================
// ModelWrapper Implementation
// ============================================================================

ModelWrapper::ModelWrapper(ModelType type)
    : type_(type), spec_(MODEL_SPECS[static_cast<size_t>(type)]) {
    output_.resize(spec_.outputSize, 0.0f);
}

bool ModelWrapper::loadWeights(const juce::File& file) {
    if (!file.existsAsFile()) {
        juce::Logger::writeToLog(juce::String("Model not found: ") + spec_.name +
                                 " (will use fallback heuristics)");
        loaded_ = true;  // Mark as loaded to enable fallback mode
        return false;
    }

#ifdef ENABLE_RTNEURAL
    try {
        // Convert JUCE File to std::string for ifstream
        std::string filePath = file.getFullPathName().toStdString();
        std::ifstream jsonStream(filePath, std::ifstream::binary);

        if (!jsonStream.is_open()) {
            juce::Logger::writeToLog(juce::String("Failed to open model file: ") +
                                     file.getFullPathName());
            loaded_ = true;  // Use fallback
            return false;
        }

        // Parse JSON using RTNeural's JSON parser
        auto model = RTNeural::json_parser::parseJson<float>(jsonStream);

        if (!model) {
            juce::Logger::writeToLog(juce::String("Failed to parse RTNeural model JSON: ") +
                                     file.getFullPathName());
            jsonStream.close();
            loaded_ = true;  // Use fallback
            return false;
        }

        // Reset model state (important for LSTM layers)
        model->reset();

        // Store the model
        rtModel_ = std::move(model);
        loaded_ = true;

        juce::Logger::writeToLog(juce::String("Loaded model: ") + spec_.name +
                                 " (" + juce::String(spec_.estimatedParams) + " params)");
        jsonStream.close();
        return true;
    }
    catch (const std::exception& e) {
        juce::Logger::writeToLog(juce::String("Exception loading model ") + spec_.name +
                                 ": " + juce::String(e.what()));
        loaded_ = true;  // Use fallback
        return false;
    }
    catch (...) {
        juce::Logger::writeToLog(juce::String("Unknown exception loading model: ") + spec_.name);
        loaded_ = true;  // Use fallback
        return false;
    }
#else
    loaded_ = true;
    juce::Logger::writeToLog(juce::String("Loaded model (fallback mode): ") + spec_.name);
    return true;
#endif
}

std::vector<float> ModelWrapper::forward(const float* input, size_t inputSize) {
    if (!loaded_ || !enabled_) {
        return output_;
    }

    if (inputSize != spec_.inputSize) {
        juce::Logger::writeToLog(juce::String("ModelWrapper: Input size mismatch. Expected ") +
                                 juce::String(spec_.inputSize) + ", got " + juce::String(inputSize));
        return output_;
    }

#ifdef ENABLE_RTNEURAL
    if (rtModel_) {
        try {
            // Run RTNeural inference
            // RTNeural's forward() takes input pointer and writes to output
            rtModel_->forward(input);

            // Get output from model
            // Note: RTNeural API may vary - check if getOutputs() exists or use copyOutput()
            const float* modelOutput = rtModel_->getOutputs();
            if (modelOutput) {
                size_t copySize = std::min(spec_.outputSize, output_.size());
                std::memcpy(output_.data(), modelOutput, copySize * sizeof(float));
            } else {
                // Fallback if getOutputs() not available
                computeFallback(input);
            }
            return output_;
        }
        catch (const std::exception& e) {
            juce::Logger::writeToLog(juce::String("Exception during inference: ") +
                                     juce::String(e.what()));
            computeFallback(input);
            return output_;
        }
    }
#endif

    // Fallback heuristic
    computeFallback(input);
    return output_;
}

void ModelWrapper::computeFallback(const float* input) {
    switch (type_) {
        case ModelType::EmotionRecognizer:
            // Extract valence (first 32) and arousal (last 32) from mel features
            for (size_t i = 0; i < 32 && i < output_.size(); ++i) {
                output_[i] = std::tanh(input[i] * 0.3f);  // Valence
                if (i + 32 < output_.size()) {
                    output_[i + 32] = std::tanh(input[i + 64] * 0.5f);  // Arousal
                }
            }
            break;

        case ModelType::MelodyTransformer:
            // Generate note probabilities based on emotion
            for (size_t i = 0; i < output_.size(); ++i) {
                float note = static_cast<float>(i % 12) / 12.0f;
                float emotionInfluence = (i < 64) ? input[i % 64] : 0.0f;
                output_[i] = 1.0f / (1.0f + std::exp(-(note * 2.0f - 1.0f + emotionInfluence * 0.5f)));
            }
            break;

        case ModelType::HarmonyPredictor:
            // Predict chord probabilities from context
            for (size_t i = 0; i < output_.size(); ++i) {
                output_[i] = std::tanh(input[i] * 0.4f);
            }
            break;

        case ModelType::DynamicsEngine:
            // Generate velocity/timing/expression parameters
            for (size_t i = 0; i < output_.size(); ++i) {
                output_[i] = 0.5f + input[i % 32] * 0.3f;
                output_[i] = std::max(0.0f, std::min(1.0f, output_[i]));  // Clamp to [0,1]
            }
            break;

        case ModelType::GroovePredictor:
            // Generate groove parameters from emotion
            for (size_t i = 0; i < output_.size(); ++i) {
                output_[i] = std::tanh(input[i % 64] * 0.6f);
            }
            break;

        default:
            break;
    }
}

// ============================================================================
// MultiModelProcessor Implementation
// ============================================================================

MultiModelProcessor::MultiModelProcessor() {
    for (size_t i = 0; i < static_cast<size_t>(ModelType::COUNT); ++i) {
        models_[i] = std::make_unique<ModelWrapper>(static_cast<ModelType>(i));
    }
}

bool MultiModelProcessor::initialize(const juce::File& modelsDir) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!modelsDir.isDirectory()) {
        juce::Logger::writeToLog("Models directory not found: " + modelsDir.getFullPathName());
        juce::Logger::writeToLog("Will use fallback heuristics for all models");
    }

    bool anyLoaded = false;

    for (size_t i = 0; i < models_.size(); ++i) {
        juce::String filename = juce::String(MODEL_SPECS[i].name).toLowerCase() + ".json";
        juce::File modelFile = modelsDir.getChildFile(filename);

        if (models_[i]->loadWeights(modelFile)) {
            anyLoaded = true;
        }
    }

    initialized_ = true;
    logStats();

    return initialized_;
}

void MultiModelProcessor::setModelEnabled(ModelType type, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    models_[static_cast<size_t>(type)]->setEnabled(enabled);
}

bool MultiModelProcessor::isModelEnabled(ModelType type) const {
    return models_[static_cast<size_t>(type)]->isEnabled();
}

std::vector<float> MultiModelProcessor::infer(ModelType type, const std::vector<float>& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    return models_[static_cast<size_t>(type)]->forward(input.data(), input.size());
}

InferenceResult MultiModelProcessor::runFullPipeline(const std::array<float, 128>& audioFeatures) {
    std::lock_guard<std::mutex> lock(mutex_);

    InferenceResult result;

    // 1. EmotionRecognizer: audio → emotion
    if (models_[0]->isEnabled()) {
        auto emotion = models_[0]->forward(audioFeatures.data(), 128);
        std::copy_n(emotion.begin(), 64, result.emotionEmbedding.begin());
    }

    // 2. MelodyTransformer: emotion → melody
    if (models_[1]->isEnabled()) {
        auto melody = models_[1]->forward(result.emotionEmbedding.data(), 64);
        std::copy_n(melody.begin(), 128, result.melodyProbabilities.begin());
    }

    // 3. HarmonyPredictor: context (emotion + audio) → harmony
    if (models_[2]->isEnabled()) {
        std::array<float, 128> context{};
        std::copy_n(result.emotionEmbedding.begin(), 64, context.begin());
        std::copy_n(audioFeatures.begin(), 64, context.begin() + 64);
        auto harmony = models_[2]->forward(context.data(), 128);
        std::copy_n(harmony.begin(), 64, result.harmonyPrediction.begin());
    }

    // 4. DynamicsEngine: compact emotion → dynamics
    if (models_[3]->isEnabled()) {
        std::array<float, 32> compact{};
        std::copy_n(result.emotionEmbedding.begin(), 32, compact.begin());
        auto dynamics = models_[3]->forward(compact.data(), 32);
        std::copy_n(dynamics.begin(), 16, result.dynamicsOutput.begin());
    }

    // 5. GroovePredictor: emotion → groove
    if (models_[4]->isEnabled()) {
        auto groove = models_[4]->forward(result.emotionEmbedding.data(), 64);
        std::copy_n(groove.begin(), 32, result.grooveParameters.begin());
    }

    result.valid = true;
    return result;
}

size_t MultiModelProcessor::getTotalParams() const {
    size_t total = 0;
    for (const auto& spec : MODEL_SPECS) {
        total += spec.estimatedParams;
    }
    return total;
}

size_t MultiModelProcessor::getTotalMemoryKB() const {
    return getTotalParams() * sizeof(float) / 1024;
}

void MultiModelProcessor::logStats() const {
    juce::Logger::writeToLog("============================================================");
    juce::Logger::writeToLog("MultiModelProcessor initialized:");
    juce::Logger::writeToLog("  Total params: " + juce::String(getTotalParams()));
    juce::Logger::writeToLog("  Total memory: " + juce::String(getTotalMemoryKB()) + " KB");
    juce::Logger::writeToLog("  Estimated inference: <10ms");
    juce::Logger::writeToLog("============================================================");
}

// ============================================================================
// AsyncMLPipeline Implementation
// ============================================================================

class InferenceThread : public juce::Thread {
public:
    explicit InferenceThread(AsyncMLPipeline& pipeline)
        : juce::Thread("Kelly ML Inference"), pipeline_(pipeline) {}

    void run() override {
        pipeline_.run();
    }

private:
    AsyncMLPipeline& pipeline_;
};

AsyncMLPipeline::AsyncMLPipeline(MultiModelProcessor& processor)
    : processor_(processor) {}

AsyncMLPipeline::~AsyncMLPipeline() {
    stop();
}

void AsyncMLPipeline::start() {
    if (running_) return;

    running_ = true;
    thread_ = std::make_unique<InferenceThread>(*this);
    static_cast<InferenceThread*>(thread_.get())->startThread();
}

void AsyncMLPipeline::stop() {
    running_ = false;

    if (thread_) {
        static_cast<InferenceThread*>(thread_.get())->stopThread(1000);
        thread_.reset();
    }
}

void AsyncMLPipeline::submitFeatures(const std::array<float, 128>& features) {
    if (!hasRequest_.load(std::memory_order_acquire)) {
        pendingFeatures_ = features;
        hasRequest_.store(true, std::memory_order_release);
    }
}

bool AsyncMLPipeline::hasResult() const {
    return hasResult_.load(std::memory_order_acquire);
}

InferenceResult AsyncMLPipeline::getResult() {
    hasResult_.store(false, std::memory_order_release);
    return latestResult_;
}

void AsyncMLPipeline::run() {
    while (running_) {
        if (hasRequest_.load(std::memory_order_acquire)) {
            hasRequest_.store(false, std::memory_order_release);

            // Run inference on background thread
            latestResult_ = processor_.runFullPipeline(pendingFeatures_);

            hasResult_.store(true, std::memory_order_release);
        } else {
            juce::Thread::sleep(1);
        }
    }
}

} // namespace ML
} // namespace Kelly
