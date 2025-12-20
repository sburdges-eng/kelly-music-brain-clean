#pragma once
/*
 * Kelly MIDI Companion - Multi-Model ML Processor
 * ================================================
 * 5-Model Architecture (~1M params, ~4MB, <10ms inference):
 *   1. EmotionRecognizer:  128→512→256→128→64  (~500K)
 *   2. MelodyTransformer:  64→256→256→256→128  (~400K)
 *   3. HarmonyPredictor:   128→256→128→64      (~100K)
 *   4. DynamicsEngine:     32→128→64→16        (~20K)
 *   5. GroovePredictor:    64→128→64→32        (~25K)
 */

#include <juce_core/juce_core.h>
#include <array>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <cmath>

#ifdef ENABLE_RTNEURAL
#include <RTNeural/RTNeural.h>
#endif

namespace Kelly {
namespace ML {

// ============================================================================
// Model Type Enumeration
// ============================================================================
enum class ModelType : size_t {
    EmotionRecognizer = 0,  // Audio → Emotion
    MelodyTransformer = 1,  // Emotion → MIDI
    HarmonyPredictor  = 2,  // Context → Chords
    DynamicsEngine    = 3,  // Context → Expression
    GroovePredictor   = 4,  // Emotion → Groove
    COUNT             = 5
};

// ============================================================================
// Model Configuration
// ============================================================================
struct ModelSpec {
    const char* name;
    size_t inputSize;
    size_t outputSize;
    size_t estimatedParams;
};

constexpr std::array<ModelSpec, 5> MODEL_SPECS = {{
    {"EmotionRecognizer", 128, 64,  497664},
    {"MelodyTransformer", 64,  128, 412672},
    {"HarmonyPredictor",  128, 64,  74048},
    {"DynamicsEngine",    32,  16,  13456},
    {"GroovePredictor",   64,  32,  19040}
}};

// ============================================================================
// Inference Result Structure
// ============================================================================
struct InferenceResult {
    std::array<float, 64>  emotionEmbedding{};    // Model 0 output
    std::array<float, 128> melodyProbabilities{}; // Model 1 output
    std::array<float, 64>  harmonyPrediction{};   // Model 2 output
    std::array<float, 16>  dynamicsOutput{};      // Model 3 output
    std::array<float, 32>  grooveParameters{};    // Model 4 output
    bool valid = false;
};

// ============================================================================
// Single Model Wrapper (Heap Allocated)
// ============================================================================
class ModelWrapper {
public:
    explicit ModelWrapper(ModelType type);
    ~ModelWrapper() = default;

    bool loadWeights(const juce::File& file);
    std::vector<float> forward(const float* input, size_t inputSize);

    bool isLoaded() const { return loaded_; }
    void setEnabled(bool e) { enabled_ = e; }
    bool isEnabled() const { return enabled_; }
    const ModelSpec& spec() const { return spec_; }

private:
    void computeFallback(const float* input);

    ModelType type_;
    ModelSpec spec_;
    std::vector<float> output_;
    bool loaded_ = false;
    bool enabled_ = true;

#ifdef ENABLE_RTNEURAL
    std::unique_ptr<RTNeural::Model<float>> rtModel_;
#endif
};

// ============================================================================
// Multi-Model Manager
// ============================================================================
class MultiModelProcessor {
public:
    MultiModelProcessor();
    ~MultiModelProcessor() = default;

    bool initialize(const juce::File& modelsDir);

    void setModelEnabled(ModelType type, bool enabled);
    bool isModelEnabled(ModelType type) const;

    // Run single model inference
    std::vector<float> infer(ModelType type, const std::vector<float>& input);

    // Run full pipeline: audio features → all outputs
    InferenceResult runFullPipeline(const std::array<float, 128>& audioFeatures);

    size_t getTotalParams() const;
    size_t getTotalMemoryKB() const;

    bool isInitialized() const { return initialized_; }

private:
    void logStats() const;

    std::array<std::unique_ptr<ModelWrapper>, 5> models_;
    std::mutex mutex_;
    bool initialized_ = false;
};

// ============================================================================
// Async Inference (Lock-Free for Audio Thread)
// ============================================================================
class InferenceThread;  // Forward declaration

class AsyncMLPipeline {
public:
    explicit AsyncMLPipeline(MultiModelProcessor& processor);
    ~AsyncMLPipeline();

    void start();
    void stop();

    // Non-blocking submit (audio thread safe)
    void submitFeatures(const std::array<float, 128>& features);

    // Non-blocking result check (audio thread safe)
    bool hasResult() const;
    InferenceResult getResult();

private:
    friend class InferenceThread;  // Allow InferenceThread to call run()
    void run();

    MultiModelProcessor& processor_;
    std::unique_ptr<juce::Thread> thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> hasRequest_{false};
    std::atomic<bool> hasResult_{false};
    std::array<float, 128> pendingFeatures_{};
    InferenceResult latestResult_;
};

} // namespace ML
} // namespace Kelly
