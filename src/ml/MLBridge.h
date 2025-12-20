#pragma once
/*
 * MLBridge.h - Connect ML Models to IntentPipeline
 * =================================================
 * Bridges the gap between:
 *   - MultiModelProcessor (neural network inference) - existing Kelly::ML::
 * namespace
 *   - IntentPipeline (wound → musical parameters) - via KellyBrain
 *
 * Flow:
 *   Audio → FeatureExtractor → EmotionRecognizer → EmotionNode
 *   EmotionNode → IntentPipeline → IntentResult
 *   IntentResult + MelodyTransformer → GeneratedMidi
 */

#include "common/KellyTypes.h"
#include "engine/KellyBrain.h"
#include "ml/MLFeatureExtractor.h"
#include "ml/MultiModelProcessor.h"
#include <array>
#include <memory>

namespace kelly {

// Forward declarations
// Note: MultiModelProcessor is in Kelly::ML:: namespace
// We use type aliases for convenience
using MLMultiModelProcessor = Kelly::ML::MultiModelProcessor;
using MLModelType = Kelly::ML::ModelType;
using MLInferenceResult = Kelly::ML::InferenceResult;
using MLAsyncPipeline = Kelly::ML::AsyncMLPipeline;

// =============================================================================
// ML → Emotion Mapping
// =============================================================================

class EmotionEmbeddingMapper {
public:
  EmotionEmbeddingMapper();
  explicit EmotionEmbeddingMapper(EmotionThesaurus *thesaurus);

  // Convert 64-dim embedding to EmotionNode (KellyTypes::EmotionNode)
  EmotionNode embeddingToEmotion(const std::array<float, 64> &embedding);

  // Convert EmotionNode to 64-dim embedding (for training)
  std::array<float, 64> emotionToEmbedding(const EmotionNode &node);

  // Batch conversion
  std::vector<EmotionNode>
  embeddingsToEmotions(const std::vector<std::array<float, 64>> &embeddings);

  // Interpretation helpers
  struct EmbeddingInterpretation {
    float valence;    // Derived from dims 0-15
    float arousal;    // Derived from dims 16-31
    float dominance;  // Derived from dims 32-47
    float complexity; // Derived from dims 48-63
    EmotionCategory primaryCategory;
    float confidence;
  };

  EmbeddingInterpretation interpret(const std::array<float, 64> &embedding);

private:
  EmotionThesaurus *thesaurus_;

  // Learned mapping weights (can be loaded from file)
  std::array<std::array<float, 64>, 8> categoryWeights_; // 8 categories
  void initializeDefaultWeights();
};

// =============================================================================
// ML-Enhanced Intent Pipeline
// =============================================================================

class MLIntentPipeline {
public:
  MLIntentPipeline();
  ~MLIntentPipeline() = default;

  // Initialize with model directory
  bool initialize(const std::string &modelsPath, const std::string &dataPath);

  // === Audio-Driven Generation ===

  // Process audio buffer → full pipeline
  // Note: Returns KellyTypes::IntentResult (KellyTypes.h is included above)
  IntentResult processAudio(const float *audioData, size_t numSamples);

  // Async version for real-time use
  void submitAudio(const float *audioData, size_t numSamples);
  bool hasResult() const;
  IntentResult getResult();

  // === Hybrid: Audio + Text ===

  // Combine audio emotion with text description
  IntentResult processHybrid(const float *audioData, size_t numSamples,
                             const std::string &textDescription);

  // === Generate from ML Outputs ===

  // Use MelodyTransformer output directly
  std::vector<MidiNote> generateMelodyFromProbabilities(
      const std::array<float, 128> &noteProbabilities,
      const IntentResult &intent, int bars = 4);

  // Use HarmonyPredictor output
  std::vector<Chord>
  generateHarmonyFromPrediction(const std::array<float, 64> &harmonyPrediction,
                                const IntentResult &intent);

  // Use DynamicsEngine output
  void applyDynamics(std::vector<MidiNote> &notes,
                     const std::array<float, 16> &dynamicsOutput);

  // Use GroovePredictor output
  void applyGroove(std::vector<MidiNote> &notes,
                   const std::array<float, 32> &grooveParams);

  // === Full Generation ===

  GeneratedMidi generateFromAudio(const float *audioData, size_t numSamples,
                                  int bars = 8);

  // Access components
  KellyBrain &brain() { return *brain_; }
  Kelly::ML::MultiModelProcessor &mlProcessor() { return *mlProcessor_; }
  EmotionEmbeddingMapper &mapper() { return *mapper_; }

  // Enable/disable ML models
  void setMLEnabled(bool enabled) { mlEnabled_ = enabled; }
  bool isMLEnabled() const { return mlEnabled_; }

  // Per-model control
  void setModelEnabled(Kelly::ML::ModelType type, bool enabled);

private:
  std::unique_ptr<KellyBrain> brain_;
  std::unique_ptr<Kelly::ML::MultiModelProcessor> mlProcessor_;
  std::unique_ptr<MLFeatureExtractor> featureExtractor_;
  std::unique_ptr<EmotionEmbeddingMapper> mapper_;

  bool mlEnabled_ = true;
  bool initialized_ = false;

  // Feature extraction settings
  double sampleRate_ = 44100.0;
  size_t fftSize_ = 2048UL;
};

// =============================================================================
// Convenience Functions
// =============================================================================

// One-shot audio → MIDI
inline GeneratedMidi audioToMidi(const float *audioData, size_t numSamples,
                                 const std::string &modelsPath = "./models",
                                 int bars = 8) {
  MLIntentPipeline pipeline;
  pipeline.initialize(modelsPath, "./data");
  return pipeline.generateFromAudio(audioData, numSamples, bars);
}

// Create ML-enhanced pipeline
inline std::unique_ptr<MLIntentPipeline> createMLPipeline() {
  return std::make_unique<MLIntentPipeline>();
}

} // namespace kelly
