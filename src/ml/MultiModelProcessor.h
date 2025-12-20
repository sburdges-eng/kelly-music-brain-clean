/**
 * @file MultiModelProcessor.h
 * @brief Orchestrates the 5-model ML pipeline for emotion-to-music generation
 * 
 * Pipeline:
 * 1. EmotionRecognizer: Audio/VAD → 64-dim emotion embedding
 * 2. MelodyTransformer: Emotion embedding → MIDI note probabilities
 * 3. HarmonyPredictor: Context → Chord probabilities
 * 4. DynamicsEngine: Intensity → Expression parameters
 * 5. GroovePredictor: Arousal → Groove parameters
 */

#pragma once

#include "ONNXInference.h"
#include "NodeMLMapper.h"
#include <juce_core/juce_core.h>
#include <memory>
#include <vector>
#include <optional>

namespace midikompanion::ml {

/**
 * @brief Generated note with all parameters
 */
struct GeneratedNote {
    int pitch{60};
    int velocity{100};
    double startBeat{0.0};
    double durationBeats{0.25};
    int channel{1};
    
    // Expression data
    float expressionValue{0.5f};
    float modulation{0.0f};
    
    // Groove adjustments
    float timingOffset{0.0f};    // Swing/groove timing
    float velocityVariation{0.0f};
};

/**
 * @brief Generated chord
 */
struct GeneratedChord {
    std::vector<int> pitches;
    juce::String chordName;
    double startBeat{0.0};
    double durationBeats{1.0};
    float voicingSpread{0.5f};
};

/**
 * @brief Complete generation result
 */
struct GenerationResult {
    bool success{false};
    juce::String errorMessage;
    
    // Generated content
    std::vector<GeneratedNote> melody;
    std::vector<GeneratedNote> bassline;
    std::vector<GeneratedChord> chords;
    std::vector<GeneratedNote> drums;
    
    // Expression parameters
    std::vector<float> dynamics;      // Per-beat dynamics
    std::vector<float> expression;    // Expression curve
    
    // Groove parameters
    float swing{0.0f};               // 0.0 = straight, 1.0 = full swing
    float humanize{0.0f};            // Timing variation
    float grooveIntensity{0.5f};     // Overall groove strength
    
    // Source information
    int sourceNodeId{-1};
    float mlConfidence{0.0f};
    double processingTimeMs{0.0};
};

/**
 * @brief Generation configuration
 */
struct GenerationConfig {
    // Content settings
    int numBars{4};
    int beatsPerBar{4};
    double tempo{120.0};
    
    // Generation options
    bool generateMelody{true};
    bool generateBass{true};
    bool generateChords{true};
    bool generateDrums{false};
    
    // ML options
    bool useMLEnhancement{true};
    float mlBlendRatio{0.7f};        // 0 = rule-based only, 1 = ML only
    float temperature{0.8f};          // Generation randomness
    
    // Musical constraints
    int keyRoot{0};                   // 0 = C, 1 = C#, etc.
    juce::String scale{"major"};
    int lowestNote{36};               // Lowest allowed MIDI note
    int highestNote{84};              // Highest allowed MIDI note
};

/**
 * @brief Multi-model processor for the 5-model emotion-to-music pipeline
 * 
 * Coordinates all 5 models and integrates with the 216-node emotion thesaurus.
 * Supports both ML-enhanced and rule-based generation with smooth blending.
 */
class MultiModelProcessor {
public:
    MultiModelProcessor();
    ~MultiModelProcessor() = default;
    
    //==========================================================================
    // Initialization
    //==========================================================================
    
    /**
     * @brief Initialize the processor
     * @return True if all components initialized successfully
     */
    bool initialize();
    
    /**
     * @brief Load all ML models from directory
     * @param modelsDir Directory containing ONNX models
     * @return True if all models loaded (or stub mode enabled)
     */
    bool loadModels(const juce::File& modelsDir);
    
    /**
     * @brief Check if processor is ready for generation
     */
    bool isReady() const { return initialized_; }
    
    /**
     * @brief Check if ML models are loaded
     */
    bool hasMLModels() const { return mlModelsLoaded_; }
    
    //==========================================================================
    // Generation - Main Interface
    //==========================================================================
    
    /**
     * @brief Generate music from emotion node
     * @param nodeId Emotion node ID (0-215)
     * @param config Generation configuration
     * @return Generation result with all tracks
     */
    GenerationResult generateFromNode(int nodeId, const GenerationConfig& config);
    
    /**
     * @brief Generate music from VAD coordinates
     * @param vad VAD coordinates
     * @param config Generation configuration
     * @return Generation result
     */
    GenerationResult generateFromVAD(const VADCoordinates& vad, const GenerationConfig& config);
    
    /**
     * @brief Generate music from ML embedding
     * @param embedding 64-dim emotion embedding
     * @param config Generation configuration
     * @return Generation result
     */
    GenerationResult generateFromEmbedding(const std::vector<float>& embedding, 
                                           const GenerationConfig& config);
    
    /**
     * @brief Generate music from audio features
     * @param audioFeatures 128-dim audio feature vector
     * @param config Generation configuration
     * @return Generation result
     */
    GenerationResult generateFromAudio(const std::vector<float>& audioFeatures,
                                       const GenerationConfig& config);
    
    //==========================================================================
    // Node Mapper Access
    //==========================================================================
    
    /**
     * @brief Get the node mapper
     */
    NodeMLMapper& getNodeMapper() { return nodeMapper_; }
    const NodeMLMapper& getNodeMapper() const { return nodeMapper_; }
    
    //==========================================================================
    // Mode Control
    //==========================================================================
    
    /**
     * @brief Enable/disable ML enhancement
     */
    void setMLEnabled(bool enabled) { mlEnabled_ = enabled; }
    bool isMLEnabled() const { return mlEnabled_; }
    
    /**
     * @brief Set ML/rule-based blend ratio
     * @param ratio 0.0 = rule-based only, 1.0 = ML only
     */
    void setMLBlendRatio(float ratio) { mlBlendRatio_ = std::clamp(ratio, 0.0f, 1.0f); }
    float getMLBlendRatio() const { return mlBlendRatio_; }
    
    //==========================================================================
    // Model Status
    //==========================================================================
    
    struct ModelStatus {
        bool emotionRecognizer{false};
        bool melodyTransformer{false};
        bool harmonyPredictor{false};
        bool dynamicsEngine{false};
        bool groovePredictor{false};
    };
    
    /**
     * @brief Get status of all models
     */
    ModelStatus getModelStatus() const;
    
private:
    //==========================================================================
    // Pipeline Stages
    //==========================================================================
    
    /**
     * @brief Stage 1: Get emotion embedding
     */
    std::vector<float> runEmotionRecognizer(const std::vector<float>& audioFeatures);
    
    /**
     * @brief Stage 2: Generate melody
     */
    std::vector<GeneratedNote> runMelodyTransformer(const std::vector<float>& embedding,
                                                    const GenerationConfig& config);
    
    /**
     * @brief Stage 3: Generate harmony
     */
    std::vector<GeneratedChord> runHarmonyPredictor(const std::vector<float>& context,
                                                    const GenerationConfig& config);
    
    /**
     * @brief Stage 4: Apply dynamics
     */
    std::vector<float> runDynamicsEngine(const std::vector<float>& intensityFeatures);
    
    /**
     * @brief Stage 5: Apply groove
     */
    void applyGroove(GenerationResult& result, const std::vector<float>& grooveParams);
    
    //==========================================================================
    // Rule-Based Fallbacks
    //==========================================================================
    
    std::vector<GeneratedNote> generateMelodyRuleBased(const EmotionNodeML& node,
                                                       const GenerationConfig& config);
    
    std::vector<GeneratedNote> generateBassRuleBased(const EmotionNodeML& node,
                                                     const GenerationConfig& config,
                                                     const std::vector<GeneratedChord>& chords);
    
    std::vector<GeneratedChord> generateChordsRuleBased(const EmotionNodeML& node,
                                                        const GenerationConfig& config);
    
    std::vector<float> generateDynamicsRuleBased(const EmotionNodeML& node,
                                                 const GenerationConfig& config);
    
    //==========================================================================
    // Utility
    //==========================================================================
    
    /**
     * @brief Blend ML and rule-based outputs
     */
    template<typename T>
    std::vector<T> blendOutputs(const std::vector<T>& mlOutput,
                                const std::vector<T>& ruleOutput,
                                float blendRatio);
    
    /**
     * @brief Apply musical constraints to notes
     */
    void applyConstraints(std::vector<GeneratedNote>& notes, const GenerationConfig& config);
    
    /**
     * @brief Get scale notes for key
     */
    std::vector<int> getScaleNotes(int keyRoot, const juce::String& scale);
    
    /**
     * @brief Quantize note to scale
     */
    int quantizeToScale(int pitch, const std::vector<int>& scaleNotes);
    
    //==========================================================================
    // Member Variables
    //==========================================================================
    
    // Models
    std::unique_ptr<ONNXInference> emotionRecognizer_;
    std::unique_ptr<ONNXInference> melodyTransformer_;
    std::unique_ptr<ONNXInference> harmonyPredictor_;
    std::unique_ptr<ONNXInference> dynamicsEngine_;
    std::unique_ptr<ONNXInference> groovePredictor_;
    
    // Node mapper
    NodeMLMapper nodeMapper_;
    
    // State
    bool initialized_{false};
    bool mlModelsLoaded_{false};
    bool mlEnabled_{true};
    float mlBlendRatio_{0.7f};
    
    // Random generator for variations
    juce::Random random_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MultiModelProcessor)
};

} // namespace midikompanion::ml
