/**
 * @file NodeMLMapper.h
 * @brief ML-Node bridge for 216-node emotion thesaurus integration
 * 
 * Bridges ML embeddings (64-dim vectors) with the existing 216-node
 * emotion thesaurus structure. Supports:
 * - Mapping ML embeddings to nearest node by VAD distance
 * - Converting nodes to ML input format
 * - Getting node context for ML models
 * - Hybrid generation (ML + rule-based)
 */

#pragma once

#include <juce_core/juce_core.h>
#include <vector>
#include <map>
#include <optional>

namespace midikompanion {
namespace ml {

/**
 * @brief VAD (Valence-Arousal-Dominance) coordinates
 */
struct VADCoordinates {
    float valence = 0.0f;    // -1.0 to 1.0: negative to positive
    float arousal = 0.5f;    // 0.0 to 1.0: calm to excited
    float dominance = 0.5f;  // 0.0 to 1.0: submissive to dominant
    float intensity = 0.5f;  // 0.0 to 1.0: subtle to extreme
    
    float distanceTo(const VADCoordinates& other) const;
    static VADCoordinates lerp(const VADCoordinates& a, const VADCoordinates& b, float t);
};

/**
 * @brief Emotion node in the 216-node thesaurus
 */
struct EmotionNode {
    int id = 0;                          // 0-215 node ID
    juce::String name;                   // e.g., "joyful", "melancholic"
    juce::String category;               // Base emotion: happy, sad, angry, etc.
    juce::String subcategory;            // Sub-emotion
    VADCoordinates vad;                  // VAD coordinates
    std::vector<int> relatedEmotions;    // Related node IDs for context
    
    // Musical attributes
    juce::String mode = "major";         // major, minor, modal
    float tempoMultiplier = 1.0f;        // Relative tempo adjustment
    float dynamicsScale = 1.0f;          // Dynamics multiplier
    
    // ML-enhanced fields (optional)
    std::optional<std::vector<float>> mlEmbedding;  // 64-dim from EmotionRecognizer
    std::optional<float> mlConfidence;              // Model confidence score
    std::map<juce::String, float> mlFeatures;       // Additional ML-derived features
};

/**
 * @brief Node context for ML model input
 */
struct NodeContext {
    std::vector<float> embedding;        // Combined embedding for ML
    std::vector<int> relatedNodeIds;     // Context from related nodes
    float contextWeight = 1.0f;          // Weight for context influence
};

/**
 * @brief ML-Node Mapper
 * 
 * Bridges ML embeddings with the 216-node emotion thesaurus.
 * Supports hybrid mode: ML-enhanced node selection with rule-based fallback.
 */
class NodeMLMapper {
public:
    NodeMLMapper();
    ~NodeMLMapper() = default;
    
    /**
     * @brief Initialize the 216-node thesaurus
     * @param jsonPath Path to emotion thesaurus JSON file
     * @return true if successful
     */
    bool loadThesaurus(const juce::File& jsonPath);
    
    /**
     * @brief Load thesaurus from embedded data
     */
    void initializeDefaultThesaurus();
    
    /**
     * @brief Map ML embedding (64-dim) to nearest EmotionNode by VAD distance
     * @param embedding 64-dim ML embedding
     * @return Nearest emotion node
     */
    EmotionNode embeddingToNode(const std::vector<float>& embedding) const;
    
    /**
     * @brief Find nearest node by VAD coordinates
     * @param vad VAD coordinates
     * @return Nearest node ID
     */
    int findNearestNode(const VADCoordinates& vad) const;
    
    /**
     * @brief Get node by ID
     * @param nodeId Node ID (0-215)
     * @return Node or nullptr if not found
     */
    const EmotionNode* getNode(int nodeId) const;
    
    /**
     * @brief Get all nodes in a category
     * @param category Base emotion category
     * @return Vector of matching nodes
     */
    std::vector<const EmotionNode*> getNodesInCategory(const juce::String& category) const;
    
    /**
     * @brief Get node context for ML models
     * @param nodeId Node ID
     * @return Context with related nodes and combined embedding
     */
    NodeContext getNodeContext(int nodeId) const;
    
    /**
     * @brief Convert node to ML input format
     * @param node Emotion node
     * @return ML-compatible input vector
     */
    std::vector<float> nodeToMLInput(const EmotionNode& node) const;
    
    /**
     * @brief Convert VAD to ML embedding space
     * @param vad VAD coordinates
     * @return 64-dim embedding
     */
    std::vector<float> vadToEmbedding(const VADCoordinates& vad) const;
    
    /**
     * @brief Convert ML embedding to VAD
     * @param embedding 64-dim embedding
     * @return Estimated VAD coordinates
     */
    VADCoordinates embeddingToVAD(const std::vector<float>& embedding) const;
    
    /**
     * @brief Attach ML embedding to node
     * @param nodeId Node ID
     * @param embedding 64-dim ML embedding
     * @param confidence Model confidence
     */
    void attachMLEmbedding(int nodeId, const std::vector<float>& embedding, float confidence);
    
    /**
     * @brief Get interpolated node between two nodes
     * @param nodeA First node ID
     * @param nodeB Second node ID
     * @param t Interpolation factor (0.0 = nodeA, 1.0 = nodeB)
     * @return Interpolated node
     */
    EmotionNode interpolateNodes(int nodeA, int nodeB, float t) const;
    
    /**
     * @brief Get transition path between nodes
     * @param fromNode Starting node ID
     * @param toNode Ending node ID
     * @param steps Number of steps
     * @return Vector of intermediate nodes
     */
    std::vector<EmotionNode> getTransitionPath(int fromNode, int toNode, int steps) const;
    
    /**
     * @brief Get total number of nodes
     */
    int getNodeCount() const { return static_cast<int>(nodes_.size()); }
    
    /**
     * @brief Get all category names
     */
    juce::StringArray getCategories() const;
    
    /**
     * @brief Check if thesaurus is loaded
     */
    bool isLoaded() const { return !nodes_.empty(); }

private:
    std::vector<EmotionNode> nodes_;
    std::map<juce::String, std::vector<int>> categoryIndex_;
    
    void buildCategoryIndex();
    float calculateNodeDistance(const EmotionNode& a, const EmotionNode& b) const;
    
    // VAD to embedding mapping matrices (learned or predefined)
    std::vector<std::vector<float>> vadToEmbeddingMatrix_;
    std::vector<std::vector<float>> embeddingToVADMatrix_;
    
    void initializeMappingMatrices();
};

/**
 * @brief Hybrid generator that combines ML and rule-based approaches
 */
class HybridGenerator {
public:
    HybridGenerator();
    ~HybridGenerator() = default;
    
    /**
     * @brief Set the node mapper
     */
    void setNodeMapper(NodeMLMapper* mapper) { nodeMapper_ = mapper; }
    
    /**
     * @brief Generate with ML enhancement
     * @param node Current emotion node
     * @param mlEmbedding ML embedding for enhancement
     * @return Enhanced generation parameters
     */
    struct GenerationParams {
        float tempo = 120.0f;
        juce::String mode = "major";
        float dynamics = 0.7f;
        float articulation = 0.5f;
        std::vector<int> suggestedNotes;
        std::vector<float> noteWeights;
    };
    
    GenerationParams generate(const EmotionNode& node, 
                               const std::vector<float>& mlEmbedding = {}) const;
    
    /**
     * @brief Generate with rule-based approach only
     */
    GenerationParams generateRuleBased(const EmotionNode& node) const;
    
    /**
     * @brief Set ML influence weight (0.0 = pure rule-based, 1.0 = pure ML)
     */
    void setMLWeight(float weight) { mlWeight_ = juce::jlimit(0.0f, 1.0f, weight); }
    float getMLWeight() const { return mlWeight_; }

private:
    NodeMLMapper* nodeMapper_ = nullptr;
    float mlWeight_ = 0.5f;  // Balance between ML and rule-based
    
    GenerationParams applyRules(const EmotionNode& node) const;
    GenerationParams applyML(const std::vector<float>& embedding) const;
    GenerationParams blend(const GenerationParams& ruleBased, 
                           const GenerationParams& mlBased, 
                           float weight) const;
};

} // namespace ml
} // namespace midikompanion
