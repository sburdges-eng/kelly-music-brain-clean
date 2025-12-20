#pragma once
/*
 * NodeMLMapper.h - Bridge Between ML Embeddings and 216-Node Emotion Thesaurus
 * ============================================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - ML Layer: Uses ONNXInference for ML model inference
 * - Engine Layer: Uses EmotionThesaurus for 216-node structure
 * - Type System: EmotionNode (with ML embeddings from KellyTypes.h)
 *
 * Purpose: Bridges ML embeddings (64-dim vectors) with the 216-node emotion thesaurus.
 *          Maps ML model outputs to emotion nodes and vice versa.
 *
 * Features:
 * - Map 64-dim ML embeddings to nearest EmotionNode by VAD distance
 * - Convert EmotionNode to ML input format
 * - Get node context (related nodes + ML features) for ML models
 * - Support hybrid mode: ML-enhanced node selection with rule-based fallback
 */

#include "common/Types.h"  // EmotionNode, EmotionThesaurus
#include "engine/EmotionThesaurus.h"  // For node lookup
#include <vector>
#include <optional>
#include <map>

namespace midikompanion {
namespace ml {

/**
 * Node-ML Mapper - Bridges ML embeddings and 216-node emotion thesaurus.
 *
 * Architecture:
 * - ML models output 64-dim embeddings
 * - Mapper finds nearest EmotionNode by VAD distance
 * - Node structure provides musical attributes and relationships
 * - Hybrid generation: ML suggests, node structure validates
 */
class NodeMLMapper {
public:
    /**
     * Constructor - Requires reference to EmotionThesaurus for node lookup.
     *
     * @param thesaurus Reference to 216-node emotion thesaurus (non-owning)
     */
    explicit NodeMLMapper(const kelly::EmotionThesaurus& thesaurus);

    /**
     * Map ML embedding (64-dim) to nearest EmotionNode by VAD distance.
     * Uses the embedding to find the closest emotion node in VAD space.
     *
     * @param embedding 64-dimensional ML embedding from EmotionRecognizer
     * @return Nearest EmotionNode with ML embedding attached, or nullopt if not found
     */
    std::optional<EmotionNode> embeddingToNode(const std::vector<float>& embedding) const;

    /**
     * Get node context (related nodes + ML features) for ML models.
     * Combines the node's VAD coordinates, musical attributes, and related nodes
     * into a feature vector suitable for ML model input.
     *
     * @param nodeId Emotion node ID (0-215)
     * @return Feature vector for ML models (size depends on model requirements)
     */
    std::vector<float> getNodeContext(int nodeId) const;

    /**
     * Convert EmotionNode to ML input format.
     * Extracts VAD coordinates and musical attributes into ML-compatible format.
     *
     * @param node EmotionNode to convert
     * @return ML input vector (size depends on model requirements)
     */
    std::vector<float> nodeToMLInput(const EmotionNode& node) const;

    /**
     * Find nearest node by VAD coordinates.
     * Helper method for mapping ML embeddings to nodes.
     *
     * @param valence Valence value (-1.0 to 1.0)
     * @param arousal Arousal value (0.0 to 1.0)
     * @param dominance Dominance value (0.0 to 1.0)
     * @return Nearest EmotionNode
     */
    EmotionNode findNearestNodeByVAD(float valence, float arousal, float dominance) const;

    /**
     * Calculate VAD distance between two nodes.
     *
     * @param node1 First emotion node
     * @param node2 Second emotion node
     * @return Euclidean distance in VAD space
     */
    float calculateVADDistance(const EmotionNode& node1, const EmotionNode& node2) const;

    /**
     * Calculate distance from VAD coordinates to node.
     *
     * @param valence Valence value
     * @param arousal Arousal value
     * @param dominance Dominance value
     * @param node Emotion node
     * @return Euclidean distance in VAD space
     */
    float calculateVADDistance(float valence, float arousal, float dominance, const EmotionNode& node) const;

    /**
     * Enhance node with ML embedding.
     * Attaches ML embedding to an existing EmotionNode.
     *
     * @param node EmotionNode to enhance
     * @param embedding 64-dim ML embedding
     * @param confidence Model confidence score (0.0-1.0)
     * @return Enhanced EmotionNode with ML data
     */
    EmotionNode enhanceNodeWithML(const EmotionNode& node,
                                   const std::vector<float>& embedding,
                                   float confidence = 1.0f) const;

private:
    // Non-owning reference to emotion thesaurus
    const kelly::EmotionThesaurus& thesaurus_;

    /**
     * Convert 64-dim ML embedding to VAD coordinates.
     * Uses a learned mapping or projection to convert embedding space to VAD space.
     * For now, uses a simple linear projection (can be improved with learned mapping).
     *
     * @param embedding 64-dim ML embedding
     * @return VAD coordinates (valence, arousal, dominance)
     */
    std::tuple<float, float, float> embeddingToVAD(const std::vector<float>& embedding) const;

    /**
     * Convert VAD coordinates to ML input format.
     * Creates a feature vector from VAD + musical attributes.
     *
     * @param valence Valence value
     * @param arousal Arousal value
     * @param dominance Dominance value
     * @param musicalAttrs Musical attributes
     * @return ML input vector
     */
    std::vector<float> vadToMLInput(float valence, float arousal, float dominance,
                                    const MusicalAttributes& musicalAttrs) const;
};

} // namespace ml
} // namespace midikompanion
