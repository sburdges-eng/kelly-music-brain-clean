#include "ml/NodeMLMapper.h"
#include "common/Types.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace midikompanion {
namespace ml {

NodeMLMapper::NodeMLMapper(const kelly::EmotionThesaurus& thesaurus)
    : thesaurus_(thesaurus)
{
}

std::optional<EmotionNode> NodeMLMapper::embeddingToNode(const std::vector<float>& embedding) const {
    if (embedding.size() != 64) {
        return std::nullopt;
    }

    // Convert ML embedding to VAD coordinates
    auto [valence, arousal, dominance] = embeddingToVAD(embedding);

    // Find nearest node by VAD distance
    EmotionNode nearestNode = findNearestNodeByVAD(valence, arousal, dominance);

    // Enhance node with ML embedding
    nearestNode = enhanceNodeWithML(nearestNode, embedding, 1.0f);

    return nearestNode;
}

std::vector<float> NodeMLMapper::getNodeContext(int nodeId) const {
    auto nodeOpt = thesaurus_.findById(nodeId);
    if (!nodeOpt.has_value()) {
        return {};  // Return empty vector if node not found
    }

    const auto& node = *nodeOpt;

    // Build context vector from node data
    std::vector<float> context;

    // Add VAD coordinates
    context.push_back(node.valence);
    context.push_back(node.arousal);
    context.push_back(node.dominance);
    context.push_back(node.intensity);

    // Add musical attributes
    context.push_back(node.musicalAttributes.tempoModifier);
    context.push_back(node.musicalAttributes.dynamics);
    context.push_back(node.musicalAttributes.articulation);
    context.push_back(node.musicalAttributes.density);
    context.push_back(node.musicalAttributes.dissonance);

    // Add related nodes' VAD coordinates (up to 5 related nodes)
    auto relatedNodes = thesaurus_.findRelated(nodeId);
    for (size_t i = 0; i < std::min(relatedNodes.size(), size_t(5)); ++i) {
        context.push_back(relatedNodes[i].valence);
        context.push_back(relatedNodes[i].arousal);
        context.push_back(relatedNodes[i].dominance);
    }

    // Pad to fixed size if needed (for models expecting fixed input)
    // Target size: 128 (for HarmonyPredictor, etc.)
    while (context.size() < 128) {
        context.push_back(0.0f);
    }
    if (context.size() > 128) {
        context.resize(128);
    }

    return context;
}

std::vector<float> NodeMLMapper::nodeToMLInput(const EmotionNode& node) const {
    return vadToMLInput(node.valence, node.arousal, node.dominance, node.musicalAttributes);
}

EmotionNode NodeMLMapper::findNearestNodeByVAD(float valence, float arousal, float dominance) const {
    // Use thesaurus's built-in nearest neighbor search
    return thesaurus_.findNearestVAD(valence, arousal, dominance);
}

float NodeMLMapper::calculateVADDistance(const EmotionNode& node1, const EmotionNode& node2) const {
    return calculateVADDistance(node1.valence, node1.arousal, node1.dominance, node2);
}

float NodeMLMapper::calculateVADDistance(float valence, float arousal, float dominance, const EmotionNode& node) const {
    float dV = valence - node.valence;
    float dA = arousal - node.arousal;
    float dD = dominance - node.dominance;

    return std::sqrt(dV * dV + dA * dA + dD * dD);
}

EmotionNode NodeMLMapper::enhanceNodeWithML(const EmotionNode& node,
                                              const std::vector<float>& embedding,
                                              float confidence) const {
    EmotionNode enhanced = node;

    // Attach ML embedding
    if (embedding.size() == 64) {
        enhanced.mlEmbedding = embedding;
    }

    // Set ML confidence
    enhanced.mlConfidence = confidence;

    // Extract additional ML features from embedding
    // For now, we'll add some simple statistics
    if (!embedding.empty()) {
        float mean = std::accumulate(embedding.begin(), embedding.end(), 0.0f) / embedding.size();
        float variance = 0.0f;
        for (float val : embedding) {
            variance += (val - mean) * (val - mean);
        }
        variance /= embedding.size();
        float stdDev = std::sqrt(variance);

        enhanced.mlFeatures["mean"] = mean;
        enhanced.mlFeatures["stdDev"] = stdDev;
        enhanced.mlFeatures["max"] = *std::max_element(embedding.begin(), embedding.end());
        enhanced.mlFeatures["min"] = *std::min_element(embedding.begin(), embedding.end());
    }

    return enhanced;
}

std::tuple<float, float, float> NodeMLMapper::embeddingToVAD(const std::vector<float>& embedding) const {
    // Simple linear projection from 64-dim embedding to VAD space
    // This is a placeholder - in production, you'd use a learned mapping
    // (e.g., a small MLP trained to map embeddings to VAD)

    if (embedding.size() != 64) {
        return {0.0f, 0.5f, 0.5f};  // Default neutral
    }

    // Simple approach: Use first 3 dimensions of embedding, normalized
    // Better approach: Train a small projection network
    float valence = embedding[0] * 2.0f - 1.0f;  // Map 0-1 to -1 to 1
    float arousal = embedding[1];  // Already 0-1
    float dominance = embedding[2];  // Already 0-1

    // Clamp to valid ranges
    valence = std::clamp(valence, -1.0f, 1.0f);
    arousal = std::clamp(arousal, 0.0f, 1.0f);
    dominance = std::clamp(dominance, 0.0f, 1.0f);

    return {valence, arousal, dominance};
}

std::vector<float> NodeMLMapper::vadToMLInput(float valence, float arousal, float dominance,
                                               const MusicalAttributes& musicalAttrs) const {
    std::vector<float> input;

    // Add VAD coordinates
    input.push_back(valence);
    input.push_back(arousal);
    input.push_back(dominance);

    // Add musical attributes
    input.push_back(musicalAttrs.tempoModifier);
    input.push_back(musicalAttrs.dynamics);
    input.push_back(musicalAttrs.articulation);
    input.push_back(musicalAttrs.density);
    input.push_back(musicalAttrs.dissonance);

    // Pad or truncate to target size based on model requirements
    // For now, return as-is (caller can pad if needed)
    // Common sizes: 32 (DynamicsEngine), 64 (MelodyTransformer), 128 (HarmonyPredictor)

    return input;
}

} // namespace ml
} // namespace midikompanion
