/**
 * @file NodeMLMapper.cpp
 * @brief ML-Node bridge implementation
 */

#include "NodeMLMapper.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace midikompanion {
namespace ml {

//==============================================================================
// VADCoordinates Implementation
//==============================================================================

float VADCoordinates::distanceTo(const VADCoordinates& other) const {
    float dv = valence - other.valence;
    float da = arousal - other.arousal;
    float dd = dominance - other.dominance;
    float di = intensity - other.intensity;
    
    // Weighted Euclidean distance (valence and arousal more important)
    return std::sqrt(2.0f * dv * dv + 2.0f * da * da + dd * dd + di * di);
}

VADCoordinates VADCoordinates::lerp(const VADCoordinates& a, const VADCoordinates& b, float t) {
    VADCoordinates result;
    result.valence = a.valence + (b.valence - a.valence) * t;
    result.arousal = a.arousal + (b.arousal - a.arousal) * t;
    result.dominance = a.dominance + (b.dominance - a.dominance) * t;
    result.intensity = a.intensity + (b.intensity - a.intensity) * t;
    return result;
}

//==============================================================================
// NodeMLMapper Implementation
//==============================================================================

NodeMLMapper::NodeMLMapper() {
    initializeMappingMatrices();
}

bool NodeMLMapper::loadThesaurus(const juce::File& jsonPath) {
    if (!jsonPath.existsAsFile()) {
        return false;
    }
    
    auto jsonContent = jsonPath.loadFileAsString();
    auto parsed = juce::JSON::parse(jsonContent);
    
    if (!parsed.isObject()) {
        return false;
    }
    
    nodes_.clear();
    
    auto* rootObj = parsed.getDynamicObject();
    if (!rootObj) return false;
    
    // Parse nodes array
    auto nodesVar = rootObj->getProperty("nodes");
    if (auto* nodesArray = nodesVar.getArray()) {
        for (const auto& nodeVar : *nodesArray) {
            if (auto* nodeObj = nodeVar.getDynamicObject()) {
                EmotionNode node;
                node.id = nodeObj->getProperty("id");
                node.name = nodeObj->getProperty("name").toString();
                node.category = nodeObj->getProperty("category").toString();
                node.subcategory = nodeObj->getProperty("subcategory").toString();
                
                // Parse VAD
                if (auto* vadObj = nodeObj->getProperty("vad").getDynamicObject()) {
                    node.vad.valence = vadObj->getProperty("valence");
                    node.vad.arousal = vadObj->getProperty("arousal");
                    node.vad.dominance = vadObj->getProperty("dominance");
                    node.vad.intensity = vadObj->getProperty("intensity");
                }
                
                // Parse related emotions
                if (auto* relatedArray = nodeObj->getProperty("relatedEmotions").getArray()) {
                    for (const auto& relatedVar : *relatedArray) {
                        node.relatedEmotions.push_back(static_cast<int>(relatedVar));
                    }
                }
                
                // Parse musical attributes
                node.mode = nodeObj->getProperty("mode").toString();
                if (node.mode.isEmpty()) node.mode = "major";
                node.tempoMultiplier = nodeObj->getProperty("tempoMultiplier");
                if (node.tempoMultiplier == 0.0f) node.tempoMultiplier = 1.0f;
                node.dynamicsScale = nodeObj->getProperty("dynamicsScale");
                if (node.dynamicsScale == 0.0f) node.dynamicsScale = 1.0f;
                
                nodes_.push_back(node);
            }
        }
    }
    
    buildCategoryIndex();
    return !nodes_.empty();
}

void NodeMLMapper::initializeDefaultThesaurus() {
    nodes_.clear();
    
    // Base emotions and their VAD profiles
    struct EmotionProfile {
        juce::String category;
        float baseValence;
        float baseArousal;
        float baseDominance;
        juce::String mode;
        float tempoMult;
    };
    
    std::vector<EmotionProfile> profiles = {
        {"happy",    0.8f, 0.6f, 0.6f, "major", 1.1f},
        {"sad",     -0.6f, 0.3f, 0.3f, "minor", 0.8f},
        {"angry",   -0.5f, 0.8f, 0.8f, "minor", 1.2f},
        {"fear",    -0.7f, 0.7f, 0.2f, "minor", 0.9f},
        {"surprise", 0.3f, 0.8f, 0.5f, "major", 1.15f},
        {"disgust", -0.6f, 0.5f, 0.6f, "minor", 0.85f}
    };
    
    int nodeId = 0;
    
    // 6 base emotions × 6 sub-emotions × 6 intensity levels = 216 nodes
    for (const auto& profile : profiles) {
        for (int sub = 0; sub < 6; ++sub) {
            for (int intensity = 0; intensity < 6; ++intensity) {
                EmotionNode node;
                node.id = nodeId++;
                node.category = profile.category;
                node.subcategory = profile.category + "_" + juce::String(sub);
                node.name = profile.category + "_" + juce::String(sub) + "_" + juce::String(intensity);
                
                // Vary VAD based on sub-emotion and intensity
                float subVariation = (sub - 2.5f) / 5.0f * 0.3f;
                float intensityScale = (intensity + 1) / 6.0f;
                
                node.vad.valence = juce::jlimit(-1.0f, 1.0f, 
                    profile.baseValence + subVariation);
                node.vad.arousal = juce::jlimit(0.0f, 1.0f, 
                    profile.baseArousal * (0.5f + intensityScale * 0.5f));
                node.vad.dominance = juce::jlimit(0.0f, 1.0f, 
                    profile.baseDominance + subVariation * 0.5f);
                node.vad.intensity = intensityScale;
                
                node.mode = profile.mode;
                node.tempoMultiplier = profile.tempoMult * (0.9f + intensityScale * 0.2f);
                node.dynamicsScale = 0.5f + intensityScale * 0.5f;
                
                // Add related emotions (nearby in the same category + adjacent categories)
                for (int r = -3; r <= 3; ++r) {
                    int relatedId = node.id + r;
                    if (relatedId >= 0 && relatedId < 216 && relatedId != node.id) {
                        node.relatedEmotions.push_back(relatedId);
                    }
                }
                
                nodes_.push_back(node);
            }
        }
    }
    
    buildCategoryIndex();
}

void NodeMLMapper::buildCategoryIndex() {
    categoryIndex_.clear();
    
    for (const auto& node : nodes_) {
        categoryIndex_[node.category].push_back(node.id);
    }
}

EmotionNode NodeMLMapper::embeddingToNode(const std::vector<float>& embedding) const {
    if (nodes_.empty()) {
        return EmotionNode();
    }
    
    // Convert embedding to VAD
    VADCoordinates vad = embeddingToVAD(embedding);
    
    // Find nearest node
    int nearestId = findNearestNode(vad);
    
    if (nearestId >= 0 && nearestId < static_cast<int>(nodes_.size())) {
        EmotionNode result = nodes_[nearestId];
        result.mlEmbedding = embedding;
        result.mlConfidence = 0.8f; // Default confidence
        return result;
    }
    
    return EmotionNode();
}

int NodeMLMapper::findNearestNode(const VADCoordinates& vad) const {
    if (nodes_.empty()) return -1;
    
    int nearestId = 0;
    float minDistance = std::numeric_limits<float>::max();
    
    for (size_t i = 0; i < nodes_.size(); ++i) {
        float distance = vad.distanceTo(nodes_[i].vad);
        if (distance < minDistance) {
            minDistance = distance;
            nearestId = static_cast<int>(i);
        }
    }
    
    return nearestId;
}

const EmotionNode* NodeMLMapper::getNode(int nodeId) const {
    if (nodeId >= 0 && nodeId < static_cast<int>(nodes_.size())) {
        return &nodes_[nodeId];
    }
    return nullptr;
}

std::vector<const EmotionNode*> NodeMLMapper::getNodesInCategory(const juce::String& category) const {
    std::vector<const EmotionNode*> result;
    
    auto it = categoryIndex_.find(category);
    if (it != categoryIndex_.end()) {
        for (int id : it->second) {
            if (id >= 0 && id < static_cast<int>(nodes_.size())) {
                result.push_back(&nodes_[id]);
            }
        }
    }
    
    return result;
}

NodeContext NodeMLMapper::getNodeContext(int nodeId) const {
    NodeContext context;
    
    const EmotionNode* node = getNode(nodeId);
    if (!node) return context;
    
    // Get related node IDs
    context.relatedNodeIds = node->relatedEmotions;
    
    // Create combined embedding from node and related nodes
    std::vector<float> combinedVAD(4);
    combinedVAD[0] = node->vad.valence;
    combinedVAD[1] = node->vad.arousal;
    combinedVAD[2] = node->vad.dominance;
    combinedVAD[3] = node->vad.intensity;
    
    // Add context from related nodes
    int relatedCount = 0;
    for (int relatedId : context.relatedNodeIds) {
        const EmotionNode* related = getNode(relatedId);
        if (related) {
            combinedVAD[0] += related->vad.valence * 0.2f;
            combinedVAD[1] += related->vad.arousal * 0.2f;
            combinedVAD[2] += related->vad.dominance * 0.2f;
            combinedVAD[3] += related->vad.intensity * 0.2f;
            ++relatedCount;
        }
    }
    
    // Normalize
    if (relatedCount > 0) {
        float norm = 1.0f + relatedCount * 0.2f;
        for (float& v : combinedVAD) {
            v /= norm;
        }
    }
    
    // Expand to 64-dim embedding
    context.embedding = vadToEmbedding({
        combinedVAD[0], combinedVAD[1], combinedVAD[2], combinedVAD[3]
    });
    
    context.contextWeight = 1.0f;
    
    return context;
}

std::vector<float> NodeMLMapper::nodeToMLInput(const EmotionNode& node) const {
    // If node has ML embedding, use it
    if (node.mlEmbedding.has_value()) {
        return node.mlEmbedding.value();
    }
    
    // Otherwise, convert VAD to embedding
    return vadToEmbedding(node.vad);
}

std::vector<float> NodeMLMapper::vadToEmbedding(const VADCoordinates& vad) const {
    std::vector<float> embedding(64, 0.0f);
    
    // Use predefined transformation (could be learned)
    // First 16 dimensions: valence variations
    for (int i = 0; i < 16; ++i) {
        float phase = static_cast<float>(i) / 16.0f * 6.28318f;
        embedding[i] = vad.valence * std::cos(phase) * 0.5f + 
                       vad.arousal * std::sin(phase) * 0.5f;
    }
    
    // Next 16: arousal variations
    for (int i = 16; i < 32; ++i) {
        float phase = static_cast<float>(i - 16) / 16.0f * 6.28318f;
        embedding[i] = vad.arousal * std::cos(phase) * 0.5f + 
                       vad.dominance * std::sin(phase) * 0.5f;
    }
    
    // Next 16: dominance variations
    for (int i = 32; i < 48; ++i) {
        float phase = static_cast<float>(i - 32) / 16.0f * 6.28318f;
        embedding[i] = vad.dominance * std::cos(phase) * 0.5f + 
                       vad.intensity * std::sin(phase) * 0.5f;
    }
    
    // Last 16: intensity and combined features
    for (int i = 48; i < 64; ++i) {
        float phase = static_cast<float>(i - 48) / 16.0f * 6.28318f;
        embedding[i] = vad.intensity * std::cos(phase) * 0.3f + 
                       (vad.valence + vad.arousal) * 0.2f * std::sin(phase);
    }
    
    return embedding;
}

VADCoordinates NodeMLMapper::embeddingToVAD(const std::vector<float>& embedding) const {
    VADCoordinates vad;
    
    if (embedding.size() < 64) {
        return vad;
    }
    
    // Reverse the transformation (simplified inverse)
    float valenceSum = 0.0f;
    float arousalSum = 0.0f;
    float dominanceSum = 0.0f;
    float intensitySum = 0.0f;
    
    for (int i = 0; i < 16; ++i) {
        float phase = static_cast<float>(i) / 16.0f * 6.28318f;
        valenceSum += embedding[i] * std::cos(phase);
        arousalSum += embedding[i] * std::sin(phase);
    }
    
    for (int i = 16; i < 32; ++i) {
        float phase = static_cast<float>(i - 16) / 16.0f * 6.28318f;
        arousalSum += embedding[i] * std::cos(phase);
        dominanceSum += embedding[i] * std::sin(phase);
    }
    
    for (int i = 32; i < 48; ++i) {
        float phase = static_cast<float>(i - 32) / 16.0f * 6.28318f;
        dominanceSum += embedding[i] * std::cos(phase);
        intensitySum += embedding[i] * std::sin(phase);
    }
    
    for (int i = 48; i < 64; ++i) {
        float phase = static_cast<float>(i - 48) / 16.0f * 6.28318f;
        intensitySum += embedding[i] * std::cos(phase);
    }
    
    // Normalize and clamp
    vad.valence = juce::jlimit(-1.0f, 1.0f, valenceSum / 8.0f);
    vad.arousal = juce::jlimit(0.0f, 1.0f, arousalSum / 16.0f + 0.5f);
    vad.dominance = juce::jlimit(0.0f, 1.0f, dominanceSum / 16.0f + 0.5f);
    vad.intensity = juce::jlimit(0.0f, 1.0f, intensitySum / 16.0f + 0.5f);
    
    return vad;
}

void NodeMLMapper::attachMLEmbedding(int nodeId, const std::vector<float>& embedding, float confidence) {
    if (nodeId >= 0 && nodeId < static_cast<int>(nodes_.size())) {
        nodes_[nodeId].mlEmbedding = embedding;
        nodes_[nodeId].mlConfidence = confidence;
    }
}

EmotionNode NodeMLMapper::interpolateNodes(int nodeA, int nodeB, float t) const {
    const EmotionNode* a = getNode(nodeA);
    const EmotionNode* b = getNode(nodeB);
    
    if (!a) return EmotionNode();
    if (!b) return *a;
    
    EmotionNode result;
    result.id = -1; // Synthetic node
    result.name = a->name + "_to_" + b->name;
    result.category = t < 0.5f ? a->category : b->category;
    
    // Interpolate VAD
    result.vad = VADCoordinates::lerp(a->vad, b->vad, t);
    
    // Interpolate musical attributes
    result.tempoMultiplier = a->tempoMultiplier + (b->tempoMultiplier - a->tempoMultiplier) * t;
    result.dynamicsScale = a->dynamicsScale + (b->dynamicsScale - a->dynamicsScale) * t;
    result.mode = t < 0.5f ? a->mode : b->mode;
    
    // Combine related emotions
    for (int rel : a->relatedEmotions) {
        result.relatedEmotions.push_back(rel);
    }
    for (int rel : b->relatedEmotions) {
        if (std::find(result.relatedEmotions.begin(), result.relatedEmotions.end(), rel) 
            == result.relatedEmotions.end()) {
            result.relatedEmotions.push_back(rel);
        }
    }
    
    return result;
}

std::vector<EmotionNode> NodeMLMapper::getTransitionPath(int fromNode, int toNode, int steps) const {
    std::vector<EmotionNode> path;
    
    if (steps < 2) steps = 2;
    
    for (int i = 0; i <= steps; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(steps);
        path.push_back(interpolateNodes(fromNode, toNode, t));
    }
    
    return path;
}

juce::StringArray NodeMLMapper::getCategories() const {
    juce::StringArray categories;
    
    for (const auto& [category, _] : categoryIndex_) {
        categories.add(category);
    }
    
    return categories;
}

void NodeMLMapper::initializeMappingMatrices() {
    // Initialize simple transformation matrices
    // In production, these would be learned from data
    
    // VAD to 64-dim embedding (4 x 64)
    vadToEmbeddingMatrix_.resize(4);
    for (auto& row : vadToEmbeddingMatrix_) {
        row.resize(64, 0.0f);
        for (float& v : row) {
            v = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    
    // 64-dim embedding to VAD (64 x 4)
    embeddingToVADMatrix_.resize(64);
    for (auto& row : embeddingToVADMatrix_) {
        row.resize(4, 0.0f);
        for (float& v : row) {
            v = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }
    }
}

float NodeMLMapper::calculateNodeDistance(const EmotionNode& a, const EmotionNode& b) const {
    return a.vad.distanceTo(b.vad);
}

//==============================================================================
// HybridGenerator Implementation
//==============================================================================

HybridGenerator::HybridGenerator() = default;

HybridGenerator::GenerationParams HybridGenerator::generate(
    const EmotionNode& node, 
    const std::vector<float>& mlEmbedding) const 
{
    // Get rule-based generation
    GenerationParams ruleBased = generateRuleBased(node);
    
    if (mlEmbedding.empty() || mlWeight_ < 0.01f) {
        return ruleBased;
    }
    
    // Get ML-based generation
    GenerationParams mlBased = applyML(mlEmbedding);
    
    // Blend results
    return blend(ruleBased, mlBased, mlWeight_);
}

HybridGenerator::GenerationParams HybridGenerator::generateRuleBased(const EmotionNode& node) const {
    return applyRules(node);
}

HybridGenerator::GenerationParams HybridGenerator::applyRules(const EmotionNode& node) const {
    GenerationParams params;
    
    // Base tempo from arousal
    params.tempo = 60.0f + node.vad.arousal * 80.0f;
    params.tempo *= node.tempoMultiplier;
    
    // Mode from valence
    params.mode = node.mode;
    if (params.mode.isEmpty()) {
        params.mode = node.vad.valence >= 0.0f ? "major" : "minor";
    }
    
    // Dynamics from intensity
    params.dynamics = 0.4f + node.vad.intensity * 0.5f;
    params.dynamics *= node.dynamicsScale;
    
    // Articulation from arousal and dominance
    params.articulation = 0.3f + node.vad.arousal * 0.3f + node.vad.dominance * 0.2f;
    
    // Suggest notes based on mode
    if (params.mode == "major") {
        params.suggestedNotes = {0, 2, 4, 5, 7, 9, 11}; // Major scale degrees
    } else {
        params.suggestedNotes = {0, 2, 3, 5, 7, 8, 10}; // Natural minor
    }
    
    // Weight notes by emotion
    params.noteWeights.resize(params.suggestedNotes.size());
    for (size_t i = 0; i < params.noteWeights.size(); ++i) {
        float weight = 1.0f;
        
        // Root and fifth are always strong
        if (i == 0 || i == 4) weight = 1.5f;
        
        // Third varies by valence
        if (i == 2) weight = 0.8f + node.vad.valence * 0.3f;
        
        // Seventh for more intensity
        if (i == 6) weight = 0.5f + node.vad.intensity * 0.5f;
        
        params.noteWeights[i] = weight;
    }
    
    return params;
}

HybridGenerator::GenerationParams HybridGenerator::applyML(const std::vector<float>& embedding) const {
    GenerationParams params;
    
    if (embedding.size() < 64) {
        return params;
    }
    
    // Extract parameters from embedding
    // These mappings would be learned in practice
    
    // Tempo from first 8 dimensions
    float tempoFeature = 0.0f;
    for (int i = 0; i < 8; ++i) {
        tempoFeature += embedding[i];
    }
    params.tempo = 80.0f + tempoFeature * 10.0f;
    
    // Mode from next 8 dimensions
    float modeFeature = 0.0f;
    for (int i = 8; i < 16; ++i) {
        modeFeature += embedding[i];
    }
    params.mode = modeFeature > 0.0f ? "major" : "minor";
    
    // Dynamics from next 8 dimensions
    float dynamicsFeature = 0.0f;
    for (int i = 16; i < 24; ++i) {
        dynamicsFeature += std::abs(embedding[i]);
    }
    params.dynamics = 0.3f + dynamicsFeature * 0.1f;
    
    // Articulation from next 8 dimensions
    float articulationFeature = 0.0f;
    for (int i = 24; i < 32; ++i) {
        articulationFeature += std::abs(embedding[i]);
    }
    params.articulation = 0.2f + articulationFeature * 0.1f;
    
    // Note suggestions from remaining dimensions
    params.suggestedNotes.clear();
    params.noteWeights.clear();
    
    for (int i = 0; i < 12; ++i) {
        float weight = 0.0f;
        for (int j = 32 + i * 2; j < 34 + i * 2 && j < 64; ++j) {
            weight += std::abs(embedding[j]);
        }
        
        if (weight > 0.1f) {
            params.suggestedNotes.push_back(i);
            params.noteWeights.push_back(weight);
        }
    }
    
    // Ensure at least some notes
    if (params.suggestedNotes.empty()) {
        params.suggestedNotes = {0, 4, 7};
        params.noteWeights = {1.0f, 0.8f, 0.9f};
    }
    
    return params;
}

HybridGenerator::GenerationParams HybridGenerator::blend(
    const GenerationParams& ruleBased,
    const GenerationParams& mlBased,
    float weight) const 
{
    GenerationParams result;
    
    float ruleWeight = 1.0f - weight;
    
    // Blend continuous parameters
    result.tempo = ruleBased.tempo * ruleWeight + mlBased.tempo * weight;
    result.dynamics = ruleBased.dynamics * ruleWeight + mlBased.dynamics * weight;
    result.articulation = ruleBased.articulation * ruleWeight + mlBased.articulation * weight;
    
    // Mode: use rule-based if weight is low, ML if high
    result.mode = weight < 0.5f ? ruleBased.mode : mlBased.mode;
    
    // Combine note suggestions
    std::map<int, float> noteWeightMap;
    
    for (size_t i = 0; i < ruleBased.suggestedNotes.size(); ++i) {
        int note = ruleBased.suggestedNotes[i];
        float w = i < ruleBased.noteWeights.size() ? ruleBased.noteWeights[i] : 1.0f;
        noteWeightMap[note] += w * ruleWeight;
    }
    
    for (size_t i = 0; i < mlBased.suggestedNotes.size(); ++i) {
        int note = mlBased.suggestedNotes[i];
        float w = i < mlBased.noteWeights.size() ? mlBased.noteWeights[i] : 1.0f;
        noteWeightMap[note] += w * weight;
    }
    
    // Convert back to vectors
    result.suggestedNotes.clear();
    result.noteWeights.clear();
    
    for (const auto& [note, w] : noteWeightMap) {
        result.suggestedNotes.push_back(note);
        result.noteWeights.push_back(w);
    }
    
    return result;
}

} // namespace ml
} // namespace midikompanion
