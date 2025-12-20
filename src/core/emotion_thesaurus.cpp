/**
 * @file emotion_thesaurus.cpp
 * @brief 216-node emotion thesaurus implementation
 */

#include "emotion_thesaurus.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cctype>

namespace kelly {

//==============================================================================
// Sub-emotion definitions for each base emotion
//==============================================================================

static const std::map<std::string, std::array<std::string, 6>> SUB_EMOTIONS = {
    {"Happy", {"Joyful", "Content", "Excited", "Hopeful", "Proud", "Grateful"}},
    {"Sad", {"Melancholic", "Lonely", "Grief", "Disappointed", "Remorseful", "Nostalgic"}},
    {"Angry", {"Frustrated", "Irritated", "Furious", "Resentful", "Bitter", "Outraged"}},
    {"Fear", {"Anxious", "Nervous", "Terrified", "Worried", "Panicked", "Uneasy"}},
    {"Surprise", {"Amazed", "Astonished", "Startled", "Confused", "Curious", "Intrigued"}},
    {"Disgust", {"Revolted", "Contemptuous", "Repulsed", "Uncomfortable", "Disapproving", "Skeptical"}}
};

// Base VAD values for each emotion
static const std::map<std::string, std::array<float, 3>> BASE_VAD = {
    {"Happy", {0.8f, 0.5f, 0.5f}},      // V, A, D
    {"Sad", {-0.7f, -0.4f, -0.3f}},
    {"Angry", {-0.5f, 0.7f, 0.6f}},
    {"Fear", {-0.6f, 0.6f, -0.5f}},
    {"Surprise", {0.3f, 0.7f, 0.0f}},
    {"Disgust", {-0.6f, 0.2f, 0.3f}}
};

//==============================================================================
// EmotionThesaurus Implementation
//==============================================================================

EmotionThesaurus::EmotionThesaurus() {
    nodes_.reserve(216);
}

void EmotionThesaurus::initialize() {
    if (initialized_) return;
    initializeDefaultThesaurus();
    setupRelationships();
    initialized_ = true;
}

void EmotionThesaurus::initializeDefaultThesaurus() {
    nodes_.clear();
    nameToIdMap_.clear();
    
    int nodeId = 0;
    
    for (const auto& baseEmotion : BASE_EMOTIONS) {
        const auto& subEmotions = SUB_EMOTIONS.at(baseEmotion);
        const auto& baseVad = BASE_VAD.at(baseEmotion);
        
        for (int subIdx = 0; subIdx < 6; ++subIdx) {
            const std::string& subName = subEmotions[subIdx];
            float subVariation = (subIdx - 2.5f) / 10.0f;
            
            for (int intensity = 0; intensity < 6; ++intensity) {
                EmotionThesaurusNode node;
                node.id = nodeId;
                node.name = subName + " (" + std::string(INTENSITY_LABELS[intensity]) + ")";
                node.category = baseEmotion;
                node.subcategory = subName;
                node.intensityLevel = intensity;
                
                // Calculate VAD with variation
                node.vad.valence = std::clamp(baseVad[0] + subVariation, -1.0f, 1.0f);
                node.vad.arousal = std::clamp(baseVad[1] + subVariation * 0.5f, -1.0f, 1.0f);
                node.vad.dominance = std::clamp(baseVad[2] + subVariation * 0.3f, -1.0f, 1.0f);
                node.vad.intensity = (intensity + 1) / 6.0f;
                
                // Calculate musical attributes
                calculateMusicalAttributes(node);
                
                // Add to collection
                nodes_.push_back(node);
                
                // Lowercase key for case-insensitive lookup
                std::string lowerName = node.name;
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(),
                              [](unsigned char c) { return std::tolower(c); });
                nameToIdMap_[lowerName] = nodeId;
                
                ++nodeId;
            }
        }
    }
}

void EmotionThesaurus::calculateMusicalAttributes(EmotionThesaurusNode& node) {
    MusicalAttributes& attrs = node.musicalAttributes;
    const VADCoordinates& vad = node.vad;
    
    // Major/minor based on valence
    attrs.isMajorMode = vad.valence >= 0.0f;
    
    // Tempo multiplier: calm emotions slower, excited faster
    attrs.tempoMultiplier = 0.7f + (vad.arousal + 1.0f) / 2.0f * 0.6f;
    
    // Dynamics based on intensity and arousal
    attrs.dynamicsLevel = 0.3f + vad.intensity * 0.5f + std::abs(vad.arousal) * 0.2f;
    
    // Rhythmic density based on arousal
    attrs.rhythmicDensity = 0.2f + (vad.arousal + 1.0f) / 2.0f * 0.6f;
    
    // Harmonic complexity based on dominance and intensity
    attrs.harmonicComplexity = 0.3f + vad.intensity * 0.3f + std::abs(vad.dominance) * 0.2f;
    
    // Melodic contour: positive valence tends upward
    attrs.melodicContour = vad.valence * 0.5f + vad.arousal * 0.3f;
    
    // Octave: higher for positive emotions
    attrs.suggestedOctave = 4 + static_cast<int>(vad.valence + 0.5f);
    
    // Scale based on mode and emotion
    if (attrs.isMajorMode) {
        if (vad.arousal > 0.3f) {
            attrs.preferredScale = "lydian";  // Bright, uplifting
        } else if (vad.arousal < -0.3f) {
            attrs.preferredScale = "mixolydian";  // Relaxed major
        } else {
            attrs.preferredScale = "major";
        }
    } else {
        if (vad.arousal > 0.3f) {
            attrs.preferredScale = "harmonic_minor";  // Tense
        } else if (vad.arousal < -0.3f) {
            attrs.preferredScale = "dorian";  // Melancholic but not too dark
        } else {
            attrs.preferredScale = "natural_minor";
        }
    }
}

void EmotionThesaurus::setupRelationships() {
    for (auto& node : nodes_) {
        // Intensity ladder: same sub-emotion at all intensities
        int baseId = (node.id / 6) * 6;
        for (int i = 0; i < 6; ++i) {
            node.intensityLadder.push_back(baseId + i);
        }
        
        // Related emotions: same category, different sub-emotions
        int categoryStart = (node.id / 36) * 36;
        for (int i = 0; i < 36; ++i) {
            int relatedId = categoryStart + i;
            if (relatedId != node.id && (relatedId / 6) != (node.id / 6)) {
                // Different sub-emotion, similar intensity
                int relatedIntensity = relatedId % 6;
                if (std::abs(relatedIntensity - node.intensityLevel) <= 1) {
                    node.relatedEmotions.push_back(relatedId);
                }
            }
        }
        
        // Opposite emotions: different category, opposite valence
        for (const auto& otherNode : nodes_) {
            if (otherNode.category != node.category) {
                // Opposite valence and similar intensity
                float valenceDiff = std::abs(node.vad.valence - (-otherNode.vad.valence));
                float intensityDiff = std::abs(node.vad.intensity - otherNode.vad.intensity);
                
                if (valenceDiff < 0.3f && intensityDiff < 0.2f) {
                    node.oppositeEmotions.push_back(otherNode.id);
                }
            }
        }
        
        // Limit relationships to reasonable size
        if (node.relatedEmotions.size() > 12) {
            node.relatedEmotions.resize(12);
        }
        if (node.oppositeEmotions.size() > 6) {
            node.oppositeEmotions.resize(6);
        }
    }
}

const EmotionThesaurusNode* EmotionThesaurus::getNode(int id) const {
    if (id >= 0 && id < static_cast<int>(nodes_.size())) {
        return &nodes_[id];
    }
    return nullptr;
}

EmotionThesaurusNode* EmotionThesaurus::getNodeMutable(int id) {
    if (id >= 0 && id < static_cast<int>(nodes_.size())) {
        return &nodes_[id];
    }
    return nullptr;
}

const EmotionThesaurusNode* EmotionThesaurus::getNodeByName(const std::string& name) const {
    std::string lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    
    auto it = nameToIdMap_.find(lowerName);
    if (it != nameToIdMap_.end()) {
        return getNode(it->second);
    }
    return nullptr;
}

const EmotionThesaurusNode* EmotionThesaurus::findNearestNode(const VADCoordinates& vad) const {
    if (nodes_.empty()) return nullptr;
    
    const EmotionThesaurusNode* nearest = &nodes_[0];
    float minDist = std::numeric_limits<float>::max();
    
    for (const auto& node : nodes_) {
        float dist = vad.distanceTo(node.vad);
        if (dist < minDist) {
            minDist = dist;
            nearest = &node;
        }
    }
    
    return nearest;
}

std::vector<std::pair<const EmotionThesaurusNode*, float>> EmotionThesaurus::findNearestNodes(
    const VADCoordinates& vad, int k) const {
    
    std::vector<std::pair<size_t, float>> distances;
    distances.reserve(nodes_.size());
    
    for (size_t i = 0; i < nodes_.size(); ++i) {
        distances.emplace_back(i, vad.distanceTo(nodes_[i].vad));
    }
    
    k = std::min(k, static_cast<int>(nodes_.size()));
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
    
    std::vector<std::pair<const EmotionThesaurusNode*, float>> result;
    result.reserve(k);
    for (int i = 0; i < k; ++i) {
        result.emplace_back(&nodes_[distances[i].first], distances[i].second);
    }
    
    return result;
}

std::vector<const EmotionThesaurusNode*> EmotionThesaurus::getNodesInCategory(
    const std::string& category) const {
    
    std::vector<const EmotionThesaurusNode*> result;
    for (const auto& node : nodes_) {
        if (node.category == category) {
            result.push_back(&node);
        }
    }
    return result;
}

std::vector<const EmotionThesaurusNode*> EmotionThesaurus::getNodesInSubcategory(
    const std::string& subcategory) const {
    
    std::vector<const EmotionThesaurusNode*> result;
    for (const auto& node : nodes_) {
        if (node.subcategory == subcategory) {
            result.push_back(&node);
        }
    }
    return result;
}

std::vector<const EmotionThesaurusNode*> EmotionThesaurus::getRelatedNodes(int nodeId) const {
    std::vector<const EmotionThesaurusNode*> result;
    const EmotionThesaurusNode* node = getNode(nodeId);
    
    if (node) {
        for (int relId : node->relatedEmotions) {
            if (const EmotionThesaurusNode* relNode = getNode(relId)) {
                result.push_back(relNode);
            }
        }
    }
    
    return result;
}

std::vector<const EmotionThesaurusNode*> EmotionThesaurus::getOppositeNodes(int nodeId) const {
    std::vector<const EmotionThesaurusNode*> result;
    const EmotionThesaurusNode* node = getNode(nodeId);
    
    if (node) {
        for (int oppId : node->oppositeEmotions) {
            if (const EmotionThesaurusNode* oppNode = getNode(oppId)) {
                result.push_back(oppNode);
            }
        }
    }
    
    return result;
}

std::vector<const EmotionThesaurusNode*> EmotionThesaurus::getIntensityLadder(int nodeId) const {
    std::vector<const EmotionThesaurusNode*> result;
    const EmotionThesaurusNode* node = getNode(nodeId);
    
    if (node) {
        for (int intId : node->intensityLadder) {
            if (const EmotionThesaurusNode* intNode = getNode(intId)) {
                result.push_back(intNode);
            }
        }
    }
    
    return result;
}

void EmotionThesaurus::updateNodeEmbedding(int nodeId,
                                            const std::vector<float>& embedding,
                                            float confidence) {
    EmotionThesaurusNode* node = getNodeMutable(nodeId);
    if (node) {
        node->mlEmbedding = embedding;
        node->mlConfidence = confidence;
    }
}

const EmotionThesaurusNode* EmotionThesaurus::findNodeFromEmbedding(
    const std::vector<float>& embedding) const {
    
    VADCoordinates vad = embeddingToVAD(embedding);
    return findNearestNode(vad);
}

std::vector<float> EmotionThesaurus::nodeToMLInput(int nodeId) const {
    const EmotionThesaurusNode* node = getNode(nodeId);
    if (!node) {
        return std::vector<float>(64, 0.0f);
    }
    
    // If node has ML embedding, use it
    if (node->mlEmbedding.has_value()) {
        return node->mlEmbedding.value();
    }
    
    // Otherwise, generate from VAD
    return vadToEmbedding(node->vad);
}

std::vector<float> EmotionThesaurus::getNodeContext(int nodeId) const {
    std::vector<float> context(128, 0.0f);
    const EmotionThesaurusNode* node = getNode(nodeId);
    
    if (!node) return context;
    
    // First 64 dims: node's embedding
    std::vector<float> nodeEmbed = nodeToMLInput(nodeId);
    std::copy(nodeEmbed.begin(), 
              nodeEmbed.begin() + std::min(size_t(64), nodeEmbed.size()),
              context.begin());
    
    // Next 64 dims: average of related nodes
    auto related = getRelatedNodes(nodeId);
    if (!related.empty()) {
        std::vector<float> relatedSum(64, 0.0f);
        
        for (const auto* relNode : related) {
            std::vector<float> relEmbed = nodeToMLInput(relNode->id);
            for (size_t i = 0; i < std::min(size_t(64), relEmbed.size()); ++i) {
                relatedSum[i] += relEmbed[i];
            }
        }
        
        for (size_t i = 0; i < 64; ++i) {
            context[64 + i] = relatedSum[i] / related.size();
        }
    }
    
    return context;
}

std::pair<VADCoordinates, MusicalAttributes> EmotionThesaurus::interpolate(
    int node1Id, int node2Id, float t) const {
    
    const EmotionThesaurusNode* node1 = getNode(node1Id);
    const EmotionThesaurusNode* node2 = getNode(node2Id);
    
    if (!node1 || !node2) {
        return {{}, {}};
    }
    
    t = std::clamp(t, 0.0f, 1.0f);
    float inv = 1.0f - t;
    
    // Interpolate VAD
    VADCoordinates vad;
    vad.valence = node1->vad.valence * inv + node2->vad.valence * t;
    vad.arousal = node1->vad.arousal * inv + node2->vad.arousal * t;
    vad.dominance = node1->vad.dominance * inv + node2->vad.dominance * t;
    vad.intensity = node1->vad.intensity * inv + node2->vad.intensity * t;
    
    // Interpolate musical attributes
    MusicalAttributes attrs;
    const auto& a1 = node1->musicalAttributes;
    const auto& a2 = node2->musicalAttributes;
    
    attrs.isMajorMode = t < 0.5f ? a1.isMajorMode : a2.isMajorMode;
    attrs.tempoMultiplier = a1.tempoMultiplier * inv + a2.tempoMultiplier * t;
    attrs.dynamicsLevel = a1.dynamicsLevel * inv + a2.dynamicsLevel * t;
    attrs.rhythmicDensity = a1.rhythmicDensity * inv + a2.rhythmicDensity * t;
    attrs.harmonicComplexity = a1.harmonicComplexity * inv + a2.harmonicComplexity * t;
    attrs.melodicContour = a1.melodicContour * inv + a2.melodicContour * t;
    attrs.suggestedOctave = static_cast<int>(a1.suggestedOctave * inv + a2.suggestedOctave * t + 0.5f);
    attrs.preferredScale = t < 0.5f ? a1.preferredScale : a2.preferredScale;
    
    return {vad, attrs};
}

VADCoordinates EmotionThesaurus::embeddingToVAD(const std::vector<float>& embedding) const {
    VADCoordinates vad;
    
    if (embedding.size() >= 4) {
        vad.valence = std::clamp(embedding[0], -1.0f, 1.0f);
        vad.arousal = std::clamp(embedding[1], -1.0f, 1.0f);
        vad.dominance = std::clamp(embedding[2], -1.0f, 1.0f);
        vad.intensity = std::clamp((embedding[3] + 1.0f) / 2.0f, 0.0f, 1.0f);
    }
    
    return vad;
}

std::vector<float> EmotionThesaurus::vadToEmbedding(const VADCoordinates& vad) const {
    std::vector<float> embedding(64, 0.0f);
    
    // First 4 dims: VAD coordinates
    embedding[0] = vad.valence;
    embedding[1] = vad.arousal;
    embedding[2] = vad.dominance;
    embedding[3] = vad.intensity * 2.0f - 1.0f;
    
    // Derived features
    embedding[4] = vad.valence * vad.arousal;
    embedding[5] = vad.arousal * vad.intensity;
    embedding[6] = std::abs(vad.valence);
    embedding[7] = std::abs(vad.arousal);
    
    // Fill with harmonic expansions
    for (int i = 8; i < 64; ++i) {
        float phase = (i - 8) / 56.0f * 3.14159f * 4.0f;
        embedding[i] = std::sin(vad.valence + phase) * 0.3f +
                       std::cos(vad.arousal + phase) * 0.3f +
                       std::sin(vad.dominance + phase * 2.0f) * 0.2f +
                       vad.intensity * std::cos(phase) * 0.2f;
    }
    
    return embedding;
}

bool EmotionThesaurus::loadFromFile(const std::string& filepath) {
    // Simplified JSON loading - in production, use a proper JSON library
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    // For now, just initialize with defaults
    initialize();
    return true;
}

bool EmotionThesaurus::saveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    // Simplified JSON output
    file << "{\n  \"nodes\": [\n";
    
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node = nodes_[i];
        file << "    {\n";
        file << "      \"id\": " << node.id << ",\n";
        file << "      \"name\": \"" << node.name << "\",\n";
        file << "      \"category\": \"" << node.category << "\",\n";
        file << "      \"subcategory\": \"" << node.subcategory << "\",\n";
        file << "      \"vad\": {\n";
        file << "        \"valence\": " << node.vad.valence << ",\n";
        file << "        \"arousal\": " << node.vad.arousal << ",\n";
        file << "        \"dominance\": " << node.vad.dominance << ",\n";
        file << "        \"intensity\": " << node.vad.intensity << "\n";
        file << "      }\n";
        file << "    }";
        if (i < nodes_.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]\n}\n";
    
    return true;
}

//==============================================================================
// EmotionJourney Implementation
//==============================================================================

EmotionJourney::EmotionJourney(const EmotionThesaurus& thesaurus)
    : thesaurus_(thesaurus) {
}

void EmotionJourney::addWaypoint(int nodeId, float duration, float transitionTime) {
    waypoints_.push_back({nodeId, duration, transitionTime});
}

VADCoordinates EmotionJourney::getVADAtTime(float timeBeats) const {
    if (waypoints_.empty()) {
        return VADCoordinates();
    }
    
    float currentTime = 0.0f;
    
    for (size_t i = 0; i < waypoints_.size(); ++i) {
        const auto& wp = waypoints_[i];
        float waypointEnd = currentTime + wp.duration;
        
        if (timeBeats < currentTime + wp.transitionTime && i > 0) {
            // In transition from previous waypoint
            const auto& prevWp = waypoints_[i - 1];
            float t = (timeBeats - (currentTime - prevWp.transitionTime)) / prevWp.transitionTime;
            t = std::clamp(t, 0.0f, 1.0f);
            
            auto result = thesaurus_.interpolate(prevWp.nodeId, wp.nodeId, t);
            return result.first;
        }
        
        if (timeBeats <= waypointEnd) {
            // At this waypoint
            const EmotionThesaurusNode* node = thesaurus_.getNode(wp.nodeId);
            return node ? node->vad : VADCoordinates();
        }
        
        currentTime = waypointEnd;
    }
    
    // Past end - return last waypoint
    const EmotionThesaurusNode* node = thesaurus_.getNode(waypoints_.back().nodeId);
    return node ? node->vad : VADCoordinates();
}

MusicalAttributes EmotionJourney::getAttributesAtTime(float timeBeats) const {
    if (waypoints_.empty()) {
        return MusicalAttributes();
    }
    
    float currentTime = 0.0f;
    
    for (size_t i = 0; i < waypoints_.size(); ++i) {
        const auto& wp = waypoints_[i];
        float waypointEnd = currentTime + wp.duration;
        
        if (timeBeats < currentTime + wp.transitionTime && i > 0) {
            const auto& prevWp = waypoints_[i - 1];
            float t = (timeBeats - (currentTime - prevWp.transitionTime)) / prevWp.transitionTime;
            t = std::clamp(t, 0.0f, 1.0f);
            
            auto result = thesaurus_.interpolate(prevWp.nodeId, wp.nodeId, t);
            return result.second;
        }
        
        if (timeBeats <= waypointEnd) {
            const EmotionThesaurusNode* node = thesaurus_.getNode(wp.nodeId);
            return node ? node->musicalAttributes : MusicalAttributes();
        }
        
        currentTime = waypointEnd;
    }
    
    const EmotionThesaurusNode* node = thesaurus_.getNode(waypoints_.back().nodeId);
    return node ? node->musicalAttributes : MusicalAttributes();
}

float EmotionJourney::getTotalDuration() const {
    float total = 0.0f;
    for (const auto& wp : waypoints_) {
        total += wp.duration;
    }
    return total;
}

void EmotionJourney::clear() {
    waypoints_.clear();
}

int EmotionJourney::findWaypointAtTime(float timeBeats) const {
    float currentTime = 0.0f;
    
    for (size_t i = 0; i < waypoints_.size(); ++i) {
        currentTime += waypoints_[i].duration;
        if (timeBeats <= currentTime) {
            return static_cast<int>(i);
        }
    }
    
    return static_cast<int>(waypoints_.size()) - 1;
}

} // namespace kelly
