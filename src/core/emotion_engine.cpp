#include "emotion_engine.h"
#include <cmath>
#include <algorithm>

namespace kelly {

EmotionEngine::EmotionEngine() {
    initializeEmotions();
}

void EmotionEngine::initializeEmotions() {
    // Initialize core emotion nodes (simplified for initial implementation)
    nodes_[0] = EmotionNode{
        0, "euphoria", EmotionCategory::Joy,
        1.0f, 1.0f, 1.0f, {},
        {1.5f, "major", 1.0f}
    };
    
    nodes_[1] = EmotionNode{
        1, "contentment", EmotionCategory::Joy,
        0.5f, 0.7f, 0.3f, {},
        {1.0f, "major", 0.5f}
    };
    
    nodes_[2] = EmotionNode{
        2, "grief", EmotionCategory::Sadness,
        1.0f, -0.9f, 0.7f, {},
        {0.7f, "minor", 0.9f}
    };
    
    nodes_[3] = EmotionNode{
        3, "melancholy", EmotionCategory::Sadness,
        0.6f, -0.6f, 0.3f, {},
        {0.8f, "minor", 0.6f}
    };
    
    nodes_[4] = EmotionNode{
        4, "rage", EmotionCategory::Anger,
        1.0f, -0.8f, 1.0f, {},
        {1.4f, "minor", 1.0f}
    };
    
    nodes_[5] = EmotionNode{
        5, "annoyance", EmotionCategory::Anger,
        0.4f, -0.4f, 0.5f, {},
        {1.1f, "minor", 0.4f}
    };
    
    nodes_[6] = EmotionNode{
        6, "terror", EmotionCategory::Fear,
        1.0f, -0.9f, 1.0f, {},
        {1.3f, "minor", 1.0f}
    };
    
    nodes_[7] = EmotionNode{
        7, "anxiety", EmotionCategory::Fear,
        0.6f, -0.5f, 0.8f, {},
        {1.2f, "minor", 0.6f}
    };
}

const EmotionNode* EmotionEngine::getEmotion(int emotionId) const {
    auto it = nodes_.find(emotionId);
    return (it != nodes_.end()) ? &it->second : nullptr;
}

const EmotionNode* EmotionEngine::findEmotionByName(const std::string& name) const {
    for (const auto& [id, node] : nodes_) {
        if (node.name == name) {
            return &node;
        }
    }
    return nullptr;
}

float EmotionEngine::calculateDistance(const EmotionNode& a, const EmotionNode& b) const {
    float dv = a.valence - b.valence;
    float da = a.arousal - b.arousal;
    float di = a.intensity - b.intensity;
    return std::sqrt(dv * dv + da * da + di * di);
}

std::vector<const EmotionNode*> EmotionEngine::getNearbyEmotions(int emotionId, float threshold) const {
    const EmotionNode* source = getEmotion(emotionId);
    if (!source) {
        return {};
    }
    
    std::vector<const EmotionNode*> nearby;
    for (const auto& [id, node] : nodes_) {
        if (id == emotionId) continue;
        
        float distance = calculateDistance(*source, node);
        if (distance < threshold) {
            nearby.push_back(&node);
        }
    }
    
    return nearby;
}

} // namespace kelly
