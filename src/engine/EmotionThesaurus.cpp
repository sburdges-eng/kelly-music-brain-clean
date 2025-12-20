#include "engine/EmotionThesaurus.h"
#include "common/MusicConstants.h"
#include "common/KellyTypes.h"  // For categoryToString
#include <algorithm>
#include <cmath>
#include <cctype>

// Conditionally include loader only if JUCE is available and not building bridge
#ifndef KELLY_BRIDGE_NO_JUCE
#include "engine/EmotionThesaurusLoader.h"
#include <juce_core/juce_core.h>
#endif

namespace kelly {

EmotionThesaurus::EmotionThesaurus() {
#ifndef KELLY_BRIDGE_NO_JUCE
    // Use the enhanced loader with fallback paths and embedded defaults
    int loaded = EmotionThesaurusLoader::loadWithFallbacks(*this);

    // If still no data loaded, use hardcoded fallback
    // Note: Hardcoded fallback uses IDs starting from 1, which is fine since
    // JSON loader also starts from 1. If JSON partially loads, we skip the fallback.
    if (loaded == 0) {
        juce::Logger::writeToLog("EmotionThesaurus: Using hardcoded fallback thesaurus");
        initializeThesaurus();
    } else {
        juce::Logger::writeToLog("EmotionThesaurus: Loaded " + juce::String(loaded) + " emotions from JSON files");
    }
#else
    // For Python bridge (no JUCE), always use embedded hardcoded thesaurus
    initializeThesaurus();
#endif
}

void EmotionThesaurus::addNode(EmotionNode node) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string lowerName = node.name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    nameIndex_[lowerName] = node.id;
    nodes_[node.id] = std::move(node);
}

void EmotionThesaurus::addSynonym(int emotionId, const std::string& synonym) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (nodes_.find(emotionId) == nodes_.end()) {
        return;  // Emotion doesn't exist
    }
    std::string lowerSynonym = synonym;
    std::transform(lowerSynonym.begin(), lowerSynonym.end(), lowerSynonym.begin(), ::tolower);
    nameIndex_[lowerSynonym] = emotionId;
}

void EmotionThesaurus::initializeThesaurus() {
    using namespace MusicConstants;

    // Helper function to create EmotionNode with correct field order for KellyTypes::EmotionNode
    auto makeNode = [](int id, const std::string& name, EmotionCategory cat, float intensity,
                       float valence, float arousal, float dominance,
                       const std::vector<int>& related) -> EmotionNode {
        EmotionNode node;
        node.id = id;
        node.name = name;
        node.categoryEnum = cat;
        node.category = categoryToString(cat);
        node.intensity = intensity;
        node.valence = valence;
        node.arousal = arousal;
        node.dominance = dominance;
        node.relatedEmotions = related;
        return node;
    };

    // SADNESS CLUSTER (Negative valence, varying arousal, low dominance)
    addNode(makeNode(1, "Grief", EmotionCategory::Sadness, INTENSITY_EXTREME, VALENCE_VERY_NEGATIVE - 0.2f, AROUSAL_LOW, 0.25f, {2, 3, 4}));
    addNode(makeNode(2, "Melancholy", EmotionCategory::Sadness, INTENSITY_MODERATE, VALENCE_NEGATIVE - 0.1f, AROUSAL_VERY_LOW, 0.30f, {1, 5, 6}));
    addNode(makeNode(3, "Despair", EmotionCategory::Sadness, INTENSITY_EXTREME, -1.0f, AROUSAL_LOW + 0.1f, 0.15f, {1, 4}));
    addNode(makeNode(4, "Sorrow", EmotionCategory::Sadness, INTENSITY_VERY_HIGH, VALENCE_VERY_NEGATIVE - 0.1f, AROUSAL_LOW, 0.25f, {1, 3, 5}));
    addNode(makeNode(5, "Wistfulness", EmotionCategory::Sadness, INTENSITY_LOW, VALENCE_SLIGHTLY_NEGATIVE, AROUSAL_VERY_LOW, 0.40f, {2, 6}));
    addNode(makeNode(6, "Longing", EmotionCategory::Sadness, INTENSITY_HIGH, VALENCE_NEGATIVE - 0.1f, INTENSITY_MODERATE, 0.35f, {2, 5, 7}));
    addNode(makeNode(7, "Heartache", EmotionCategory::Sadness, INTENSITY_VERY_HIGH, VALENCE_VERY_NEGATIVE - 0.2f, AROUSAL_LOW + 0.1f, 0.25f, {1, 4, 6}));
    addNode(makeNode(8, "Emptiness", EmotionCategory::Sadness, INTENSITY_HIGH, VALENCE_NEGATIVE, AROUSAL_VERY_LOW - 0.1f, 0.20f, {2, 9}));
    addNode(makeNode(9, "Numbness", EmotionCategory::Sadness, INTENSITY_LOW + 0.1f, VALENCE_SLIGHTLY_NEGATIVE, 0.05f, 0.25f, {8, 10}));
    addNode(makeNode(10, "Resignation", EmotionCategory::Sadness, INTENSITY_MODERATE, VALENCE_NEGATIVE - 0.1f, AROUSAL_VERY_LOW + 0.05f, 0.20f, {9, 2}));

    // ANGER CLUSTER (Negative valence, high arousal, high dominance)
    addNode(makeNode(20, "Rage", EmotionCategory::Anger, 1.0f, -0.8f, 1.0f, 0.90f, {21, 22}));
    addNode(makeNode(21, "Fury", EmotionCategory::Anger, 0.95f, -0.9f, 0.95f, 0.85f, {20, 22}));
    addNode(makeNode(22, "Frustration", EmotionCategory::Anger, 0.6f, -0.5f, 0.7f, 0.70f, {20, 23, 24}));
    addNode(makeNode(23, "Irritation", EmotionCategory::Anger, 0.4f, -0.3f, 0.5f, 0.60f, {22, 24}));
    addNode(makeNode(24, "Annoyance", EmotionCategory::Anger, 0.3f, -0.2f, 0.4f, 0.55f, {22, 23}));
    addNode(makeNode(25, "Resentment", EmotionCategory::Anger, 0.7f, -0.6f, 0.5f, 0.65f, {22, 26}));
    addNode(makeNode(26, "Bitterness", EmotionCategory::Anger, 0.6f, -0.7f, 0.4f, 0.60f, {25, 10}));
    addNode(makeNode(27, "Hostility", EmotionCategory::Anger, 0.8f, -0.7f, 0.8f, 0.80f, {20, 21}));

    // FEAR CLUSTER (Negative valence, high arousal, defensive)
    addNode(makeNode(40, "Terror", EmotionCategory::Fear, 1.0f, -0.9f, 1.0f, 0.15f, {41, 42}));
    addNode(makeNode(41, "Panic", EmotionCategory::Fear, 0.95f, -0.8f, 0.95f, 0.18f, {40, 42}));
    addNode(makeNode(42, "Anxiety", EmotionCategory::Fear, 0.6f, -0.5f, 0.7f, 0.22f, {41, 43, 44}));
    addNode(makeNode(43, "Worry", EmotionCategory::Fear, 0.4f, -0.3f, 0.5f, 0.25f, {42, 44}));
    addNode(makeNode(44, "Unease", EmotionCategory::Fear, 0.3f, -0.2f, 0.4f, 0.28f, {42, 43}));
    addNode(makeNode(45, "Dread", EmotionCategory::Fear, 0.8f, -0.7f, 0.6f, 0.20f, {40, 46}));
    addNode(makeNode(46, "Apprehension", EmotionCategory::Fear, 0.5f, -0.4f, 0.5f, 0.25f, {45, 43}));
    addNode(makeNode(47, "Vulnerability", EmotionCategory::Fear, 0.6f, -0.4f, 0.3f, 0.18f, {44, 8}));

    // JOY CLUSTER (Positive valence, varying arousal)
    addNode(makeNode(60, "Ecstasy", EmotionCategory::Joy, 1.0f, 1.0f, 1.0f, 0.85f, {61, 62}));
    addNode(makeNode(61, "Elation", EmotionCategory::Joy, 0.9f, 0.9f, 0.9f, 0.80f, {60, 62}));
    addNode(makeNode(62, "Happiness", EmotionCategory::Joy, 0.7f, 0.7f, 0.7f, 0.75f, {61, 63, 64}));
    addNode(makeNode(63, "Contentment", EmotionCategory::Joy, 0.5f, 0.6f, 0.3f, 0.70f, {62, 64}));
    addNode(makeNode(64, "Serenity", EmotionCategory::Joy, 0.4f, 0.5f, 0.2f, 0.65f, {63, 65}));
    addNode(makeNode(65, "Peace", EmotionCategory::Joy, 0.3f, 0.4f, 0.15f, 0.60f, {64, 66}));
    addNode(makeNode(66, "Calm", EmotionCategory::Joy, 0.2f, 0.3f, 0.1f, 0.55f, {65}));
    addNode(makeNode(67, "Relief", EmotionCategory::Joy, 0.6f, 0.5f, 0.5f, 0.70f, {62, 63}));
    addNode(makeNode(68, "Gratitude", EmotionCategory::Joy, 0.6f, 0.6f, 0.4f, 0.72f, {62, 80}));
    addNode(makeNode(69, "Hope", EmotionCategory::Joy, 0.7f, 0.6f, 0.6f, 0.73f, {62, 90}));

    // TRUST CLUSTER (Positive valence, moderate arousal)
    addNode(makeNode(80, "Love", EmotionCategory::Trust, 0.9f, 0.8f, 0.7f, 0.75f, {81, 82, 68}));
    addNode(makeNode(81, "Adoration", EmotionCategory::Trust, 0.85f, 0.85f, 0.75f, 0.78f, {80, 82}));
    addNode(makeNode(82, "Affection", EmotionCategory::Trust, 0.6f, 0.6f, 0.5f, 0.70f, {80, 83}));
    addNode(makeNode(83, "Tenderness", EmotionCategory::Trust, 0.5f, 0.5f, 0.4f, 0.68f, {82, 84}));
    addNode(makeNode(84, "Warmth", EmotionCategory::Trust, 0.4f, 0.5f, 0.35f, 0.65f, {83, 63}));
    addNode(makeNode(85, "Acceptance", EmotionCategory::Trust, 0.5f, 0.4f, 0.3f, 0.68f, {84, 63}));
    addNode(makeNode(86, "Belonging", EmotionCategory::Trust, 0.6f, 0.5f, 0.4f, 0.72f, {80, 85}));

    // ANTICIPATION CLUSTER (Positive valence, high arousal, forward-looking)
    addNode(makeNode(90, "Excitement", EmotionCategory::Anticipation, 0.9f, 0.7f, 0.9f, 0.78f, {91, 69}));
    addNode(makeNode(91, "Eagerness", EmotionCategory::Anticipation, 0.7f, 0.6f, 0.75f, 0.75f, {90, 92}));
    addNode(makeNode(92, "Curiosity", EmotionCategory::Anticipation, 0.5f, 0.4f, 0.6f, 0.68f, {91, 93}));
    addNode(makeNode(93, "Interest", EmotionCategory::Anticipation, 0.4f, 0.3f, 0.5f, 0.65f, {92}));
    addNode(makeNode(94, "Vigilance", EmotionCategory::Anticipation, 0.6f, 0.2f, 0.7f, 0.70f, {46, 93}));

    // SURPRISE CLUSTER (Neutral valence, high arousal spike)
    addNode(makeNode(100, "Amazement", EmotionCategory::Surprise, 0.9f, 0.5f, 0.95f, 0.60f, {101}));
    addNode(makeNode(101, "Astonishment", EmotionCategory::Surprise, 0.85f, 0.4f, 0.9f, 0.58f, {100, 102}));
    addNode(makeNode(102, "Shock", EmotionCategory::Surprise, 0.8f, -0.1f, 0.85f, 0.45f, {101, 40}));
    addNode(makeNode(103, "Startle", EmotionCategory::Surprise, 0.7f, 0.0f, 0.8f, 0.48f, {102}));
    addNode(makeNode(104, "Wonder", EmotionCategory::Surprise, 0.6f, 0.6f, 0.6f, 0.65f, {100, 92}));

    // DISGUST CLUSTER (Negative valence, rejection)
    addNode(makeNode(110, "Revulsion", EmotionCategory::Disgust, 0.9f, -0.8f, 0.7f, 0.75f, {111}));
    addNode(makeNode(111, "Loathing", EmotionCategory::Disgust, 0.85f, -0.85f, 0.6f, 0.72f, {110, 112}));
    addNode(makeNode(112, "Contempt", EmotionCategory::Disgust, 0.7f, -0.6f, 0.5f, 0.70f, {111, 26}));
    addNode(makeNode(113, "Distaste", EmotionCategory::Disgust, 0.4f, -0.3f, 0.4f, 0.60f, {112}));
    addNode(makeNode(114, "Disapproval", EmotionCategory::Disgust, 0.5f, -0.4f, 0.45f, 0.65f, {112, 113}));

    // COMPOUND / COMPLEX EMOTIONS
    addNode(makeNode(200, "Bittersweetness", EmotionCategory::Sadness, 0.6f, 0.1f, 0.4f, 0.40f, {2, 62, 6}));
    addNode(makeNode(201, "Nostalgia", EmotionCategory::Sadness, 0.5f, 0.2f, 0.3f, 0.42f, {5, 6, 200}));
    addNode(makeNode(202, "Catharsis", EmotionCategory::Joy, 0.8f, 0.3f, 0.7f, 0.70f, {1, 67}));
    addNode(makeNode(203, "Yearning", EmotionCategory::Anticipation, 0.7f, -0.2f, 0.6f, 0.38f, {6, 91}));
    addNode(makeNode(204, "Ambivalence", EmotionCategory::Surprise, 0.5f, 0.0f, 0.5f, 0.50f, {42, 69}));
    addNode(makeNode(205, "Melancholic Hope", EmotionCategory::Sadness, 0.6f, 0.1f, 0.5f, 0.45f, {2, 69}));
    addNode(makeNode(206, "Tender Grief", EmotionCategory::Sadness, 0.75f, -0.3f, 0.35f, 0.30f, {1, 83}));
    addNode(makeNode(207, "Defiant Joy", EmotionCategory::Joy, 0.8f, 0.4f, 0.8f, 0.80f, {62, 20}));
    addNode(makeNode(208, "Quiet Rage", EmotionCategory::Anger, 0.7f, -0.6f, 0.3f, 0.75f, {20, 9}));
    addNode(makeNode(209, "Anxious Hope", EmotionCategory::Fear, 0.6f, 0.1f, 0.65f, 0.28f, {42, 69}));
}

std::optional<EmotionNode> EmotionThesaurus::findById(int id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(id);
    if (it != nodes_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<EmotionNode> EmotionThesaurus::findByName(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string lowerName = name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

    auto it = nameIndex_.find(lowerName);
    if (it != nameIndex_.end()) {
        auto nodeIt = nodes_.find(it->second);
        if (nodeIt != nodes_.end()) {
            return nodeIt->second;
        }
    }
    return std::nullopt;
}

std::vector<EmotionNode> EmotionThesaurus::findByCategory(EmotionCategory category) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<EmotionNode> results;
    for (const auto& [id, node] : nodes_) {
        if (node.categoryEnum == category) {
            results.push_back(node);
        }
    }
    return results;
}

float EmotionThesaurus::distance(const EmotionNode& node, float v, float ar, float i) {
    float dv = node.valence - v;
    float da = node.arousal - ar;
    float di = node.intensity - i;
    return std::sqrt(dv * dv + da * da + di * di);
}

float EmotionThesaurus::distanceVAD(const EmotionNode& node, float v, float ar, float d) {
    float dv = node.valence - v;
    float da = node.arousal - ar;
    float dd = node.dominance - d;
    return std::sqrt(dv * dv + da * da + dd * dd);
}

EmotionNode EmotionThesaurus::findNearest(float valence, float arousal, float intensity) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (nodes_.empty()) {
        // Return a default emotion if thesaurus is empty
        EmotionNode defaultNode;
        defaultNode.id = 0;
        defaultNode.name = "Neutral";
        defaultNode.categoryEnum = EmotionCategory::Joy;
        defaultNode.category = "Joy";
        defaultNode.intensity = 0.5f;
        defaultNode.valence = 0.0f;
        defaultNode.arousal = 0.5f;
        defaultNode.dominance = 0.5f;
        return defaultNode;
        defaultNode.id = 0;
        defaultNode.name = "Neutral";
        defaultNode.categoryEnum = EmotionCategory::Joy;
        defaultNode.category = "Joy";
        defaultNode.intensity = 0.5f;
        defaultNode.valence = 0.0f;
        defaultNode.arousal = 0.5f;
        defaultNode.dominance = 0.5f;
        return defaultNode;
    }

    EmotionNode nearest = nodes_.begin()->second;
    float minDist = distance(nearest, valence, arousal, intensity);

    for (const auto& [id, node] : nodes_) {
        float d = distance(node, valence, arousal, intensity);
        if (d < minDist) {
            minDist = d;
            nearest = node;
        }
    }
    return nearest;
}

EmotionNode EmotionThesaurus::findNearestVAD(float valence, float arousal, float dominance) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (nodes_.empty()) {
        // Return a default emotion if thesaurus is empty
        EmotionNode defaultNode;
        defaultNode.id = 0;
        defaultNode.name = "Neutral";
        defaultNode.categoryEnum = EmotionCategory::Joy;
        defaultNode.category = "Joy";
        defaultNode.intensity = 0.5f;
        defaultNode.valence = 0.0f;
        defaultNode.arousal = 0.5f;
        defaultNode.dominance = 0.5f;
        return defaultNode;
    }

    EmotionNode nearest = nodes_.begin()->second;
    float minDist = distanceVAD(nearest, valence, arousal, dominance);

    for (const auto& [id, node] : nodes_) {
        float d = distanceVAD(node, valence, arousal, dominance);
        if (d < minDist) {
            minDist = d;
            nearest = node;
        }
    }
    return nearest;
}

std::vector<EmotionNode> EmotionThesaurus::findRelated(int emotionId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<EmotionNode> results;
    auto it = nodes_.find(emotionId);
    if (it == nodes_.end()) return results;

    for (int relatedId : it->second.relatedEmotions) {
        auto relatedIt = nodes_.find(relatedId);
        if (relatedIt != nodes_.end()) {
            results.push_back(relatedIt->second);
        }
    }
    return results;
}

std::string EmotionThesaurus::suggestMode(const EmotionNode& emotion) const {
    if (emotion.valence < -0.7f) {
        return emotion.arousal > 0.6f ? "phrygian" : "minor";
    } else if (emotion.valence < -0.3f) {
        return emotion.arousal > 0.5f ? "dorian" : "aeolian";
    } else if (emotion.valence < 0.3f) {
        return emotion.arousal > 0.5f ? "mixolydian" : "dorian";
    } else if (emotion.valence < 0.7f) {
        return emotion.arousal > 0.6f ? "lydian" : "major";
    } else {
        return emotion.arousal > 0.7f ? "lydian" : "ionian";
    }
}

float EmotionThesaurus::suggestTempoModifier(const EmotionNode& emotion) const {
    float arousalFactor = 0.5f + (emotion.arousal * 1.0f);
    float intensityBoost = emotion.intensity * 0.3f;
    return std::clamp(arousalFactor + intensityBoost, 0.5f, 2.0f);
}

float EmotionThesaurus::suggestDynamicRange(const EmotionNode& emotion) const {
    float base = emotion.intensity;

    if (emotion.valence < -0.5f && emotion.arousal > 0.7f) {
        return std::min(1.0f, base + 0.3f);
    }

    if (emotion.valence > 0.5f && emotion.arousal < 0.3f) {
        return std::max(0.2f, base - 0.2f);
    }

    return base;
}

} // namespace kelly
