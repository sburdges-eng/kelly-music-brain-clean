#pragma once

/**
 * @file emotion_thesaurus.h
 * @brief 216-node emotion thesaurus with ML enhancement support
 * 
 * Implements the hierarchical emotion structure:
 * - 6 Base Emotions (Happy, Sad, Angry, Fear, Surprise, Disgust)
 * - 36 Sub-Emotions (6 per base)
 * - 216 Nodes (6 intensity levels per sub-emotion)
 * - 1296 Intensity Tiers (6 levels per node)
 * 
 * ML Enhancement:
 * - Optional 64-dim embeddings from EmotionRecognizer
 * - VAD coordinates for node lookup
 * - Musical attributes per node
 */

#include <string>
#include <vector>
#include <array>
#include <map>
#include <optional>
#include <memory>
#include <functional>

namespace kelly {

/**
 * @brief VAD (Valence-Arousal-Dominance) coordinates
 */
struct VADCoordinates {
    float valence = 0.0f;    ///< Negative (-1) to Positive (+1)
    float arousal = 0.0f;    ///< Calm (-1) to Excited (+1)
    float dominance = 0.0f;  ///< Submissive (-1) to Dominant (+1)
    float intensity = 0.5f;  ///< Subtle (0) to Extreme (1)
    
    float distanceTo(const VADCoordinates& other) const {
        float dv = valence - other.valence;
        float da = arousal - other.arousal;
        float dd = dominance - other.dominance;
        float di = intensity - other.intensity;
        return std::sqrt(dv*dv + da*da + dd*dd + di*di);
    }
};

/**
 * @brief Musical attributes derived from emotion
 */
struct MusicalAttributes {
    bool isMajorMode = true;           ///< Major (true) vs Minor (false)
    float tempoMultiplier = 1.0f;      ///< Relative tempo adjustment (0.5-1.5)
    float dynamicsLevel = 0.5f;        ///< Soft (0) to Loud (1)
    float rhythmicDensity = 0.5f;      ///< Sparse (0) to Dense (1)
    float harmonicComplexity = 0.5f;   ///< Simple (0) to Complex (1)
    float melodicContour = 0.0f;       ///< Descending (-1) to Ascending (+1)
    int suggestedOctave = 4;           ///< Base octave for melody
    std::string preferredScale = "major"; ///< Scale suggestion
};

/**
 * @brief Emotion node in the 216-node thesaurus
 */
struct EmotionThesaurusNode {
    // Core identity
    int id = 0;                         ///< Node ID (0-215)
    std::string name;                   ///< Human-readable name (e.g., "Joyful (Strong)")
    std::string category;               ///< Base emotion (e.g., "Happy")
    std::string subcategory;            ///< Sub-emotion (e.g., "Joyful")
    int intensityLevel = 0;             ///< Intensity tier (0-5)
    
    // VAD coordinates
    VADCoordinates vad;
    
    // Relationships (node IDs)
    std::vector<int> relatedEmotions;   ///< Similar emotions
    std::vector<int> oppositeEmotions;  ///< Contrasting emotions
    std::vector<int> intensityLadder;   ///< Same emotion at different intensities
    
    // Musical mapping
    MusicalAttributes musicalAttributes;
    
    // ML enhancement (optional)
    std::optional<std::vector<float>> mlEmbedding;  ///< 64-dim from EmotionRecognizer
    std::optional<float> mlConfidence;               ///< Model confidence (0-1)
    std::map<std::string, float> mlFeatures;         ///< Additional ML-derived features
};

/**
 * @brief Intensity tier descriptions
 */
constexpr std::array<const char*, 6> INTENSITY_LABELS = {
    "Subtle", "Mild", "Moderate", "Strong", "Intense", "Extreme"
};

/**
 * @brief Base emotion categories
 */
constexpr std::array<const char*, 6> BASE_EMOTIONS = {
    "Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"
};

/**
 * @brief 216-node emotion thesaurus
 * 
 * Provides:
 * - Node lookup by ID, name, or VAD coordinates
 * - Node relationships for emotional journeys
 * - Musical attribute derivation
 * - ML embedding integration
 */
class EmotionThesaurus {
public:
    EmotionThesaurus();
    ~EmotionThesaurus() = default;
    
    // Initialization
    /**
     * @brief Initialize with default 216-node structure
     */
    void initialize();
    
    /**
     * @brief Load thesaurus from JSON file
     * @param filepath Path to JSON file
     * @return true if loaded successfully
     */
    bool loadFromFile(const std::string& filepath);
    
    /**
     * @brief Save thesaurus to JSON file
     * @param filepath Path to output file
     * @return true if saved successfully
     */
    bool saveToFile(const std::string& filepath) const;
    
    // Node access
    /**
     * @brief Get node by ID
     * @param id Node ID (0-215)
     * @return Node pointer or nullptr
     */
    const EmotionThesaurusNode* getNode(int id) const;
    
    /**
     * @brief Get mutable node by ID
     */
    EmotionThesaurusNode* getNodeMutable(int id);
    
    /**
     * @brief Get node by name (case-insensitive)
     * @param name Full node name
     * @return Node pointer or nullptr
     */
    const EmotionThesaurusNode* getNodeByName(const std::string& name) const;
    
    /**
     * @brief Find nearest node by VAD coordinates
     * @param vad Target VAD coordinates
     * @return Nearest node
     */
    const EmotionThesaurusNode* findNearestNode(const VADCoordinates& vad) const;
    
    /**
     * @brief Find top-k nearest nodes by VAD
     * @param vad Target coordinates
     * @param k Number of nodes to return
     * @return Vector of (node, distance) pairs
     */
    std::vector<std::pair<const EmotionThesaurusNode*, float>> findNearestNodes(
        const VADCoordinates& vad, int k = 5) const;
    
    /**
     * @brief Get all nodes in a category
     * @param category Base emotion name
     * @return Vector of nodes
     */
    std::vector<const EmotionThesaurusNode*> getNodesInCategory(const std::string& category) const;
    
    /**
     * @brief Get all nodes with specific sub-emotion
     * @param subcategory Sub-emotion name
     * @return Vector of nodes (one per intensity)
     */
    std::vector<const EmotionThesaurusNode*> getNodesInSubcategory(const std::string& subcategory) const;
    
    // Relationships
    /**
     * @brief Get related emotions
     * @param nodeId Source node ID
     * @return Vector of related nodes
     */
    std::vector<const EmotionThesaurusNode*> getRelatedNodes(int nodeId) const;
    
    /**
     * @brief Get opposite emotions
     * @param nodeId Source node ID
     * @return Vector of contrasting nodes
     */
    std::vector<const EmotionThesaurusNode*> getOppositeNodes(int nodeId) const;
    
    /**
     * @brief Get intensity ladder (same emotion, different intensities)
     * @param nodeId Source node ID
     * @return Vector of nodes ordered by intensity
     */
    std::vector<const EmotionThesaurusNode*> getIntensityLadder(int nodeId) const;
    
    // ML Integration
    /**
     * @brief Update node's ML embedding
     * @param nodeId Node ID
     * @param embedding 64-dim embedding vector
     * @param confidence Model confidence
     */
    void updateNodeEmbedding(int nodeId, 
                              const std::vector<float>& embedding,
                              float confidence);
    
    /**
     * @brief Find node from ML embedding
     * @param embedding 64-dim embedding from EmotionRecognizer
     * @return Nearest node based on embedding-to-VAD mapping
     */
    const EmotionThesaurusNode* findNodeFromEmbedding(const std::vector<float>& embedding) const;
    
    /**
     * @brief Convert node to ML input format
     * @param nodeId Node ID
     * @return 64-dim feature vector
     */
    std::vector<float> nodeToMLInput(int nodeId) const;
    
    /**
     * @brief Get node context for ML models
     * @param nodeId Node ID
     * @return 128-dim context vector including related nodes
     */
    std::vector<float> getNodeContext(int nodeId) const;
    
    // Utility
    /**
     * @brief Get total node count
     */
    size_t getNodeCount() const { return nodes_.size(); }
    
    /**
     * @brief Get all nodes
     */
    const std::vector<EmotionThesaurusNode>& getAllNodes() const { return nodes_; }
    
    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Interpolate between two nodes
     * @param node1Id First node ID
     * @param node2Id Second node ID
     * @param t Interpolation factor (0 = node1, 1 = node2)
     * @return Interpolated VAD and musical attributes
     */
    std::pair<VADCoordinates, MusicalAttributes> interpolate(int node1Id, int node2Id, float t) const;

private:
    std::vector<EmotionThesaurusNode> nodes_;
    std::map<std::string, int> nameToIdMap_;
    bool initialized_ = false;
    
    // Initialization helpers
    void initializeDefaultThesaurus();
    void calculateMusicalAttributes(EmotionThesaurusNode& node);
    void setupRelationships();
    
    // ML helpers
    VADCoordinates embeddingToVAD(const std::vector<float>& embedding) const;
    std::vector<float> vadToEmbedding(const VADCoordinates& vad) const;
};

/**
 * @brief Emotion journey - sequence of emotional transitions
 */
class EmotionJourney {
public:
    struct Waypoint {
        int nodeId;
        float duration;  // in beats
        float transitionTime;  // in beats
    };
    
    EmotionJourney(const EmotionThesaurus& thesaurus);
    
    /**
     * @brief Add waypoint to journey
     */
    void addWaypoint(int nodeId, float duration, float transitionTime = 1.0f);
    
    /**
     * @brief Get interpolated VAD at time position
     */
    VADCoordinates getVADAtTime(float timeBeats) const;
    
    /**
     * @brief Get interpolated musical attributes at time
     */
    MusicalAttributes getAttributesAtTime(float timeBeats) const;
    
    /**
     * @brief Get total journey duration
     */
    float getTotalDuration() const;
    
    /**
     * @brief Clear all waypoints
     */
    void clear();
    
    /**
     * @brief Get waypoint count
     */
    size_t getWaypointCount() const { return waypoints_.size(); }

private:
    const EmotionThesaurus& thesaurus_;
    std::vector<Waypoint> waypoints_;
    
    int findWaypointAtTime(float timeBeats) const;
};

} // namespace kelly
