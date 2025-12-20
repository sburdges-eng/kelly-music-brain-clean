#pragma once
/**
 * IntentProcessor.h
 * 
 * Ported from Python: intent_processor.py
 * Implements three-phase therapeutic intent processing:
 *   Phase 1: Wound identification
 *   Phase 2: Emotional mapping  
 *   Phase 3: Musical rule-breaking for expression
 */

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <algorithm>
#include <cctype>
#include "EmotionMapper.h"

namespace kelly {

// =============================================================================
// WOUND (from Python Wound dataclass)
// =============================================================================

struct Wound {
    std::string description;
    float intensity = 0.7f;     // 0.0-1.0
    std::string source = "internal";  // "internal" or "external"
    double timestamp = 0.0;
    
    // Optional additional context
    std::string context;
    std::vector<std::string> triggers;
};

// =============================================================================
// RULE BREAK (from Python RuleBreak dataclass)
// =============================================================================

enum class RuleBreakType {
    Harmony,
    Rhythm,
    Dynamics,
    Structure,
    Voice_Leading,
    Texture,
    Range
};

struct RuleBreak {
    RuleBreakType type;
    float severity = 0.5f;      // 0.0-1.0
    std::string description;
    std::string emotionalJustification;
    
    // Musical impact parameters
    bool allowDissonance = false;
    bool allowParallelMotion = false;
    bool allowUnresolvedTensions = false;
    float clusterProbability = 0.0f;
    float syncopationLevel = 0.0f;
    bool irregularMeters = false;
    std::pair<int, int> velocityRange = {0, 127};
    bool suddenDynamicChanges = false;
    float restProbability = 0.0f;
    bool crossHandVoicing = false;
};

// =============================================================================
// EMOTION NODE (216-node thesaurus entry)
// =============================================================================

struct EmotionNode {
    int id;
    std::string name;
    EmotionCategory category;
    float valence;
    float arousal;
    float intensity;
    
    // Musical attributes (from Python emotion.musical_attributes)
    float tempoModifier = 1.0f;
    std::string preferredMode = "Aeolian";
    float dynamicsBase = 0.7f;
    float harmonicComplexity = 0.5f;
    
    // Related emotions for transitions
    std::vector<int> nearbyEmotionIds;
    
    // Technical descriptions (for UI)
    std::string technicalDescription;
    std::vector<std::string> musicalTechniques;
};

// =============================================================================
// INTENT RESULT (complete processing output)
// =============================================================================

struct IntentResult {
    Wound wound;
    EmotionNode emotion;
    std::vector<RuleBreak> ruleBreaks;
    MusicalParameters musicalParams;
    
    // Summary for UI
    std::string summary() const {
        std::string result = "Wound: " + wound.description + "\n";
        result += "Emotion: " + emotion.name + " (";
        result += "V:" + std::to_string(emotion.valence).substr(0,4) + ", ";
        result += "A:" + std::to_string(emotion.arousal).substr(0,4) + ", ";
        result += "I:" + std::to_string(emotion.intensity).substr(0,4) + ")\n";
        result += "Rule Breaks: " + std::to_string(ruleBreaks.size()) + " applied\n";
        result += "Tempo: " + std::to_string(musicalParams.tempoSuggested) + " BPM\n";
        result += "Mode: " + musicalParams.modeSuggested;
        return result;
    }
};

// =============================================================================
// EMOTION THESAURUS (216-node system)
// =============================================================================

class EmotionThesaurus {
public:
    EmotionThesaurus() {
        initializeNodes();
    }
    
    const EmotionNode* findById(int id) const {
        auto it = nodesById_.find(id);
        return it != nodesById_.end() ? &it->second : nullptr;
    }
    
    const EmotionNode* findByName(const std::string& name) const {
        std::string lower = toLower(name);
        auto it = nodesByName_.find(lower);
        return it != nodesByName_.end() ? findById(it->second) : nullptr;
    }
    
    /**
     * Find nearest emotion by valence/arousal/intensity
     */
    const EmotionNode* findNearest(float valence, float arousal, float intensity) const {
        const EmotionNode* nearest = nullptr;
        float minDistance = std::numeric_limits<float>::max();
        
        for (const auto& pair : nodesById_) {
            const EmotionNode& node = pair.second;
            float dv = node.valence - valence;
            float da = node.arousal - arousal;
            float di = node.intensity - intensity;
            float distance = std::sqrt(dv*dv + da*da + di*di);
            
            if (distance < minDistance) {
                minDistance = distance;
                nearest = &pair.second;
            }
        }
        
        return nearest;
    }
    
    /**
     * Get emotions within a certain distance
     */
    std::vector<const EmotionNode*> getNearby(
        float valence, float arousal, float intensity,
        float threshold = 0.5f
    ) const {
        std::vector<const EmotionNode*> result;
        
        for (const auto& pair : nodesById_) {
            const EmotionNode& node = pair.second;
            float dv = node.valence - valence;
            float da = node.arousal - arousal;
            float di = node.intensity - intensity;
            float distance = std::sqrt(dv*dv + da*da + di*di);
            
            if (distance <= threshold) {
                result.push_back(&pair.second);
            }
        }
        
        return result;
    }
    
    /**
     * Get all emotions in a category
     */
    std::vector<const EmotionNode*> getByCategory(EmotionCategory category) const {
        std::vector<const EmotionNode*> result;
        for (const auto& pair : nodesById_) {
            if (pair.second.category == category) {
                result.push_back(&pair.second);
            }
        }
        return result;
    }
    
    size_t size() const { return nodesById_.size(); }

private:
    std::map<int, EmotionNode> nodesById_;
    std::map<std::string, int> nodesByName_;
    
    static std::string toLower(const std::string& s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
    void addNode(const EmotionNode& node) {
        nodesById_[node.id] = node;
        nodesByName_[toLower(node.name)] = node.id;
    }
    
    void initializeNodes() {
        // =====================================================================
        // JOY CLUSTER (IDs 60-69) - Positive valence, variable arousal
        // =====================================================================
        addNode({60, "Ecstasy", EmotionCategory::Joy, 1.0f, 1.0f, 1.0f,
                 1.3f, "Lydian", 0.9f, 0.3f, {61, 62},
                 "Peak positive state", {"Major 7ths", "Rising arpeggios"}});
        
        addNode({61, "Joy", EmotionCategory::Joy, 0.9f, 0.8f, 0.7f,
                 1.2f, "Ionian", 0.8f, 0.4f, {60, 62, 63},
                 "Pure happiness", {"Major triads", "Bright timbres"}});
        
        addNode({62, "Happiness", EmotionCategory::Joy, 0.8f, 0.7f, 0.6f,
                 1.1f, "Mixolydian", 0.75f, 0.4f, {61, 63},
                 "General positivity", {"Open voicings", "Legato"}});
        
        addNode({63, "Contentment", EmotionCategory::Joy, 0.6f, 0.3f, 0.4f,
                 0.9f, "Ionian", 0.6f, 0.3f, {62, 64},
                 "Peaceful satisfaction", {"Sustained pads", "Simple harmony"}});
        
        addNode({64, "Serenity", EmotionCategory::Joy, 0.7f, 0.2f, 0.3f,
                 0.85f, "Lydian", 0.5f, 0.2f, {63, 65},
                 "Deep peace", {"Long tones", "Minimal movement"}});
        
        addNode({65, "Hope", EmotionCategory::Joy, 0.5f, 0.5f, 0.6f,
                 1.0f, "Ionian", 0.7f, 0.5f, {62, 66},
                 "Optimistic anticipation", {"Rising phrases", "Suspensions resolving"}});
        
        addNode({66, "Gratitude", EmotionCategory::Joy, 0.7f, 0.4f, 0.5f,
                 0.95f, "Ionian", 0.65f, 0.4f, {63, 65},
                 "Thankful warmth", {"Warm chords", "Gentle dynamics"}});
        
        addNode({67, "Elation", EmotionCategory::Joy, 0.85f, 0.9f, 0.8f,
                 1.25f, "Lydian", 0.85f, 0.4f, {60, 61},
                 "Excited joy", {"Staccato", "Syncopation"}});
        
        addNode({68, "Amusement", EmotionCategory::Joy, 0.6f, 0.6f, 0.5f,
                 1.1f, "Mixolydian", 0.7f, 0.5f, {62, 67},
                 "Lighthearted fun", {"Playful rhythms", "Grace notes"}});
        
        addNode({69, "Pride", EmotionCategory::Joy, 0.7f, 0.6f, 0.7f,
                 1.05f, "Ionian", 0.8f, 0.5f, {61, 65},
                 "Confident satisfaction", {"Strong bass", "Full chords"}});
        
        // =====================================================================
        // GRIEF/SADNESS CLUSTER (IDs 1-10) - Negative valence, low arousal
        // =====================================================================
        addNode({1, "Grief", EmotionCategory::Sadness, -0.9f, 0.3f, 0.9f,
                 0.7f, "Aeolian", 0.5f, 0.6f, {2, 3},
                 "Deep loss processing", {"Minor 2nds", "Descending lines", "Modal mixture"}});
        
        addNode({2, "Sorrow", EmotionCategory::Sadness, -0.8f, 0.25f, 0.8f,
                 0.75f, "Aeolian", 0.45f, 0.5f, {1, 3},
                 "Profound sadness", {"Suspended chords", "Slow resolution"}});
        
        addNode({3, "Melancholy", EmotionCategory::Sadness, -0.5f, 0.3f, 0.5f,
                 0.85f, "Dorian", 0.5f, 0.5f, {1, 4, 5},
                 "Bittersweet reflection", {"Add9 chords", "6ths"}});
        
        addNode({4, "Sadness", EmotionCategory::Sadness, -0.6f, 0.35f, 0.6f,
                 0.8f, "Aeolian", 0.55f, 0.45f, {3, 5},
                 "General unhappiness", {"Minor triads", "Legato"}});
        
        addNode({5, "Despair", EmotionCategory::Sadness, -1.0f, 0.2f, 1.0f,
                 0.6f, "Phrygian", 0.4f, 0.7f, {1, 6},
                 "Hopelessness", {"Tritones", "Unresolved dissonance"}});
        
        addNode({6, "Loneliness", EmotionCategory::Sadness, -0.6f, 0.2f, 0.6f,
                 0.75f, "Aeolian", 0.4f, 0.4f, {4, 7},
                 "Isolated emptiness", {"Sparse texture", "Solo voice"}});
        
        addNode({7, "Disappointment", EmotionCategory::Sadness, -0.5f, 0.4f, 0.5f,
                 0.9f, "Dorian", 0.5f, 0.4f, {3, 8},
                 "Unmet expectations", {"Deceptive cadences", "Sighing motifs"}});
        
        addNode({8, "Nostalgia", EmotionCategory::Sadness, -0.3f, 0.3f, 0.5f,
                 0.85f, "Dorian", 0.55f, 0.5f, {3, 9},
                 "Bittersweet memory", {"Major-minor mixture", "Vintage sounds"}});
        
        addNode({9, "Regret", EmotionCategory::Sadness, -0.6f, 0.35f, 0.6f,
                 0.8f, "Aeolian", 0.5f, 0.5f, {7, 10},
                 "Wishing for change", {"Unresolved phrases", "Repeated patterns"}});
        
        addNode({10, "Emptiness", EmotionCategory::Sadness, -0.7f, 0.15f, 0.7f,
                  0.65f, "Locrian", 0.35f, 0.3f, {5, 6},
                  "Void state", {"Minimal notes", "Silence as instrument"}});
        
        // =====================================================================
        // ANGER CLUSTER (IDs 20-29) - Negative valence, high arousal
        // =====================================================================
        addNode({20, "Rage", EmotionCategory::Anger, -0.9f, 1.0f, 1.0f,
                 1.4f, "Phrygian", 0.95f, 0.7f, {21, 22},
                 "Explosive fury", {"Parallel 5ths", "Clusters", "Fortissimo"}});
        
        addNode({21, "Anger", EmotionCategory::Anger, -0.7f, 0.85f, 0.8f,
                 1.3f, "Aeolian", 0.85f, 0.6f, {20, 22, 23},
                 "Active hostility", {"Sharp accents", "Staccato"}});
        
        addNode({22, "Frustration", EmotionCategory::Anger, -0.6f, 0.7f, 0.7f,
                 1.2f, "Dorian", 0.75f, 0.5f, {21, 24},
                 "Blocked goals", {"Repeated motifs", "Crescendos"}});
        
        addNode({23, "Irritation", EmotionCategory::Anger, -0.4f, 0.6f, 0.5f,
                 1.1f, "Mixolydian", 0.65f, 0.4f, {22, 24},
                 "Minor annoyance", {"Dissonant ornaments", "Edgy timbre"}});
        
        addNode({24, "Resentment", EmotionCategory::Anger, -0.6f, 0.5f, 0.6f,
                 1.0f, "Aeolian", 0.6f, 0.5f, {22, 25},
                 "Held grievance", {"Ostinato", "Building tension"}});
        
        addNode({25, "Defiance", EmotionCategory::Anger, -0.3f, 0.8f, 0.7f,
                 1.25f, "Mixolydian", 0.8f, 0.5f, {21, 26},
                 "Rebellious resistance", {"Strong downbeats", "Modal mixture"}});
        
        addNode({26, "Contempt", EmotionCategory::Anger, -0.5f, 0.4f, 0.6f,
                 0.95f, "Locrian", 0.55f, 0.6f, {24, 27},
                 "Superior disdain", {"Tritones", "Chromatic motion"}});
        
        addNode({27, "Bitterness", EmotionCategory::Anger, -0.7f, 0.4f, 0.7f,
                 0.9f, "Phrygian", 0.5f, 0.6f, {24, 26},
                 "Lasting resentment", {"Minor 2nds", "Slow dissonance"}});
        
        // =====================================================================
        // FEAR CLUSTER (IDs 40-49) - Negative valence, high arousal
        // =====================================================================
        addNode({40, "Terror", EmotionCategory::Fear, -1.0f, 1.0f, 1.0f,
                 1.35f, "Locrian", 0.9f, 0.8f, {41, 42},
                 "Overwhelming dread", {"Tritones", "Sudden dynamics", "Tremolo"}});
        
        addNode({41, "Fear", EmotionCategory::Fear, -0.8f, 0.9f, 0.8f,
                 1.2f, "Phrygian", 0.8f, 0.7f, {40, 42, 43},
                 "Active threat response", {"Chromatic lines", "Staccato"}});
        
        addNode({42, "Panic", EmotionCategory::Fear, -0.9f, 1.0f, 0.9f,
                 1.4f, "Locrian", 0.95f, 0.75f, {40, 41},
                 "Overwhelmed flight", {"Fast arpeggios", "Clusters"}});
        
        addNode({43, "Anxiety", EmotionCategory::Fear, -0.6f, 0.7f, 0.7f,
                 1.15f, "Dorian", 0.7f, 0.6f, {41, 44},
                 "Anticipated threat", {"Ostinato", "Unresolved suspensions"}});
        
        addNode({44, "Worry", EmotionCategory::Fear, -0.5f, 0.6f, 0.5f,
                 1.0f, "Aeolian", 0.6f, 0.5f, {43, 45},
                 "Persistent concern", {"Repeated figures", "Minor 7ths"}});
        
        addNode({45, "Unease", EmotionCategory::Fear, -0.4f, 0.5f, 0.4f,
                 1.0f, "Dorian", 0.55f, 0.45f, {44, 46},
                 "Background discomfort", {"Subtle dissonance", "Unstable bass"}});
        
        addNode({46, "Dread", EmotionCategory::Fear, -0.8f, 0.6f, 0.8f,
                 0.85f, "Phrygian", 0.6f, 0.7f, {40, 47},
                 "Anticipated doom", {"Descending bass", "Slow crescendo"}});
        
        addNode({47, "Nervousness", EmotionCategory::Fear, -0.4f, 0.7f, 0.5f,
                 1.1f, "Mixolydian", 0.65f, 0.45f, {43, 45},
                 "Restless apprehension", {"Syncopation", "Quick ornaments"}});
        
        // =====================================================================
        // TRUST CLUSTER (IDs 80-89)
        // =====================================================================
        addNode({80, "Admiration", EmotionCategory::Trust, 0.7f, 0.5f, 0.6f,
                 1.0f, "Ionian", 0.7f, 0.4f, {81, 82},
                 "Respectful appreciation", {"Full chords", "Rising lines"}});
        
        addNode({81, "Trust", EmotionCategory::Trust, 0.6f, 0.4f, 0.5f,
                 0.95f, "Ionian", 0.6f, 0.35f, {80, 82, 83},
                 "Secure confidence", {"Consonant intervals", "Steady rhythm"}});
        
        addNode({82, "Acceptance", EmotionCategory::Trust, 0.5f, 0.3f, 0.4f,
                 0.9f, "Mixolydian", 0.55f, 0.3f, {81, 84},
                 "Open receiving", {"Open voicings", "Gentle dynamics"}});
        
        addNode({83, "Love", EmotionCategory::Trust, 0.9f, 0.6f, 0.8f,
                 0.95f, "Lydian", 0.75f, 0.5f, {80, 84},
                 "Deep connection", {"Rich harmony", "Legato phrasing"}});
        
        addNode({84, "Compassion", EmotionCategory::Trust, 0.6f, 0.4f, 0.6f,
                 0.9f, "Dorian", 0.6f, 0.45f, {82, 83, 85},
                 "Empathetic care", {"Warm timbres", "Gentle movement"}});
        
        addNode({85, "Safety", EmotionCategory::Trust, 0.5f, 0.2f, 0.4f,
                 0.85f, "Ionian", 0.5f, 0.25f, {82, 86},
                 "Protected security", {"Pedal tones", "Consonance"}});
        
        addNode({86, "Belonging", EmotionCategory::Trust, 0.6f, 0.4f, 0.5f,
                 0.9f, "Ionian", 0.6f, 0.4f, {84, 85},
                 "Connected inclusion", {"Unison passages", "Ensemble texture"}});
        
        // =====================================================================
        // ANTICIPATION CLUSTER (IDs 90-99)
        // =====================================================================
        addNode({90, "Anticipation", EmotionCategory::Anticipation, 0.3f, 0.7f, 0.6f,
                 1.1f, "Mixolydian", 0.7f, 0.5f, {91, 92},
                 "Eager expectation", {"Dominant prolongation", "Building texture"}});
        
        addNode({91, "Excitement", EmotionCategory::Anticipation, 0.6f, 0.9f, 0.8f,
                 1.25f, "Lydian", 0.85f, 0.5f, {90, 67},
                 "Thrilled energy", {"Syncopation", "Rising sequences"}});
        
        addNode({92, "Interest", EmotionCategory::Anticipation, 0.3f, 0.5f, 0.4f,
                 1.0f, "Dorian", 0.6f, 0.45f, {90, 93},
                 "Engaged curiosity", {"Modal exploration", "Question phrases"}});
        
        addNode({93, "Vigilance", EmotionCategory::Anticipation, 0.2f, 0.7f, 0.6f,
                 1.05f, "Mixolydian", 0.7f, 0.5f, {90, 94},
                 "Alert readiness", {"Staccato", "Clear articulation"}});
        
        addNode({94, "Curiosity", EmotionCategory::Anticipation, 0.4f, 0.6f, 0.5f,
                 1.05f, "Lydian", 0.65f, 0.5f, {92, 93},
                 "Exploratory wonder", {"Unexpected intervals", "Modal shifts"}});
        
        // =====================================================================
        // SURPRISE CLUSTER (IDs 100-109)
        // =====================================================================
        addNode({100, "Amazement", EmotionCategory::Surprise, 0.5f, 0.9f, 0.9f,
                  1.2f, "Lydian", 0.85f, 0.6f, {101, 102},
                  "Overwhelmed wonder", {"Sudden modulations", "Wide intervals"}});
        
        addNode({101, "Surprise", EmotionCategory::Surprise, 0.2f, 0.8f, 0.7f,
                  1.15f, "Mixolydian", 0.75f, 0.55f, {100, 102, 103},
                  "Unexpected reaction", {"Sudden dynamics", "Sforzando"}});
        
        addNode({102, "Shock", EmotionCategory::Surprise, -0.3f, 0.95f, 0.9f,
                  1.3f, "Phrygian", 0.9f, 0.7f, {100, 101},
                  "Jarring disruption", {"Cluster chords", "Silence"}});
        
        addNode({103, "Confusion", EmotionCategory::Surprise, -0.2f, 0.6f, 0.5f,
                  1.0f, "Locrian", 0.6f, 0.6f, {101, 104},
                  "Disoriented uncertainty", {"Chromatic wandering", "Irregular meter"}});
        
        addNode({104, "Distraction", EmotionCategory::Surprise, 0.0f, 0.5f, 0.3f,
                  1.05f, "Mixolydian", 0.55f, 0.45f, {103},
                  "Attention diverted", {"Fragmentary phrases", "Interruptions"}});
        
        // =====================================================================
        // DISGUST CLUSTER (IDs 110-119)
        // =====================================================================
        addNode({110, "Loathing", EmotionCategory::Disgust, -0.9f, 0.6f, 0.9f,
                  1.1f, "Locrian", 0.7f, 0.7f, {111, 112},
                  "Intense aversion", {"Harsh timbres", "Grating intervals"}});
        
        addNode({111, "Disgust", EmotionCategory::Disgust, -0.7f, 0.5f, 0.7f,
                  1.0f, "Phrygian", 0.6f, 0.6f, {110, 112, 113},
                  "Strong rejection", {"Tritones", "Rough texture"}});
        
        addNode({112, "Revulsion", EmotionCategory::Disgust, -0.8f, 0.7f, 0.8f,
                  1.15f, "Locrian", 0.75f, 0.7f, {110, 111},
                  "Physical rejection", {"Cluster dissonance", "Abrupt phrases"}});
        
        addNode({113, "Aversion", EmotionCategory::Disgust, -0.5f, 0.4f, 0.5f,
                  0.95f, "Phrygian", 0.55f, 0.5f, {111, 114},
                  "Avoidance response", {"Chromatic movement", "Unstable harmony"}});
        
        addNode({114, "Contempt", EmotionCategory::Disgust, -0.6f, 0.3f, 0.6f,
                  0.9f, "Locrian", 0.5f, 0.55f, {111, 113},
                  "Superior rejection", {"Cold timbre", "Sparse texture"}});
        
        // =====================================================================
        // COMPLEX EMOTIONS (IDs 200-220)
        // =====================================================================
        addNode({200, "Bittersweet", EmotionCategory::Complex, 0.0f, 0.4f, 0.6f,
                  0.9f, "Dorian", 0.55f, 0.6f, {3, 65},
                  "Joy and sorrow mixed", {"Major-minor shifts", "Suspensions"}});
        
        addNode({201, "Catharsis", EmotionCategory::Complex, 0.3f, 0.7f, 0.9f,
                  1.1f, "Mixolydian", 0.8f, 0.65f, {1, 60},
                  "Emotional release", {"Building to resolution", "Dynamic arc"}});
        
        addNode({202, "Awe", EmotionCategory::Complex, 0.5f, 0.6f, 0.8f,
                  0.95f, "Lydian", 0.7f, 0.55f, {100, 64},
                  "Transcendent wonder", {"Wide voicings", "Sustained harmony"}});
        
        addNode({203, "Yearning", EmotionCategory::Complex, -0.2f, 0.5f, 0.7f,
                  0.9f, "Dorian", 0.6f, 0.55f, {8, 65},
                  "Intense longing", {"Unresolved 7ths", "Rising phrases"}});
        
        addNode({204, "Tenderness", EmotionCategory::Complex, 0.6f, 0.3f, 0.5f,
                  0.85f, "Ionian", 0.5f, 0.4f, {83, 84},
                  "Gentle affection", {"Soft dynamics", "Legato"}});
        
        addNode({205, "Vulnerability", EmotionCategory::Complex, -0.1f, 0.3f, 0.7f,
                  0.8f, "Aeolian", 0.45f, 0.5f, {6, 85},
                  "Exposed openness", {"Sparse texture", "Fragile phrases"}});
        
        addNode({206, "Resilience", EmotionCategory::Complex, 0.4f, 0.6f, 0.7f,
                  1.0f, "Mixolydian", 0.7f, 0.5f, {65, 25},
                  "Strengthened recovery", {"Steady pulse", "Modal strength"}});
        
        addNode({207, "Wistfulness", EmotionCategory::Complex, -0.2f, 0.3f, 0.5f,
                  0.85f, "Dorian", 0.5f, 0.5f, {8, 3},
                  "Gentle longing", {"6/9 chords", "Gentle syncopation"}});
        
        addNode({208, "Overwhelm", EmotionCategory::Complex, -0.4f, 0.8f, 0.9f,
                  1.2f, "Aeolian", 0.85f, 0.7f, {42, 22},
                  "Flooded sensation", {"Dense texture", "Layered voices"}});
        
        addNode({209, "Peace", EmotionCategory::Complex, 0.6f, 0.1f, 0.3f,
                  0.75f, "Lydian", 0.4f, 0.2f, {64, 85},
                  "Deep tranquility", {"Long tones", "Perfect consonance"}});
    }
};

// =============================================================================
// INTENT PROCESSOR CLASS
// =============================================================================

class IntentProcessor {
public:
    IntentProcessor() : emotionMapper_() {}
    
    /**
     * Phase 1: Process wound and map to emotion.
     * Ported from Python process_wound()
     */
    const EmotionNode* processWound(const Wound& wound) {
        woundHistory_.push_back(wound);
        
        // Keyword-based emotion mapping (from Python)
        std::string desc = toLower(wound.description);
        
        // Check for grief/loss keywords
        if (contains(desc, "loss") || contains(desc, "grief") || 
            contains(desc, "death") || contains(desc, "gone") ||
            contains(desc, "miss") || contains(desc, "died")) {
            return thesaurus_.findByName("grief");
        }
        
        // Anger keywords
        if (contains(desc, "anger") || contains(desc, "rage") ||
            contains(desc, "furious") || contains(desc, "hate") ||
            contains(desc, "mad")) {
            return wound.intensity > 0.8f 
                ? thesaurus_.findByName("rage")
                : thesaurus_.findByName("anger");
        }
        
        // Fear/anxiety keywords
        if (contains(desc, "fear") || contains(desc, "anxiety") ||
            contains(desc, "scared") || contains(desc, "terrified") ||
            contains(desc, "panic") || contains(desc, "worry")) {
            return wound.intensity > 0.8f
                ? thesaurus_.findByName("terror")
                : thesaurus_.findByName("anxiety");
        }
        
        // Joy keywords
        if (contains(desc, "happy") || contains(desc, "joy") ||
            contains(desc, "excited") || contains(desc, "love") ||
            contains(desc, "wonderful")) {
            return thesaurus_.findByName("joy");
        }
        
        // Sadness keywords
        if (contains(desc, "sad") || contains(desc, "depressed") ||
            contains(desc, "hopeless") || contains(desc, "empty") ||
            contains(desc, "lonely")) {
            return thesaurus_.findByName("sadness");
        }
        
        // Complex emotions
        if (contains(desc, "bittersweet") || contains(desc, "nostalgic")) {
            return thesaurus_.findByName("bittersweet");
        }
        
        if (contains(desc, "overwhelm")) {
            return thesaurus_.findByName("overwhelm");
        }
        
        // Default to melancholy
        return thesaurus_.findByName("melancholy");
    }
    
    /**
     * Phase 2-3: Convert emotion to musical rule-breaks.
     * Ported from Python emotion_to_rule_breaks()
     */
    std::vector<RuleBreak> emotionToRuleBreaks(const EmotionNode& emotion) {
        std::vector<RuleBreak> ruleBreaks;
        
        // High intensity → Dynamics rule-breaks
        if (emotion.intensity > 0.8f) {
            RuleBreak rb;
            rb.type = RuleBreakType::Dynamics;
            rb.severity = emotion.intensity;
            rb.description = "Extreme dynamic contrasts";
            rb.emotionalJustification = "High intensity demands dramatic expression";
            rb.velocityRange = {10, 127};
            rb.suddenDynamicChanges = true;
            ruleBreaks.push_back(rb);
        }
        
        // Negative valence → Harmony rule-breaks
        if (emotion.valence < -0.5f) {
            RuleBreak rb;
            rb.type = RuleBreakType::Harmony;
            rb.severity = std::abs(emotion.valence);
            rb.description = "Dissonant intervals and clusters";
            rb.emotionalJustification = "Pain expressed through harmonic tension";
            rb.allowDissonance = true;
            rb.clusterProbability = std::abs(emotion.valence) * 0.5f;
            ruleBreaks.push_back(rb);
        }
        
        // High arousal → Rhythm rule-breaks
        if (emotion.arousal > 0.7f) {
            RuleBreak rb;
            rb.type = RuleBreakType::Rhythm;
            rb.severity = emotion.arousal;
            rb.description = "Irregular rhythms and syncopation";
            rb.emotionalJustification = "Agitation requires rhythmic disruption";
            rb.syncopationLevel = emotion.arousal;
            rb.irregularMeters = emotion.arousal > 0.85f;
            ruleBreaks.push_back(rb);
        }
        
        // Grief-specific: Voice leading violations
        if (emotion.category == EmotionCategory::Sadness && emotion.intensity > 0.7f) {
            RuleBreak rb;
            rb.type = RuleBreakType::Voice_Leading;
            rb.severity = emotion.intensity * 0.8f;
            rb.description = "Parallel motion and unresolved tensions";
            rb.emotionalJustification = "Grief breaks conventional resolution";
            rb.allowParallelMotion = true;
            rb.allowUnresolvedTensions = true;
            ruleBreaks.push_back(rb);
        }
        
        // Anger-specific: Texture rule-breaks
        if (emotion.category == EmotionCategory::Anger) {
            RuleBreak rb;
            rb.type = RuleBreakType::Texture;
            rb.severity = emotion.intensity;
            rb.description = "Aggressive layering and collision";
            rb.emotionalJustification = "Anger demands forceful expression";
            rb.crossHandVoicing = true;
            ruleBreaks.push_back(rb);
        }
        
        // Low arousal + negative valence → Rest/silence rule-breaks
        if (emotion.arousal < 0.3f && emotion.valence < -0.3f) {
            RuleBreak rb;
            rb.type = RuleBreakType::Structure;
            rb.severity = std::abs(emotion.valence);
            rb.description = "Intentional silence and fragmentation";
            rb.emotionalJustification = "Emptiness expressed through absence";
            rb.restProbability = 0.3f + (1.0f - emotion.arousal) * 0.3f;
            ruleBreaks.push_back(rb);
        }
        
        ruleBreaks_.insert(ruleBreaks_.end(), ruleBreaks.begin(), ruleBreaks.end());
        return ruleBreaks;
    }
    
    /**
     * Complete three-phase processing.
     * Ported from Python process_intent()
     */
    IntentResult processIntent(const Wound& wound) {
        IntentResult result;
        result.wound = wound;
        
        // Phase 1: Wound → Emotion
        const EmotionNode* emotion = processWound(wound);
        if (emotion) {
            result.emotion = *emotion;
        }
        
        // Phase 2-3: Emotion → Rule-breaks
        result.ruleBreaks = emotionToRuleBreaks(result.emotion);
        
        // Compile musical parameters
        EmotionalState state;
        state.valence = result.emotion.valence;
        state.arousal = result.emotion.arousal;
        state.intensity = result.emotion.intensity;
        state.primaryEmotion = result.emotion.name;
        
        result.musicalParams = emotionMapper_.mapToParameters(state);
        
        // Apply rule-break modifications
        for (const auto& rb : result.ruleBreaks) {
            applyRuleBreakToParams(rb, result.musicalParams);
        }
        
        return result;
    }
    
    const EmotionThesaurus& thesaurus() const { return thesaurus_; }
    const std::vector<Wound>& woundHistory() const { return woundHistory_; }
    const std::vector<RuleBreak>& ruleBreaks() const { return ruleBreaks_; }
    
    void clearHistory() {
        woundHistory_.clear();
        ruleBreaks_.clear();
    }

private:
    EmotionThesaurus thesaurus_;
    EmotionMapper emotionMapper_;
    std::vector<Wound> woundHistory_;
    std::vector<RuleBreak> ruleBreaks_;
    
    static std::string toLower(const std::string& s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
    static bool contains(const std::string& haystack, const std::string& needle) {
        return haystack.find(needle) != std::string::npos;
    }
    
    void applyRuleBreakToParams(const RuleBreak& rb, MusicalParameters& params) {
        switch (rb.type) {
            case RuleBreakType::Dynamics:
                params.velocityMin = rb.velocityRange.first;
                params.velocityMax = rb.velocityRange.second;
                params.dynamicsRange = std::max(params.dynamicsRange, rb.severity);
                break;
                
            case RuleBreakType::Harmony:
                params.dissonance = std::max(params.dissonance, rb.clusterProbability);
                break;
                
            case RuleBreakType::Rhythm:
                if (rb.syncopationLevel > 0.5f) {
                    params.timingFeel = TimingFeel::Pushed;
                }
                break;
                
            case RuleBreakType::Structure:
                params.spaceProbability = std::max(params.spaceProbability, rb.restProbability);
                params.density = std::min(params.density, 1.0f - rb.restProbability);
                break;
                
            default:
                break;
        }
    }
};

} // namespace kelly
