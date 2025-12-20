#pragma once
/*
 * EngineIntelligenceBridge.h - Python Engine Intelligence Bridge
 * ===============================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: Python engine-level intelligence modules
 * - Engine Layer: Used by MelodyEngine, BassEngine, DrumGrooveEngine, etc.
 * - MIDI Layer: Engines use this bridge for intelligent suggestions
 * - Used By: All MIDI engines (MelodyEngine, BassEngine, PadEngine, etc.)
 *
 * Purpose: C++ interface to Python engine-level intelligence for getting intelligent
 *          suggestions for individual C++ engines from Python intelligence modules.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Python calls are cached to avoid blocking audio thread
 * - Suggestions are returned as JSON strings for easy parsing
 */

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace kelly {

/**
 * EngineIntelligenceBridge - C++ interface to Python engine-level intelligence
 *
 * Provides methods to get intelligent suggestions for individual C++ engines
 * (MelodyEngine, BassEngine, DrumGrooveEngine, etc.) from Python intelligence modules.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Python calls are cached to avoid blocking audio thread
 * - Suggestions are returned as JSON strings for easy parsing
 */
class EngineIntelligenceBridge {
public:
    EngineIntelligenceBridge();
    ~EngineIntelligenceBridge();

    /**
     * Get suggestions for a specific engine type.
     *
     * @param engineType Engine type: "melody", "bass", "drum", "pad", "string", etc.
     * @param currentStateJson JSON string with current state:
     *   {
     *     "emotion": "grief",
     *     "key": "C",
     *     "mode": "minor",
     *     "chords": ["Am", "Dm", "F", "C"],
     *     "parameters": {"complexity": 0.4, "density": 0.3, ...},
     *     "context": {"emotion_category": "negative_low_energy", ...}
     *   }
     * @return JSON string with engine-specific suggestions:
     *   {
     *     "contour": "descending",
     *     "density": "sparse",
     *     "velocity_range": [40, 75],
     *     "register_range": [55, 75],
     *     "parameter_adjustments": {"complexity": 0.3, "density": 0.2},
     *     "confidence": 0.85
     *   }
     */
    std::string getEngineSuggestions(
        const std::string& engineType,
        const std::string& currentStateJson
    );

    /**
     * Get suggestions for multiple engines at once (batch operation).
     *
     * @param engineTypes Vector of engine types
     * @param currentStateJson Current state JSON
     * @return JSON string with suggestions for all engines:
     *   {
     *     "melody": {...},
     *     "bass": {...},
     *     "drum": {...}
     *   }
     */
    std::string getBatchEngineSuggestions(
        const std::vector<std::string>& engineTypes,
        const std::string& currentStateJson
    );

    /**
     * Record that engine suggestions were applied.
     *
     * @param engineType Engine type
     * @param suggestionJson The suggestion that was applied
     * @param resultJson Result of applying the suggestion (for learning)
     */
    void recordSuggestionApplied(
        const std::string& engineType,
        const std::string& suggestionJson,
        const std::string& resultJson = "{}"
    );

    /**
     * Check if Python bridge is available.
     */
    bool isAvailable() const { return available_; }

    /**
     * Clear suggestion cache (force fresh suggestions on next call).
     */
    void clearCache();

    /**
     * Report engine state to Python (for learning/feedback).
     */
    void reportEngineState(
        const std::string& engineType,
        const std::string& stateJson,
        const std::string& generatedNotesJson
    );

private:
    bool available_;

    // Python function pointers (if Python is embedded)
    void* getEngineSuggestionsFunc_;
    void* getBatchSuggestionsFunc_;
    void* recordAppliedFunc_;
    void* reportEngineStateFunc_;

    // Suggestion cache (key: engineType + state hash, value: cached suggestion)
    struct CachedSuggestion {
        std::string suggestionJson;
        std::string stateHash;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::map<std::string, CachedSuggestion> suggestionCache_;
    static constexpr int CACHE_TTL_MS = 1000;  // Cache for 1 second

    bool initializePython();
    void shutdownPython();
    std::string getCachedSuggestion(const std::string& cacheKey);
    void cacheSuggestion(const std::string& cacheKey, const std::string& suggestionJson);
    std::string hashState(const std::string& stateJson);
    void pruneCache();

    static constexpr size_t MAX_CACHE_SIZE = 100;
};

} // namespace kelly
