#pragma once
/*
 * IntentBridge.h - Python-C++ Intent Processing Bridge
 * ====================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: intent_processor.py (Python intent processing)
 * - Engine Layer: IntentPipeline (C++ intent processing)
 * - Type System: KellyTypes.h (unified types for C++/Python conversion)
 * - Used By: IntentPipeline (optional Python processing for complex intents)
 *
 * Purpose: Bridge between Python intent_processor and C++ IntentPipeline
 *          Allows complex intents to be processed in Python when needed
 */

#include "common/KellyTypes.h"  // Unified type system
#include <string>
#include <memory>

namespace kelly {

/**
 * IntentBridge - C++ interface to Python intent_processor
 *
 * Provides methods to process intents using Python intent_processor and convert
 * between Python CompleteSongIntent and C++ IntentResult formats.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Python calls are made from worker thread, not audio thread
 * - Results are cached to avoid repeated processing
 */
class IntentBridge {
public:
    IntentBridge();
    ~IntentBridge();

    /**
     * Process intent using Python intent_processor.
     *
     * @param intentJson JSON string with Python CompleteSongIntent format:
     *   {
     *     "phase_0": {
     *       "core_event": "...",
     *       "core_resistance": "...",
     *       "core_longing": "..."
     *     },
     *     "phase_1": {
     *       "mood_primary": "grief",
     *       "vulnerability_scale": 0.8,
     *       ...
     *     },
     *     "phase_2": {
     *       "technical_genre": "ambient",
     *       "technical_key": "C",
     *       "technical_rule_to_break": ["HARMONY_AvoidTonicResolution"],
     *       ...
     *     }
     *   }
     * @return JSON string with C++ IntentResult format:
     *   {
     *     "key": "C",
     *     "mode": "minor",
     *     "tempoBpm": 82,
     *     "chordProgression": ["Am", "Dm", "F", "C"],
     *     "ruleBreaks": [...],
     *     "melodicRange": 0.6,
     *     ...
     *   }
     */
    std::string processIntent(const std::string& intentJson);

    /**
     * Convert Python CompleteSongIntent to C++ IntentResult.
     *
     * @param intentJson Python intent JSON
     * @return C++ IntentResult object
     */
    IntentResult convertToCppIntent(const std::string& intentJson);

    /**
     * Convert C++ IntentResult to Python CompleteSongIntent format.
     *
     * @param intent C++ IntentResult
     * @return JSON string with Python intent format
     */
    std::string convertToPythonIntent(const IntentResult& intent);

    /**
     * Validate intent processing result.
     *
     * @param resultJson Result JSON from Python
     * @return true if result is valid
     */
    bool validateResult(const std::string& resultJson);

    /**
     * Get suggested rule breaks for an emotion.
     *
     * @param emotion Emotion name (e.g., "grief", "longing")
     * @return JSON string with suggested rule breaks:
     *   {
     *     "rule_breaks": ["HARMONY_AvoidTonicResolution", ...],
     *     "justifications": {...}
     *   }
     */
    std::string getSuggestedRuleBreaks(const std::string& emotion);

    /**
     * Check if Python bridge is available.
     */
    bool isAvailable() const { return available_; }

private:
    bool available_;

    // Python function pointers
    void* processIntentFunc_;
    void* convertToCppFunc_;
    void* convertToPythonFunc_;
    void* validateResultFunc_;
    void* getRuleBreaksFunc_;

    // Result cache (key: intent hash, value: cached result)
    struct CachedResult {
        std::string resultJson;
        std::string intentHash;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::map<std::string, CachedResult> resultCache_;
    static constexpr int CACHE_TTL_MS = 5000;  // Cache for 5 seconds

    bool initializePython();
    void shutdownPython();
    std::string getCachedResult(const std::string& cacheKey);
    void cacheResult(const std::string& cacheKey, const std::string& resultJson);
    std::string hashIntent(const std::string& intentJson);
    IntentResult parseIntentResult(const std::string& resultJson);
};

} // namespace kelly
