#pragma once
/*
 * SuggestionBridge.h - Python Suggestion Engine Bridge
 * ====================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: SuggestionEngine (intelligent suggestion generation)
 * - UI Layer: Used by EmotionWorkstation (SuggestionOverlay)
 * - Engine Layer: Provides suggestions based on current musical state
 * - Used By: EmotionWorkstation, MidiGenerator (for intelligent suggestions)
 *
 * Purpose: C++ interface to Python SuggestionEngine for intelligent suggestions
 *          without requiring direct Python embedding in audio thread.
 */

#include <string>
#include <vector>
#include <memory>

namespace kelly {

/**
 * SuggestionBridge - C++ interface to Python SuggestionEngine
 *
 * Provides methods to get intelligent suggestions from Python SuggestionEngine
 * without requiring direct Python embedding in audio thread.
 */
class SuggestionBridge {
public:
    SuggestionBridge();
    ~SuggestionBridge();

    /**
     * Get suggestions based on current musical state.
     *
     * @param currentStateJson JSON string with current state:
     *   {
     *     "parameters": {"valence": -0.5, "arousal": 0.4, ...},
     *     "emotion": "grief",
     *     "rule_breaks": ["HARMONY_AvoidTonicResolution"]
     *   }
     * @param maxSuggestions Maximum number of suggestions to return
     * @return JSON string with suggestions array
     */
    std::string getSuggestions(
        const std::string& currentStateJson,
        int maxSuggestions = 5
    );

    /**
     * Record that a suggestion was shown to the user.
     */
    void recordSuggestionShown(
        const std::string& suggestionId,
        const std::string& suggestionType,
        const std::string& contextJson = "{}"
    );

    /**
     * Record that user accepted (applied) a suggestion.
     */
    void recordSuggestionAccepted(const std::string& suggestionId);

    /**
     * Record that user dismissed a suggestion.
     */
    void recordSuggestionDismissed(const std::string& suggestionId);

    /**
     * Check if Python bridge is available.
     */
    bool isAvailable() const { return available_; }

private:
    bool available_;

    // Python function pointers (if Python is embedded)
    // These will be set up if Python is available
    void* getSuggestionsFunc_;
    void* recordShownFunc_;
    void* recordAcceptedFunc_;
    void* recordDismissedFunc_;

    bool initializePython();
    void shutdownPython();
};

} // namespace kelly
