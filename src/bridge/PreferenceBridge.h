#pragma once
/*
 * PreferenceBridge.h - C++ Bridge to Python UserPreferenceModel
 * =============================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: UserPreferenceModel (preference learning system)
 * - UI Layer: Used by EmotionWorkstation (parameter tracking)
 * - Learning Layer: PreferenceTracker (C++ preference tracking)
 * - Used By: PluginProcessor (via EmotionWorkstation)
 *
 * Purpose: Thread-safe bridge between C++ UI and Python preference learning system.
 *          Tracks user parameter adjustments and learns preferences over time.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Parameter tracking is batched (collects changes, flushes periodically)
 * - Python calls are made from a dedicated thread to avoid blocking UI
 */

#include <juce_core/juce_core.h>
#include <memory>
#include <mutex>
#include <vector>
#include <map>
#include <string>
#include <atomic>

namespace kelly {

/**
 * PreferenceBridge - C++ bridge to Python UserPreferenceModel
 *
 * Provides thread-safe communication between C++ UI and Python preference learning system.
 * Uses Python C API to call Python functions from C++.
 *
 * Thread Safety:
 * - All methods are thread-safe
 * - Parameter tracking is batched (collects changes, flushes periodically)
 * - Python calls are made from a dedicated thread to avoid blocking UI
 */
class PreferenceBridge {
public:
    PreferenceBridge();
    ~PreferenceBridge();

    // Initialize Python interpreter and load UserPreferenceModel
    bool initialize();
    void shutdown();

    // Parameter tracking
    void recordParameterAdjustment(
        const juce::String& parameterName,
        float oldValue,
        float newValue,
        const std::map<std::string, std::string>& context = {}
    );

    // Emotion tracking
    void recordEmotionSelection(
        const juce::String& emotionName,
        float valence,
        float arousal,
        const std::map<std::string, std::string>& context = {}
    );

    // MIDI generation tracking
    juce::String recordMidiGeneration(
        const juce::String& intentText,
        const std::map<juce::String, float>& parameters,
        const juce::String& emotion = {},
        const std::vector<juce::String>& ruleBreaks = {}
    );

    // MIDI feedback
    void recordMidiFeedback(const juce::String& generationId, bool accepted);

    // MIDI modification tracking
    void recordMidiModification(
        const juce::String& generationId,
        const juce::String& parameterName,
        float oldValue,
        float newValue
    );

    // Rule break tracking
    void recordRuleBreakModification(
        const juce::String& ruleBreak,
        const juce::String& action,  // "added" or "removed"
        const std::map<std::string, std::string>& context = {}
    );

    // Statistics retrieval (cached, updated periodically)
    struct ParameterStatistics {
        float averageValue = 0.0f;
        float medianValue = 0.0f;
        float minValue = 0.0f;
        float maxValue = 0.0f;
        int adjustmentCount = 0;
    };

    std::map<std::string, ParameterStatistics> getParameterStatistics();
    std::map<std::string, int> getEmotionPreferences();
    std::map<std::string, std::pair<float, float>> getPreferredParameterRanges();
    float getAcceptanceRate();

    // Check if bridge is initialized and ready
    bool isInitialized() const { return initialized_.load(); }

    // Force flush pending operations (for testing or explicit save)
    void flush();

private:
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdownRequested_{false};

    // Python objects (protected by pythonMutex_)
    void* pythonModule_ = nullptr;  // PyObject* (void* to avoid including Python.h in header)
    void* preferenceModel_ = nullptr;  // PyObject*

    // Pending operations queue (batched for performance)
    struct PendingOperation {
        enum Type {
            ParameterAdjustment,
            EmotionSelection,
            MidiGeneration,
            MidiFeedback,
            MidiModification,
            RuleBreakModification
        };
        Type type;
        std::map<std::string, std::string> data;  // Serialized operation data
    };

    std::vector<PendingOperation> pendingOperations_;
    std::mutex pendingMutex_;
    static constexpr size_t MAX_PENDING_OPERATIONS = 100;
    static constexpr int FLUSH_INTERVAL_MS = 500;  // Flush every 500ms

    // Cached statistics (updated periodically)
    std::map<std::string, ParameterStatistics> cachedStats_;
    std::map<std::string, int> cachedEmotionPrefs_;
    std::map<std::string, std::pair<float, float>> cachedRanges_;
    float cachedAcceptanceRate_ = 0.0f;
    std::mutex cacheMutex_;
    juce::int64 lastCacheUpdate_ = 0;
    static constexpr juce::int64 CACHE_UPDATE_INTERVAL_MS = 2000;  // Update cache every 2s

    // Worker thread for Python calls
    class WorkerThread;
    std::unique_ptr<WorkerThread> workerThread_;

    // Internal methods
    bool initializePython();
    void shutdownPython();
    void processPendingOperations();
    void updateCachedStatistics();
    juce::String generateId();

    // Python C API helpers (implemented in .cpp)
    bool callPythonMethod(
        const char* methodName,
        const std::vector<std::string>& args = {},
        const std::map<std::string, std::string>& kwargs = {}
    );
    bool callPythonMethodWithResult(
        const char* methodName,
        void* resultOut,  // Type depends on method
        const std::vector<std::string>& args = {},
        const std::map<std::string, std::string>& kwargs = {}
    );

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PreferenceBridge)
};

} // namespace kelly
