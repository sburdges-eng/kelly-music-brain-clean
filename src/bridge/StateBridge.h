#pragma once
/*
 * StateBridge.h - State Synchronization Bridge
 * ============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: Python intelligence modules (state synchronization)
 * - MIDI Layer: Used by MidiGenerator and engines (state emission)
 * - Engine Layer: All engines emit state updates via this bridge
 * - Used By: MidiGenerator, MelodyEngine, BassEngine, and other engines
 *
 * Purpose: C++ interface for state synchronization with Python intelligence modules.
 *          Engines emit state updates, Python can query current state.
 *
 * Thread Safety:
 * - State emission is lock-free (uses lock-free queue)
 * - Safe to call from audio thread
 * - State updates are batched and processed asynchronously
 */

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <atomic>
#include <mutex>
#include <queue>

namespace kelly {

/**
 * StateBridge - C++ interface for state synchronization with Python
 *
 * Provides methods to emit state updates from C++ engines to Python intelligence
 * modules, and to query current state from Python.
 *
 * Thread Safety:
 * - State emission is lock-free (uses lock-free queue)
 * - Safe to call from audio thread
 * - State updates are batched and processed asynchronously
 */
class StateBridge {
public:
    StateBridge();
    ~StateBridge();

    /**
     * Emit state update from C++ engine.
     *
     * This is safe to call from audio thread.
     *
     * @param engineType Engine type: "melody", "bass", "drum", "midi_generator", etc.
     * @param stateJson JSON string with state update:
     *   {
     *     "chords": ["Am", "Dm", "F", "C"],
     *     "notes": [...],
     *     "parameters": {"complexity": 0.4, ...},
     *     "timestamp": 1234567890
     *   }
     */
    void emitStateUpdate(
        const std::string& engineType,
        const std::string& stateJson
    );

    /**
     * Get current state from Python.
     *
     * @return JSON string with current state:
     *   {
     *     "emotion": "grief",
     *     "chords": [...],
     *     "parameters": {...},
     *     "context": {...}
     *   }
     */
    std::string getCurrentState();

    /**
     * Query state for specific engine.
     *
     * @param engineType Engine type
     * @return JSON string with engine state
     */
    std::string getEngineState(const std::string& engineType);

    /**
     * Initialize state tracking.
     */
    bool initialize();

    /**
     * Shutdown state bridge and flush pending updates.
     */
    void shutdown();

    /**
     * Check if Python bridge is available.
     */
    bool isAvailable() const { return available_.load(); }

    /**
     * Force flush pending state updates (for testing or explicit save).
     */
    void flush();

private:
    std::atomic<bool> available_{false};
    std::atomic<bool> shutdownRequested_{false};

    // Python function pointers
    void* emitStateFunc_;
    void* getCurrentStateFunc_;
    void* getEngineStateFunc_;

    // Lock-free queue for state updates (audio thread safe)
    struct StateUpdate {
        std::string engineType;
        std::string stateJson;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::queue<StateUpdate> stateQueue_;
    std::mutex queueMutex_;
    static constexpr size_t MAX_QUEUE_SIZE = 1000;

    // Worker thread for processing state updates
    class StateWorkerThread;
    std::unique_ptr<StateWorkerThread> workerThread_;

    bool initializePython();
    void shutdownPython();
    void processStateQueue();
};

} // namespace kelly
