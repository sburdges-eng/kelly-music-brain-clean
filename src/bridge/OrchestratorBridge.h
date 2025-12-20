#pragma once
/*
 * OrchestratorBridge.h - Python AI Orchestrator Bridge
 * ===================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Python Layer: AI Orchestrator (multi-stage generation pipelines)
 * - MIDI Layer: Used by MidiGenerator (orchestrates multi-engine generation)
 * - Engine Layer: Coordinates Python intelligence with C++ engine execution
 * - Used By: MidiGenerator (for complex multi-stage generation)
 *
 * Purpose: C++ interface to Python AI Orchestrator for executing multi-stage
 *          generation pipelines that combine Python intelligence with C++ engines.
 *
 * Thread Safety:
 * - Pipeline execution is asynchronous (non-blocking)
 * - Results are returned via callback or polling
 * - Python calls are made from worker thread, not audio thread
 */

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace kelly {

/**
 * OrchestratorBridge - C++ interface to Python AI Orchestrator
 *
 * Provides methods to execute Python orchestrator pipelines from C++ MidiGenerator.
 * The orchestrator can coordinate multi-stage generation pipelines that combine
 * Python intelligence with C++ engine execution.
 *
 * Thread Safety:
 * - Pipeline execution is asynchronous (non-blocking)
 * - Results are returned via callback or polling
 * - Python calls are made from worker thread, not audio thread
 */
class OrchestratorBridge {
public:
    OrchestratorBridge();
    ~OrchestratorBridge();

    /**
     * Execute a Python orchestrator pipeline.
     *
     * @param pipelineName Name of pipeline to execute (e.g., "idaw_bridge", "full_generation")
     * @param inputDataJson JSON string with input data:
     *   {
     *     "mood_primary": "grief",
     *     "emotion": "grief",
     *     "technical_key": "C",
     *     "technical_mode": "minor",
     *     "text_prompt": "I feel lost",
     *     "bars": 8,
     *     "complexity": 0.5,
     *     ...
     *   }
     * @return JSON string with execution result:
     *   {
     *     "success": true,
     *     "execution_id": "uuid",
     *     "final_output": {...},
     *     "stage_results": [...],
     *     "error": null
     *   }
     */
    std::string executePipeline(
        const std::string& pipelineName,
        const std::string& inputDataJson
    );

    /**
     * Execute pipeline asynchronously with callback.
     *
     * @param pipelineName Name of pipeline
     * @param inputDataJson Input data JSON
     * @param callback Function called when execution completes (called from worker thread)
     */
    void executePipelineAsync(
        const std::string& pipelineName,
        const std::string& inputDataJson,
        std::function<void(const std::string& resultJson)> callback
    );

    /**
     * Check execution status of a running pipeline.
     *
     * @param executionId Execution ID returned from executePipeline
     * @return JSON string with status:
     *   {
     *     "status": "running" | "completed" | "failed",
     *     "progress": 0.0-1.0,
     *     "current_stage": "harmony",
     *     "result": {...}  // if completed
     *   }
     */
    std::string getExecutionStatus(const std::string& executionId);

    /**
     * Cancel a running pipeline execution.
     *
     * @param executionId Execution ID to cancel
     * @return true if cancellation successful
     */
    bool cancelExecution(const std::string& executionId);

    /**
     * Register a C++ engine callback for orchestrator to call.
     *
     * The orchestrator can call back to C++ engines during pipeline execution.
     * This allows Python to coordinate C++ engine execution.
     *
     * @param engineType Engine type: "melody", "bass", "drum", etc.
     * @param callback Function that orchestrator can call:
     *   callback(engineType, configJson) -> resultJson
     */
    void registerEngineCallback(
        const std::string& engineType,
        std::function<std::string(const std::string&, const std::string&)> callback
    );

    /**
     * Check if Python bridge is available.
     */
    bool isAvailable() const { return available_; }

private:
    bool available_;

    // Python function pointers
    void* executePipelineFunc_;
    void* executePipelineAsyncFunc_;
    void* getStatusFunc_;
    void* cancelExecutionFunc_;

    // Registered engine callbacks
    std::map<std::string, std::function<std::string(const std::string&, const std::string&)>> engineCallbacks_;

    // Active executions (for status tracking)
    struct Execution {
        std::string executionId;
        std::string status;
        std::chrono::steady_clock::time_point startTime;
    };
    std::map<std::string, Execution> activeExecutions_;

    bool initializePython();
    void shutdownPython();
    std::string generateExecutionId();
};

} // namespace kelly
