#pragma once

#include "ml/LockFreeRingBuffer.h"
#include "ml/RTNeuralProcessor.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <juce_core/juce_core.h>

namespace kelly {

/**
 * InferenceRequest - Request for ML inference.
 */
struct InferenceRequest {
    std::array<float, 128> features;
    int64_t timestamp;

    InferenceRequest() : timestamp(0) {
        features.fill(0.0f);
    }
};

/**
 * InferenceResult - Result from ML inference.
 */
struct InferenceResult {
    std::array<float, 64> emotionVector;
    int64_t timestamp;

    InferenceResult() : timestamp(0) {
        emotionVector.fill(0.0f);
    }
};

/**
 * InferenceThreadManager - Manages ML inference in separate thread.
 *
 * Provides non-blocking inference for real-time audio processing.
 * Uses lock-free ring buffers for thread-safe communication.
 */
class InferenceThreadManager {
public:
    static constexpr size_t BUFFER_SIZE = 256;

    InferenceThreadManager() : running_(false) {}

    ~InferenceThreadManager() {
        stop();
    }

    /**
     * Start inference thread and load model.
     * @param modelPath Path to model file
     */
    void start(const juce::File& modelPath) {
        if (running_.load()) {
            stop();
        }

        if (!processor_.loadModel(modelPath)) {
            juce::Logger::writeToLog("Failed to load ML model: " + modelPath.getFullPathName());
            return;
        }

        running_.store(true);
        inferenceThread_ = std::thread(&InferenceThreadManager::inferenceLoop, this);
    }

    /**
     * Stop inference thread.
     */
    void stop() {
        running_.store(false);
        if (inferenceThread_.joinable()) {
            inferenceThread_.join();
        }
    }

    /**
     * Submit inference request (called from audio thread - never blocks).
     * @param request Inference request
     * @return true if submitted successfully
     */
    bool submitRequest(const InferenceRequest& request) {
        return requestBuffer_.push(&request, 1);
    }

    /**
     * Get inference result (called from audio thread - never blocks).
     * @param result Output result
     * @return true if result available
     */
    bool getResult(InferenceResult& result) {
        return resultBuffer_.pop(&result, 1);
    }

    /**
     * Check if inference thread is running.
     */
    bool isRunning() const {
        return running_.load();
    }

    /**
     * Get number of pending requests.
     */
    size_t getPendingRequests() const {
        return requestBuffer_.availableToRead();
    }

    /**
     * Get number of available results.
     */
    size_t getAvailableResults() const {
        return resultBuffer_.availableToRead();
    }

private:
    /**
     * Inference loop running in separate thread.
     */
    void inferenceLoop() {
        InferenceRequest request;

        while (running_.load()) {
            if (requestBuffer_.pop(&request, 1)) {
                // Perform inference
                InferenceResult result;
                result.emotionVector = processor_.inferEmotion(request.features);
                result.timestamp = request.timestamp;

                // Push result (may fail if buffer full, but that's OK)
                resultBuffer_.push(&result, 1);
            } else {
                // No requests available, sleep briefly
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }

    RTNeuralProcessor processor_;
    LockFreeRingBuffer<InferenceRequest, BUFFER_SIZE> requestBuffer_;
    LockFreeRingBuffer<InferenceResult, BUFFER_SIZE> resultBuffer_;
    std::thread inferenceThread_;
    std::atomic<bool> running_;
};

} // namespace kelly
