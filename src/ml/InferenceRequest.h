#pragma once

#include <array>
#include <cstdint>

namespace kelly {

/**
 * Request structure for ML inference.
 * Sent from audio thread to inference thread.
 */
struct InferenceRequest {
    std::array<float, 128> features;  // Extracted audio features
    int64_t timestamp;                 // Sample position timestamp

    InferenceRequest() : timestamp(0) {
        features.fill(0.0f);
    }
};

/**
 * Result structure from ML inference.
 * Sent from inference thread back to audio thread.
 */
struct InferenceResult {
    std::array<float, 64> emotionVector;  // Emotion embedding vector
    int64_t timestamp;                      // Matching timestamp

    InferenceResult() : timestamp(0) {
        emotionVector.fill(0.0f);
    }
};

} // namespace kelly
