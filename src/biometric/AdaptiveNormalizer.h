#pragma once

/**
 * AdaptiveNormalizer - Adaptive normalization based on user baseline
 * ===================================================================
 *
 * Establishes historical baseline and adapts biometric readings
 * to emotion parameters based on individual user patterns.
 */

#include "biometric/BiometricInput.h"
#include <vector>
#include <deque>
#include <chrono>
#include <algorithm>
#include <numeric>

namespace kelly {
namespace biometric {

/**
 * AdaptiveNormalizer - Normalizes biometric data based on user baseline
 */
class AdaptiveNormalizer {
public:
    AdaptiveNormalizer();
    ~AdaptiveNormalizer() = default;

    /**
     * Add biometric reading to history
     */
    void addReading(const BiometricInput::BiometricData& data);

    /**
     * Calculate baseline from historical data
     * @param days Number of days to use for baseline
     */
    struct Baseline {
        float heartRate = 70.0f;
        float heartRateVariability = 50.0f;
        float skinConductance = 5.0f;
        float temperature = 36.5f;
        int sampleCount = 0;
    };

    Baseline calculateBaseline(int days = 7);

    /**
     * Normalize biometric data relative to baseline
     * @param data Raw biometric data
     * @param baseline User's baseline
     * @return Normalized data (0.0-1.0 range)
     */
    struct NormalizedData {
        float heartRate = 0.5f;      // 0.0 = very low, 1.0 = very high
        float hrv = 0.5f;            // 0.0 = low HRV (stressed), 1.0 = high HRV (calm)
        float skinConductance = 0.5f;
        float temperature = 0.5f;
    };

    NormalizedData normalize(const BiometricInput::BiometricData& data, const Baseline& baseline);

    /**
     * Get current baseline (cached)
     */
    Baseline getCurrentBaseline() const { return currentBaseline_; }

    /**
     * Update baseline (recalculate from history)
     */
    void updateBaseline(int days = 7);

    /**
     * Clear history
     */
    void clearHistory();

    /**
     * Get number of readings in history
     */
    size_t getHistorySize() const { return history_.size(); }

private:
    struct TimestampedData {
        BiometricInput::BiometricData data;
        std::chrono::system_clock::time_point timestamp;
    };

    std::deque<TimestampedData> history_;
    Baseline currentBaseline_;

    static constexpr size_t MAX_HISTORY_SIZE = 10000;  // ~10 days at 1 reading/minute

    /**
     * Filter history by time window
     */
    std::vector<BiometricInput::BiometricData> getDataInWindow(int days);
};

} // namespace biometric
} // namespace kelly
