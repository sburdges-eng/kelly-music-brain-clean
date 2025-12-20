#pragma once

#include "engine/VADCalculator.h"
#include <vector>
#include <deque>
#include <optional>

namespace kelly {

/**
 * Predictive Trend Analyzer
 * 
 * Analyzes VAD trends over time and predicts future states.
 * Uses simple linear regression and moving averages.
 */
struct TrendPrediction {
    VADState predictedState;      // Predicted VAD at next time point
    float confidence;              // 0.0-1.0, confidence in prediction
    float trendStrength;           // 0.0-1.0, strength of trend (how consistent)
    std::string trendDescription;  // Human-readable trend description
    
    TrendPrediction() : confidence(0.5f), trendStrength(0.5f) {}
};

struct TrendMetrics {
    float valenceTrend;      // -1.0 to 1.0, direction and strength
    float arousalTrend;     // -1.0 to 1.0
    float dominanceTrend;   // -1.0 to 1.0
    float overallTrend;     // -1.0 to 1.0, combined trend
    
    TrendMetrics() : valenceTrend(0.0f), arousalTrend(0.0f), 
                    dominanceTrend(0.0f), overallTrend(0.0f) {}
};

/**
 * Predictive Trend Analyzer
 */
class PredictiveTrendAnalyzer {
public:
    PredictiveTrendAnalyzer(size_t historySize = 20);
    
    /**
     * Add a new VAD state to the history
     */
    void addVADState(const VADState& state);
    
    /**
     * Predict next VAD state based on history
     * @param stepsAhead Number of steps to predict (default 1)
     * @return Prediction with confidence
     */
    TrendPrediction predictNext(size_t stepsAhead = 1) const;
    
    /**
     * Calculate current trend metrics
     * @return Trend metrics showing direction and strength
     */
    TrendMetrics calculateTrends() const;
    
    /**
     * Get smoothed VAD state (moving average)
     * @param windowSize Size of moving average window
     * @return Smoothed VAD state
     */
    VADState getSmoothedState(size_t windowSize = 5) const;
    
    /**
     * Detect if there's a significant change in trend
     * @param threshold Minimum change to consider significant (default 0.3)
     * @return true if significant change detected
     */
    bool detectTrendChange(float threshold = 0.3f) const;
    
    /**
     * Get rate of change for each dimension
     * @return Vector of rates: [valence_rate, arousal_rate, dominance_rate]
     */
    std::vector<float> getRatesOfChange() const;
    
    /**
     * Clear history
     */
    void clear();
    
    /**
     * Get current history size
     */
    size_t size() const { return history_.size(); }
    
private:
    std::deque<VADState> history_;
    size_t maxHistorySize_;
    
    // Linear regression helpers
    float linearRegressionSlope(const std::vector<float>& x, const std::vector<float>& y) const;
    float linearRegressionIntercept(const std::vector<float>& x, const std::vector<float>& y, float slope) const;
    
    // Trend description generation
    std::string generateTrendDescription(const TrendMetrics& metrics) const;
};

} // namespace kelly
