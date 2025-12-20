#include "engine/PredictiveTrendAnalyzer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace kelly {

PredictiveTrendAnalyzer::PredictiveTrendAnalyzer(size_t historySize)
    : maxHistorySize_(historySize) {
}

void PredictiveTrendAnalyzer::addVADState(const VADState& state) {
    history_.push_back(state);
    if (history_.size() > maxHistorySize_) {
        history_.pop_front();
    }
}

TrendPrediction PredictiveTrendAnalyzer::predictNext(size_t stepsAhead) const {
    TrendPrediction prediction;
    
    if (history_.size() < 3) {
        // Not enough data, return last state with low confidence
        if (!history_.empty()) {
            prediction.predictedState = history_.back();
        }
        prediction.confidence = 0.2f;
        prediction.trendStrength = 0.0f;
        prediction.trendDescription = "Insufficient data";
        return prediction;
    }
    
    // Build time series
    std::vector<float> timePoints;
    std::vector<float> valenceValues, arousalValues, dominanceValues;
    
    for (size_t i = 0; i < history_.size(); ++i) {
        timePoints.push_back(static_cast<float>(i));
        valenceValues.push_back(history_[i].valence);
        arousalValues.push_back(history_[i].arousal);
        dominanceValues.push_back(history_[i].dominance);
    }
    
    // Linear regression for each dimension
    float vSlope = linearRegressionSlope(timePoints, valenceValues);
    float aSlope = linearRegressionSlope(timePoints, arousalValues);
    float dSlope = linearRegressionSlope(timePoints, dominanceValues);
    
    // Predict next value
    float nextTime = static_cast<float>(history_.size() + stepsAhead - 1);
    
    float vIntercept = linearRegressionIntercept(timePoints, valenceValues, vSlope);
    float aIntercept = linearRegressionIntercept(timePoints, arousalValues, aSlope);
    float dIntercept = linearRegressionIntercept(timePoints, dominanceValues, dSlope);
    
    prediction.predictedState.valence = vSlope * nextTime + vIntercept;
    prediction.predictedState.arousal = aSlope * nextTime + aIntercept;
    prediction.predictedState.dominance = dSlope * nextTime + dIntercept;
    prediction.predictedState.clamp();
    
    // Calculate confidence based on R-squared (goodness of fit)
    // Simplified: use variance of residuals
    float vVariance = 0.0f, aVariance = 0.0f, dVariance = 0.0f;
    for (size_t i = 0; i < history_.size(); ++i) {
        float vPred = vSlope * timePoints[i] + vIntercept;
        float aPred = aSlope * timePoints[i] + aIntercept;
        float dPred = dSlope * timePoints[i] + dIntercept;
        
        vVariance += (valenceValues[i] - vPred) * (valenceValues[i] - vPred);
        aVariance += (arousalValues[i] - aPred) * (arousalValues[i] - aPred);
        dVariance += (dominanceValues[i] - dPred) * (dominanceValues[i] - dPred);
    }
    
    vVariance /= history_.size();
    aVariance /= history_.size();
    dVariance /= history_.size();
    
    // Lower variance = higher confidence
    float avgVariance = (vVariance + aVariance + dVariance) / 3.0f;
    prediction.confidence = std::max(0.0f, 1.0f - avgVariance * 2.0f);  // Normalize
    prediction.confidence = std::clamp(prediction.confidence, 0.0f, 1.0f);
    
    // Trend strength: how consistent is the slope?
    float vTrendStrength = std::abs(vSlope);
    float aTrendStrength = std::abs(aSlope);
    float dTrendStrength = std::abs(dSlope);
    prediction.trendStrength = (vTrendStrength + aTrendStrength + dTrendStrength) / 3.0f;
    prediction.trendStrength = std::clamp(prediction.trendStrength, 0.0f, 1.0f);
    
    // Generate description
    TrendMetrics metrics;
    metrics.valenceTrend = vSlope;
    metrics.arousalTrend = aSlope;
    metrics.dominanceTrend = dSlope;
    prediction.trendDescription = generateTrendDescription(metrics);
    
    return prediction;
}

TrendMetrics PredictiveTrendAnalyzer::calculateTrends() const {
    TrendMetrics metrics;
    
    if (history_.size() < 2) {
        return metrics;  // No trend if insufficient data
    }
    
    // Simple linear regression
    std::vector<float> timePoints;
    std::vector<float> valenceValues, arousalValues, dominanceValues;
    
    for (size_t i = 0; i < history_.size(); ++i) {
        timePoints.push_back(static_cast<float>(i));
        valenceValues.push_back(history_[i].valence);
        arousalValues.push_back(history_[i].arousal);
        dominanceValues.push_back(history_[i].dominance);
    }
    
    metrics.valenceTrend = linearRegressionSlope(timePoints, valenceValues);
    metrics.arousalTrend = linearRegressionSlope(timePoints, arousalValues);
    metrics.dominanceTrend = linearRegressionSlope(timePoints, dominanceValues);
    
    // Normalize trends to -1.0 to 1.0 range
    // Assuming max slope over history size is about 0.1 per step
    float maxSlope = 0.1f;
    metrics.valenceTrend = std::clamp(metrics.valenceTrend / maxSlope, -1.0f, 1.0f);
    metrics.arousalTrend = std::clamp(metrics.arousalTrend / maxSlope, -1.0f, 1.0f);
    metrics.dominanceTrend = std::clamp(metrics.dominanceTrend / maxSlope, -1.0f, 1.0f);
    
    // Overall trend: weighted average
    metrics.overallTrend = (metrics.valenceTrend * 0.4f + 
                           metrics.arousalTrend * 0.4f + 
                           metrics.dominanceTrend * 0.2f);
    
    return metrics;
}

VADState PredictiveTrendAnalyzer::getSmoothedState(size_t windowSize) const {
    if (history_.empty()) {
        return VADState(0.0f, 0.5f, 0.5f);
    }
    
    size_t actualWindow = std::min(windowSize, history_.size());
    size_t startIdx = history_.size() - actualWindow;
    
    VADState smoothed;
    for (size_t i = startIdx; i < history_.size(); ++i) {
        smoothed.valence += history_[i].valence;
        smoothed.arousal += history_[i].arousal;
        smoothed.dominance += history_[i].dominance;
    }
    
    smoothed.valence /= actualWindow;
    smoothed.arousal /= actualWindow;
    smoothed.dominance /= actualWindow;
    
    if (!history_.empty()) {
        smoothed.timestamp = history_.back().timestamp;
    }
    
    smoothed.clamp();
    return smoothed;
}

bool PredictiveTrendAnalyzer::detectTrendChange(float threshold) const {
    if (history_.size() < 6) {
        return false;  // Need enough data to detect change
    }
    
    // Compare recent trend vs earlier trend
    size_t midPoint = history_.size() / 2;
    
    // Recent trend
    std::vector<float> recentTime, recentV, recentA, recentD;
    for (size_t i = midPoint; i < history_.size(); ++i) {
        recentTime.push_back(static_cast<float>(i - midPoint));
        recentV.push_back(history_[i].valence);
        recentA.push_back(history_[i].arousal);
        recentD.push_back(history_[i].dominance);
    }
    
    // Earlier trend
    std::vector<float> earlierTime, earlierV, earlierA, earlierD;
    for (size_t i = 0; i < midPoint; ++i) {
        earlierTime.push_back(static_cast<float>(i));
        earlierV.push_back(history_[i].valence);
        earlierA.push_back(history_[i].arousal);
        earlierD.push_back(history_[i].dominance);
    }
    
    float recentVSlope = linearRegressionSlope(recentTime, recentV);
    float recentASlope = linearRegressionSlope(recentTime, recentA);
    float recentDSlope = linearRegressionSlope(recentTime, recentD);
    
    float earlierVSlope = linearRegressionSlope(earlierTime, earlierV);
    float earlierASlope = linearRegressionSlope(earlierTime, earlierA);
    float earlierDSlope = linearRegressionSlope(earlierTime, earlierD);
    
    // Check if slopes changed significantly
    float vChange = std::abs(recentVSlope - earlierVSlope);
    float aChange = std::abs(recentASlope - earlierASlope);
    float dChange = std::abs(recentDSlope - earlierDSlope);
    
    float maxChange = std::max({vChange, aChange, dChange});
    return maxChange > threshold;
}

std::vector<float> PredictiveTrendAnalyzer::getRatesOfChange() const {
    std::vector<float> rates(3, 0.0f);
    
    if (history_.size() < 2) {
        return rates;
    }
    
    // Calculate average rate of change over recent history
    size_t window = std::min(size_t(5), history_.size() - 1);
    
    float vTotalChange = 0.0f, aTotalChange = 0.0f, dTotalChange = 0.0f;
    float timeTotal = 0.0f;
    
    for (size_t i = history_.size() - window; i < history_.size(); ++i) {
        if (i > 0) {
            double timeDelta = history_[i].timestamp - history_[i-1].timestamp;
            if (timeDelta > 0.0) {
                vTotalChange += (history_[i].valence - history_[i-1].valence) / timeDelta;
                aTotalChange += (history_[i].arousal - history_[i-1].arousal) / timeDelta;
                dTotalChange += (history_[i].dominance - history_[i-1].dominance) / timeDelta;
                timeTotal += 1.0f;
            }
        }
    }
    
    if (timeTotal > 0.0f) {
        rates[0] = vTotalChange / timeTotal;
        rates[1] = aTotalChange / timeTotal;
        rates[2] = dTotalChange / timeTotal;
    }
    
    return rates;
}

void PredictiveTrendAnalyzer::clear() {
    history_.clear();
}

float PredictiveTrendAnalyzer::linearRegressionSlope(
    const std::vector<float>& x,
    const std::vector<float>& y
) const {
    if (x.size() != y.size() || x.size() < 2) {
        return 0.0f;
    }
    
    float n = static_cast<float>(x.size());
    float sumX = std::accumulate(x.begin(), x.end(), 0.0f);
    float sumY = std::accumulate(y.begin(), y.end(), 0.0f);
    float sumXY = 0.0f;
    float sumX2 = 0.0f;
    
    for (size_t i = 0; i < x.size(); ++i) {
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }
    
    float denominator = n * sumX2 - sumX * sumX;
    if (std::abs(denominator) < 1e-6f) {
        return 0.0f;
    }
    
    return (n * sumXY - sumX * sumY) / denominator;
}

float PredictiveTrendAnalyzer::linearRegressionIntercept(
    const std::vector<float>& x,
    const std::vector<float>& y,
    float slope
) const {
    if (x.size() != y.size() || x.empty()) {
        return 0.0f;
    }
    
    float sumX = std::accumulate(x.begin(), x.end(), 0.0f);
    float sumY = std::accumulate(y.begin(), y.end(), 0.0f);
    float n = static_cast<float>(x.size());
    
    return (sumY - slope * sumX) / n;
}

std::string PredictiveTrendAnalyzer::generateTrendDescription(const TrendMetrics& metrics) const {
    std::ostringstream oss;
    
    // Valence trend
    if (metrics.valenceTrend > 0.3f) {
        oss << "Improving mood";
    } else if (metrics.valenceTrend < -0.3f) {
        oss << "Declining mood";
    } else {
        oss << "Stable mood";
    }
    
    oss << ", ";
    
    // Arousal trend
    if (metrics.arousalTrend > 0.3f) {
        oss << "increasing energy";
    } else if (metrics.arousalTrend < -0.3f) {
        oss << "decreasing energy";
    } else {
        oss << "stable energy";
    }
    
    oss << ", ";
    
    // Dominance trend
    if (metrics.dominanceTrend > 0.3f) {
        oss << "gaining control";
    } else if (metrics.dominanceTrend < -0.3f) {
        oss << "losing control";
    } else {
        oss << "stable control";
    }
    
    return oss.str();
}

} // namespace kelly
