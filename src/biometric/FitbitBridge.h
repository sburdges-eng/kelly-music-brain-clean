#pragma once

/**
 * FitbitBridge - Fitbit API Integration
 * ======================================
 *
 * Provides access to Fitbit health data via REST API:
 * - Heart rate (HR)
 * - Heart rate variability (HRV)
 * - Activity data
 * - Sleep data
 *
 * Note: Requires OAuth 2.0 authentication with Fitbit API.
 */

#include <juce_core/juce_core.h>
#include <memory>
#include <functional>
#include <string>
#include <map>

namespace kelly {
namespace biometric {

struct FitbitData {
    float heartRate = 0.0f;
    float heartRateVariability = 0.0f;
    float restingHeartRate = 0.0f;
    int steps = 0;
    float calories = 0.0f;
    double timestamp = 0.0;

    bool isValid() const {
        return heartRate > 0.0f && timestamp > 0.0;
    }
};

/**
 * FitbitBridge - Interface to Fitbit API
 */
class FitbitBridge {
public:
    FitbitBridge();
    ~FitbitBridge();

    /**
     * Authenticate with Fitbit (OAuth 2.0)
     * @param clientId Fitbit app client ID
     * @param clientSecret Fitbit app client secret
     * @return true if authenticated
     */
    bool authenticate(const std::string& clientId, const std::string& clientSecret);

    /**
     * Check if authenticated
     */
    bool isAuthenticated() const { return authenticated_; }

    /**
     * Get latest heart rate data
     */
    FitbitData getLatestHeartRate();

    /**
     * Get heart rate for specific date
     * @param date Date string (YYYY-MM-DD)
     */
    FitbitData getHeartRateForDate(const std::string& date);

    /**
     * Get historical baseline
     * @param days Number of days
     */
    FitbitData getHistoricalBaseline(int days = 7);

    /**
     * Start polling for updates
     * @param intervalSeconds Polling interval in seconds
     * @param callback Function called when new data arrives
     */
    void startPolling(int intervalSeconds, std::function<void(const FitbitData&)> callback);

    /**
     * Stop polling
     */
    void stopPolling();

    /**
     * Calculate adaptive normalization factors
     */
    struct NormalizationFactors {
        float hrScale = 1.0f;
        float hrvScale = 1.0f;
        float hrOffset = 0.0f;
        float hrvOffset = 0.0f;
    };

    NormalizationFactors calculateNormalizationFactors();

private:
    bool authenticated_ = false;
    std::string accessToken_;
    std::string refreshToken_;
    std::string clientId_;
    std::string clientSecret_;

    bool polling_ = false;
    int pollingInterval_ = 60;
    std::function<void(const FitbitData&)> dataCallback_;

    std::vector<FitbitData> historicalData_;

    /**
     * Make API request to Fitbit
     */
    juce::var makeAPIRequest(const std::string& endpoint);

    /**
     * Refresh access token
     */
    bool refreshAccessToken();

    /**
     * Parse heart rate data from JSON response
     */
    FitbitData parseHeartRateData(const juce::var& jsonData);
};

} // namespace biometric
} // namespace kelly
