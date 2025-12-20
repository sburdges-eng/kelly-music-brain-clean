#pragma once

#include "common/Types.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace kelly {

/**
 * Biometric Input - Real-time biometric data integration.
 *
 * v2.0 feature: Integrates heart rate, skin conductance, etc. to influence
 * emotion parameters in real-time.
 */
class BiometricInput {
public:
  BiometricInput();
  ~BiometricInput();

  /**
   * Biometric data structure
   */
  struct BiometricData {
    std::optional<float> heartRate; // BPM
    std::optional<float>
        heartRateVariability;             // HRV in milliseconds (RMSSD or SDNN)
    std::optional<float> skinConductance; // EDA in Microsiemens
    std::optional<float> temperature;     // Celsius
    std::optional<float> movement;        // 0.0 to 1.0 (accelerometer)
    double timestamp;                     // Seconds since start
  };

  /**
   * Process biometric data and convert to emotion parameters.
   * @param data The biometric reading
   * @return Emotion parameters (valence, arousal, intensity) derived from
   * biometrics
   */
  struct EmotionFromBiometrics {
    float valence;   // -1.0 to 1.0
    float arousal;   // 0.0 to 1.0
    float intensity; // 0.0 to 1.0
  };

  EmotionFromBiometrics processBiometricData(const BiometricData &data);

  /**
   * Enable/disable biometric input
   */
  void setEnabled(bool enabled) { enabled_ = enabled; }
  bool isEnabled() const { return enabled_; }

  /**
   * Set callback for when biometric data is received
   */
  void onBiometricData(std::function<void(const BiometricData &)> callback);

  /**
   * Simulate biometric input (for testing without hardware)
   */
  void simulateInput(const BiometricData &data);

  /**
   * Get smoothed/averaged biometric values
   */
  BiometricData getSmoothedData() const;

  /**
   * Platform-specific hardware integration
   */

  /**
   * Initialize HealthKit integration (macOS/iOS)
   * @return true if HealthKit is available and initialized
   */
  bool initializeHealthKit();

  /**
   * Initialize Fitbit API integration
   * @param accessToken Fitbit OAuth access token
   * @return true if connection successful
   */
  bool initializeFitbit(const std::string &accessToken);

  /**
   * Start real-time data streaming from hardware
   * @return true if streaming started
   */
  bool startStreaming();

  /**
   * Stop real-time data streaming
   */
  void stopStreaming();

  /**
   * Check if hardware streaming is active
   */
  bool isStreaming() const { return streamingActive_; }

  /**
   * Get last hardware reading timestamp
   */
  double getLastReadingTime() const { return lastReadingTime_; }

  /**
   * Enable adaptive normalization (uses historical baseline)
   */
  void setAdaptiveNormalization(bool enabled) {
    adaptiveNormalization_ = enabled;
  }
  bool isAdaptiveNormalizationEnabled() const { return adaptiveNormalization_; }

  /**
   * Set baseline for adaptive normalization
   */
  void setBaseline(const BiometricData &baseline);
  BiometricData getBaseline() const { return baseline_; }

private:
  bool adaptiveNormalization_ = false;
  BiometricData baseline_;
  bool enabled_ = false;
  std::function<void(const BiometricData &)> onDataCallback_;

  // Hardware integration state
  bool streamingActive_ = false;
  double lastReadingTime_ = 0.0;
  std::string fitbitAccessToken_;
  bool healthKitInitialized_ = false;
  bool fitbitInitialized_ = false;

  // Smoothing/averaging
  std::vector<BiometricData> dataHistory_;
  static constexpr size_t HISTORY_SIZE = 10;

  // Platform-specific hardware reading (implemented per platform)
  void readHealthKitData(); // macOS/iOS implementation
  void readFitbitData();    // Cross-platform via API

  // Bridge instances for hardware integration
  // Note: These are forward-declared in header, full definitions included in
  // .cpp
  class HealthKitBridge;
  class FitbitBridge;
  // Use void* to avoid incomplete type issues with unique_ptr in header
  // Will be cast to proper types in implementation
  void *healthKitBridge_;
  void *fitbitBridge_;

  // Streaming thread
  std::thread streamingThread_;
  std::atomic<bool> shouldStream_;
  void streamingLoop();

  void addToHistory(const BiometricData &data);

  /** Convert heart rate to arousal */
  float heartRateToArousal(float bpm) const;

  /** Convert skin conductance to intensity */
  float conductanceToIntensity(float conductance) const;

  /** Convert movement to arousal */
  float movementToArousal(float movement) const;
};

} // namespace kelly
