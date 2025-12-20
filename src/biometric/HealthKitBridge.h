#pragma once

/**
 * HealthKitBridge - macOS HealthKit Integration
 * =============================================
 *
 * Provides real-time access to health data from Apple HealthKit:
 * - Heart rate (HR)
 * - Heart rate variability (HRV)
 * - Heart rate recovery
 * - Activity data
 *
 * Note: Requires HealthKit framework and proper entitlements.
 */

#include <chrono>
#include <functional>
#include <juce_core/juce_core.h>
#include <memory>
#include <vector>

#ifdef JUCE_MAC
// HealthKit requires Objective-C++, but we're compiling as C++
// Define a stub interface that can be used from C++ code
// The actual implementation would need to be in a .mm file
#define HEALTHKIT_AVAILABLE 0 // Disable for now to allow C++ compilation
// Note: To enable HealthKit, create HealthKitBridge.mm (Objective-C++) file
// and move HealthKit-specific code there
#else
#define HEALTHKIT_AVAILABLE 0
#endif

namespace kelly {
namespace biometric {

struct HealthKitData {
  float heartRate = 0.0f;            // BPM
  float heartRateVariability = 0.0f; // ms
  float restingHeartRate = 0.0f;     // BPM
  float activeEnergyBurned = 0.0f;   // kcal
  double timestamp = 0.0;

  bool isValid() const { return heartRate > 0.0f && timestamp > 0.0; }
};

/**
 * HealthKitBridge - Interface to Apple HealthKit
 */
class HealthKitBridge {
public:
  HealthKitBridge();
  ~HealthKitBridge();

  /**
   * Request HealthKit authorization
   * @return true if authorized or authorization granted
   */
  bool requestAuthorization();

  /**
   * Check if HealthKit is available and authorized
   */
  bool isAvailable() const;
  bool isAuthorized() const;

  /**
   * Get latest heart rate data
   */
  HealthKitData getLatestHeartRate();

  /**
   * Get heart rate variability
   */
  float getHeartRateVariability();

  /**
   * Get resting heart rate (baseline)
   */
  float getRestingHeartRate();

  /**
   * Start continuous heart rate monitoring
   * @param callback Function called when new data arrives
   */
  void startMonitoring(std::function<void(const HealthKitData &)> callback);

  /**
   * Stop monitoring
   */
  void stopMonitoring();

  /**
   * Get historical baseline (average over last N days)
   * @param days Number of days to average
   */
  HealthKitData getHistoricalBaseline(int days = 7);

  /**
   * Calculate adaptive normalization factors based on user's baseline
   */
  struct NormalizationFactors {
    float hrScale = 1.0f;
    float hrvScale = 1.0f;
    float hrOffset = 0.0f;
    float hrvOffset = 0.0f;
  };

  NormalizationFactors calculateNormalizationFactors();

private:
  bool authorized_ = false;
  bool monitoring_ = false;

  std::function<void(const HealthKitData &)> dataCallback_;

  // Historical data cache
  std::vector<HealthKitData> historicalData_;
  std::chrono::system_clock::time_point lastUpdate_;

  void updateHistoricalData();

#if HEALTHKIT_AVAILABLE
  // HealthKit store (Objective-C object, forward declared)
  void *healthStore_ = nullptr;
#endif
};

} // namespace biometric
} // namespace kelly
