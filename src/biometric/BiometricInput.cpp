#include "biometric/BiometricInput.h"
// Include bridge headers to provide complete types for unique_ptr
// Note: HealthKitBridge is disabled (requires Objective-C++)
// FitbitBridge should be available
#include "biometric/FitbitBridge.h"
#if HEALTHKIT_AVAILABLE
#include "biometric/HealthKitBridge.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <thread>

namespace kelly {

BiometricInput::~BiometricInput() {
  // Clean up bridge instances
#if HEALTHKIT_AVAILABLE
  if (healthKitBridge_) {
    delete static_cast<biometric::HealthKitBridge *>(healthKitBridge_);
  }
#endif
  if (fitbitBridge_) {
    delete static_cast<biometric::FitbitBridge *>(fitbitBridge_);
  }
  if (streamingActive_) {
    stopStreaming();
  }
}

BiometricInput::BiometricInput()
    : enabled_(false), adaptiveNormalization_(false), streamingActive_(false),
      lastReadingTime_(0.0), healthKitInitialized_(false),
      fitbitInitialized_(false), healthKitBridge_(nullptr),
      fitbitBridge_(nullptr), shouldStream_(false) {
  dataHistory_.reserve(HISTORY_SIZE);

  // Initialize default baseline
  baseline_.heartRate = 70.0f;
  baseline_.heartRateVariability = 50.0f;
  baseline_.skinConductance = 5.0f;
  baseline_.temperature = 36.5f;
}

BiometricInput::EmotionFromBiometrics
BiometricInput::processBiometricData(const BiometricData &data) {
  if (!enabled_) {
    return {0.0f, 0.5f, 0.5f}; // Neutral default
  }

  EmotionFromBiometrics result;

  // Process heart rate -> arousal
  if (data.heartRate) {
    result.arousal = heartRateToArousal(*data.heartRate);
  } else {
    result.arousal = 0.5f; // Default
  }

  // Process HRV if available (affects both arousal and intensity)
  if (data.heartRateVariability) {
    float hrv = *data.heartRateVariability;
    // HRV affects arousal: high HRV = lower stress = lower arousal
    // Normal HRV: 20-60ms, high HRV (>40ms) = calm, low HRV (<20ms) = stressed
    if (hrv > 40.0f) {
      result.arousal *= 0.8f; // High HRV = calmer
    } else if (hrv < 20.0f) {
      result.arousal *= 1.2f; // Low HRV = more stressed/aroused
      result.arousal = std::clamp(result.arousal, 0.0f, 1.0f);
    }

    // HRV also affects intensity: low HRV = higher emotional intensity
    if (data.skinConductance) {
      // Combine EDA and HRV for intensity
      float edaIntensity = conductanceToIntensity(*data.skinConductance);
      float hrvIntensity =
          1.0f - (hrv / 60.0f); // Inverse: low HRV = high intensity
      hrvIntensity = std::clamp(hrvIntensity, 0.0f, 1.0f);
      result.intensity = (edaIntensity + hrvIntensity) / 2.0f;
    } else {
      // Use HRV alone for intensity
      result.intensity = 1.0f - (hrv / 60.0f);
      result.intensity = std::clamp(result.intensity, 0.0f, 1.0f);
    }
  } else {
    // Process skin conductance -> intensity (fallback if no HRV)
    if (data.skinConductance) {
      result.intensity = conductanceToIntensity(*data.skinConductance);
    } else {
      result.intensity = 0.5f; // Default
    }
  }

  // Movement can also affect arousal
  if (data.movement) {
    float movementArousal = movementToArousal(*data.movement);
    result.arousal = (result.arousal + movementArousal) / 2.0f; // Average
  }

  // Valence is harder to determine from basic biometrics
  // Could use temperature (warmer = more positive) or other signals
  if (data.temperature) {
    // Normal body temp ~37°C, warmer might indicate positive state
    float tempNorm =
        (*data.temperature - 36.0f) / 2.0f; // Normalize around 36-38°C
    result.valence = std::clamp(tempNorm, -1.0f, 1.0f);
  } else {
    result.valence = 0.0f; // Neutral
  }

  // Clamp all values
  result.valence = std::clamp(result.valence, -1.0f, 1.0f);
  result.arousal = std::clamp(result.arousal, 0.0f, 1.0f);
  result.intensity = std::clamp(result.intensity, 0.0f, 1.0f);

  // Add to history for smoothing
  addToHistory(data);

  // Trigger callback
  if (onDataCallback_) {
    onDataCallback_(data);
  }

  return result;
}

void BiometricInput::simulateInput(const BiometricData &data) {
  if (enabled_) {
    processBiometricData(data);
  }
}

BiometricInput::BiometricData BiometricInput::getSmoothedData() const {
  if (dataHistory_.empty()) {
    return {};
  }

  BiometricData smoothed;
  size_t count = dataHistory_.size();

  // Average heart rate
  float totalHR = 0.0f;
  int hrCount = 0;
  for (const auto &d : dataHistory_) {
    if (d.heartRate) {
      totalHR += *d.heartRate;
      hrCount++;
    }
  }
  if (hrCount > 0)
    smoothed.heartRate = totalHR / hrCount;

  // Average HRV (Heart Rate Variability)
  float totalHRV = 0.0f;
  int hrvCount = 0;
  for (const auto &d : dataHistory_) {
    if (d.heartRateVariability) {
      totalHRV += *d.heartRateVariability;
      hrvCount++;
    }
  }
  if (hrvCount > 0)
    smoothed.heartRateVariability = totalHRV / hrvCount;

  // Average skin conductance
  float totalSC = 0.0f;
  int scCount = 0;
  for (const auto &d : dataHistory_) {
    if (d.skinConductance) {
      totalSC += *d.skinConductance;
      scCount++;
    }
  }
  if (scCount > 0)
    smoothed.skinConductance = totalSC / scCount;

  // Average temperature
  float totalTemp = 0.0f;
  int tempCount = 0;
  for (const auto &d : dataHistory_) {
    if (d.temperature) {
      totalTemp += *d.temperature;
      tempCount++;
    }
  }
  if (tempCount > 0)
    smoothed.temperature = totalTemp / tempCount;

  // Average movement
  float totalMove = 0.0f;
  int moveCount = 0;
  for (const auto &d : dataHistory_) {
    if (d.movement) {
      totalMove += *d.movement;
      moveCount++;
    }
  }
  if (moveCount > 0)
    smoothed.movement = totalMove / moveCount;

  // Use most recent timestamp
  smoothed.timestamp = dataHistory_.back().timestamp;

  return smoothed;
}

void BiometricInput::onBiometricData(
    std::function<void(const BiometricData &)> callback) {
  onDataCallback_ = callback;
}

void BiometricInput::addToHistory(const BiometricData &data) {
  dataHistory_.push_back(data);
  if (dataHistory_.size() > HISTORY_SIZE) {
    dataHistory_.erase(dataHistory_.begin());
  }
}

float BiometricInput::heartRateToArousal(float bpm) const {
  // Normal resting HR: 60-100 BPM
  // High arousal: >100 BPM
  // Low arousal: <60 BPM (or very calm)

  if (bpm < 60.0f) {
    return 0.2f; // Very calm
  } else if (bpm < 80.0f) {
    return 0.3f + ((bpm - 60.0f) / 20.0f) * 0.2f; // 0.3 to 0.5
  } else if (bpm < 100.0f) {
    return 0.5f + ((bpm - 80.0f) / 20.0f) * 0.3f; // 0.5 to 0.8
  } else {
    return 0.8f + std::min((bpm - 100.0f) / 60.0f, 0.2f); // 0.8 to 1.0
  }
}

float BiometricInput::conductanceToIntensity(float conductance) const {
  // Skin conductance typically 1-20 microsiemens
  // Higher = more intense emotional state
  return std::clamp(conductance / 20.0f, 0.0f, 1.0f);
}

float BiometricInput::movementToArousal(float movement) const {
  // Movement already normalized 0.0 to 1.0
  // More movement = higher arousal
  return movement;
}

void BiometricInput::setBaseline(const BiometricData &baseline) {
  baseline_ = baseline;

  // Update baseline from smoothed history if adaptive normalization is enabled
  if (adaptiveNormalization_ && !dataHistory_.empty()) {
    auto smoothed = getSmoothedData();
    if (smoothed.heartRate)
      baseline_.heartRate = *smoothed.heartRate;
    if (smoothed.heartRateVariability)
      baseline_.heartRateVariability = *smoothed.heartRateVariability;
    if (smoothed.skinConductance)
      baseline_.skinConductance = *smoothed.skinConductance;
    if (smoothed.temperature)
      baseline_.temperature = *smoothed.temperature;
  }
}

bool BiometricInput::initializeHealthKit() {
#if HEALTHKIT_AVAILABLE && (JUCE_MAC || JUCE_IOS)
  if (!healthKitBridge_) {
    healthKitBridge_ = new biometric::HealthKitBridge();
  }

  auto *bridge = static_cast<biometric::HealthKitBridge *>(healthKitBridge_);
  if (!bridge->isAvailable()) {
    healthKitInitialized_ = false;
    return false;
  }

  // Request authorization
  if (bridge->requestAuthorization()) {
    healthKitInitialized_ = true;
    return true;
  }

  healthKitInitialized_ = false;
  return false;
#else
  // HealthKit not available (requires Objective-C++ compilation)
  healthKitInitialized_ = false;
  return false;
#endif
}

bool BiometricInput::initializeFitbit(const std::string &accessToken) {
  if (accessToken.empty()) {
    return false;
  }

  fitbitAccessToken_ = accessToken;

  // TODO: Implement Fitbit API integration
  // This requires:
  // 1. HTTP client for Fitbit API calls
  // 2. OAuth token management
  // 3. WebSocket connection for real-time data (if available)
  // 4. API endpoints:
  //    - GET /1/user/-/activities/heart/date/{date}/1d/{detail-level}.json
  //    - GET /1/user/-/hrv/date/{date}.json

  fitbitInitialized_ = false;
  return false; // Not yet implemented
}

bool BiometricInput::startStreaming() {
  if (!enabled_) {
    return false;
  }

  if (healthKitInitialized_) {
    // Start HealthKit streaming
    readHealthKitData();
    streamingActive_ = true;
    return true;
  } else if (fitbitInitialized_) {
    // Start Fitbit streaming (polling or WebSocket)
    // For now, use polling
    streamingActive_ = true;
    return true;
  }

  return false;
}

void BiometricInput::stopStreaming() { streamingActive_ = false; }

void BiometricInput::readHealthKitData() {
#if HEALTHKIT_AVAILABLE && (JUCE_MAC || JUCE_IOS)
  // TODO: Read from HealthKit
  // Example:
  // HKQuantityType* heartRateType = [HKQuantityType
  // quantityTypeForIdentifier:HKQuantityTypeIdentifierHeartRate]; Query latest
  // sample and convert to BiometricData Note: This requires Objective-C++ (.mm
  // file) for HealthKit integration
#else
  // HealthKit not available
  (void)0; // Suppress unused function warning
#endif
}

void BiometricInput::readFitbitData() {
  if (fitbitAccessToken_.empty()) {
    return;
  }

  // TODO: Implement Fitbit API call
  // Example HTTP request:
  // GET
  // https://api.fitbit.com/1/user/-/activities/heart/date/today/1d/1sec.json
  // Headers: Authorization: Bearer {accessToken}

  // Parse JSON response and convert to BiometricData
  // Update lastReadingTime_
}

} // namespace kelly
