#pragma once

#include "engine/VADCalculator.h"
#include "biometric/BiometricInput.h"
#include "common/Types.h"
#include <vector>
#include <deque>
#include <cmath>

namespace kelly {

/**
 * Resonance Calculator
 * 
 * Calculates coherence/resonance between:
 * - Biometric data and emotion states
 * - Multiple biometric signals
 * - VAD states over time
 * 
 * Resonance indicates alignment between physiological state and emotional state.
 * High resonance = biometrics match emotions (authentic state)
 * Low resonance = mismatch (possible suppression, masking, or sensor issues)
 */
struct ResonanceMetrics {
    float coherence;           // 0.0-1.0, overall coherence between signals
    float biometricCoherence;  // 0.0-1.0, coherence within biometric signals
    float emotionBiometricMatch; // 0.0-1.0, match between emotion and biometrics
    float temporalStability;    // 0.0-1.0, stability of VAD over time
    
    ResonanceMetrics() : coherence(0.5f), biometricCoherence(0.5f), 
                         emotionBiometricMatch(0.5f), temporalStability(0.5f) {}
};

/**
 * Resonance Calculator
 */
class ResonanceCalculator {
public:
    ResonanceCalculator();
    
    /**
     * Calculate resonance between emotion VAD and biometric VAD
     * @param emotionVAD VAD calculated from emotion
     * @param biometricVAD VAD calculated from biometrics
     * @return Resonance metrics
     */
    ResonanceMetrics calculateResonance(
        const VADState& emotionVAD,
        const VADState& biometricVAD
    ) const;
    
    /**
     * Calculate coherence between multiple biometric signals
     * @param biometricData Current biometric reading
     * @param history Recent biometric history (for variance calculation)
     * @return Coherence score (0.0-1.0)
     */
    float calculateBiometricCoherence(
        const BiometricInput::BiometricData& biometricData,
        const std::vector<BiometricInput::BiometricData>& history
    ) const;
    
    /**
     * Calculate temporal stability of VAD states
     * @param vadHistory Recent VAD states (chronological order)
     * @return Stability score (0.0-1.0, higher = more stable)
     */
    float calculateTemporalStability(
        const std::vector<VADState>& vadHistory
    ) const;
    
    /**
     * Calculate cross-correlation between two VAD dimensions
     * @param states1 First set of VAD states
     * @param states2 Second set of VAD states
     * @param dimension 0=valence, 1=arousal, 2=dominance
     * @return Correlation coefficient (-1.0 to 1.0)
     */
    float calculateCorrelation(
        const std::vector<VADState>& states1,
        const std::vector<VADState>& states2,
        int dimension
    ) const;
    
    /**
     * Detect anomalies in VAD sequence
     * @param vadHistory Recent VAD states
     * @return Vector of anomaly scores (0.0-1.0, higher = more anomalous)
     */
    std::vector<float> detectAnomalies(const std::vector<VADState>& vadHistory) const;
    
private:
    // Helper functions
    float calculateVariance(const std::vector<float>& values) const;
    float calculateMean(const std::vector<float>& values) const;
    float calculateStandardDeviation(const std::vector<float>& values) const;
    
    // Expected ranges for biometric coherence
    static constexpr float EXPECTED_HR_MIN = 50.0f;
    static constexpr float EXPECTED_HR_MAX = 120.0f;
    static constexpr float EXPECTED_EDA_MIN = 1.0f;
    static constexpr float EXPECTED_EDA_MAX = 20.0f;
    static constexpr float EXPECTED_TEMP_MIN = 36.0f;
    static constexpr float EXPECTED_TEMP_MAX = 37.5f;
};

} // namespace kelly
