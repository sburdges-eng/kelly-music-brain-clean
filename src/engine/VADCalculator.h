#pragma once

#include "common/Types.h"
#include "engine/EmotionThesaurus.h"
#include "biometric/BiometricInput.h"
#include <vector>
#include <deque>
#include <chrono>
#include <optional>

namespace kelly {

/**
 * VAD (Valence-Arousal-Dominance) Calculator
 * 
 * Calculates VAD coordinates from:
 * - Emotion IDs (primary method)
 * - Biometric data (HR, HRV, EDA, temperature)
 * - Context-aware adjustments (circadian, time-of-day)
 * 
 * Dominance represents sense of control/power:
 * - High dominance: feeling in control, powerful, assertive
 * - Low dominance: feeling submissive, powerless, overwhelmed
 */
struct VADState {
    float valence;      // -1.0 (negative) to 1.0 (positive)
    float arousal;      // 0.0 (calm) to 1.0 (excited)
    float dominance;    // 0.0 (submissive) to 1.0 (dominant)
    
    double timestamp;   // When this state was calculated
    
    VADState() : valence(0.0f), arousal(0.5f), dominance(0.5f), timestamp(0.0) {}
    VADState(float v, float a, float d) : valence(v), arousal(a), dominance(d), timestamp(0.0) {}
    
    void clamp() {
        valence = std::clamp(valence, -1.0f, 1.0f);
        arousal = std::clamp(arousal, 0.0f, 1.0f);
        dominance = std::clamp(dominance, 0.0f, 1.0f);
    }
    
    // Calculate distance in 3D VAD space
    float distanceTo(const VADState& other) const {
        float dv = valence - other.valence;
        float da = arousal - other.arousal;
        float dd = dominance - other.dominance;
        return std::sqrt(dv*dv + da*da + dd*dd);
    }
    
    // Helper method to get intensity from dominance
    float getIntensity() const {
        return dominance;  // Use dominance as intensity proxy
    }
};

/**
 * VAD Calculator - Core VAD calculation engine
 */
class VADCalculator {
public:
    explicit VADCalculator(const EmotionThesaurus* thesaurus = nullptr);
    
    /**
     * Calculate VAD from emotion ID (primary method)
     * @param emotionId The emotion ID from EmotionThesaurus
     * @param intensityModifier Optional intensity adjustment (0.0-2.0, default 1.0)
     * @return VAD state calculated from emotion
     */
    VADState calculateFromEmotionId(int emotionId, float intensityModifier = 1.0f) const;
    
    /**
     * Calculate VAD from emotion node directly
     */
    VADState calculateFromEmotion(const EmotionNode& emotion, float intensityModifier = 1.0f) const;
    
    /**
     * Calculate VAD from biometric data
     * @param biometricData Biometric readings (HR, HRV, EDA, temp)
     * @return VAD state derived from biometrics
     */
    VADState calculateFromBiometrics(const BiometricInput::BiometricData& biometricData) const;
    
    /**
     * Blend multiple VAD states with weights
     * @param states Vector of VAD states
     * @param weights Vector of weights (must sum to 1.0, or will be normalized)
     * @return Weighted average VAD state
     */
    VADState blendStates(const std::vector<VADState>& states, const std::vector<float>& weights) const;
    
    /**
     * Apply context-aware adjustments to VAD state
     * @param state Base VAD state
     * @param hourOfDay 0-23 (current hour)
     * @param dayOfWeek 0-6 (0=Sunday, 6=Saturday)
     * @param circadianPhase 0.0-1.0 (0=midnight, 0.5=noon, 1.0=midnight next day)
     * @return Adjusted VAD state
     */
    VADState applyContextAdjustments(
        const VADState& state,
        int hourOfDay,
        int dayOfWeek = 0,
        float circadianPhase = -1.0f  // -1 means calculate from hourOfDay
    ) const;
    
    /**
     * Calculate dominance from emotion characteristics
     * Dominance is inferred from:
     * - High arousal + positive valence = higher dominance (joy, excitement)
     * - High arousal + negative valence = lower dominance (fear, panic)
     * - Low arousal + positive valence = moderate dominance (contentment)
     * - Low arousal + negative valence = lower dominance (sadness, grief)
     */
    static float calculateDominanceFromEmotion(const EmotionNode& emotion);
    
    /**
     * Calculate dominance from biometrics
     * Uses HRV (Heart Rate Variability) as primary indicator:
     * - High HRV = higher dominance (autonomic flexibility = sense of control)
     * - Low HRV = lower dominance (stress response = feeling overwhelmed)
     */
    static float calculateDominanceFromBiometrics(const BiometricInput::BiometricData& data);
    
    /**
     * Smooth VAD state using moving average
     * @param states History of VAD states
     * @param windowSize Number of states to average (default: 5)
     * @return Smoothed VAD state
     */
    static VADState smoothMovingAverage(
        const std::vector<VADState>& states,
        size_t windowSize = 5
    );
    
    /**
     * Smooth VAD state using exponential smoothing
     * @param current Current VAD state
     * @param previous Previous smoothed state
     * @param alpha Smoothing factor (0.0-1.0, higher = more responsive)
     * @return Exponentially smoothed VAD state
     */
    static VADState smoothExponential(
        const VADState& current,
        const VADState& previous,
        float alpha = 0.3f
    );
    
    /**
     * Smooth VAD state using weighted moving average
     * @param states History of VAD states
     * @param weights Weights for each state (most recent = highest weight)
     * @return Weighted smoothed VAD state
     */
    static VADState smoothWeighted(
        const std::vector<VADState>& states,
        const std::vector<float>& weights
    );
    
    /**
     * Apply Kalman filter smoothing to VAD state
     * @param current Current measurement
     * @param previous Previous estimate
     * @param processNoise Process noise covariance (default: 0.01)
     * @param measurementNoise Measurement noise covariance (default: 0.1)
     * @return Kalman filtered VAD state
     */
    static VADState smoothKalman(
        const VADState& current,
        const VADState& previous,
        float processNoise = 0.01f,
        float measurementNoise = 0.1f
    );
    
    /**
     * Validate emotion ID
     */
    bool isValidEmotionId(int emotionId) const;
    
    /**
     * Validate VAD state (check ranges)
     */
    static bool isValidVADState(const VADState& state);
    
    /**
     * Normalize VAD state (ensure all values in valid ranges)
     */
    static VADState normalizeVADState(const VADState& state);
    
    /**
     * Calculate VAD from emotion name (alternative to ID)
     */
    VADState calculateFromEmotionName(const std::string& emotionName, float intensityModifier = 1.0f) const;
    
    /**
     * Interpolate between two VAD states
     * @param state1 First state
     * @param state2 Second state
     * @param t Interpolation factor (0.0 = state1, 1.0 = state2)
     * @return Interpolated VAD state
     */
    static VADState interpolate(const VADState& state1, const VADState& state2, float t);
    
    /**
     * Calculate rate of change between two VAD states
     * @param state1 First state
     * @param state2 Second state
     * @param deltaTime Time difference in seconds
     * @return Rate of change per second
     */
    static VADState calculateRateOfChange(
        const VADState& state1,
        const VADState& state2,
        float deltaTime
    );
    
    /**
     * Get thesaurus accessor
     */
    const EmotionThesaurus* getThesaurus() const { return thesaurus_; }
    
private:
    const EmotionThesaurus* thesaurus_;  // Non-owning pointer - lifetime guaranteed by caller
    
    // Dominance calculation helpers
    float calculateDominanceFromVA(float valence, float arousal) const;
    
    // Context adjustment functions
    float getCircadianArousalModifier(int hourOfDay) const;
    float getCircadianValenceModifier(int hourOfDay) const;
    float getCircadianDominanceModifier(int hourOfDay) const;
    float getDayOfWeekModifier(int dayOfWeek) const;
};

} // namespace kelly
