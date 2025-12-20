#pragma once

#include "common/Types.h"
#include <algorithm>
#include <cmath>

namespace kelly {

/**
 * EmotionMusicMapper - Precise mathematical formulas for emotion-to-music
 * translation.
 *
 * Implements research-based formulas:
 * - tempo = 60 + 120 * arousal
 * - velocity = 60 + 67 * dominance
 * - mode = major if valence > 0 else minor
 * - reward = 0.4*E + 0.3*C + 0.2*N + 0.1*F (therapeutic effectiveness)
 * - resonance = 0.3*Δhrv + 0.2*Δeda + 0.3*v + 0.2*c (biometric feedback)
 */
class EmotionMusicMapper {
public:
  EmotionMusicMapper() = default;

  //==========================================================================
  // Core Emotion-to-Music Formulas
  //==========================================================================

  /**
   * Calculate tempo from arousal.
   * Formula: tempo = 60 + 120 * arousal
   * Range: 60 BPM (arousal=0.0) to 180 BPM (arousal=1.0)
   */
  static int calculateTempo(float arousal) {
    arousal = std::clamp(arousal, 0.0f, 1.0f);
    return static_cast<int>(60.0f + 120.0f * arousal);
  }

  /**
   * Calculate MIDI velocity from dominance.
   * Formula: velocity = 60 + 67 * dominance
   * Range: 60 (dominance=0.0) to 127 (dominance=1.0)
   */
  static int calculateVelocity(float dominance) {
    dominance = std::clamp(dominance, 0.0f, 1.0f);
    return static_cast<int>(60.0f + 67.0f * dominance);
  }

  /**
   * Determine mode from valence.
   * Formula: mode = major if valence > 0 else minor
   */
  static std::string calculateMode(float valence) {
    return valence > 0.0f ? "major" : "minor";
  }

  /**
   * Determine detailed mode with more nuance.
   * Uses valence and arousal for richer modal selection.
   */
  static std::string calculateDetailedMode(float valence, float arousal) {
    if (valence > 0.5f) {
      // Positive emotions
      if (arousal > 0.7f)
        return "lydian"; // Bright, excited
      if (arousal > 0.4f)
        return "ionian";   // Standard major
      return "mixolydian"; // Relaxed major
    } else if (valence > 0.0f) {
      // Mildly positive
      return arousal > 0.5f ? "mixolydian" : "dorian";
    } else if (valence > -0.5f) {
      // Mildly negative
      return arousal > 0.5f ? "dorian" : "aeolian";
    } else {
      // Strongly negative
      if (arousal > 0.7f)
        return "phrygian"; // Intense, dark
      return "aeolian";    // Natural minor
    }
  }

  //==========================================================================
  // Therapeutic Effectiveness (Reward Function)
  //==========================================================================

  struct TherapeuticFactors {
    float emotionalExpression; // E: How well emotion is expressed
    float catharsis;           // C: Release/relief achieved
    float narrative;           // N: Story coherence
    float flow;                // F: Musical flow quality
  };

  /**
   * Calculate therapeutic reward (effectiveness).
   * Formula: reward = 0.4*E + 0.3*C + 0.2*N + 0.1*F
   *
   * @param factors Therapeutic effectiveness factors (each 0.0 to 1.0)
   * @return Reward value 0.0 to 1.0
   */
  static float calculateReward(const TherapeuticFactors &factors) {
    return 0.4f * factors.emotionalExpression + 0.3f * factors.catharsis +
           0.2f * factors.narrative + 0.1f * factors.flow;
  }

  //==========================================================================
  // Biometric Resonance
  //==========================================================================

  struct BiometricState {
    float deltaHRV;   // Δhrv: Change in heart rate variability
    float deltaEDA;   // Δeda: Change in electrodermal activity
    float valence;    // v: Emotional valence
    float complexity; // c: Musical complexity
  };

  /**
   * Calculate biometric resonance (body-music alignment).
   * Formula: resonance = 0.3*Δhrv + 0.2*Δeda + 0.3*v + 0.2*c
   *
   * @param state Biometric and musical state (each normalized 0.0 to 1.0)
   * @return Resonance value 0.0 to 1.0
   */
  static float calculateResonance(const BiometricState &state) {
    return 0.3f * state.deltaHRV + 0.2f * state.deltaEDA +
           0.3f * state.valence + 0.2f * state.complexity;
  }

  //==========================================================================
  // Dominance Calculation (PAD Model)
  //==========================================================================

  /**
   * Calculate dominance from emotion category and intensity.
   * Used when dominance is not directly available.
   *
   * Based on emotion category:
   * - Anger: High dominance (assertive, powerful)
   * - Fear: Low dominance (submissive, powerless)
   * - Joy: Medium-high dominance (confident)
   * - Sadness: Low-medium dominance (withdrawn)
   */
  static float estimateDominance(EmotionCategory category, float intensity) {
    float baseDominance = 0.5f;

    switch (category) {
    case EmotionCategory::Anger:
      baseDominance = 0.8f; // High dominance
      break;
    case EmotionCategory::Fear:
      baseDominance = 0.2f; // Low dominance
      break;
    case EmotionCategory::Joy:
      baseDominance = 0.7f; // Medium-high dominance
      break;
    case EmotionCategory::Sadness:
      baseDominance = 0.3f; // Low-medium dominance
      break;
    case EmotionCategory::Disgust:
      baseDominance = 0.6f; // Medium-high dominance
      break;
    case EmotionCategory::Surprise:
      baseDominance = 0.5f; // Neutral dominance
      break;
    case EmotionCategory::Trust:
      baseDominance = 0.6f; // Medium-high dominance
      break;
    case EmotionCategory::Anticipation:
      baseDominance = 0.6f; // Medium-high dominance
      break;
    }

    // Scale by intensity
    return baseDominance * (0.5f + 0.5f * intensity);
  }

  //==========================================================================
  // Complete Mapping
  //==========================================================================

  struct MusicalParameters {
    int tempo;                // BPM (60-180)
    int velocity;             // MIDI velocity (60-127)
    std::string mode;         // Mode name
    std::string detailedMode; // Detailed mode with more options
    float reward;             // Therapeutic effectiveness (0.0-1.0)
    float resonance;          // Biometric resonance (0.0-1.0)
  };

  /**
   * Map emotion to complete musical parameters.
   * Uses all formulas to generate comprehensive music settings.
   */
  static MusicalParameters
  mapEmotion(const EmotionNode &emotion,
             const TherapeuticFactors *therapeuticFactors = nullptr,
             const BiometricState *biometricState = nullptr) {
    MusicalParameters params;

    // Core formulas
    params.tempo = calculateTempo(emotion.arousal);

    // Use dominance if available, otherwise estimate
    float dominance = emotion.dominance;
    if (dominance == 0.5f) { // Default value, estimate instead
      // Use categoryEnum if available, otherwise default to Joy
      EmotionCategory cat = (emotion.categoryEnum != EmotionCategory::COUNT)
                                ? emotion.categoryEnum
                                : EmotionCategory::Joy;
      dominance = estimateDominance(cat, emotion.intensity);
    }
    params.velocity = calculateVelocity(dominance);

    params.mode = calculateMode(emotion.valence);
    params.detailedMode =
        calculateDetailedMode(emotion.valence, emotion.arousal);

    // Optional therapeutic factors
    if (therapeuticFactors) {
      params.reward = calculateReward(*therapeuticFactors);
    } else {
      // Estimate based on intensity (higher intensity = more expression)
      TherapeuticFactors estimated;
      estimated.emotionalExpression = emotion.intensity;
      estimated.catharsis = emotion.intensity * 0.8f;
      estimated.narrative = 0.7f; // Assume reasonable coherence
      estimated.flow = 0.6f;      // Assume decent flow
      params.reward = calculateReward(estimated);
    }

    // Optional biometric state
    if (biometricState) {
      params.resonance = calculateResonance(*biometricState);
    } else {
      // Estimate based on emotion alignment
      BiometricState estimated;
      estimated.deltaHRV = emotion.arousal * 0.5f;   // Assume some HRV change
      estimated.deltaEDA = emotion.intensity * 0.6f; // Assume some EDA change
      estimated.valence = (emotion.valence + 1.0f) / 2.0f; // Normalize to 0-1
      estimated.complexity = 0.5f; // Assume medium complexity
      params.resonance = calculateResonance(estimated);
    }

    return params;
  }
};

} // namespace kelly
