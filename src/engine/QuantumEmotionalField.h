#pragma once

#include "common/Types.h"
#include "engine/VADCalculator.h"
#include <complex>
#include <map>
#include <string>
#include <vector>

namespace kelly {

/**
 * Quantum Emotional Field
 *
 * Implements quantum superposition model for emotions:
 * - Emotion superposition: |Ψ_E⟩ = Σ α_i |e_i⟩
 * - Emotional interference
 * - Emotional entanglement
 * - Collapse functions (observation/interaction)
 * - Quantum emotional energy
 */

// Complex number type for quantum amplitudes
using Complex = std::complex<float>;
using EmotionBasis = std::string; // "Joy", "Fear", "Anger", etc.

/**
 * Emotion basis state with quantum amplitude
 * Note: This is the quantum EmotionState, distinct from
 * KellyTypes::EmotionState
 */
struct QuantumEmotionBasisState {
  EmotionBasis basis; // "Joy", "Fear", "Anger", etc.
  Complex amplitude;  // α_i = complex amplitude (intensity + phase)
  float probability;  // |α_i|² (probability of experiencing this emotion)

  QuantumEmotionBasisState() : amplitude(0.0f, 0.0f), probability(0.0f) {}
  QuantumEmotionBasisState(const EmotionBasis &b, const Complex &amp)
      : basis(b), amplitude(amp), probability(std::norm(amp)) {}
};

/**
 * Quantum Emotional Field State
 */
struct QuantumEmotionalState {
  std::vector<QuantumEmotionBasisState> states; // Superposition of emotions
  float coherence; // Emotional coherence C = |Σ α_i|
  float entropy;   // Emotional entropy S = -Σ p_i log(p_i)
  double timestamp;

  QuantumEmotionalState() : coherence(0.0f), entropy(0.0f), timestamp(0.0) {}

  // Normalize amplitudes (ensure Σ |α_i|² = 1)
  void normalize();

  // Calculate expectation value of VAD from quantum state
  VADState toVAD(const std::map<EmotionBasis, VADState> &basisVADMap) const;
};

/**
 * Quantum Emotional Field Calculator
 */
class QuantumEmotionalField {
public:
  QuantumEmotionalField();

  /**
   * Initialize emotion basis states with VAD coordinates
   */
  void initializeBasis(const std::map<EmotionBasis, VADState> &basisMap);

  /**
   * Create quantum superposition from VAD state
   * Projects classical VAD onto quantum basis
   */
  QuantumEmotionalState createSuperposition(const VADState &vad) const;

  /**
   * Calculate interference between two emotional fields
   * I = |Ψ₁ + Ψ₂|² = |Ψ₁|² + |Ψ₂|² + 2Re(Ψ₁*Ψ₂)
   */
  float calculateInterference(const QuantumEmotionalState &psi1,
                              const QuantumEmotionalState &psi2) const;

  /**
   * Create entangled state between two agents
   * |Ψ_AB⟩ = (1/√2)(|e_A, e_B⟩ + |e'_A, e'_B⟩)
   */
  struct EntangledState {
    QuantumEmotionalState agentA;
    QuantumEmotionalState agentB;
    float entanglementStrength; // 0.0-1.0
  };

  EntangledState createEntanglement(const QuantumEmotionalState &stateA,
                                    const QuantumEmotionalState &stateB) const;

  /**
   * Collapse quantum state to classical emotion (observation/interaction)
   * Returns emotion with probability |α_j|²
   */
  EmotionBasis collapse(const QuantumEmotionalState &state,
                        float randomValue = -1.0f) const;

  /**
   * Calculate quantum emotional energy
   * E_emotion = ℏω(n + 1/2)
   */
  float
  calculateEnergy(const QuantumEmotionalState &state,
                  float omega = 1.0f, // Frequency of emotional fluctuation
                  float hbar = 1.0f   // Emotional sensitivity constant
  ) const;

  /**
   * Calculate emotional temperature from energy
   * T_E = k_B^(-1) * E_emotion
   */
  float calculateTemperature(float energy, float kB = 1.0f) const;

  /**
   * Evolve quantum state over time
   * dΨ/dt = -iĤΨ (simplified version)
   */
  QuantumEmotionalState
  evolve(const QuantumEmotionalState &initialState, float deltaTime,
         const std::map<EmotionBasis, float> &hamiltonian // Ĥ diagonal elements
  ) const;

  /**
   * Calculate resonance with external stimulus
   * R = Re(Ψ* · Φ_stim)
   */
  float calculateResonance(const QuantumEmotionalState &state,
                           const QuantumEmotionalState &stimulus) const;

  /**
   * Get basis VAD map
   */
  const std::map<EmotionBasis, VADState> &getBasisVADMap() const {
    return basisVADMap_;
  }

private:
  std::map<EmotionBasis, VADState>
      basisVADMap_; // VAD coordinates for each basis emotion

  // Helper functions
  float
  calculateCoherence(const std::vector<QuantumEmotionBasisState> &states) const;
  float
  calculateEntropy(const std::vector<QuantumEmotionBasisState> &states) const;
  Complex innerProduct(const QuantumEmotionalState &psi1,
                       const QuantumEmotionalState &psi2) const;
};

/**
 * Classical VAD Formulas
 */
namespace ClassicalVAD {
/**
 * Energy level: E_n = A × (1 + |V|)
 */
inline float calculateEnergy(float valence, float arousal) {
  return arousal * (1.0f + std::abs(valence));
}

/**
 * Emotional tension: T = |V| × (1 - D)
 */
inline float calculateTension(float valence, float dominance) {
  return std::abs(valence) * (1.0f - dominance);
}

/**
 * Stability index: S = 1 - √((V² + A² + D²) / 3)
 */
inline float calculateStability(const VADState &vad) {
  float sumSq = vad.valence * vad.valence + vad.arousal * vad.arousal +
                vad.dominance * vad.dominance;
  return 1.0f - std::sqrt(sumSq / 3.0f);
}
} // namespace ClassicalVAD

} // namespace kelly
