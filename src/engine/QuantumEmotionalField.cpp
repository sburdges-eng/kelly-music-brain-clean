#include "engine/QuantumEmotionalField.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace kelly {

// Use the renamed type from header
using QuantumEmotionState = QuantumEmotionBasisState;

QuantumEmotionalField::QuantumEmotionalField() {
  // Initialize default basis emotions with VAD coordinates
  // Based on Plutchik's wheel and common emotional models
  basisVADMap_ = {{"Joy", VADState(0.8f, 0.7f, 0.6f)},
                  {"Sadness", VADState(-0.7f, 0.3f, 0.2f)},
                  {"Anger", VADState(-0.6f, 0.9f, 0.4f)},
                  {"Fear", VADState(-0.6f, 0.8f, 0.1f)},
                  {"Trust", VADState(0.5f, 0.4f, 0.5f)},
                  {"Surprise", VADState(0.0f, 0.9f, 0.3f)},
                  {"Disgust", VADState(-0.7f, 0.5f, 0.3f)},
                  {"Anticipation", VADState(0.3f, 0.6f, 0.5f)}};
}

void QuantumEmotionalField::initializeBasis(
    const std::map<EmotionBasis, VADState> &basisMap) {
  basisVADMap_ = basisMap;
}

void QuantumEmotionalState::normalize() {
  float totalProb = 0.0f;
  for (const auto &state : states) {
    totalProb += state.probability;
  }

  if (totalProb > 0.0f) {
    float norm = std::sqrt(totalProb);
    for (auto &state : states) {
      state.amplitude /= norm;
      state.probability = std::norm(state.amplitude);
    }
  }
}

VADState QuantumEmotionalState::toVAD(
    const std::map<EmotionBasis, VADState> &basisVADMap) const {
  VADState result;
  float totalProb = 0.0f;

  for (const auto &qState : states) {
    auto it = basisVADMap.find(qState.basis);
    if (it != basisVADMap.end()) {
      float weight = qState.probability;
      result.valence += it->second.valence * weight;
      result.arousal += it->second.arousal * weight;
      result.dominance += it->second.dominance * weight;
      totalProb += weight;
    }
  }

  if (totalProb > 0.0f) {
    result.valence /= totalProb;
    result.arousal /= totalProb;
    result.dominance /= totalProb;
  }

  result.clamp();
  return result;
}

QuantumEmotionalState
QuantumEmotionalField::createSuperposition(const VADState &vad) const {
  QuantumEmotionalState qState;

  // Project VAD onto each basis emotion
  // Amplitude is proportional to similarity (inverse distance)
  for (const auto &[basis, basisVAD] : basisVADMap_) {
    float distance = vad.distanceTo(basisVAD);
    float maxDistance = std::sqrt(3.0f); // Max distance in unit cube

    // Convert distance to amplitude (closer = stronger)
    float similarity = 1.0f - (distance / maxDistance);
    similarity = std::max(0.0f, similarity);

    // Use similarity as magnitude, phase from VAD
    float phase = std::atan2(vad.arousal - 0.5f, vad.valence);
    Complex amplitude(similarity * std::cos(phase),
                      similarity * std::sin(phase));

    QuantumEmotionBasisState eState(basis, amplitude);
    qState.states.push_back(eState);
  }

  // Normalize
  qState.normalize();

  // Calculate coherence and entropy
  qState.coherence = calculateCoherence(qState.states);
  qState.entropy = calculateEntropy(qState.states);

  return qState;
}

float QuantumEmotionalField::calculateInterference(
    const QuantumEmotionalState &psi1,
    const QuantumEmotionalState &psi2) const {
  // I = |Ψ₁ + Ψ₂|² = |Ψ₁|² + |Ψ₂|² + 2Re(Ψ₁*Ψ₂)

  float psi1Norm = 0.0f, psi2Norm = 0.0f;
  Complex crossTerm(0.0f, 0.0f);

  // Create map for efficient lookup
  std::map<EmotionBasis, Complex> psi1Map, psi2Map;
  for (const auto &state : psi1.states) {
    psi1Map[state.basis] = state.amplitude;
    psi1Norm +=
        std::norm(state.amplitude); // Calculate probability from amplitude
  }
  for (const auto &state : psi2.states) {
    psi2Map[state.basis] = state.amplitude;
    psi2Norm +=
        std::norm(state.amplitude); // Calculate probability from amplitude
  }

  // Calculate cross term: Σ (α₁* · α₂)
  for (const auto &[basis, amp1] : psi1Map) {
    auto it = psi2Map.find(basis);
    if (it != psi2Map.end()) {
      crossTerm += std::conj(amp1) * it->second;
    }
  }

  float interference = psi1Norm + psi2Norm + 2.0f * std::real(crossTerm);

  // Normalize to 0-1 range
  float maxInterference =
      psi1Norm + psi2Norm + 2.0f * std::sqrt(psi1Norm * psi2Norm);
  if (maxInterference > 0.0f) {
    interference /= maxInterference;
  }

  return std::clamp(interference, 0.0f, 1.0f);
}

QuantumEmotionalField::EntangledState QuantumEmotionalField::createEntanglement(
    const QuantumEmotionalState &stateA,
    const QuantumEmotionalState &stateB) const {
  EntangledState entangled;

  // Create Bell-like state: |Ψ_AB⟩ = (1/√2)(|e_A, e_B⟩ + |e'_A, e'_B⟩)
  // Simplified: mix states with equal weight

  entangled.agentA = stateA;
  entangled.agentB = stateB;

  // Calculate entanglement strength from correlation
  Complex correlation = innerProduct(stateA, stateB);
  entangled.entanglementStrength = std::abs(correlation);

  // Normalize entanglement strength
  entangled.entanglementStrength =
      std::clamp(entangled.entanglementStrength, 0.0f, 1.0f);

  return entangled;
}

EmotionBasis QuantumEmotionalField::collapse(const QuantumEmotionalState &state,
                                             float randomValue) const {
  if (state.states.empty()) {
    return "Neutral";
  }

  // Generate random value if not provided
  if (randomValue < 0.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    randomValue = dis(gen);
  }

  // Collapse to emotion with probability |α_j|²
  float cumulative = 0.0f;
  for (const auto &eState : state.states) {
    cumulative += std::norm(eState.amplitude); // probability = |amplitude|²
    if (randomValue <= cumulative) {
      return eState.basis;
    }
  }

  // Fallback to highest probability
  auto maxIt = std::max_element(
      state.states.begin(), state.states.end(),
      [](const QuantumEmotionBasisState &a, const QuantumEmotionBasisState &b) {
        return std::norm(a.amplitude) < std::norm(b.amplitude);
      });

  return (maxIt != state.states.end()) ? maxIt->basis : "Neutral";
}

float QuantumEmotionalField::calculateEnergy(const QuantumEmotionalState &state,
                                             float omega, float hbar) const {
  // E_emotion = ℏω(n + 1/2)
  // n = emotional excitation level (sum of probabilities weighted by arousal)

  float n = 0.0f;
  for (const auto &eState : state.states) {
    auto it = basisVADMap_.find(eState.basis);
    if (it != basisVADMap_.end()) {
      // Weight by arousal (higher arousal = higher excitation)
      n += eState.probability * it->second.arousal;
    }
  }

  return hbar * omega * (n + 0.5f);
}

float QuantumEmotionalField::calculateTemperature(float energy,
                                                  float kB) const {
  if (kB <= 0.0f)
    return 0.0f;
  return energy / kB;
}

QuantumEmotionalState QuantumEmotionalField::evolve(
    const QuantumEmotionalState &initialState, float deltaTime,
    const std::map<EmotionBasis, float> &hamiltonian) const {
  QuantumEmotionalState evolved = initialState;

  // Simplified evolution: dΨ/dt = -iĤΨ
  // For diagonal Hamiltonian: α_i(t) = α_i(0) * e^(-i*H_ii*t)

  for (auto &eState : evolved.states) {
    auto it = hamiltonian.find(eState.basis);
    if (it != hamiltonian.end()) {
      float H_ii = it->second;
      Complex phase = std::exp(Complex(0.0f, -H_ii * deltaTime));
      eState.amplitude *= phase;
      eState.probability = std::norm(eState.amplitude);
    }
  }

  evolved.normalize();
  evolved.coherence = calculateCoherence(evolved.states);
  evolved.entropy = calculateEntropy(evolved.states);

  return evolved;
}

float QuantumEmotionalField::calculateResonance(
    const QuantumEmotionalState &state,
    const QuantumEmotionalState &stimulus) const {
  // R = Re(Ψ* · Φ_stim)
  Complex correlation = innerProduct(state, stimulus);
  return std::real(correlation);
}

float QuantumEmotionalField::calculateCoherence(
    const std::vector<QuantumEmotionBasisState> &states) const {
  // C = |Σ α_i|
  Complex sum(0.0f, 0.0f);
  for (const auto &state : states) {
    sum += state.amplitude;
  }
  return std::abs(sum);
}

float QuantumEmotionalField::calculateEntropy(
    const std::vector<QuantumEmotionBasisState> &states) const {
  // S = -Σ p_i log(p_i)
  float entropy = 0.0f;
  for (const auto &state : states) {
    float prob = std::norm(state.amplitude); // probability = |amplitude|²
    if (prob > 0.0f) {
      entropy -= prob * std::log2(prob);
    }
  }
  return entropy;
}

Complex
QuantumEmotionalField::innerProduct(const QuantumEmotionalState &psi1,
                                    const QuantumEmotionalState &psi2) const {
  Complex result(0.0f, 0.0f);

  std::map<EmotionBasis, Complex> psi2Map;
  for (const auto &state : psi2.states) {
    psi2Map[state.basis] = state.amplitude;
  }

  for (const auto &state : psi1.states) {
    auto it = psi2Map.find(state.basis);
    if (it != psi2Map.end()) {
      result += std::conj(state.amplitude) * it->second;
    }
  }

  return result;
}

} // namespace kelly
