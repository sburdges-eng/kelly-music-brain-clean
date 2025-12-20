#include "engine/QuantumEntropy.h"
#include <algorithm>
#include <cmath>
#include <complex>

namespace kelly {

float QuantumEntropy::calculateEntropy(const QuantumEmotionalState& qState) const {
    // S_E = -Σ_i P_i ln(P_i)
    float entropy = 0.0f;
    
    for (const auto& state : qState.states) {
        if (state.probability > 0.0f) {
            entropy -= state.probability * std::log(state.probability);
        }
    }
    
    return entropy;
}

float QuantumEntropy::calculateMutualInformation(
    const QuantumEmotionalState& stateA,
    const QuantumEmotionalState& stateB
) const {
    // I(E_A; E_B) = Σ_i,j P(E_A, E_B) ln(P(E_A, E_B) / (P(E_A) P(E_B)))
    float mutualInfo = 0.0f;
    
    for (const auto& aState : stateA.states) {
        for (const auto& bState : stateB.states) {
            float jointProb = calculateJointProbability(aState, bState);
            float marginalA = aState.probability;
            float marginalB = bState.probability;
            
            if (jointProb > 0.0f && marginalA > 0.0f && marginalB > 0.0f) {
                float ratio = jointProb / (marginalA * marginalB);
                if (ratio > 0.0f) {
                    mutualInfo += jointProb * std::log(ratio);
                }
            }
        }
    }
    
    return mutualInfo;
}

float QuantumEntropy::calculateDecoherence(
    float initialCoherence,
    float time,
    float decoherenceRate
) const {
    // ρ(t) = ρ₀ e^(-Γt)
    return initialCoherence * std::exp(-decoherenceRate * time);
}

float QuantumEntropy::calculateVonNeumannEntropy(
    const QuantumEmotionalState& qState
) const {
    // Simplified: use probabilities as eigenvalues
    // S_vN = -Σ_i λ_i ln(λ_i) where λ_i are eigenvalues
    // For pure states, use probabilities directly
    
    float entropy = 0.0f;
    for (const auto& state : qState.states) {
        if (state.probability > 0.0f) {
            entropy -= state.probability * std::log(state.probability);
        }
    }
    
    return entropy;
}

float QuantumEntropy::calculateInformationContent(float probability) const {
    // I = -log₂(P)
    if (probability <= 0.0f) return 0.0f;
    return -std::log2(probability);
}

float QuantumEntropy::calculateTotalInformation(const QuantumEmotionalState& qState) const {
    // Total information = Σ_i I_i P_i
    float totalInfo = 0.0f;
    
    for (const auto& state : qState.states) {
        if (state.probability > 0.0f) {
            float info = calculateInformationContent(state.probability);
            totalInfo += info * state.probability;
        }
    }
    
    return totalInfo;
}

float QuantumEntropy::calculateJointProbability(
    const EmotionState& stateA,
    const EmotionState& stateB
) const {
    // Simplified: if same basis, use product; otherwise use interference
    if (stateA.basis == stateB.basis) {
        return stateA.probability * stateB.probability;
    } else {
        // Interference term
        Complex overlap = std::conj(stateA.amplitude) * stateB.amplitude;
        return std::norm(overlap);
    }
}

} // namespace kelly
