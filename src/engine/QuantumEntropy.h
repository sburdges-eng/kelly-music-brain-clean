#pragma once

#include "engine/QuantumEmotionalField.h"
#include <vector>

namespace kelly {

/**
 * Quantum Entropy and Information Theory
 * 
 * Implements:
 * - Emotional entropy
 * - Mutual information
 * - Decoherence
 * - Information measures
 */

/**
 * Quantum Entropy Calculator
 */
class QuantumEntropy {
public:
    /**
     * Calculate emotional entropy
     * S_E = -Σ_i P_i ln(P_i)
     */
    float calculateEntropy(const QuantumEmotionalState& qState) const;
    
    /**
     * Calculate mutual information between two emotional states
     * I(E_A; E_B) = Σ_i,j P(E_A, E_B) ln(P(E_A, E_B) / (P(E_A) P(E_B)))
     */
    float calculateMutualInformation(
        const QuantumEmotionalState& stateA,
        const QuantumEmotionalState& stateB
    ) const;
    
    /**
     * Calculate decoherence over time
     * ρ(t) = ρ₀ e^(-Γt)
     */
    float calculateDecoherence(
        float initialCoherence,
        float time,
        float decoherenceRate = 0.1f
    ) const;
    
    /**
     * Calculate von Neumann entropy
     * S_vN = -Tr(ρ ln ρ)
     */
    float calculateVonNeumannEntropy(
        const QuantumEmotionalState& qState
    ) const;
    
    /**
     * Calculate information content
     * I = -log₂(P)
     */
    float calculateInformationContent(float probability) const;
    
    /**
     * Calculate total information in quantum state
     */
    float calculateTotalInformation(const QuantumEmotionalState& qState) const;
    
private:
    // Helper functions
    float calculateJointProbability(
        const EmotionState& stateA,
        const EmotionState& stateB
    ) const;
};

} // namespace kelly
