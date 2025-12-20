#pragma once

#include "engine/QuantumEmotionalField.h"
#include <complex>

namespace kelly {

/**
 * Hybrid AI-Human Emotional Coupling
 * 
 * Implements:
 * - Ψ_hybrid = αΨ_AI + βΨ_human
 * - Cross-influence term: ΔH = κ Re(Ψ_AI* Ψ_human)
 * - Normalization: |α|² + |β|² = 1
 */

/**
 * Hybrid Coupling Calculator
 */
class HybridCoupling {
public:
    HybridCoupling(float alpha = 0.5f, float beta = 0.5f, float kappa = 0.3f);
    
    /**
     * Create hybrid emotional state
     * Ψ_hybrid = αΨ_AI + βΨ_human
     */
    QuantumEmotionalState createHybridState(
        const QuantumEmotionalState& aiState,
        const QuantumEmotionalState& humanState
    ) const;
    
    /**
     * Calculate cross-influence term
     * ΔH = κ Re(Ψ_AI* Ψ_human)
     */
    float calculateCrossInfluence(
        const QuantumEmotionalState& aiState,
        const QuantumEmotionalState& humanState
    ) const;
    
    /**
     * Calculate coherence between AI and human states
     */
    float calculateCoherence(
        const QuantumEmotionalState& aiState,
        const QuantumEmotionalState& humanState
    ) const;
    
    /**
     * Set coupling parameters
     */
    void setParameters(float alpha, float beta, float kappa) {
        alpha_ = alpha;
        beta_ = beta;
        kappa_ = kappa;
        normalizeWeights();
    }
    
    float getAlpha() const { return alpha_; }
    float getBeta() const { return beta_; }
    float getKappa() const { return kappa_; }
    
private:
    float alpha_, beta_, kappa_;  // Coupling parameters
    
    void normalizeWeights();
};

} // namespace kelly
