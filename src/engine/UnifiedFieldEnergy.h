#pragma once

#include "engine/QuantumEmotionalField.h"
#include "engine/EmotionalPotentialEnergy.h"
#include "engine/NetworkDynamics.h"
#include "engine/PhysiologicalResonance.h"
#include <vector>
#include <map>

namespace kelly {

/**
 * Unified Field Energy System
 * 
 * Implements the complete Quantum Emotional Field Lagrangian:
 * L_QEF = (1/2)|∇Ψ_E|² - U_E + g_bio E_bio + g_net Σ K_ij(E_j - E_i)² + g_res |Ψ_E|⁴
 */

/**
 * Unified Field Energy Calculator
 */
class UnifiedFieldEnergy {
public:
    UnifiedFieldEnergy();
    
    /**
     * Calculate Lagrangian
     * L_QEF = (1/2)|∇Ψ_E|² - U_E + g_bio E_bio + g_net Σ K_ij(E_j - E_i)² + g_res |Ψ_E|⁴
     */
    float calculateLagrangian(
        const QuantumEmotionalState& qState,
        const VADState& vad,
        const PhysiologicalEnergy& bioEnergy,
        const std::vector<Agent>& network,
        float gBio = 0.5f,
        float gNet = 0.3f,
        float gRes = 0.2f
    ) const;
    
    /**
     * Calculate total field energy
     * E_total = E_emotion + E_music + E_voice + E_bio + E_network + E_resonance
     */
    float calculateTotalEnergy(
        float emotionalEnergy,
        float musicEnergy,
        float voiceEnergy,
        const PhysiologicalEnergy& bioEnergy,
        float networkEnergy,
        float resonanceEnergy
    ) const;
    
    /**
     * Calculate gradient term: |∇Ψ_E|²
     */
    float calculateGradientTerm(const QuantumEmotionalState& qState) const;
    
    /**
     * Calculate network coupling term: Σ K_ij(E_j - E_i)²
     */
    float calculateNetworkCouplingTerm(
        const Agent& agent,
        const std::vector<Agent>& neighbors
    ) const;
    
    /**
     * Calculate resonance term: |Ψ_E|⁴
     */
    float calculateResonanceTerm(const QuantumEmotionalState& qState) const;
    
    /**
     * Calculate self-organization index
     * Ω = ⟨|Ψ|²⟩² / ⟨|Ψ|⁴⟩
     */
    float calculateSelfOrganizationIndex(const QuantumEmotionalState& qState) const;
    
    /**
     * Set coupling constants
     */
    void setCouplingConstants(float gBio, float gNet, float gRes) {
        gBio_ = gBio;
        gNet_ = gNet;
        gRes_ = gRes;
    }
    
private:
    float gBio_, gNet_, gRes_;  // Coupling constants
    NetworkDynamics networkDynamics_;
    EmotionalPotentialEnergy potentialEnergy_;
};

} // namespace kelly
