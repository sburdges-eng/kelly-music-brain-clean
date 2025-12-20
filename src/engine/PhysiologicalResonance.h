#pragma once

#include "engine/VADCalculator.h"
#include "biometric/BiometricInput.h"
#include <vector>

namespace kelly {

/**
 * Physiological Resonance System
 * 
 * Implements coupling between emotional energy and biological signals:
 * - E_bio(t) = α_H H(t) + α_R R(t) + α_G G(t)
 * - k_bio = dE/dE_bio
 * - Neural phase synchrony
 * - Biofield feedback loop
 */

struct PhysiologicalEnergy {
    float heartRateComponent;      // α_H H(t)
    float respirationComponent;    // α_R R(t)
    float galvanicComponent;       // α_G G(t)
    float totalEnergy;             // E_bio total
    
    PhysiologicalEnergy() 
        : heartRateComponent(0.0f), respirationComponent(0.0f),
          galvanicComponent(0.0f), totalEnergy(0.0f) {}
};

/**
 * Physiological Resonance Calculator
 */
class PhysiologicalResonance {
public:
    PhysiologicalResonance(
        float alphaH = 0.4f,  // Heart rate coupling coefficient
        float alphaR = 0.3f,  // Respiration coupling coefficient
        float alphaG = 0.3f    // Galvanic skin response coupling coefficient
    ) : alphaH_(alphaH), alphaR_(alphaR), alphaG_(alphaG) {}
    
    /**
     * Calculate physiological energy
     * E_bio(t) = α_H H(t) + α_R R(t) + α_G G(t)
     */
    PhysiologicalEnergy calculateBioEnergy(
        const BiometricInput::BiometricData& biometricData
    ) const;
    
    /**
     * Calculate emotion coupling constant
     * k_bio = dE/dE_bio = ΔE / Δ(α_H H + α_R R + α_G G)
     */
    float calculateCouplingConstant(
        float deltaEmotionalEnergy,
        const PhysiologicalEnergy& deltaBioEnergy
    ) const;
    
    /**
     * Calculate neural phase synchrony
     * Φ(t) = cos(Δφ(t)) = cos(φ₁(t) - φ₂(t))
     */
    float calculateNeuralSynchrony(
        float phase1,
        float phase2
    ) const;
    
    /**
     * Calculate neural coherence strength
     * C_neural = (1/T) ∫₀ᵀ Φ(t) dt
     */
    float calculateNeuralCoherence(
        const std::vector<float>& synchronyHistory
    ) const;
    
    /**
     * Calculate total biofield energy
     * E_total = E_emotion + β E_bio + γ E_env
     */
    float calculateTotalBiofieldEnergy(
        float emotionalEnergy,
        const PhysiologicalEnergy& bioEnergy,
        float environmentalEnergy = 0.0f,
        float beta = 0.5f,  // Bio coupling
        float gamma = 0.3f  // Environment coupling
    ) const;
    
    /**
     * Set coupling coefficients
     */
    void setCouplingCoefficients(float alphaH, float alphaR, float alphaG) {
        alphaH_ = alphaH;
        alphaR_ = alphaR;
        alphaG_ = alphaG;
    }
    
private:
    float alphaH_, alphaR_, alphaG_;  // Coupling coefficients
};

} // namespace kelly
