#include "engine/PhysiologicalResonance.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace kelly {

PhysiologicalEnergy PhysiologicalResonance::calculateBioEnergy(
    const BiometricInput::BiometricData& biometricData
) const {
    PhysiologicalEnergy energy;
    
    // E_bio(t) = α_H H(t) + α_R R(t) + α_G G(t)
    
    // Heart rate component (normalized 60-100 BPM to 0-1)
    if (biometricData.heartRate) {
        float hr = *biometricData.heartRate;
        float hrNorm = (hr - 60.0f) / 40.0f;  // Normalize to 0-1
        hrNorm = std::clamp(hrNorm, 0.0f, 1.0f);
        energy.heartRateComponent = alphaH_ * hrNorm;
    }
    
    // Respiration rate (inferred from HRV or movement, normalized)
    // For now, use movement as proxy
    if (biometricData.movement) {
        energy.respirationComponent = alphaR_ * (*biometricData.movement);
    }
    
    // Galvanic skin response (EDA)
    if (biometricData.skinConductance) {
        float eda = *biometricData.skinConductance;
        float edaNorm = eda / 20.0f;  // Normalize (typical range 1-20 microsiemens)
        edaNorm = std::clamp(edaNorm, 0.0f, 1.0f);
        energy.galvanicComponent = alphaG_ * edaNorm;
    }
    
    energy.totalEnergy = energy.heartRateComponent + 
                        energy.respirationComponent + 
                        energy.galvanicComponent;
    
    return energy;
}

float PhysiologicalResonance::calculateCouplingConstant(
    float deltaEmotionalEnergy,
    const PhysiologicalEnergy& deltaBioEnergy
) const {
    // k_bio = dE/dE_bio = ΔE / ΔE_bio
    float deltaBioTotal = deltaBioEnergy.totalEnergy;
    
    if (std::abs(deltaBioTotal) < 1e-6f) {
        return 0.0f;  // Avoid division by zero
    }
    
    return deltaEmotionalEnergy / deltaBioTotal;
}

float PhysiologicalResonance::calculateNeuralSynchrony(
    float phase1,
    float phase2
) const {
    // Φ(t) = cos(Δφ(t)) = cos(φ₁(t) - φ₂(t))
    float phaseDiff = phase1 - phase2;
    return std::cos(phaseDiff);
}

float PhysiologicalResonance::calculateNeuralCoherence(
    const std::vector<float>& synchronyHistory
) const {
    // C_neural = (1/T) ∫₀ᵀ Φ(t) dt
    if (synchronyHistory.empty()) {
        return 0.0f;
    }
    
    float sum = std::accumulate(synchronyHistory.begin(), synchronyHistory.end(), 0.0f);
    return sum / synchronyHistory.size();
}

float PhysiologicalResonance::calculateTotalBiofieldEnergy(
    float emotionalEnergy,
    const PhysiologicalEnergy& bioEnergy,
    float environmentalEnergy,
    float beta,
    float gamma
) const {
    // E_total = E_emotion + β E_bio + γ E_env
    return emotionalEnergy + beta * bioEnergy.totalEnergy + gamma * environmentalEnergy;
}

} // namespace kelly
