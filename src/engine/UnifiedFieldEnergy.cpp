#include "engine/UnifiedFieldEnergy.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace kelly {

UnifiedFieldEnergy::UnifiedFieldEnergy()
    : gBio_(0.5f), gNet_(0.3f), gRes_(0.2f) {
}

float UnifiedFieldEnergy::calculateLagrangian(
    const QuantumEmotionalState& qState,
    const VADState& vad,
    const PhysiologicalEnergy& bioEnergy,
    const std::vector<Agent>& network,
    float gBio,
    float gNet,
    float gRes
) const {
    // L_QEF = (1/2)|∇Ψ_E|² - U_E + g_bio E_bio + g_net Σ K_ij(E_j - E_i)² + g_res |Ψ_E|⁴
    
    // Gradient term
    float gradientTerm = 0.5f * calculateGradientTerm(qState);
    
    // Potential energy term
    float potentialTerm = -potentialEnergy_.calculatePotential(vad);
    
    // Bio energy term
    float bioTerm = gBio * bioEnergy.totalEnergy;
    
    // Network coupling term
    float networkTerm = 0.0f;
    if (!network.empty()) {
        Agent centerAgent = network[0];  // Simplified: use first agent as center
        std::vector<Agent> neighbors(network.begin() + 1, network.end());
        networkTerm = gNet * calculateNetworkCouplingTerm(centerAgent, neighbors);
    }
    
    // Resonance term
    float resonanceTerm = gRes * calculateResonanceTerm(qState);
    
    return gradientTerm + potentialTerm + bioTerm + networkTerm + resonanceTerm;
}

float UnifiedFieldEnergy::calculateTotalEnergy(
    float emotionalEnergy,
    float musicEnergy,
    float voiceEnergy,
    const PhysiologicalEnergy& bioEnergy,
    float networkEnergy,
    float resonanceEnergy
) const {
    // E_total = E_emotion + E_music + E_voice + E_bio + E_network + E_resonance
    return emotionalEnergy + musicEnergy + voiceEnergy + 
           bioEnergy.totalEnergy + networkEnergy + resonanceEnergy;
}

float UnifiedFieldEnergy::calculateGradientTerm(const QuantumEmotionalState& qState) const {
    // |∇Ψ_E|² ≈ Σ_i |α_i|² (simplified gradient)
    float gradientSquared = 0.0f;
    
    for (const auto& state : qState.states) {
        gradientSquared += state.probability;
    }
    
    return gradientSquared;
}

float UnifiedFieldEnergy::calculateNetworkCouplingTerm(
    const Agent& agent,
    const std::vector<Agent>& neighbors
) const {
    // Σ K_ij(E_j - E_i)²
    float couplingSum = 0.0f;
    
    for (const auto& neighbor : neighbors) {
        float connectivity = networkDynamics_.calculateConnectivity(agent, neighbor);
        float distance = agent.emotionalState.distanceTo(neighbor.emotionalState);
        couplingSum += connectivity * distance * distance;
    }
    
    return couplingSum;
}

float UnifiedFieldEnergy::calculateResonanceTerm(const QuantumEmotionalState& qState) const {
    // |Ψ_E|⁴ = (|Ψ_E|²)²
    float normSquared = 0.0f;
    
    for (const auto& state : qState.states) {
        normSquared += state.probability;
    }
    
    return normSquared * normSquared;
}

float UnifiedFieldEnergy::calculateSelfOrganizationIndex(const QuantumEmotionalState& qState) const {
    // Ω = ⟨|Ψ|²⟩² / ⟨|Ψ|⁴⟩
    float avgNormSquared = 0.0f;
    float avgNormFourth = 0.0f;
    
    for (const auto& state : qState.states) {
        float prob = state.probability;
        avgNormSquared += prob;
        avgNormFourth += prob * prob;
    }
    
    if (qState.states.size() > 0) {
        avgNormSquared /= qState.states.size();
        avgNormFourth /= qState.states.size();
    }
    
    if (avgNormFourth > 0.0f) {
        return (avgNormSquared * avgNormSquared) / avgNormFourth;
    }
    
    return 0.0f;
}

} // namespace kelly
