#include "engine/NetworkDynamics.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace kelly {

NetworkDynamics::NetworkDynamics() {
}

VADState NetworkDynamics::calculateCoupling(
    const Agent& agent,
    const std::vector<Agent>& neighbors,
    const std::map<std::pair<int, int>, float>& couplingMatrix
) const {
    VADState coupling;
    
    for (const auto& neighbor : neighbors) {
        auto key = std::make_pair(agent.id, neighbor.id);
        auto it = couplingMatrix.find(key);
        float k_ij = (it != couplingMatrix.end()) ? it->second : 0.1f;  // Default coupling
        
        // dE_i/dt = Σ_j k_ij(E_j - E_i)
        VADState diff;
        diff.valence = neighbor.emotionalState.valence - agent.emotionalState.valence;
        diff.arousal = neighbor.emotionalState.arousal - agent.emotionalState.arousal;
        diff.dominance = neighbor.emotionalState.dominance - agent.emotionalState.dominance;
        
        coupling.valence += k_ij * diff.valence;
        coupling.arousal += k_ij * diff.arousal;
        coupling.dominance += k_ij * diff.dominance;
    }
    
    return coupling;
}

float NetworkDynamics::calculateCoherence(const std::vector<Agent>& agents) const {
    if (agents.size() < 2) return 1.0f;
    
    // C = (1/N²) Σ_i,j cos(θ_i - θ_j)
    // Extract phases from quantum states
    std::vector<float> phases;
    for (const auto& agent : agents) {
        float phase = extractPhase(agent.quantumState);
        phases.push_back(phase);
    }
    
    float coherence = 0.0f;
    size_t n = agents.size();
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float phaseDiff = phases[i] - phases[j];
            coherence += std::cos(phaseDiff);
        }
    }
    
    return coherence / (n * n);
}

float NetworkDynamics::calculatePhaseLockingValue(
    const std::vector<float>& phases1,
    const std::vector<float>& phases2
) const {
    if (phases1.size() != phases2.size() || phases1.empty()) {
        return 0.0f;
    }
    
    // PLV = |(1/N) Σ_n e^(i(φ₁(n) - φ₂(n)))|
    std::complex<float> sum(0.0f, 0.0f);
    
    for (size_t i = 0; i < phases1.size(); ++i) {
        float phaseDiff = phases1[i] - phases2[i];
        sum += std::exp(std::complex<float>(0.0f, phaseDiff));
    }
    
    sum /= static_cast<float>(phases1.size());
    return std::abs(sum);
}

float NetworkDynamics::calculateConnectivity(
    const Agent& agent1,
    const Agent& agent2,
    float correlationLength
) const {
    // K_ij = e^(-||x_i - x_j||/L) / (1 + |E_i - E_j|)
    float distance = calculateDistance(agent1, agent2);
    float emotionalDistance = agent1.emotionalState.distanceTo(agent2.emotionalState);
    
    float spatialTerm = std::exp(-distance / correlationLength);
    float emotionalTerm = 1.0f / (1.0f + emotionalDistance);
    
    return spatialTerm * emotionalTerm;
}

VADState NetworkDynamics::simulateDiffusion(
    const Agent& agent,
    const std::vector<Agent>& neighbors,
    float diffusionRate,
    float selfRegulationRate,
    float noise
) const {
    // ∂E_i/∂t = D_E ∇²E_i - λ(E_i - Ē) + η_i(t)
    
    // Calculate Laplacian (second derivative approximation)
    VADState laplacian;
    VADState average = calculateAverageVAD(neighbors);
    
    // ∇²E ≈ (E_neighbors - E_agent)
    laplacian.valence = average.valence - agent.emotionalState.valence;
    laplacian.arousal = average.arousal - agent.emotionalState.arousal;
    laplacian.dominance = average.dominance - agent.emotionalState.dominance;
    
    // Calculate neighborhood average
    VADState neighborhoodAvg = calculateAverageVAD(neighbors);
    if (!neighbors.empty()) {
        neighborhoodAvg.valence /= neighbors.size();
        neighborhoodAvg.arousal /= neighbors.size();
        neighborhoodAvg.dominance /= neighbors.size();
    }
    
    VADState result;
    
    // Diffusion term: D_E ∇²E_i
    result.valence = diffusionRate * laplacian.valence;
    result.arousal = diffusionRate * laplacian.arousal;
    result.dominance = diffusionRate * laplacian.dominance;
    
    // Self-regulation term: -λ(E_i - Ē)
    result.valence -= selfRegulationRate * (agent.emotionalState.valence - neighborhoodAvg.valence);
    result.arousal -= selfRegulationRate * (agent.emotionalState.arousal - neighborhoodAvg.arousal);
    result.dominance -= selfRegulationRate * (agent.emotionalState.dominance - neighborhoodAvg.dominance);
    
    // Noise term: η_i(t)
    if (noise > 0.0f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0f, noise);
        
        result.valence += dis(gen);
        result.arousal += dis(gen);
        result.dominance += dis(gen);
    }
    
    return result;
}

float NetworkDynamics::calculatePropagationSpeed(
    float emotionalStiffness,
    float emotionalMass
) const {
    // c_E = √(k_E / m_E)
    if (emotionalMass <= 0.0f) return 0.0f;
    return std::sqrt(emotionalStiffness / emotionalMass);
}

float NetworkDynamics::calculateResonanceFrequency(
    float emotionalStiffness,
    float emotionalMass
) const {
    // f_res = (1/2π)√(k_E / m_E)
    if (emotionalMass <= 0.0f) return 0.0f;
    return (1.0f / (2.0f * 3.14159f)) * std::sqrt(emotionalStiffness / emotionalMass);
}

float NetworkDynamics::calculateQualityFactor(
    float resonanceFreq,
    float bandwidth
) const {
    // Q_E = f_res / Δf
    if (bandwidth <= 0.0f) return 0.0f;
    return resonanceFreq / bandwidth;
}

float NetworkDynamics::calculateDistance(const Agent& a1, const Agent& a2) const {
    float dx = a1.position[0] - a2.position[0];
    float dy = a1.position[1] - a2.position[1];
    float dz = a1.position[2] - a2.position[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

float NetworkDynamics::calculateAverageVAD(const std::vector<Agent>& agents) const {
    if (agents.empty()) return 0.0f;
    
    VADState sum;
    for (const auto& agent : agents) {
        sum.valence += agent.emotionalState.valence;
        sum.arousal += agent.emotionalState.arousal;
        sum.dominance += agent.emotionalState.dominance;
    }
    
    // Return average arousal as single value (for simplicity)
    return sum.arousal / agents.size();
}

float NetworkDynamics::extractPhase(const QuantumEmotionalState& qState) const {
    // Extract dominant phase from quantum state
    if (qState.states.empty()) return 0.0f;
    
    // Find state with highest probability
    auto maxIt = std::max_element(
        qState.states.begin(),
        qState.states.end(),
        [](const EmotionState& a, const EmotionState& b) {
            return a.probability < b.probability;
        }
    );
    
    if (maxIt != qState.states.end()) {
        return std::arg(maxIt->amplitude);
    }
    
    return 0.0f;
}

} // namespace kelly
