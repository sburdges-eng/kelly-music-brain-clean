#include "engine/HybridCoupling.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <set>

namespace kelly {

HybridCoupling::HybridCoupling(float alpha, float beta, float kappa)
    : alpha_(alpha), beta_(beta), kappa_(kappa) {
    normalizeWeights();
}

void HybridCoupling::normalizeWeights() {
    // |α|² + |β|² = 1
    float norm = std::sqrt(alpha_ * alpha_ + beta_ * beta_);
    if (norm > 0.0f) {
        alpha_ /= norm;
        beta_ /= norm;
    }
}

QuantumEmotionalState HybridCoupling::createHybridState(
    const QuantumEmotionalState& aiState,
    const QuantumEmotionalState& humanState
) const {
    // Ψ_hybrid = αΨ_AI + βΨ_human
    QuantumEmotionalState hybrid;
    
    // Create map for efficient lookup
    std::map<EmotionBasis, Complex> aiMap, humanMap;
    
    for (const auto& state : aiState.states) {
        aiMap[state.basis] = state.amplitude;
    }
    
    for (const auto& state : humanState.states) {
        humanMap[state.basis] = state.amplitude;
    }
    
    // Combine all unique bases
    std::set<EmotionBasis> allBases;
    for (const auto& state : aiState.states) {
        allBases.insert(state.basis);
    }
    for (const auto& state : humanState.states) {
        allBases.insert(state.basis);
    }
    
    // Create hybrid amplitudes
    for (const auto& basis : allBases) {
        Complex aiAmp = (aiMap.find(basis) != aiMap.end()) ? aiMap[basis] : Complex(0.0f, 0.0f);
        Complex humanAmp = (humanMap.find(basis) != humanMap.end()) ? humanMap[basis] : Complex(0.0f, 0.0f);
        
        Complex hybridAmp = alpha_ * aiAmp + beta_ * humanAmp;
        
        EmotionState eState(basis, hybridAmp);
        eState.probability = std::norm(hybridAmp);
        hybrid.states.push_back(eState);
    }
    
    hybrid.normalize();
    
    // Calculate coherence and entropy
    float coherence = 0.0f;
    Complex sum(0.0f, 0.0f);
    for (const auto& state : hybrid.states) {
        sum += state.amplitude;
    }
    hybrid.coherence = std::abs(sum);
    
    float entropy = 0.0f;
    for (const auto& state : hybrid.states) {
        if (state.probability > 0.0f) {
            entropy -= state.probability * std::log2(state.probability);
        }
    }
    hybrid.entropy = entropy;
    
    return hybrid;
}

float HybridCoupling::calculateCrossInfluence(
    const QuantumEmotionalState& aiState,
    const QuantumEmotionalState& humanState
) const {
    // ΔH = κ Re(Ψ_AI* Ψ_human)
    Complex innerProduct(0.0f, 0.0f);
    
    std::map<EmotionBasis, Complex> aiMap;
    for (const auto& state : aiState.states) {
        aiMap[state.basis] = state.amplitude;
    }
    
    for (const auto& state : humanState.states) {
        auto it = aiMap.find(state.basis);
        if (it != aiMap.end()) {
            innerProduct += std::conj(it->second) * state.amplitude;
        }
    }
    
    return kappa_ * std::real(innerProduct);
}

float HybridCoupling::calculateCoherence(
    const QuantumEmotionalState& aiState,
    const QuantumEmotionalState& humanState
) const {
    // Calculate overlap/coherence between states
    Complex overlap(0.0f, 0.0f);
    
    std::map<EmotionBasis, Complex> aiMap;
    for (const auto& state : aiState.states) {
        aiMap[state.basis] = state.amplitude;
    }
    
    for (const auto& state : humanState.states) {
        auto it = aiMap.find(state.basis);
        if (it != aiMap.end()) {
            overlap += std::conj(it->second) * state.amplitude;
        }
    }
    
    return std::abs(overlap);
}

} // namespace kelly
