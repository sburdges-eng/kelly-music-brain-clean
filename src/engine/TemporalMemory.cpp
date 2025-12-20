#include "engine/TemporalMemory.h"
#include "engine/EmotionalPotentialEnergy.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace kelly {

TemporalMemory::TemporalMemory(float decayRate, float halfLife)
    : decayRate_(decayRate), halfLife_(halfLife) {
}

VADState TemporalMemory::calculateHysteresis(
    const VADState& currentState,
    const std::deque<VADState>& history,
    const std::function<float(float)>& memoryKernel
) const {
    // E(t) = E₀ + ∫₀ᵗ K(τ) S(t-τ) dτ
    VADState result = currentState;
    
    if (history.empty()) {
        return result;
    }
    
    // Discrete approximation of convolution integral
    float currentTime = currentState.timestamp;
    
    for (size_t i = 0; i < history.size(); ++i) {
        float tau = currentTime - history[i].timestamp;
        if (tau <= 0.0) continue;
        
        float kernelValue = memoryKernel(tau);
        
        // Weighted contribution from past state
        result.valence += kernelValue * (history[i].valence - currentState.valence) * 0.1f;
        result.arousal += kernelValue * (history[i].arousal - currentState.arousal) * 0.1f;
        result.dominance += kernelValue * (history[i].dominance - currentState.dominance) * 0.1f;
    }
    
    result.clamp();
    return result;
}

VADState TemporalMemory::applyDecay(
    const VADState& state,
    float deltaTime,
    float halfLife
) const {
    // E(t+Δt) = E(t) e^(-Δt/τ_E)
    // τ_E = halfLife / ln(2)
    float tau = halfLife / 0.693147f;  // ln(2) ≈ 0.693
    float decayFactor = std::exp(-deltaTime / tau);
    
    VADState decayed = state;
    
    // Decay toward neutral
    VADState neutral;
    neutral.valence = 0.0f;
    neutral.arousal = 0.5f;
    neutral.dominance = 0.5f;
    
    decayed.valence = neutral.valence + (state.valence - neutral.valence) * decayFactor;
    decayed.arousal = neutral.arousal + (state.arousal - neutral.arousal) * decayFactor;
    decayed.dominance = neutral.dominance + (state.dominance - neutral.dominance) * decayFactor;
    
    decayed.clamp();
    return decayed;
}

TemporalMemory::EmotionalMomentum TemporalMemory::calculateMomentum(
    const VADState& currentState,
    const VADState& previousState,
    float deltaTime,
    float emotionalMass
) const {
    // p_E = m_E dE/dt
    EmotionalMomentum momentum;
    
    if (deltaTime > 0.0f) {
        float dV = (currentState.valence - previousState.valence) / deltaTime;
        float dA = (currentState.arousal - previousState.arousal) / deltaTime;
        float dD = (currentState.dominance - previousState.dominance) / deltaTime;
        
        momentum.valenceMomentum = emotionalMass * dV;
        momentum.arousalMomentum = emotionalMass * dA;
        momentum.dominanceMomentum = emotionalMass * dD;
    }
    
    return momentum;
}

EmotionalForce TemporalMemory::calculateForceFromMomentum(
    const EmotionalMomentum& currentMomentum,
    const EmotionalMomentum& previousMomentum,
    float deltaTime
) const {
    // F_E = dp_E/dt
    EmotionalForce force;
    
    if (deltaTime > 0.0f) {
        force.valenceForce = (currentMomentum.valenceMomentum - previousMomentum.valenceMomentum) / deltaTime;
        force.arousalForce = (currentMomentum.arousalMomentum - previousMomentum.arousalMomentum) / deltaTime;
        force.dominanceForce = (currentMomentum.dominanceMomentum - previousMomentum.dominanceMomentum) / deltaTime;
    }
    
    return force;
}

float TemporalMemory::exponentialKernel(float tau, float memoryTime) {
    // K(τ) = e^(-τ/τ_mem)
    return std::exp(-tau / memoryTime);
}

float TemporalMemory::gaussianKernel(float tau, float sigma) {
    // K(τ) = e^(-τ²/(2σ²))
    return std::exp(-(tau * tau) / (2.0f * sigma * sigma));
}

float TemporalMemory::powerLawKernel(float tau, float alpha) {
    // K(τ) = τ^(-α)
    if (tau <= 0.0f) return 0.0f;
    return std::pow(tau, -alpha);
}

} // namespace kelly
