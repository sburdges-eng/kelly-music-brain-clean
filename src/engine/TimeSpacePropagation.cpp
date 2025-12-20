#include "engine/TimeSpacePropagation.h"
#include <algorithm>
#include <cmath>

namespace kelly {

QuantumEmotionalState TimeSpacePropagation::evolveField(
    const QuantumEmotionalState& currentState,
    const QuantumEmotionalState& previousState,
    const std::vector<QuantumEmotionalState>& neighbors,
    float deltaTime,
    const std::vector<FieldSource>& sources
) const {
    // ∂²Ψ_E/∂t² = c_E²∇²Ψ_E - γ∂Ψ_E/∂t - μ²Ψ_E + S(x,t)
    
    // Calculate Laplacian
    QuantumEmotionalState laplacian = calculateLaplacian(currentState, neighbors);
    
    // Calculate time derivatives (finite difference)
    QuantumEmotionalState dPsi_dt;
    QuantumEmotionalState d2Psi_dt2;
    
    // First derivative: ∂Ψ/∂t ≈ (Ψ(t) - Ψ(t-Δt)) / Δt
    if (deltaTime > 0.0f) {
        for (size_t i = 0; i < currentState.states.size() && i < previousState.states.size(); ++i) {
            Complex diff = currentState.states[i].amplitude - previousState.states[i].amplitude;
            dPsi_dt.states.push_back(EmotionState(
                currentState.states[i].basis,
                diff / deltaTime
            ));
        }
    }
    
    // Second derivative approximation (simplified)
    // ∂²Ψ/∂t² ≈ (Ψ(t+Δt) - 2Ψ(t) + Ψ(t-Δt)) / Δt²
    // For now, use Laplacian as approximation
    
    // Calculate source term
    float dummyPos[3] = {0.0f, 0.0f, 0.0f};
    QuantumEmotionalState sourceTerm = calculateSourceTerm(sources, currentState.timestamp, dummyPos);
    
    // Combine terms
    QuantumEmotionalState evolved = currentState;
    
    // Evolve each basis state
    for (size_t i = 0; i < evolved.states.size(); ++i) {
        Complex laplacianAmp = (i < laplacian.states.size()) ? laplacian.states[i].amplitude : Complex(0.0f, 0.0f);
        Complex dPsiAmp = (i < dPsi_dt.states.size()) ? dPsi_dt.states[i].amplitude : Complex(0.0f, 0.0f);
        Complex sourceAmp = (i < sourceTerm.states.size()) ? sourceTerm.states[i].amplitude : Complex(0.0f, 0.0f);
        
        // ∂²Ψ/∂t² = c_E²∇²Ψ - γ∂Ψ/∂t - μ²Ψ + S
        Complex d2Psi = cE_ * cE_ * laplacianAmp - 
                        gamma_ * dPsiAmp - 
                        mu_ * mu_ * currentState.states[i].amplitude +
                        sourceAmp;
        
        // Update amplitude: Ψ(t+Δt) ≈ Ψ(t) + Δt ∂Ψ/∂t + (Δt²/2) ∂²Ψ/∂t²
        evolved.states[i].amplitude = currentState.states[i].amplitude +
                                     deltaTime * dPsiAmp +
                                     0.5f * deltaTime * deltaTime * d2Psi;
        
        evolved.states[i].probability = std::norm(evolved.states[i].amplitude);
    }
    
    evolved.normalize();
    return evolved;
}

QuantumEmotionalState TimeSpacePropagation::calculateLaplacian(
    const QuantumEmotionalState& center,
    const std::vector<QuantumEmotionalState>& neighbors
) const {
    // ∇²Ψ ≈ Σ_neighbors (Ψ_neighbor - Ψ_center)
    QuantumEmotionalState laplacian = center;
    
    // Initialize to zero
    for (auto& state : laplacian.states) {
        state.amplitude = Complex(0.0f, 0.0f);
        state.probability = 0.0f;
    }
    
    if (neighbors.empty()) {
        return laplacian;
    }
    
    // Sum differences
    for (const auto& neighbor : neighbors) {
        for (size_t i = 0; i < center.states.size() && i < neighbor.states.size(); ++i) {
            if (center.states[i].basis == neighbor.states[i].basis) {
                laplacian.states[i].amplitude += neighbor.states[i].amplitude - center.states[i].amplitude;
            }
        }
    }
    
    // Average
    float n = static_cast<float>(neighbors.size());
    if (n > 0.0f) {
        for (auto& state : laplacian.states) {
            state.amplitude /= n;
            state.probability = std::norm(state.amplitude);
        }
    }
    
    return laplacian;
}

QuantumEmotionalState TimeSpacePropagation::calculateSourceTerm(
    const std::vector<FieldSource>& sources,
    float currentTime,
    float position[3]
) const {
    QuantumEmotionalState sourceTerm;
    
    // Sum contributions from all sources
    for (const auto& source : sources) {
        // Simple distance-based attenuation
        float dx = position[0] - source.position[0];
        float dy = position[1] - source.position[1];
        float dz = position[2] - source.position[2];
        float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        // Time decay
        float timeDiff = std::abs(currentTime - source.time);
        float attenuation = std::exp(-distance - timeDiff);
        
        // Create quantum state from VAD
        // Simplified: create single state
        Complex amplitude(source.amplitude * attenuation, 0.0f);
        EmotionState eState("Source", amplitude);
        eState.probability = std::norm(amplitude);
        sourceTerm.states.push_back(eState);
    }
    
    return sourceTerm;
}

} // namespace kelly
