#pragma once

#include "engine/QuantumEmotionalField.h"
#include <vector>

namespace kelly {

/**
 * Time-Space Propagation System
 * 
 * Implements wave equation for emotional field propagation:
 * ∂²Ψ_E/∂t² = c_E²∇²Ψ_E - γ∂Ψ_E/∂t - μ²Ψ_E + S(x,t)
 */

struct FieldSource {
    float position[3];  // Spatial position
    float time;         // Time of source
    float amplitude;    // Source amplitude
    VADState emotionalContent;  // Emotional content of source
};

/**
 * Time-Space Propagation Calculator
 */
class TimeSpacePropagation {
public:
    TimeSpacePropagation(
        float propagationSpeed = 1.0f,  // c_E
        float damping = 0.1f,           // γ
        float massTerm = 0.5f           // μ
    ) : cE_(propagationSpeed), gamma_(damping), mu_(massTerm) {}
    
    /**
     * Evolve field according to wave equation
     * ∂²Ψ_E/∂t² = c_E²∇²Ψ_E - γ∂Ψ_E/∂t - μ²Ψ_E + S(x,t)
     */
    QuantumEmotionalState evolveField(
        const QuantumEmotionalState& currentState,
        const QuantumEmotionalState& previousState,
        const std::vector<QuantumEmotionalState>& neighbors,
        float deltaTime,
        const std::vector<FieldSource>& sources
    ) const;
    
    /**
     * Calculate Laplacian (spatial second derivative)
     */
    QuantumEmotionalState calculateLaplacian(
        const QuantumEmotionalState& center,
        const std::vector<QuantumEmotionalState>& neighbors
    ) const;
    
    /**
     * Calculate source term contribution
     */
    QuantumEmotionalState calculateSourceTerm(
        const std::vector<FieldSource>& sources,
        float currentTime,
        float position[3]
    ) const;
    
    /**
     * Set propagation parameters
     */
    void setParameters(float cE, float gamma, float mu) {
        cE_ = cE;
        gamma_ = gamma;
        mu_ = mu;
    }
    
private:
    float cE_;      // Propagation speed
    float gamma_;  // Damping constant
    float mu_;     // Mass term
};

} // namespace kelly
