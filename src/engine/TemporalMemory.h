#pragma once

#include "engine/VADCalculator.h"
#include <vector>
#include <deque>
#include <functional>

namespace kelly {

/**
 * Temporal Memory System
 * 
 * Implements:
 * - Emotional hysteresis (memory of feeling)
 * - Temporal decay
 * - Emotional momentum
 * - Memory kernel functions
 */

/**
 * Temporal Memory Calculator
 */
class TemporalMemory {
public:
    TemporalMemory(float decayRate = 0.1f, float halfLife = 5.0f);
    
    /**
     * Calculate emotional hysteresis (memory of feeling)
     * E(t) = E₀ + ∫₀ᵗ K(τ) S(t-τ) dτ
     */
    VADState calculateHysteresis(
        const VADState& currentState,
        const std::deque<VADState>& history,
        const std::function<float(float)>& memoryKernel
    ) const;
    
    /**
     * Calculate temporal decay
     * E(t+Δt) = E(t) e^(-Δt/τ_E)
     */
    VADState applyDecay(
        const VADState& state,
        float deltaTime,
        float halfLife
    ) const;
    
    /**
     * Calculate emotional momentum
     * p_E = m_E dE/dt
     */
    struct EmotionalMomentum {
        float valenceMomentum;
        float arousalMomentum;
        float dominanceMomentum;
        
        EmotionalMomentum() 
            : valenceMomentum(0.0f), arousalMomentum(0.0f), dominanceMomentum(0.0f) {}
    };
    
    EmotionalMomentum calculateMomentum(
        const VADState& currentState,
        const VADState& previousState,
        float deltaTime,
        float emotionalMass = 1.0f
    ) const;
    
    /**
     * Calculate force from momentum change
     * F_E = dp_E/dt
     */
    EmotionalForce calculateForceFromMomentum(
        const EmotionalMomentum& currentMomentum,
        const EmotionalMomentum& previousMomentum,
        float deltaTime
    ) const;
    
    /**
     * Exponential memory kernel
     * K(τ) = e^(-τ/τ_mem)
     */
    static float exponentialKernel(float tau, float memoryTime = 2.0f);
    
    /**
     * Gaussian memory kernel
     * K(τ) = e^(-τ²/(2σ²))
     */
    static float gaussianKernel(float tau, float sigma = 1.0f);
    
    /**
     * Power-law memory kernel
     * K(τ) = τ^(-α)
     */
    static float powerLawKernel(float tau, float alpha = 0.5f);
    
private:
    float decayRate_;
    float halfLife_;
};

} // namespace kelly
