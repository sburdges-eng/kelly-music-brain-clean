#pragma once

#include "engine/VADCalculator.h"
#include <array>

namespace kelly {

/**
 * Emotional Potential Energy System
 * 
 * Implements classical potential energy and force calculations:
 * - U_E = (1/2)k_V V² + (1/2)k_A A² + (1/2)k_D D²
 * - F_E = -∇U_E = [-k_V V, -k_A A, -k_D D]
 */

struct EmotionalForce {
    float valenceForce;    // -k_V V
    float arousalForce;    // -k_A A
    float dominanceForce;  // -k_D D
    
    EmotionalForce() : valenceForce(0.0f), arousalForce(0.0f), dominanceForce(0.0f) {}
    
    // Magnitude of force vector
    float magnitude() const {
        return std::sqrt(valenceForce * valenceForce + 
                        arousalForce * arousalForce + 
                        dominanceForce * dominanceForce);
    }
};

/**
 * Emotional Potential Energy Calculator
 */
class EmotionalPotentialEnergy {
public:
    EmotionalPotentialEnergy(
        float kV = 1.0f,  // Valence stiffness constant
        float kA = 1.0f,  // Arousal stiffness constant
        float kD = 1.0f   // Dominance stiffness constant
    ) : kV_(kV), kA_(kA), kD_(kD) {}
    
    /**
     * Calculate potential energy
     * U_E = (1/2)k_V V² + (1/2)k_A A² + (1/2)k_D D²
     */
    float calculatePotential(const VADState& vad) const;
    
    /**
     * Calculate emotional force (gradient of potential)
     * F_E = -∇U_E = [-k_V V, -k_A A, -k_D D]
     */
    EmotionalForce calculateForce(const VADState& vad) const;
    
    /**
     * Calculate equilibrium point (where force = 0)
     */
    VADState getEquilibrium() const;
    
    /**
     * Calculate work done moving from state1 to state2
     * W = U_E(state2) - U_E(state1)
     */
    float calculateWork(const VADState& state1, const VADState& state2) const;
    
    /**
     * Set stiffness constants
     */
    void setStiffness(float kV, float kA, float kD) {
        kV_ = kV;
        kA_ = kA;
        kD_ = kD;
    }
    
    float getKV() const { return kV_; }
    float getKA() const { return kA_; }
    float getKD() const { return kD_; }
    
private:
    float kV_, kA_, kD_;  // Stiffness constants
};

} // namespace kelly
