#pragma once

#include "engine/VADCalculator.h"
#include <vector>

namespace kelly {

/**
 * Geometric and Topological Analysis
 * 
 * Implements:
 * - Emotional manifold curvature
 * - Emotional distance metrics
 * - Emotional attractors (stable states)
 */

/**
 * Geometric Topology Calculator
 */
class GeometricTopology {
public:
    /**
     * Calculate emotional manifold curvature
     * κ = ||Ė × Ë|| / ||Ė||³
     */
    float calculateCurvature(
        const VADState& state,
        const VADState& velocity,  // First derivative
        const VADState& acceleration  // Second derivative
    ) const;
    
    /**
     * Calculate emotional distance
     * d(E₁, E₂) = √((V₁-V₂)² + (A₁-A₂)² + (D₁-D₂)²)
     */
    float calculateDistance(const VADState& state1, const VADState& state2) const;
    
    /**
     * Find emotional attractors (stable equilibrium points)
     * ∇U_E = 0, det(∇²U_E) > 0
     */
    struct Attractor {
        VADState position;
        float stability;  // Stability measure
        std::string name;  // e.g., "Joy", "Peace", "Flow"
    };
    
    std::vector<Attractor> findAttractors(
        const std::vector<VADState>& candidateStates
    ) const;
    
    /**
     * Calculate stability of a state
     * Based on potential energy second derivative
     */
    float calculateStability(const VADState& state) const;
    
    /**
     * Calculate emotional volume in VAD space
     * Volume of region defined by states
     */
    float calculateVolume(const std::vector<VADState>& states) const;
    
private:
    // Helper functions
    VADState crossProduct(const VADState& a, const VADState& b) const;
    float dotProduct(const VADState& a, const VADState& b) const;
};

} // namespace kelly
