#include "engine/GeometricTopology.h"
#include "engine/EmotionalPotentialEnergy.h"
#include <algorithm>
#include <cmath>

namespace kelly {

float GeometricTopology::calculateCurvature(
    const VADState& state,
    const VADState& velocity,
    const VADState& acceleration
) const {
    // κ = ||Ė × Ë|| / ||Ė||³
    VADState cross = crossProduct(velocity, acceleration);
    float crossMagnitude = std::sqrt(
        cross.valence * cross.valence +
        cross.arousal * cross.arousal +
        cross.dominance * cross.dominance
    );
    
    float velocityMagnitude = std::sqrt(
        velocity.valence * velocity.valence +
        velocity.arousal * velocity.arousal +
        velocity.dominance * velocity.dominance
    );
    
    if (velocityMagnitude < 1e-6f) {
        return 0.0f;  // Avoid division by zero
    }
    
    float velocityMagnitudeCubed = velocityMagnitude * velocityMagnitude * velocityMagnitude;
    return crossMagnitude / velocityMagnitudeCubed;
}

float GeometricTopology::calculateDistance(const VADState& state1, const VADState& state2) const {
    // d(E₁, E₂) = √((V₁-V₂)² + (A₁-A₂)² + (D₁-D₂)²)
    return state1.distanceTo(state2);
}

std::vector<GeometricTopology::Attractor> GeometricTopology::findAttractors(
    const std::vector<VADState>& candidateStates
) const {
    std::vector<Attractor> attractors;
    
    EmotionalPotentialEnergy potential;
    
    for (const auto& state : candidateStates) {
        // Check if force is near zero (equilibrium)
        EmotionalForce force = potential.calculateForce(state);
        float forceMagnitude = force.magnitude();
        
        if (forceMagnitude < 0.1f) {  // Near equilibrium
            Attractor attractor;
            attractor.position = state;
            attractor.stability = calculateStability(state);
            
            // Simple naming based on position
            if (state.valence > 0.5f && state.arousal > 0.5f) {
                attractor.name = "Joy";
            } else if (state.valence < -0.5f && state.arousal < 0.5f) {
                attractor.name = "Sadness";
            } else if (state.valence > 0.0f && state.arousal < 0.5f) {
                attractor.name = "Peace";
            } else if (state.valence > 0.0f && state.dominance > 0.5f) {
                attractor.name = "Flow";
            } else {
                attractor.name = "Equilibrium";
            }
            
            attractors.push_back(attractor);
        }
    }
    
    return attractors;
}

float GeometricTopology::calculateStability(const VADState& state) const {
    // Stability based on potential energy curvature
    // Higher stability = lower potential energy variation
    
    EmotionalPotentialEnergy potential;
    float basePotential = potential.calculatePotential(state);
    
    // Test small perturbations
    VADState perturbed = state;
    float maxVariation = 0.0f;
    
    for (int i = 0; i < 6; ++i) {
        perturbed = state;
        float epsilon = 0.01f;
        
        switch (i) {
            case 0: perturbed.valence += epsilon; break;
            case 1: perturbed.valence -= epsilon; break;
            case 2: perturbed.arousal += epsilon; break;
            case 3: perturbed.arousal -= epsilon; break;
            case 4: perturbed.dominance += epsilon; break;
            case 5: perturbed.dominance -= epsilon; break;
        }
        
        float perturbedPotential = potential.calculatePotential(perturbed);
        float variation = std::abs(perturbedPotential - basePotential);
        maxVariation = std::max(maxVariation, variation);
    }
    
    // Stability is inverse of variation
    return 1.0f / (1.0f + maxVariation);
}

float GeometricTopology::calculateVolume(const std::vector<VADState>& states) const {
    if (states.size() < 4) {
        return 0.0f;  // Need at least 4 points for 3D volume
    }
    
    // Simple approximation: bounding box volume
    float minV = states[0].valence, maxV = states[0].valence;
    float minA = states[0].arousal, maxA = states[0].arousal;
    float minD = states[0].dominance, maxD = states[0].dominance;
    
    for (const auto& state : states) {
        minV = std::min(minV, state.valence);
        maxV = std::max(maxV, state.valence);
        minA = std::min(minA, state.arousal);
        maxA = std::max(maxA, state.arousal);
        minD = std::min(minD, state.dominance);
        maxD = std::max(maxD, state.dominance);
    }
    
    float volume = (maxV - minV) * (maxA - minA) * (maxD - minD);
    return volume;
}

VADState GeometricTopology::crossProduct(const VADState& a, const VADState& b) const {
    VADState result;
    result.valence = a.arousal * b.dominance - a.dominance * b.arousal;
    result.arousal = a.dominance * b.valence - a.valence * b.dominance;
    result.dominance = a.valence * b.arousal - a.arousal * b.valence;
    return result;
}

float GeometricTopology::dotProduct(const VADState& a, const VADState& b) const {
    return a.valence * b.valence + a.arousal * b.arousal + a.dominance * b.dominance;
}

} // namespace kelly
