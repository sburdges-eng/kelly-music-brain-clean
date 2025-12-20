#include "engine/EmotionalPotentialEnergy.h"
#include <cmath>

namespace kelly {

float EmotionalPotentialEnergy::calculatePotential(const VADState& vad) const {
    // U_E = (1/2)k_V V² + (1/2)k_A A² + (1/2)k_D D²
    float vTerm = 0.5f * kV_ * vad.valence * vad.valence;
    float aTerm = 0.5f * kA_ * vad.arousal * vad.arousal;
    float dTerm = 0.5f * kD_ * vad.dominance * vad.dominance;
    
    return vTerm + aTerm + dTerm;
}

EmotionalForce EmotionalPotentialEnergy::calculateForce(const VADState& vad) const {
    EmotionalForce force;
    
    // F_E = -∇U_E = [-k_V V, -k_A A, -k_D D]
    force.valenceForce = -kV_ * vad.valence;
    force.arousalForce = -kA_ * vad.arousal;
    force.dominanceForce = -kD_ * vad.dominance;
    
    return force;
}

VADState EmotionalPotentialEnergy::getEquilibrium() const {
    // Equilibrium is where force = 0, so V = 0, A = 0, D = 0
    VADState equilibrium;
    equilibrium.valence = 0.0f;
    equilibrium.arousal = 0.5f;  // Neutral arousal
    equilibrium.dominance = 0.5f;  // Neutral dominance
    return equilibrium;
}

float EmotionalPotentialEnergy::calculateWork(const VADState& state1, const VADState& state2) const {
    // W = U_E(state2) - U_E(state1)
    float u1 = calculatePotential(state1);
    float u2 = calculatePotential(state2);
    return u2 - u1;
}

} // namespace kelly
