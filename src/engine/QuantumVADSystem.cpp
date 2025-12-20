#include "engine/QuantumVADSystem.h"
#include <algorithm>

namespace kelly {

QuantumVADSystem::QuantumVADSystem(const EmotionThesaurus* thesaurus)
    : vadSystem_(thesaurus) {
    initializeDefaultHamiltonian();
}

QuantumVADSystem::QuantumProcessingResult QuantumVADSystem::processEmotionQuantum(
    int emotionId,
    float intensityModifier,
    bool generateOSC
) {
    QuantumProcessingResult result;
    result.success = false;
    
    // Get classical VAD
    auto classicalResult = vadSystem_.processEmotionId(emotionId, intensityModifier, generateOSC);
    if (!classicalResult.success) {
        return result;
    }
    
    result.classicalVAD = classicalResult.vad;
    
    // Create quantum superposition from VAD
    result.quantumState = quantumField_.createSuperposition(result.classicalVAD);
    
    // Calculate quantum metrics
    result.energy = quantumField_.calculateEnergy(result.quantumState);
    result.temperature = quantumField_.calculateTemperature(result.energy);
    result.coherence = result.quantumState.coherence;
    result.entropy = result.quantumState.entropy;
    
    // Map to music
    result.frequency = musicMapper_.quantumStateToFrequency(result.quantumState);
    result.voice = musicMapper_.quantumStateToVoice(result.quantumState);
    
    // Get resonance
    result.resonance = classicalResult.resonance;
    
    result.success = true;
    return result;
}

QuantumVADSystem::QuantumProcessingResult QuantumVADSystem::processBiometricsQuantum(
    const BiometricInput::BiometricData& biometricData,
    bool generateOSC
) {
    QuantumProcessingResult result;
    result.success = false;
    
    // Get classical VAD
    auto classicalResult = vadSystem_.processBiometrics(biometricData, generateOSC);
    if (!classicalResult.success) {
        return result;
    }
    
    result.classicalVAD = classicalResult.vad;
    
    // Create quantum superposition
    result.quantumState = quantumField_.createSuperposition(result.classicalVAD);
    
    // Calculate quantum metrics
    result.energy = quantumField_.calculateEnergy(result.quantumState);
    result.temperature = quantumField_.calculateTemperature(result.energy);
    result.coherence = result.quantumState.coherence;
    result.entropy = result.quantumState.entropy;
    
    // Map to music
    result.frequency = musicMapper_.quantumStateToFrequency(result.quantumState);
    result.voice = musicMapper_.quantumStateToVoice(result.quantumState);
    
    // Get resonance
    result.resonance = classicalResult.resonance;
    
    result.success = true;
    return result;
}

float QuantumVADSystem::calculateInterference(int emotionId1, int emotionId2) const {
    // Get VAD for both emotions using the VAD system
    VADState vad1 = vadSystem_.processEmotionId(emotionId1, 1.0f, false).vad;
    VADState vad2 = vadSystem_.processEmotionId(emotionId2, 1.0f, false).vad;
    
    // Create quantum states
    QuantumEmotionalState qState1 = quantumField_.createSuperposition(vad1);
    QuantumEmotionalState qState2 = quantumField_.createSuperposition(vad2);
    
    return quantumField_.calculateInterference(qState1, qState2);
}

QuantumEmotionalField::EntangledState QuantumVADSystem::createEmotionalEntanglement(
    int emotionId1,
    int emotionId2
) const {
    // Get VAD for both emotions
    VADState vad1 = vadSystem_.processEmotionId(emotionId1, 1.0f, false).vad;
    VADState vad2 = vadSystem_.processEmotionId(emotionId2, 1.0f, false).vad;
    
    QuantumEmotionalState qState1 = quantumField_.createSuperposition(vad1);
    QuantumEmotionalState qState2 = quantumField_.createSuperposition(vad2);
    
    return quantumField_.createEntanglement(qState1, qState2);
}

QuantumEmotionalState QuantumVADSystem::evolveState(
    const QuantumEmotionalState& initialState,
    float deltaTime
) const {
    return quantumField_.evolve(initialState, deltaTime, defaultHamiltonian_);
}

QuantumVADSystem::ClassicalMetrics QuantumVADSystem::calculateClassicalMetrics(
    const VADState& vad
) const {
    ClassicalMetrics metrics;
    metrics.energy = ClassicalVAD::calculateEnergy(vad.valence, vad.arousal);
    metrics.tension = ClassicalVAD::calculateTension(vad.valence, vad.dominance);
    metrics.stability = ClassicalVAD::calculateStability(vad);
    return metrics;
}

void QuantumVADSystem::initializeDefaultHamiltonian() {
    // Default Hamiltonian diagonal elements (energy levels for each emotion)
    defaultHamiltonian_ = {
        {"Joy", 1.0f},
        {"Sadness", 0.5f},
        {"Anger", 1.5f},
        {"Fear", 1.2f},
        {"Trust", 0.8f},
        {"Surprise", 1.3f},
        {"Disgust", 0.9f},
        {"Anticipation", 1.0f}
    };
}

} // namespace kelly
