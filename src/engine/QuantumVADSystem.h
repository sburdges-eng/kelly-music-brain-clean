#pragma once

/**
 * QuantumVADSystem.h
 * 
 * Integrated system combining:
 * - Classical VAD calculations
 * - Quantum emotional field
 * - Emotion-to-music mapping
 * - Voice synthesis parameters
 */

#include "engine/VADSystem.h"
#include "engine/QuantumEmotionalField.h"
#include "engine/EmotionToMusicMapper.h"
#include <optional>

namespace kelly {

/**
 * Quantum VAD System - Full integration of classical and quantum models
 */
class QuantumVADSystem {
public:
    QuantumVADSystem(const EmotionThesaurus* thesaurus = nullptr);
    
    /**
     * Process emotion with full quantum field model
     */
    struct QuantumProcessingResult {
        VADState classicalVAD;              // Classical VAD coordinates
        QuantumEmotionalState quantumState; // Quantum superposition
        MusicalFrequency frequency;         // Musical frequency mapping
        VoiceParameters voice;             // Voice synthesis parameters
        float energy;                        // Quantum emotional energy
        float temperature;                  // Emotional temperature
        float coherence;                    // Emotional coherence
        float entropy;                       // Emotional entropy
        ResonanceMetrics resonance;         // Resonance metrics
        bool success;
    };
    
    QuantumProcessingResult processEmotionQuantum(
        int emotionId,
        float intensityModifier = 1.0f,
        bool generateOSC = false
    );
    
    /**
     * Process biometrics with quantum field
     */
    QuantumProcessingResult processBiometricsQuantum(
        const BiometricInput::BiometricData& biometricData,
        bool generateOSC = false
    );
    
    /**
     * Calculate interference between two emotional states
     */
    float calculateInterference(
        int emotionId1,
        int emotionId2
    ) const;
    
    /**
     * Create entangled state between two emotions
     */
    QuantumEmotionalField::EntangledState createEmotionalEntanglement(
        int emotionId1,
        int emotionId2
    ) const;
    
    /**
     * Evolve quantum state over time
     */
    QuantumEmotionalState evolveState(
        const QuantumEmotionalState& initialState,
        float deltaTime
    ) const;
    
    /**
     * Get classical VAD formulas
     */
    struct ClassicalMetrics {
        float energy;      // E_n = A × (1 + |V|)
        float tension;     // T = |V| × (1 - D)
        float stability;   // S = 1 - √((V² + A² + D²) / 3)
    };
    
    ClassicalMetrics calculateClassicalMetrics(const VADState& vad) const;
    
    /**
     * Get quantum field accessors
     */
    QuantumEmotionalField& quantumField() { return quantumField_; }
    const QuantumEmotionalField& quantumField() const { return quantumField_; }
    
    EmotionToMusicMapper& musicMapper() { return musicMapper_; }
    const EmotionToMusicMapper& musicMapper() const { return musicMapper_; }
    
    VADSystem& vadSystem() { return vadSystem_; }
    const VADSystem& vadSystem() const { return vadSystem_; }
    
private:
    VADSystem vadSystem_;
    QuantumEmotionalField quantumField_;
    EmotionToMusicMapper musicMapper_;
    
    // Default Hamiltonian for evolution
    std::map<EmotionBasis, float> defaultHamiltonian_;
    
    void initializeDefaultHamiltonian();
};

} // namespace kelly
