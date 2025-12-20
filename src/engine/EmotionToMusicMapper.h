#pragma once

#include "engine/VADCalculator.h"
#include "engine/QuantumEmotionalField.h"
#include "common/MusicConstants.h"
#include <string>
#include <map>
#include <vector>

namespace kelly {
using namespace MusicConstants;

/**
 * Emotion-to-Music Frequency Mapping
 * 
 * Maps emotions to musical frequencies, chords, scales, and timbres
 * based on VAD coordinates and quantum emotional field.
 */

/**
 * Voice synthesis parameters
 */
struct VoiceParameters {
    float pitch;              // Fundamental frequency (Hz)
    float amplitude;          // Volume (0-1)
    float formant1;           // F1 (Hz)
    float formant2;           // F2 (Hz)
    float formant3;           // F3 (Hz)
    float spectralTilt;      // Spectral slope (dB/oct)
    float vibratoRate;        // Vibrato frequency (Hz)
    float vibratoDepth;       // Vibrato depth (semitones)
    float speechRate;         // Speech rate multiplier
    float jitter;             // Pitch jitter (%)
    float shimmer;            // Amplitude shimmer (%)
    
    VoiceParameters() 
        : pitch(200.0f), amplitude(0.7f),
          formant1(800.0f), formant2(1200.0f), formant3(2400.0f),
          spectralTilt(0.0f), vibratoRate(5.0f), vibratoDepth(2.0f),
          speechRate(1.0f), jitter(0.0f), shimmer(0.0f) {}
};

/**
 * Musical frequency parameters
 */
struct MusicalFrequency {
    float baseFrequency;      // Base frequency (Hz)
    std::vector<float> harmonics;  // Harmonic frequencies
    std::string scale;        // Musical scale name
    std::vector<int> chord;   // Chord intervals (semitones)
    
    MusicalFrequency() : baseFrequency(440.0f) {}  // A4
};

/**
 * Emotion-to-Music Mapper
 */
class EmotionToMusicMapper {
public:
    EmotionToMusicMapper();
    
    /**
     * Map emotion to frequency based on VAD
     * Formulas from quantum emotional field model
     */
    float emotionToFrequency(
        const std::string& emotion,
        float valence,
        float arousal,
        float baseFreq = 440.0f
    ) const;
    
    /**
     * Calculate frequency from VAD directly
     */
    float vadToFrequency(const VADState& vad, float baseFreq = 440.0f) const;
    
    /**
     * Generate musical frequency from quantum emotional state
     */
    MusicalFrequency quantumStateToFrequency(
        const QuantumEmotionalState& qState,
        float baseFreq = 440.0f
    ) const;
    
    /**
     * Calculate voice parameters from VAD
     */
    VoiceParameters vadToVoice(const VADState& vad) const;
    
    /**
     * Calculate voice parameters from quantum state
     */
    VoiceParameters quantumStateToVoice(const QuantumEmotionalState& qState) const;
    
    /**
     * Get chord from emotion
     */
    std::vector<int> emotionToChord(const std::string& emotion, float valence) const;
    
    /**
     * Get scale from emotion
     */
    std::string emotionToScale(const std::string& emotion) const;
    
    /**
     * Calculate tempo from arousal
     */
    int arousalToTempo(float arousal, int baseTempo = TEMPO_MODERATE) const;
    
    /**
     * Calculate volume from dominance
     */
    int dominanceToVelocity(float dominance, int baseVelocity = 70) const;
    
    /**
     * Generate quantum harmonic field
     * Ψ_music(t) = Σ α_i e^(i2πf_i t + φ_i)
     */
    struct HarmonicField {
        std::vector<float> frequencies;
        std::vector<float> amplitudes;
        std::vector<float> phases;
        float time;
    };
    
    HarmonicField generateHarmonicField(
        const QuantumEmotionalState& qState,
        float time,
        float baseFreq = 440.0f
    ) const;
    
    /**
     * Calculate resonance between two emotional frequencies
     * R = cos(2π(f₁ - f₂)t)
     */
    float calculateFrequencyResonance(
        float freq1,
        float freq2,
        float time
    ) const;
    
    /**
     * Calculate resonance energy
     * E_res = |Σ a_i e^(iφ_i)|²
     */
    float calculateResonanceEnergy(
        const std::vector<float>& amplitudes,
        const std::vector<float>& phases
    ) const;
    
private:
    // Emotion-to-frequency formulas
    float joyFrequency(float valence, float arousal, float baseFreq) const;
    float sadnessFrequency(float valence, float arousal, float baseFreq) const;
    float fearFrequency(float valence, float arousal, float baseFreq) const;
    float angerFrequency(float valence, float arousal, float baseFreq) const;
    float trustFrequency(float valence, float arousal, float baseFreq) const;
    
    // Emotion-to-scale mapping
    std::map<std::string, std::string> emotionScaleMap_;
    
    // Emotion-to-chord mapping
    std::map<std::string, std::vector<int>> emotionChordMap_;
};

} // namespace kelly
