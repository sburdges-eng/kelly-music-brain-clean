#pragma once
/**
 * EmotionMapper.h
 * 
 * Ported from Python: kellymidicompanion_emotional_mapping.py
 * Maps emotional states to musical parameters for MIDI generation.
 */

#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

namespace kelly {

// =============================================================================
// ENUMS
// =============================================================================

enum class TimingFeel {
    Straight,
    Swung,
    LaidBack,
    Pushed,
    Rubato
};

enum class EmotionCategory {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Trust,
    Anticipation,
    Complex
};

// =============================================================================
// EMOTIONAL STATE (from Python EmotionalState dataclass)
// =============================================================================

struct EmotionalState {
    float valence = 0.0f;      // -1.0 (negative) to 1.0 (positive)
    float arousal = 0.5f;      // 0.0 (calm) to 1.0 (excited)
    float intensity = 0.5f;    // 0.0 (subtle) to 1.0 (extreme)
    std::string primaryEmotion = "neutral";
    std::vector<std::string> secondaryEmotions;
    
    void clamp() {
        valence = std::clamp(valence, -1.0f, 1.0f);
        arousal = std::clamp(arousal, 0.0f, 1.0f);
        intensity = std::clamp(intensity, 0.0f, 1.0f);
    }
    
    // Calculate emotional distance in 3D space
    float distanceTo(const EmotionalState& other) const {
        float dv = valence - other.valence;
        float da = arousal - other.arousal;
        float di = intensity - other.intensity;
        return std::sqrt(dv*dv + da*da + di*di);
    }
};

// =============================================================================
// MUSICAL PARAMETERS (from Python MusicalParameters dataclass)
// =============================================================================

struct MusicalParameters {
    // Tempo
    int tempoSuggested = 100;
    int tempoMin = 60;
    int tempoMax = 180;
    
    // Key and Mode
    std::string keySuggested = "C";
    std::string modeSuggested = "major";
    std::vector<std::string> modeChoices;
    
    // Expression
    float dissonance = 0.0f;        // 0-1, harmonic tension
    float density = 0.5f;           // 0-1, arrangement density
    float spaceProbability = 0.2f;  // 0-1, probability of rests
    TimingFeel timingFeel = TimingFeel::Straight;
    float swingAmount = 0.0f;       // 0-1 for swing feel
    
    // Dynamics
    float dynamicsRange = 0.5f;     // 0-1, dynamic variation
    int velocityMin = 60;
    int velocityMax = 100;
    
    // Effects suggestions
    float reverbAmount = 0.3f;      // 0-1
    float reverbDecay = 1.5f;       // seconds
    float brightness = 0.5f;        // 0-1, EQ brightness
    float compressionRatio = 2.0f;
    float saturation = 0.0f;        // 0-1
    
    // Humanization
    float timingVariation = 0.02f;  // ±percentage of beat
    float velocityVariation = 0.1f; // ±percentage of velocity
};

// =============================================================================
// VALENCE/AROUSAL MAPPING QUADRANTS (from Python VALENCE_AROUSAL_MAPPINGS)
// =============================================================================

struct QuadrantMapping {
    std::pair<int, int> tempoRange;
    std::vector<std::string> modes;
    float dissonance;
    float brightness;
    float reverb;
    float density;
    TimingFeel defaultFeel;
};

// =============================================================================
// EMOTION MAPPER CLASS
// =============================================================================

class EmotionMapper {
public:
    EmotionMapper() {
        initializeQuadrants();
    }
    
    /**
     * Convert emotional state to musical parameters.
     * Core algorithm ported from Python get_parameters_for_state()
     */
    MusicalParameters mapToParameters(const EmotionalState& state) const {
        MusicalParameters params;
        
        // Determine quadrant (high/low valence × high/low arousal)
        int valenceSign = state.valence >= 0 ? 1 : -1;
        int arousalLevel = state.arousal >= 0.5f ? 1 : 0;
        
        const QuadrantMapping& quadrant = getQuadrant(valenceSign, arousalLevel);
        
        // --- TEMPO CALCULATION ---
        // Base tempo from quadrant, modified by intensity
        float tempoBase = (quadrant.tempoRange.first + quadrant.tempoRange.second) / 2.0f;
        float tempoRange = (quadrant.tempoRange.second - quadrant.tempoRange.first) / 2.0f;
        
        // Higher arousal = faster, higher intensity = more extreme
        float tempoModifier = (state.arousal - 0.5f) * 2.0f * tempoRange;
        tempoModifier *= (0.5f + state.intensity * 0.5f);
        
        params.tempoSuggested = static_cast<int>(std::clamp(
            tempoBase + tempoModifier,
            static_cast<float>(quadrant.tempoRange.first),
            static_cast<float>(quadrant.tempoRange.second)
        ));
        params.tempoMin = quadrant.tempoRange.first;
        params.tempoMax = quadrant.tempoRange.second;
        
        // --- MODE SELECTION ---
        params.modeChoices = quadrant.modes;
        if (!quadrant.modes.empty()) {
            // Select mode based on valence intensity
            size_t modeIndex = static_cast<size_t>(
                std::abs(state.valence) * (quadrant.modes.size() - 1)
            );
            modeIndex = std::min(modeIndex, quadrant.modes.size() - 1);
            params.modeSuggested = quadrant.modes[modeIndex];
        }
        
        // --- KEY SELECTION ---
        // Negative valence: flat keys (Eb, Bb, F)
        // Positive valence: sharp keys (G, D, A)
        // Neutral: C
        if (state.valence < -0.5f) {
            params.keySuggested = state.intensity > 0.7f ? "Eb" : "F";
        } else if (state.valence < 0.0f) {
            params.keySuggested = "Dm";
        } else if (state.valence < 0.5f) {
            params.keySuggested = "C";
        } else {
            params.keySuggested = state.intensity > 0.7f ? "A" : "G";
        }
        
        // --- DISSONANCE ---
        // Base from quadrant, increased by intensity and negative valence
        params.dissonance = quadrant.dissonance;
        if (state.valence < 0) {
            params.dissonance += std::abs(state.valence) * 0.3f * state.intensity;
        }
        params.dissonance = std::clamp(params.dissonance, 0.0f, 1.0f);
        
        // --- DENSITY ---
        // Higher arousal = denser arrangement
        params.density = 0.3f + state.arousal * 0.5f;
        params.spaceProbability = 1.0f - params.density;
        
        // --- DYNAMICS ---
        // Higher intensity = wider dynamic range
        params.dynamicsRange = 0.3f + state.intensity * 0.7f;
        params.velocityMin = static_cast<int>(40 + (1.0f - params.dynamicsRange) * 40);
        params.velocityMax = static_cast<int>(80 + params.dynamicsRange * 47);
        
        // --- TIMING FEEL ---
        params.timingFeel = quadrant.defaultFeel;
        
        // Override based on specific conditions
        if (state.valence < -0.3f && state.arousal < 0.4f) {
            // Sad, low energy: laid back
            params.timingFeel = TimingFeel::LaidBack;
            params.swingAmount = 0.0f;
        } else if (state.arousal > 0.7f && state.valence > 0.3f) {
            // Excited, positive: pushed/driving
            params.timingFeel = TimingFeel::Pushed;
            params.swingAmount = 0.0f;
        } else if (state.valence > 0.2f && state.arousal > 0.4f && state.arousal < 0.7f) {
            // Moderate positive: swing
            params.timingFeel = TimingFeel::Swung;
            params.swingAmount = 0.66f;
        }
        
        // --- EFFECTS ---
        params.reverbAmount = quadrant.reverb;
        params.reverbDecay = 1.0f + (1.0f - state.arousal) * 2.0f; // Slower decay for calmer
        params.brightness = quadrant.brightness;
        
        // Compression: more for high intensity
        params.compressionRatio = 2.0f + state.intensity * 4.0f;
        
        // Saturation: for anger/intensity
        if (state.valence < -0.3f && state.intensity > 0.6f) {
            params.saturation = state.intensity * 0.5f;
        }
        
        // --- HUMANIZATION ---
        // More variation for higher intensity and arousal
        params.timingVariation = 0.01f + state.intensity * 0.03f;
        params.velocityVariation = 0.05f + state.intensity * 0.15f;
        
        return params;
    }
    
    /**
     * Get the emotional quadrant mapping
     */
    const QuadrantMapping& getQuadrant(int valenceSign, int arousalLevel) const {
        if (valenceSign > 0 && arousalLevel > 0) return quadrantHighHigh_;
        if (valenceSign > 0 && arousalLevel == 0) return quadrantHighLow_;
        if (valenceSign <= 0 && arousalLevel > 0) return quadrantLowHigh_;
        return quadrantLowLow_;
    }
    
private:
    // Quadrant mappings (ported from Python VALENCE_AROUSAL_MAPPINGS)
    QuadrantMapping quadrantHighHigh_;  // Joy, Excitement
    QuadrantMapping quadrantHighLow_;   // Peace, Contentment
    QuadrantMapping quadrantLowHigh_;   // Anger, Fear
    QuadrantMapping quadrantLowLow_;    // Sadness, Grief
    
    void initializeQuadrants() {
        // High valence, high arousal (Joy, Excitement)
        quadrantHighHigh_ = {
            {120, 160},
            {"Ionian", "Lydian", "Mixolydian"},
            0.1f,   // dissonance
            0.8f,   // brightness
            0.3f,   // reverb
            0.7f,   // density
            TimingFeel::Straight
        };
        
        // High valence, low arousal (Peace, Contentment)
        quadrantHighLow_ = {
            {60, 90},
            {"Ionian", "Lydian"},
            0.0f,
            0.5f,
            0.5f,
            0.4f,
            TimingFeel::LaidBack
        };
        
        // Low valence, high arousal (Anger, Fear)
        quadrantLowHigh_ = {
            {100, 180},
            {"Phrygian", "Locrian", "Aeolian"},
            0.7f,
            0.6f,
            0.2f,
            0.8f,
            TimingFeel::Pushed
        };
        
        // Low valence, low arousal (Sadness, Grief)
        quadrantLowLow_ = {
            {50, 80},
            {"Aeolian", "Dorian", "Phrygian"},
            0.3f,
            0.3f,
            0.6f,
            0.3f,
            TimingFeel::Rubato
        };
    }
};

} // namespace kelly
