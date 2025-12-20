#include "engine/EmotionToMusicMapper.h"
#include "common/MusicConstants.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <complex>

namespace kelly {
using namespace MusicConstants;

EmotionToMusicMapper::EmotionToMusicMapper() {
    // Initialize emotion-to-scale mapping
    emotionScaleMap_ = {
        {"Joy", "Lydian"},
        {"Sadness", "Aeolian"},
        {"Fear", "Phrygian"},
        {"Anger", "Locrian"},
        {"Trust", "Mixolydian"},
        {"Surprise", "Lydian"},
        {"Disgust", "Locrian"},
        {"Anticipation", "Ionian"}
    };

    // Initialize emotion-to-chord mapping
    emotionChordMap_ = {
        {"Joy", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH}},           // Major triad
        {"Sadness", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FIFTH}},        // Minor triad
        {"Anger", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_TRITONE}},          // Diminished
        {"Fear", {INTERVAL_UNISON, INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH}},           // Suspended
        {"Trust", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SEVENTH}},      // Major 7th
        {"Surprise", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SEVENTH}},   // Lydian chord
        {"Disgust", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_TRITONE}},        // Diminished
        {"Anticipation", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH}} // Add9
    };
}

float EmotionToMusicMapper::emotionToFrequency(
    const std::string& emotion,
    float valence,
    float arousal,
    float baseFreq
) const {
    // Map emotion name to frequency formula
    if (emotion == "Joy") {
        return joyFrequency(valence, arousal, baseFreq);
    } else if (emotion == "Sadness") {
        return sadnessFrequency(valence, arousal, baseFreq);
    } else if (emotion == "Fear") {
        return fearFrequency(valence, arousal, baseFreq);
    } else if (emotion == "Anger") {
        return angerFrequency(valence, arousal, baseFreq);
    } else if (emotion == "Trust") {
        return trustFrequency(valence, arousal, baseFreq);
    }

    // Default: use VAD-based formula
    return vadToFrequency(VADState(valence, arousal, 0.5f), baseFreq);
}

float EmotionToMusicMapper::vadToFrequency(const VADState& vad, float baseFreq) const {
    // General formula: f = f₀(1 + k₁V + k₂A)
    // Adjusted for emotional mapping
    float freq = baseFreq * (1.0f + 0.3f * vad.valence + 0.5f * vad.arousal);
    return std::max(80.0f, std::min(2000.0f, freq));  // Clamp to reasonable range
}

float EmotionToMusicMapper::joyFrequency(float valence, float arousal, float baseFreq) const {
    // f_J = f₀(1 + V + 0.5A)
    return baseFreq * (1.0f + valence + 0.5f * arousal);
}

float EmotionToMusicMapper::sadnessFrequency(float valence, float arousal, float baseFreq) const {
    // f_S = f₀(1 - |V|)
    return baseFreq * (1.0f - std::abs(valence));
}

float EmotionToMusicMapper::fearFrequency(float valence, float arousal, float baseFreq) const {
    // f_F = f₀(1 + 0.3A - 0.6V)
    return baseFreq * (1.0f + 0.3f * arousal - 0.6f * valence);
}

float EmotionToMusicMapper::angerFrequency(float valence, float arousal, float baseFreq) const {
    // f_A = f₀(1 + 0.8A)sin(πV)
    return baseFreq * (1.0f + 0.8f * arousal) * std::sin(3.14159f * (valence + 1.0f) / 2.0f);
}

float EmotionToMusicMapper::trustFrequency(float valence, float arousal, float baseFreq) const {
    // f_T = f₀(1 + 0.2V + 0.2A)
    return baseFreq * (1.0f + 0.2f * valence + 0.2f * arousal);
}

MusicalFrequency EmotionToMusicMapper::quantumStateToFrequency(
    const QuantumEmotionalState& qState,
    float baseFreq
) const {
    MusicalFrequency result;
    result.baseFrequency = baseFreq;

    // Calculate weighted average frequency from quantum state
    float totalProb = 0.0f;
    float weightedFreq = 0.0f;

    for (const auto& eState : qState.states) {
        // Get VAD for this emotion basis
        // For now, use simple mapping
        float valence = (eState.basis == "Joy" || eState.basis == "Trust") ? 0.5f : -0.5f;
        float arousal = 0.5f;

        float freq = emotionToFrequency(eState.basis, valence, arousal, baseFreq);
        float weight = eState.probability;

        weightedFreq += freq * weight;
        totalProb += weight;
    }

    if (totalProb > 0.0f) {
        result.baseFrequency = weightedFreq / totalProb;
    }

    // Generate harmonics
    for (int i = 1; i <= 5; ++i) {
        result.harmonics.push_back(result.baseFrequency * static_cast<float>(i));
    }

    return result;
}

VoiceParameters EmotionToMusicMapper::vadToVoice(const VADState& vad) const {
    VoiceParameters voice;

    // Pitch: f₀ = f_base(1 + 0.5A + 0.3V)
    voice.pitch = 200.0f * (1.0f + 0.5f * vad.arousal + 0.3f * vad.valence);
    voice.pitch = std::clamp(voice.pitch, 80.0f, 400.0f);

    // Volume: A = A_base(1 + 0.4D + 0.3A)
    voice.amplitude = 0.7f * (1.0f + 0.4f * vad.dominance + 0.3f * vad.arousal);
    voice.amplitude = std::clamp(voice.amplitude, 0.0f, 1.0f);

    // Formant shift: F_i' = F_i(1 + 0.2V - 0.1D)
    float formantShift = 1.0f + 0.2f * vad.valence - 0.1f * vad.dominance;
    voice.formant1 = 800.0f * formantShift;
    voice.formant2 = 1200.0f * formantShift;
    voice.formant3 = 2400.0f * formantShift;

    // Spectral tilt: T_s' = T_s + (6V - 4A)
    voice.spectralTilt = 0.0f + (6.0f * vad.valence - 4.0f * vad.arousal);
    voice.spectralTilt = std::clamp(voice.spectralTilt, -12.0f, 6.0f);

    // Vibrato rate: v_r' = 5 + 3A
    voice.vibratoRate = 5.0f + 3.0f * vad.arousal;
    voice.vibratoRate = std::clamp(voice.vibratoRate, 4.0f, 8.0f);

    // Vibrato depth: v_d' = 2 + V + 0.5A
    voice.vibratoDepth = 2.0f + vad.valence + 0.5f * vad.arousal;
    voice.vibratoDepth = std::clamp(voice.vibratoDepth, 1.0f, 3.0f);

    // Speech rate: R = R₀(1 + 0.7A - 0.4V)
    voice.speechRate = 1.0f * (1.0f + 0.7f * vad.arousal - 0.4f * vad.valence);
    voice.speechRate = std::clamp(voice.speechRate, 0.5f, 2.0f);

    // Jitter and shimmer (higher for negative emotions)
    voice.jitter = std::abs(vad.valence) * 2.0f;  // More jitter for extreme emotions
    voice.shimmer = std::abs(vad.valence) * 1.5f;

    return voice;
}

VoiceParameters EmotionToMusicMapper::quantumStateToVoice(const QuantumEmotionalState& qState) const {
    // Convert quantum state to VAD, then to voice
    // This requires basis VAD map - for now use default
    QuantumEmotionalField field;
    VADState vad = qState.toVAD(field.getBasisVADMap());
    return vadToVoice(vad);
}

std::vector<int> EmotionToMusicMapper::emotionToChord(const std::string& emotion, float valence) const {
    auto it = emotionChordMap_.find(emotion);
    if (it != emotionChordMap_.end()) {
        std::vector<int> chord = it->second;

        // Adjust for valence: positive = major shift, negative = minor shift
        if (valence > VALENCE_NEUTRAL && chord.size() >= 3) {
            // Shift to major (raise third)
            if (chord[1] == INTERVAL_MINOR_THIRD) chord[1] = INTERVAL_MAJOR_THIRD;
        } else if (valence < VALENCE_NEUTRAL && chord.size() >= 3) {
            // Shift to minor (lower third)
            if (chord[1] == INTERVAL_MAJOR_THIRD) chord[1] = INTERVAL_MINOR_THIRD;
        }

        return chord;
    }

    // Default: major triad
    return {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH};
}

std::string EmotionToMusicMapper::emotionToScale(const std::string& emotion) const {
    auto it = emotionScaleMap_.find(emotion);
    return (it != emotionScaleMap_.end()) ? it->second : "Ionian";
}

int EmotionToMusicMapper::arousalToTempo(float arousal, int baseTempo) const {
    // T = T₀(1 - A) but we want higher arousal = faster tempo
    // So: T = T₀(1 + A)
    return static_cast<int>(baseTempo * (1.0f + arousal));
}

int EmotionToMusicMapper::dominanceToVelocity(float dominance, int baseVelocity) const {
    // Volume = 70 + 30D (MIDI velocity)
    return static_cast<int>(baseVelocity + 30.0f * dominance);
}

EmotionToMusicMapper::HarmonicField EmotionToMusicMapper::generateHarmonicField(
    const QuantumEmotionalState& qState,
    float time,
    float baseFreq
) const {
    HarmonicField field;
    field.time = time;

    // Generate frequencies and amplitudes from quantum state
    for (const auto& eState : qState.states) {
        float freq = emotionToFrequency(eState.basis, 0.0f, 0.5f, baseFreq);
        float amplitude = std::abs(eState.amplitude);
        float phase = std::arg(eState.amplitude);

        field.frequencies.push_back(freq);
        field.amplitudes.push_back(amplitude);
        field.phases.push_back(phase);
    }

    return field;
}

float EmotionToMusicMapper::calculateFrequencyResonance(
    float freq1,
    float freq2,
    float time
) const {
    // R = cos(2π(f₁ - f₂)t)
    float beatFreq = freq1 - freq2;
    return std::cos(2.0f * 3.14159f * beatFreq * time);
}

float EmotionToMusicMapper::calculateResonanceEnergy(
    const std::vector<float>& amplitudes,
    const std::vector<float>& phases
) const {
    // E_res = |Σ a_i e^(iφ_i)|²
    if (amplitudes.size() != phases.size()) {
        return 0.0f;
    }

    std::complex<float> sum(0.0f, 0.0f);
    for (size_t i = 0; i < amplitudes.size(); ++i) {
        std::complex<float> term(amplitudes[i] * std::cos(phases[i]),
                     amplitudes[i] * std::sin(phases[i]));
        sum += term;
    }

    return std::norm(sum);
}

} // namespace kelly
