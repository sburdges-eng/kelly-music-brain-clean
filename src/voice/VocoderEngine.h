#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <cstdint>

namespace kelly {

/**
 * VocoderEngine - Formant-based voice synthesis engine
 *
 * Implements a sophisticated vocoder using formant synthesis to generate
 * realistic vocal sounds. Uses bandpass filters to shape formants (vocal
 * resonances) that give speech its characteristic timbre.
 */
class VocoderEngine {
public:
    VocoderEngine();
    ~VocoderEngine() = default;

    /**
     * Prepare the vocoder for synthesis
     * @param sampleRate Audio sample rate (e.g., 44100.0)
     */
    void prepare(double sampleRate);

    /**
     * Synthesize a single sample of vocal audio
     * @param pitch Current pitch in Hz
     * @param formants Formant frequencies [F1, F2, F3, F4] in Hz
     * @param formantBandwidths Formant bandwidths [B1, B2, B3, B4] in Hz
     * @param vibratoDepth Vibrato depth (0.0 to 1.0)
     * @param vibratoRate Vibrato rate in Hz
     * @param breathiness Amount of noise (0.0 to 1.0)
     * @param brightness Brightness/timbre (0.0 to 1.0)
     * @return Synthesized audio sample
     */
    float processSample(
        float pitch,
        const std::array<float, 4>& formants,
        const std::array<float, 4>& formantBandwidths,
        float vibratoDepth = 0.0f,
        float vibratoRate = 5.0f,
        float breathiness = 0.0f,
        float brightness = 0.5f
    );

    /**
     * Set target formants for smooth interpolation (for phoneme transitions)
     * @param targetFormants Target formant frequencies [F1, F2, F3, F4]
     * @param targetBandwidths Target formant bandwidths [B1, B2, B3, B4]
     * @param transitionTime Transition time in seconds
     */
    void setTargetFormants(
        const std::array<float, 4>& targetFormants,
        const std::array<float, 4>& targetBandwidths,
        float transitionTime = 0.05f
    );

    /**
     * Get current interpolated formants (for smooth transitions)
     * @return Current formant frequencies and bandwidths
     */
    std::pair<std::array<float, 4>, std::array<float, 4>> getCurrentFormants() const;

    /**
     * Reset internal state (e.g., for new note)
     */
    void reset();

    /**
     * Set formant shift (for different voice types)
     * @param shift Multiplier for formant frequencies (1.0 = normal, >1.0 = higher, <1.0 = lower)
     */
    void setFormantShift(float shift) { formantShift_ = shift; }

    /**
     * Set glottal pulse shape (affects brightness)
     * @param shape 0.0 = smooth, 1.0 = sharp
     */
    void setGlottalShape(float shape) { glottalShape_ = shape; }

private:
    double sampleRate_ = 44100.0;
    float formantShift_ = 1.0f;
    float glottalShape_ = 0.5f;

    // Vibrato oscillator state
    float vibratoPhase_ = 0.0f;

    // Glottal pulse phase (for periodic waveform generation)
    float glottalPhase_ = 0.0f;

    // Brightness filter state (for high-frequency emphasis)
    float brightnessState_ = 0.0f;

    // Formant filter states (one-pole filters approximated as second-order resonators)
    struct FormantFilter {
        float y1 = 0.0f;  // Previous output
        float y2 = 0.0f;  // Two samples ago
        float x1 = 0.0f;  // Previous input
        float x2 = 0.0f;  // Two samples ago

        float b0 = 0.0f;  // Filter coefficients
        float b1 = 0.0f;
        float b2 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;

        void reset() {
            y1 = y2 = x1 = x2 = 0.0f;
        }
    };

    std::array<FormantFilter, 4> formantFilters_;

    // Formant interpolation state (for smooth phoneme transitions)
    std::array<float, 4> currentFormants_ = {500.0f, 1500.0f, 2500.0f, 3300.0f};
    std::array<float, 4> currentBandwidths_ = {60.0f, 90.0f, 120.0f, 150.0f};
    std::array<float, 4> targetFormants_ = {500.0f, 1500.0f, 2500.0f, 3300.0f};
    std::array<float, 4> targetBandwidths_ = {60.0f, 90.0f, 120.0f, 150.0f};
    float formantInterpolationRate_ = 0.0f;  // Samples per formant unit change
    bool isInterpolating_ = false;

    // White noise generator state
    uint32_t noiseSeed_ = 12345;

    // Generate white noise
    float generateNoise();

    // Generate glottal pulse (source signal)
    float generateGlottalPulse(float phase, float shape);

    // Update formant filter coefficients
    void updateFormantFilter(
        FormantFilter& filter,
        float frequency,
        float bandwidth,
        float sampleRate
    );

    // Apply formant filter
    float applyFormantFilter(FormantFilter& filter, float input);
};

/**
 * VowelFormantDatabase - Database of vowel formant frequencies
 *
 * Stores formant frequencies (F1-F4) and bandwidths for different vowels.
 * Formants are the resonant frequencies that characterize vowel sounds.
 */
class VowelFormantDatabase {
public:
    enum class Vowel {
        AH,  // "father"
        EH,  // "bed"
        IH,  // "bit"
        OH,  // "boat"
        OO,  // "boot"
        AA,  // "cat"
        EE,  // "beet"
        OW,  // "cow"
        UH,  // "put"
        AY,  // "bait"
        IY,  // "beat"
        EY,  // "bay"
        OY,  // "boy"
        AW,  // "bought"
        ER   // "her"
    };

    struct FormantData {
        std::array<float, 4> frequencies;  // F1, F2, F3, F4 in Hz
        std::array<float, 4> bandwidths;   // B1, B2, B3, B4 in Hz
    };

    /**
     * Get formant data for a vowel
     */
    static FormantData getFormants(Vowel vowel);

    /**
     * Interpolate between two vowels
     * @param v1 First vowel
     * @param v2 Second vowel
     * @param t Interpolation factor (0.0 = v1, 1.0 = v2)
     */
    static FormantData interpolate(Vowel v1, Vowel v2, float t);

    /**
     * Get formants for a pitch (approximate vowel selection based on pitch)
     */
    static FormantData getFormantsForPitch(int midiPitch);

    /**
     * Interpolate formants smoothly over time (for phoneme transitions).
     * @param formants1 First formant set
     * @param formants2 Second formant set
     * @param t Interpolation factor (0.0 = formants1, 1.0 = formants2)
     * @return Interpolated formant data
     */
    static FormantData interpolateFormants(
        const FormantData& formants1,
        const FormantData& formants2,
        float t
    );
};

/**
 * ADSR Envelope Generator for note shaping
 */
class ADSREnvelope {
public:
    ADSREnvelope();

    void prepare(double sampleRate);

    /**
     * Trigger envelope (start note)
     */
    void trigger();

    /**
     * Release envelope (end note)
     */
    void release();

    /**
     * Get current envelope value (0.0 to 1.0)
     */
    float getValue();

    /**
     * Check if envelope is finished
     */
    bool isFinished() const { return state_ == State::Idle; }

    // Envelope parameters (in seconds)
    float attackTime = 0.01f;   // Attack time
    float decayTime = 0.05f;    // Decay time
    float sustainLevel = 0.7f;  // Sustain level (0.0 to 1.0)
    float releaseTime = 0.1f;   // Release time

private:
    enum class State {
        Idle,
        Attack,
        Decay,
        Sustain,
        Release
    };

    State state_ = State::Idle;
    float currentValue_ = 0.0f;
    double sampleRate_ = 44100.0;

    // Rate calculations
    float attackRate_ = 0.0f;
    float decayRate_ = 0.0f;
    float releaseRate_ = 0.0f;

    void calculateRates();
};

/**
 * Portamento/Glide generator for smooth pitch transitions
 */
class PortamentoGenerator {
public:
    PortamentoGenerator();

    void prepare(double sampleRate);

    /**
     * Set target pitch (in Hz)
     */
    void setTargetPitch(float pitch);

    /**
     * Get current pitch with portamento applied
     */
    float getCurrentPitch();

    /**
     * Set portamento time (in seconds)
     */
    void setPortamentoTime(float time) { portamentoTime_ = time; calculateRate(); }

    /**
     * Enable/disable portamento
     */
    void setEnabled(bool enabled) { enabled_ = enabled; }

    void reset() { currentPitch_ = targetPitch_; }

private:
    float currentPitch_ = 440.0f;
    float targetPitch_ = 440.0f;
    float portamentoTime_ = 0.1f;
    float portamentoRate_ = 0.0f;
    bool enabled_ = true;
    double sampleRate_ = 44100.0;

    void calculateRate();
};

} // namespace kelly
