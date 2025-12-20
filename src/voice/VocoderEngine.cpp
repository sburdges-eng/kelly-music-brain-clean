#include "voice/VocoderEngine.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace kelly {

//==============================================================================
// VocoderEngine Implementation
//==============================================================================

VocoderEngine::VocoderEngine() {
    for (auto& filter : formantFilters_) {
        filter.reset();
    }
}

void VocoderEngine::prepare(double sampleRate) {
    sampleRate_ = sampleRate;
    vibratoPhase_ = 0.0f;
    glottalPhase_ = 0.0f;
    brightnessState_ = 0.0f;
    isInterpolating_ = false;
    formantInterpolationRate_ = 0.0f;

    // Initialize formants to neutral values
    currentFormants_ = {500.0f, 1500.0f, 2500.0f, 3300.0f};
    currentBandwidths_ = {60.0f, 90.0f, 120.0f, 150.0f};
    targetFormants_ = currentFormants_;
    targetBandwidths_ = currentBandwidths_;

    for (auto& filter : formantFilters_) {
        filter.reset();
    }

    // Initialize noise seed
    noiseSeed_ = 12345;
}

void VocoderEngine::reset() {
    vibratoPhase_ = 0.0f;
    glottalPhase_ = 0.0f;
    brightnessState_ = 0.0f;
    isInterpolating_ = false;
    formantInterpolationRate_ = 0.0f;
    for (auto& filter : formantFilters_) {
        filter.reset();
    }
}

float VocoderEngine::generateNoise() {
    // Linear congruential generator for white noise
    noiseSeed_ = noiseSeed_ * 1103515245 + 12345;
    // Convert to float in range [-1.0, 1.0]
    return (static_cast<int32_t>(noiseSeed_) & 0x7FFFFFFF) / 2147483647.5f - 1.0f;
}

float VocoderEngine::generateGlottalPulse(float phase, float shape) {
    // Generate a periodic glottal pulse using a simplified model
    // Phase should be in range [0, 2*PI]

    // Clamp shape
    shape = std::clamp(shape, 0.0f, 1.0f);

    // Simplified Rosenberg glottal model
    // Returns a pulse waveform that approximates vocal cord vibration

    if (phase < 0.0f) phase += 2.0f * static_cast<float>(M_PI);
    if (phase >= 2.0f * static_cast<float>(M_PI)) phase -= 2.0f * static_cast<float>(M_PI);

    // Open phase (0 to T_open)
    float T_open = (1.0f - shape * 0.3f) * static_cast<float>(M_PI);

    if (phase < T_open) {
        // Rising phase: 0.5 * (1 - cos(pi * t / T_open))
        return 0.5f * (1.0f - std::cos(static_cast<float>(M_PI) * phase / T_open));
    } else {
        // Closing phase: exponential decay
        float t_close = phase - T_open;
        float decay_rate = 2.0f + shape * 5.0f; // Faster decay for sharper pulses
        return std::exp(-decay_rate * t_close) * 0.5f * (1.0f - std::cos(static_cast<float>(M_PI)));
    }
}

void VocoderEngine::updateFormantFilter(
    FormantFilter& filter,
    float frequency,
    float bandwidth,
    float sampleRate)
{
    // Design a bandpass filter using bilinear transform
    // Using a second-order resonator (bandpass) for each formant

    if (frequency <= 0.0f || bandwidth <= 0.0f || sampleRate <= 0.0f) {
        // Invalid parameters - set to passthrough
        filter.b0 = 1.0f;
        filter.b1 = filter.b2 = filter.a1 = filter.a2 = 0.0f;
        return;
    }

    float w = 2.0f * static_cast<float>(M_PI) * frequency / sampleRate;
    float r = std::exp(-static_cast<float>(M_PI) * bandwidth / sampleRate);

    // Second-order bandpass filter coefficients
    // Using a simple resonator structure
    float r2 = r * r;
    float cosw = std::cos(w);

    filter.b0 = 0.5f * (1.0f - r2);
    filter.b1 = 0.0f;
    filter.b2 = -filter.b0;
    filter.a1 = -2.0f * r * cosw;
    filter.a2 = r2;
}

float VocoderEngine::applyFormantFilter(FormantFilter& filter, float input) {
    // Direct Form II transposed structure (more efficient)
    float output = filter.b0 * input + filter.y1;

    filter.y1 = filter.b1 * input - filter.a1 * output + filter.y2;
    filter.y2 = filter.b2 * input - filter.a2 * output;

    // Prevent numerical overflow
    constexpr float maxValue = 10.0f;
    output = std::clamp(output, -maxValue, maxValue);
    filter.y1 = std::clamp(filter.y1, -maxValue, maxValue);
    filter.y2 = std::clamp(filter.y2, -maxValue, maxValue);

    return output;
}

void VocoderEngine::setTargetFormants(
    const std::array<float, 4>& targetFormants,
    const std::array<float, 4>& targetBandwidths,
    float transitionTime)
{
    targetFormants_ = targetFormants;
    targetBandwidths_ = targetBandwidths;

    if (transitionTime > 0.0f && sampleRate_ > 0.0) {
        // Calculate interpolation rate (formant units per sample)
        // Use exponential interpolation for smoother transitions
        float samplesPerTransition = static_cast<float>(sampleRate_) * transitionTime;
        formantInterpolationRate_ = 1.0f / samplesPerTransition;
        isInterpolating_ = true;
    } else {
        // Instant transition
        currentFormants_ = targetFormants;
        currentBandwidths_ = targetBandwidths;
        isInterpolating_ = false;
    }
}

std::pair<std::array<float, 4>, std::array<float, 4>> VocoderEngine::getCurrentFormants() const {
    return {currentFormants_, currentBandwidths_};
}

float VocoderEngine::processSample(
    float pitch,
    const std::array<float, 4>& formants,
    const std::array<float, 4>& formantBandwidths,
    float vibratoDepth,
    float vibratoRate,
    float breathiness,
    float brightness)
{
    // Update formant interpolation if active
    if (isInterpolating_) {
        bool allReached = true;
        for (size_t i = 0; i < 4; ++i) {
            float diff = targetFormants_[i] - currentFormants_[i];
            if (std::abs(diff) > 1.0f) {  // 1 Hz threshold
                // Exponential interpolation for smooth transition
                currentFormants_[i] += diff * formantInterpolationRate_;
                allReached = false;
            } else {
                currentFormants_[i] = targetFormants_[i];
            }

            float bwDiff = targetBandwidths_[i] - currentBandwidths_[i];
            if (std::abs(bwDiff) > 0.1f) {
                currentBandwidths_[i] += bwDiff * formantInterpolationRate_;
                allReached = false;
            } else {
                currentBandwidths_[i] = targetBandwidths_[i];
            }
        }

        if (allReached) {
            isInterpolating_ = false;
        }
    } else {
        // No interpolation active, use provided formants directly
        currentFormants_ = formants;
        currentBandwidths_ = formantBandwidths;
    }

    // Update vibrato phase
    float vibratoPhaseInc = 2.0f * static_cast<float>(M_PI) * vibratoRate / static_cast<float>(sampleRate_);
    vibratoPhase_ += vibratoPhaseInc;
    if (vibratoPhase_ >= 2.0f * static_cast<float>(M_PI)) {
        vibratoPhase_ -= 2.0f * static_cast<float>(M_PI);
    }

    // Apply vibrato to pitch
    float vibratoOffset = vibratoDepth * std::sin(vibratoPhase_) * 0.02f; // ±2% pitch modulation
    float currentPitch = pitch * (1.0f + vibratoOffset);

    // Generate source signal (glottal pulse)
    float phaseInc = 2.0f * static_cast<float>(M_PI) * currentPitch / static_cast<float>(sampleRate_);
    glottalPhase_ += phaseInc;
    if (glottalPhase_ >= 2.0f * static_cast<float>(M_PI)) {
        glottalPhase_ -= 2.0f * static_cast<float>(M_PI);
    }

    float glottalSignal = generateGlottalPulse(glottalPhase_, glottalShape_);

    // Mix in breathiness (noise)
    float noise = generateNoise();
    float source = glottalSignal * (1.0f - breathiness) + noise * breathiness;

    // Apply formant filters (series connection)
    float output = source;

    // Update and apply each formant filter using interpolated formants
    for (size_t i = 0; i < formantFilters_.size(); ++i) {
        float freq = currentFormants_[i] * formantShift_;
        float bw = currentBandwidths_[i];

        updateFormantFilter(formantFilters_[i], freq, bw, static_cast<float>(sampleRate_));
        output = applyFormantFilter(formantFilters_[i], output);
    }

    // Apply brightness (high-frequency emphasis)
    // Simple high-frequency boost for brightness
    if (brightness > 0.0f) {
        // Approximate high-shelf filter using a simple high-pass characteristic
        float brightnessCoeff = brightness * 0.3f;
        float brightOutput = output + brightnessCoeff * (output - brightnessState_);
        brightnessState_ = output;

        // Mix original and brightened signal
        output = output * (1.0f - brightness * 0.5f) + brightOutput * (brightness * 0.5f);
    }

    // Normalize to prevent clipping
    output *= 0.3f; // Scale down to reasonable level

    return output;
}

//==============================================================================
// VowelFormantDatabase Implementation
//==============================================================================

VowelFormantDatabase::FormantData VowelFormantDatabase::getFormants(Vowel vowel) {
    // Formant frequencies in Hz (typical adult male voice)
    // Format: {F1, F2, F3, F4}, {B1, B2, B3, B4}

    switch (vowel) {
        case Vowel::AH:  // "father" - /ɑ/
            return {{730.0f, 1090.0f, 2440.0f, 3300.0f},
                    {80.0f, 90.0f, 120.0f, 150.0f}};

        case Vowel::EH:  // "bed" - /ɛ/
            return {{610.0f, 1900.0f, 2480.0f, 3200.0f},
                    {70.0f, 100.0f, 120.0f, 150.0f}};

        case Vowel::IH:  // "bit" - /ɪ/
            return {{400.0f, 1920.0f, 2560.0f, 3400.0f},
                    {60.0f, 100.0f, 120.0f, 150.0f}};

        case Vowel::OH:  // "boat" - /oʊ/
            return {{570.0f, 840.0f, 2410.0f, 3250.0f},
                    {70.0f, 80.0f, 120.0f, 150.0f}};

        case Vowel::OO:  // "boot" - /u/
            return {{300.0f, 870.0f, 2240.0f, 3100.0f},
                    {50.0f, 80.0f, 120.0f, 150.0f}};

        case Vowel::AA:  // "cat" - /æ/
            return {{660.0f, 1720.0f, 2410.0f, 3300.0f},
                    {70.0f, 100.0f, 120.0f, 150.0f}};

        case Vowel::EE:  // "beet" - /i/
            return {{270.0f, 2290.0f, 3010.0f, 3600.0f},
                    {40.0f, 100.0f, 120.0f, 150.0f}};

        case Vowel::OW:  // "cow" - /aʊ/
            return {{640.0f, 1200.0f, 2400.0f, 3200.0f},
                    {70.0f, 90.0f, 120.0f, 150.0f}};

        case Vowel::UH:  // "put" - /ʊ/
            return {{440.0f, 1020.0f, 2240.0f, 3100.0f},
                    {60.0f, 80.0f, 120.0f, 150.0f}};

        case Vowel::AY:  // "bait" - /eɪ/
            return {{530.0f, 1840.0f, 2480.0f, 3300.0f},
                    {60.0f, 100.0f, 120.0f, 150.0f}};

        case Vowel::IY:  // "beat" - /iː/
            return {{280.0f, 2250.0f, 3100.0f, 3600.0f},
                    {40.0f, 100.0f, 120.0f, 150.0f}};

        case Vowel::EY:  // "bay" - /eɪ/
            return {{520.0f, 1850.0f, 2500.0f, 3300.0f},
                    {60.0f, 100.0f, 120.0f, 150.0f}};

        case Vowel::OY:  // "boy" - /ɔɪ/
            return {{500.0f, 950.0f, 2400.0f, 3200.0f},
                    {60.0f, 85.0f, 120.0f, 150.0f}};

        case Vowel::AW:  // "bought" - /ɔ/
            return {{570.0f, 840.0f, 2410.0f, 3250.0f},
                    {70.0f, 80.0f, 120.0f, 150.0f}};

        case Vowel::ER:  // "her" - /ɜr/
            return {{490.0f, 1350.0f, 1690.0f, 2600.0f},
                    {60.0f, 90.0f, 100.0f, 140.0f}};

        default:
            // Default to neutral vowel
            return {{500.0f, 1500.0f, 2500.0f, 3300.0f},
                    {60.0f, 90.0f, 120.0f, 150.0f}};
    }
}

VowelFormantDatabase::FormantData VowelFormantDatabase::interpolate(Vowel v1, Vowel v2, float t) {
    t = std::clamp(t, 0.0f, 1.0f);

    FormantData d1 = getFormants(v1);
    FormantData d2 = getFormants(v2);

    FormantData result;
    for (size_t i = 0; i < 4; ++i) {
        result.frequencies[i] = d1.frequencies[i] * (1.0f - t) + d2.frequencies[i] * t;
        result.bandwidths[i] = d1.bandwidths[i] * (1.0f - t) + d2.bandwidths[i] * t;
    }

    return result;
}

VowelFormantDatabase::FormantData VowelFormantDatabase::getFormantsForPitch(int midiPitch) {
    // Approximate vowel selection based on pitch
    // Lower pitches tend to use more open vowels, higher pitches use closer vowels

    if (midiPitch < 60) {  // Below middle C
        return getFormants(Vowel::AH);  // Open vowel
    } else if (midiPitch < 72) {  // C4 to C5
        return getFormants(Vowel::EH);  // Mid-open
    } else if (midiPitch < 84) {  // C5 to C6
        return getFormants(Vowel::IH);  // Mid-close
    } else {  // Above C6
        return getFormants(Vowel::EE);  // Close vowel
    }
}

VowelFormantDatabase::FormantData VowelFormantDatabase::interpolateFormants(
    const FormantData& formants1,
    const FormantData& formants2,
    float t)
{
    t = std::clamp(t, 0.0f, 1.0f);

    FormantData result;
    for (size_t i = 0; i < 4; ++i) {
        result.frequencies[i] = formants1.frequencies[i] * (1.0f - t) + formants2.frequencies[i] * t;
        result.bandwidths[i] = formants1.bandwidths[i] * (1.0f - t) + formants2.bandwidths[i] * t;
    }

    return result;
}

//==============================================================================
// ADSREnvelope Implementation
//==============================================================================

ADSREnvelope::ADSREnvelope() {
    calculateRates();
}

void ADSREnvelope::prepare(double sampleRate) {
    sampleRate_ = sampleRate;
    calculateRates();
}

void ADSREnvelope::calculateRates() {
    if (sampleRate_ > 0.0) {
        attackRate_ = (attackTime > 0.0f) ? 1.0f / (static_cast<float>(sampleRate_) * attackTime) : 1.0f;
        decayRate_ = (decayTime > 0.0f) ? (1.0f - sustainLevel) / (static_cast<float>(sampleRate_) * decayTime) : 1.0f;
        releaseRate_ = (releaseTime > 0.0f) ? sustainLevel / (static_cast<float>(sampleRate_) * releaseTime) : 1.0f;
    }
}

void ADSREnvelope::trigger() {
    state_ = State::Attack;
    currentValue_ = 0.0f;
}

void ADSREnvelope::release() {
    if (state_ != State::Idle) {
        state_ = State::Release;
    }
}

float ADSREnvelope::getValue() {
    switch (state_) {
        case State::Idle:
            return 0.0f;

        case State::Attack:
            currentValue_ += attackRate_;
            if (currentValue_ >= 1.0f) {
                currentValue_ = 1.0f;
                state_ = State::Decay;
            }
            return currentValue_;

        case State::Decay:
            currentValue_ -= decayRate_;
            if (currentValue_ <= sustainLevel) {
                currentValue_ = sustainLevel;
                state_ = State::Sustain;
            }
            return currentValue_;

        case State::Sustain:
            return sustainLevel;

        case State::Release:
            currentValue_ -= releaseRate_;
            if (currentValue_ <= 0.0f) {
                currentValue_ = 0.0f;
                state_ = State::Idle;
            }
            return currentValue_;
    }

    return 0.0f;
}

//==============================================================================
// PortamentoGenerator Implementation
//==============================================================================

PortamentoGenerator::PortamentoGenerator() {
    calculateRate();
}

void PortamentoGenerator::prepare(double sampleRate) {
    sampleRate_ = sampleRate;
    calculateRate();
}

void PortamentoGenerator::setTargetPitch(float pitch) {
    targetPitch_ = pitch;
    if (!enabled_) {
        currentPitch_ = targetPitch_;
    }
}

float PortamentoGenerator::getCurrentPitch() {
    if (!enabled_) {
        return targetPitch_;
    }

    // Linear interpolation (can be replaced with exponential for smoother sound)
    float diff = targetPitch_ - currentPitch_;

    if (std::abs(diff) < 0.1f) {
        currentPitch_ = targetPitch_;
    } else {
        if (diff > 0.0f) {
            currentPitch_ += portamentoRate_;
            if (currentPitch_ > targetPitch_) currentPitch_ = targetPitch_;
        } else {
            currentPitch_ -= portamentoRate_;
            if (currentPitch_ < targetPitch_) currentPitch_ = targetPitch_;
        }
    }

    return currentPitch_;
}

void PortamentoGenerator::calculateRate() {
    if (sampleRate_ > 0.0 && portamentoTime_ > 0.0f) {
        // Calculate rate to reach target in portamentoTime seconds
        // This is approximate - actual pitch transitions use exponential curves
        portamentoRate_ = 1000.0f / (static_cast<float>(sampleRate_) * portamentoTime_);
    } else {
        portamentoRate_ = 10000.0f; // Very fast (instantaneous)
    }
}

} // namespace kelly
