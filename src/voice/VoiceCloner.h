#pragma once

#include "voice/VocoderEngine.h"
#include <vector>
#include <array>
#include <complex>
#include <string>

namespace kelly {

/**
 * VoiceCloner - Extract formant characteristics from voice samples
 *
 * Analyzes recorded voice samples to extract formant frequencies
 * and bandwidths, allowing the vocoder to clone a specific voice.
 *
 * Uses Linear Predictive Coding (LPC) analysis to extract formants.
 */
class VoiceCloner {
public:
    VoiceCloner();
    ~VoiceCloner() = default;

    /**
     * Analyze audio sample to extract formant characteristics
     * @param audioSamples Audio samples (mono, float, normalized to [-1, 1])
     * @param sampleRate Sample rate of audio (e.g., 44100.0)
     * @param pitchEstimate Optional pitch estimate in Hz (if provided, improves accuracy)
     * @return Formant data structure with F1-F4 frequencies and bandwidths
     */
    struct FormantProfile {
        std::array<float, 4> frequencies = {500.0f, 1500.0f, 2500.0f, 3300.0f};
        std::array<float, 4> bandwidths = {60.0f, 90.0f, 120.0f, 150.0f};
        float formantShift = 1.0f;  // Overall formant shift relative to default
        float glottalShape = 0.5f;  // Glottal pulse shape characteristic
        float brightness = 0.5f;    // Voice brightness
    };

    FormantProfile analyzeVoice(
        const std::vector<float>& audioSamples,
        double sampleRate,
        float pitchEstimate = 0.0f
    );

    /**
     * Extract formants from a vowel segment
     * @param audioSamples Audio samples of a vowel sound
     * @param sampleRate Sample rate
     * @return Formant frequencies and bandwidths
     */
    std::pair<std::array<float, 4>, std::array<float, 4>> extractFormants(
        const std::vector<float>& audioSamples,
        double sampleRate
    );

    /**
     * Save voice profile to file
     * @param profile Formant profile to save
     * @param filePath Path to save JSON file
     * @return true if saved successfully
     */
    bool saveProfile(const FormantProfile& profile, const std::string& filePath);

    /**
     * Load voice profile from file
     * @param filePath Path to JSON file
     * @return Formant profile, or default profile if loading fails
     */
    FormantProfile loadProfile(const std::string& filePath);

    /**
     * Blend two voice profiles
     * @param profile1 First profile
     * @param profile2 Second profile
     * @param blendFactor 0.0 = profile1, 1.0 = profile2
     * @return Blended profile
     */
    static FormantProfile blendProfiles(
        const FormantProfile& profile1,
        const FormantProfile& profile2,
        float blendFactor
    );

private:
    /**
     * Linear Predictive Coding (LPC) analysis
     * @param samples Audio samples
     * @param order LPC order (typically 10-16)
     * @return LPC coefficients
     */
    std::vector<float> computeLPC(
        const std::vector<float>& samples,
        int order
    );

    /**
     * Find roots of LPC polynomial (formant frequencies)
     * @param lpcCoeffs LPC coefficients
     * @param sampleRate Sample rate
     * @return Vector of formant frequencies and bandwidths
     */
    std::vector<std::pair<float, float>> findFormantRoots(
        const std::vector<float>& lpcCoeffs,
        double sampleRate
    );

    /**
     * Estimate pitch from audio samples
     * @param samples Audio samples
     * @param sampleRate Sample rate
     * @return Pitch estimate in Hz, or 0.0 if estimation fails
     */
    float estimatePitch(
        const std::vector<float>& samples,
        double sampleRate
    );

    /**
     * Pre-emphasize audio (high-pass filter to enhance formants)
     * @param samples Input samples
     * @return Pre-emphasized samples
     */
    std::vector<float> preEmphasize(const std::vector<float>& samples);

    // Window size for analysis (in samples)
    static constexpr size_t ANALYSIS_WINDOW_SIZE = 2048;

    // LPC order
    static constexpr int LPC_ORDER = 14;
};

} // namespace kelly
