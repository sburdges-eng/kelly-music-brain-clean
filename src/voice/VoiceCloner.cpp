#include "voice/VoiceCloner.h"
#include <juce_core/juce_core.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <complex>
#include <fstream>

namespace kelly {

VoiceCloner::VoiceCloner() {
}

VoiceCloner::FormantProfile VoiceCloner::analyzeVoice(
    const std::vector<float>& audioSamples,
    double sampleRate,
    float pitchEstimate)
{
    FormantProfile profile;

    if (audioSamples.empty() || sampleRate <= 0.0) {
        return profile;  // Return default profile
    }

    // Use a windowed segment for analysis (middle portion for stability)
    size_t startSample = audioSamples.size() / 4;
    size_t endSample = (audioSamples.size() * 3) / 4;
    size_t windowSize = std::min(endSample - startSample, ANALYSIS_WINDOW_SIZE);

    if (windowSize < LPC_ORDER * 2) {
        return profile;  // Too short for analysis
    }

    std::vector<float> window(audioSamples.begin() + startSample,
                              audioSamples.begin() + startSample + windowSize);

    // Apply window function (Hanning window)
    for (size_t i = 0; i < window.size(); ++i) {
        float hann = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (window.size() - 1)));
        window[i] *= hann;
    }

    // Pre-emphasize
    auto preEmphasized = preEmphasize(window);

    // Compute LPC coefficients
    auto lpcCoeffs = computeLPC(preEmphasized, LPC_ORDER);

    // Find formants
    auto formants = findFormantRoots(lpcCoeffs, sampleRate);

    // Extract top 4 formants (by frequency)
    std::sort(formants.begin(), formants.end(),
              [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
                  return a.first < b.first;
              });

    for (size_t i = 0; i < std::min(4UL, formants.size()); ++i) {
        profile.frequencies[i] = formants[i].first;
        profile.bandwidths[i] = formants[i].second;
    }

    // Calculate formant shift relative to default male voice
    float avgF1 = profile.frequencies[0];
    float defaultF1 = 500.0f;  // Default male F1
    profile.formantShift = avgF1 / defaultF1;

    // Estimate glottal shape from spectral characteristics
    // Higher harmonics relative to fundamental indicate sharper glottal pulse
    float pitch = pitchEstimate > 0.0f ? pitchEstimate : estimatePitch(preEmphasized, sampleRate);
    if (pitch > 0.0f) {
        // Analyze spectral envelope (simplified)
        profile.glottalShape = 0.5f;  // Default, could be refined with spectral analysis
    }

    return profile;
}

std::pair<std::array<float, 4>, std::array<float, 4>> VoiceCloner::extractFormants(
    const std::vector<float>& audioSamples,
    double sampleRate)
{
    auto profile = analyzeVoice(audioSamples, sampleRate);
    return {profile.frequencies, profile.bandwidths};
}

bool VoiceCloner::saveProfile(const FormantProfile& profile, const std::string& filePath) {
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();

    // Create frequencies array
    auto* freqArray = new juce::Array<juce::var>();
    for (float f : profile.frequencies) {
        freqArray->add(f);
    }
    obj->setProperty("frequencies", juce::var(*freqArray));
    delete freqArray;

    // Create bandwidths array
    auto* bwArray = new juce::Array<juce::var>();
    for (float bw : profile.bandwidths) {
        bwArray->add(bw);
    }
    obj->setProperty("bandwidths", juce::var(*bwArray));
    delete bwArray;

    obj->setProperty("formantShift", profile.formantShift);
    obj->setProperty("glottalShape", profile.glottalShape);
    obj->setProperty("brightness", profile.brightness);

    juce::var root(obj);
    juce::String jsonString = juce::JSON::toString(root);

    juce::File file(filePath);
    return file.replaceWithText(jsonString);
}

VoiceCloner::FormantProfile VoiceCloner::loadProfile(const std::string& filePath) {
    FormantProfile profile;

    juce::File file(filePath);
    if (!file.existsAsFile()) {
        return profile;  // Return default
    }

    juce::String jsonString = file.loadFileAsString();
    auto var = juce::JSON::parse(jsonString);

    if (auto* obj = var.getDynamicObject()) {
        // Load frequencies
        if (auto* freqVar = obj->getProperty("frequencies").getArray()) {
            for (int i = 0; i < std::min(4, freqVar->size()); ++i) {
                profile.frequencies[i] = static_cast<float>(freqVar->getUnchecked(i));
            }
        }

        // Load bandwidths
        if (auto* bwVar = obj->getProperty("bandwidths").getArray()) {
            for (int i = 0; i < std::min(4, bwVar->size()); ++i) {
                profile.bandwidths[i] = static_cast<float>(bwVar->getUnchecked(i));
            }
        }

        profile.formantShift = static_cast<float>(obj->getProperty("formantShift", 1.0));
        profile.glottalShape = static_cast<float>(obj->getProperty("glottalShape", 0.5));
        profile.brightness = static_cast<float>(obj->getProperty("brightness", 0.5));
    }

    return profile;
}

VoiceCloner::FormantProfile VoiceCloner::blendProfiles(
    const FormantProfile& profile1,
    const FormantProfile& profile2,
    float blendFactor)
{
    blendFactor = std::clamp(blendFactor, 0.0f, 1.0f);
    float invBlend = 1.0f - blendFactor;

    FormantProfile blended;
    for (size_t i = 0; i < 4; ++i) {
        blended.frequencies[i] = profile1.frequencies[i] * invBlend + profile2.frequencies[i] * blendFactor;
        blended.bandwidths[i] = profile1.bandwidths[i] * invBlend + profile2.bandwidths[i] * blendFactor;
    }

    blended.formantShift = profile1.formantShift * invBlend + profile2.formantShift * blendFactor;
    blended.glottalShape = profile1.glottalShape * invBlend + profile2.glottalShape * blendFactor;
    blended.brightness = profile1.brightness * invBlend + profile2.brightness * blendFactor;

    return blended;
}

std::vector<float> VoiceCloner::computeLPC(const std::vector<float>& samples, int order) {
    // Autocorrelation method for LPC
    size_t N = samples.size();
    if (N < static_cast<size_t>(order * 2)) {
        return std::vector<float>(order + 1, 0.0f);
    }

    // Compute autocorrelation
    std::vector<float> autocorr(order + 1, 0.0f);
    for (int k = 0; k <= order; ++k) {
        for (size_t n = 0; n < N - k; ++n) {
            autocorr[k] += samples[n] * samples[n + k];
        }
    }

    // Levinson-Durbin recursion
    std::vector<float> a(order + 1, 0.0f);
    a[0] = 1.0f;

    float error = autocorr[0];
    if (error <= 0.0f) {
        return std::vector<float>(order + 1, 0.0f);
    }

    for (int i = 1; i <= order; ++i) {
        float sum = 0.0f;
        for (int j = 1; j < i; ++j) {
            sum += a[j] * autocorr[i - j];
        }

        float k = (autocorr[i] - sum) / error;
        a[i] = k;

        for (int j = 1; j < i; ++j) {
            a[j] -= k * a[i - j];
        }

        error *= (1.0f - k * k);
        if (error <= 0.0f) {
            break;
        }
    }

    return a;
}

std::vector<std::pair<float, float>> VoiceCloner::findFormantRoots(
    const std::vector<float>& lpcCoeffs,
    double sampleRate)
{
    std::vector<std::pair<float, float>> formants;

    if (lpcCoeffs.size() < 2) {
        return formants;
    }

    // Find roots of LPC polynomial (using simplified method)
    // For each complex root, compute frequency and bandwidth

    // Use polynomial root finding (simplified - in production, use a proper root finder)
    // For now, use a simplified approach: find peaks in LPC spectrum

    size_t fftSize = 4096;
    std::vector<std::complex<float>> fftInput(fftSize, 0.0f);

    // Evaluate LPC polynomial on unit circle (frequency response)
    for (size_t k = 0; k < fftSize / 2; ++k) {
        float omega = 2.0f * static_cast<float>(M_PI) * k / fftSize;
        std::complex<float> sum(1.0f, 0.0f);
        for (size_t i = 1; i < lpcCoeffs.size(); ++i) {
            sum += lpcCoeffs[i] * std::exp(std::complex<float>(0.0f, -static_cast<float>(i) * omega));
        }
        float magnitude = 1.0f / std::abs(sum);
        fftInput[k] = magnitude;
    }

    // Find peaks (formants) in the magnitude spectrum
    // Simple peak detection
    for (size_t k = 2; k < fftSize / 2 - 2; ++k) {
        float mag = std::abs(fftInput[k]);
        if (mag > std::abs(fftInput[k-1]) && mag > std::abs(fftInput[k+1]) &&
            mag > std::abs(fftInput[k-2]) && mag > std::abs(fftInput[k+2])) {

            float frequency = static_cast<float>(k) * static_cast<float>(sampleRate) / fftSize;

            // Estimate bandwidth (simplified: half-power bandwidth)
            float peakMag = mag;
            float halfPower = peakMag * 0.707f;

            // Find -3dB points
            size_t lowerK = k;
            while (lowerK > 0 && std::abs(fftInput[lowerK]) > halfPower) {
                lowerK--;
            }

            size_t upperK = k;
            while (upperK < fftSize / 2 - 1 && std::abs(fftInput[upperK]) > halfPower) {
                upperK++;
            }

            float bandwidth = static_cast<float>(upperK - lowerK) * static_cast<float>(sampleRate) / fftSize;
            bandwidth = std::max(bandwidth, 20.0f);  // Minimum bandwidth

            // Filter out unrealistic formants
            if (frequency > 100.0f && frequency < sampleRate / 2.0f && bandwidth < 500.0f) {
                formants.push_back({frequency, bandwidth});
            }
        }
    }

    // Sort by frequency and limit to reasonable number
    std::sort(formants.begin(), formants.end(),
              [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
                  return a.first < b.first;
              });

    // Limit to top formants (within vocal range)
    formants.erase(
        std::remove_if(formants.begin(), formants.end(),
                       [](const std::pair<float, float>& f) { return f.first > 5000.0f; }),
        formants.end()
    );

    return formants;
}

float VoiceCloner::estimatePitch(const std::vector<float>& samples, double sampleRate) {
    // Simple autocorrelation pitch detection
    if (samples.size() < 512) {
        return 0.0f;
    }

    size_t minPeriod = static_cast<size_t>(sampleRate / 800.0);  // Max 800 Hz
    size_t maxPeriod = static_cast<size_t>(sampleRate / 50.0);   // Min 50 Hz

    float maxCorr = 0.0f;
    size_t bestPeriod = 0;

    for (size_t period = minPeriod; period < std::min(maxPeriod, samples.size() / 2); ++period) {
        float corr = 0.0f;
        size_t count = 0;
        for (size_t i = 0; i < samples.size() - period; ++i) {
            corr += samples[i] * samples[i + period];
            count++;
        }
        if (count > 0) {
            corr /= count;
            if (corr > maxCorr) {
                maxCorr = corr;
                bestPeriod = period;
            }
        }
    }

    if (bestPeriod > 0) {
        return static_cast<float>(sampleRate) / bestPeriod;
    }

    return 0.0f;
}

std::vector<float> VoiceCloner::preEmphasize(const std::vector<float>& samples) {
    // First-order high-pass filter: y[n] = x[n] - 0.97 * x[n-1]
    std::vector<float> output(samples.size());
    float preEmphCoeff = 0.97f;

    output[0] = samples[0];
    for (size_t i = 1; i < samples.size(); ++i) {
        output[i] = samples[i] - preEmphCoeff * samples[i-1];
    }

    return output;
}

} // namespace kelly
