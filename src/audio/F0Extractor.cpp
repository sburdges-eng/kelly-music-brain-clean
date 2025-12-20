#include "audio/F0Extractor.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace midikompanion {
namespace audio {

F0Extractor::F0Extractor()
    : minFreq_(50.0f)
    , maxFreq_(2000.0f)
    , threshold_(0.1f)
    , confidence_(0.0f)
{
}

float F0Extractor::extractPitch(const juce::AudioBuffer<float>& audio, double sampleRate) {
    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) {
        confidence_ = 0.0f;
        return 0.0f;
    }

    // Use first channel for mono analysis
    const float* channelData = audio.getReadPointer(0);
    return extractPitch(channelData, audio.getNumSamples(), sampleRate);
}

float F0Extractor::extractPitch(const float* samples, int numSamples, double sampleRate) {
    if (numSamples < 1024) {
        // Need at least 1024 samples for reliable pitch detection
        confidence_ = 0.0f;
        return 0.0f;
    }

    // Calculate lag bounds from frequency range
    int minLag = static_cast<int>(sampleRate / maxFreq_);
    int maxLag = static_cast<int>(sampleRate / minFreq_);
    maxLag = std::min(maxLag, numSamples / 2);  // Don't exceed half the buffer

    if (maxLag <= minLag) {
        confidence_ = 0.0f;
        return 0.0f;
    }

    // Run YIN algorithm
    int period = yinAlgorithm(samples, numSamples, sampleRate);

    if (period > 0) {
        float f0 = static_cast<float>(sampleRate) / static_cast<float>(period);

        // Clamp to valid frequency range
        f0 = std::clamp(f0, minFreq_, maxFreq_);

        // Calculate confidence based on how well the period matches
        // (simplified - in production, use CMNDF value)
        confidence_ = 1.0f - (threshold_ * 2.0f);  // Higher threshold = lower confidence
        confidence_ = std::clamp(confidence_, 0.0f, 1.0f);

        return f0;
    }

    confidence_ = 0.0f;
    return 0.0f;
}

std::vector<float> F0Extractor::extractPitchTrajectory(const juce::AudioBuffer<float>& audio,
                                                        double sampleRate,
                                                        int frameSize,
                                                        int hopSize) {
    std::vector<float> trajectory;

    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) {
        return trajectory;
    }

    const float* channelData = audio.getReadPointer(0);
    int numSamples = audio.getNumSamples();

    // Process in overlapping frames
    for (int start = 0; start + frameSize <= numSamples; start += hopSize) {
        float f0 = extractPitch(channelData + start, frameSize, sampleRate);
        trajectory.push_back(f0);
    }

    return trajectory;
}

int F0Extractor::yinAlgorithm(const float* samples, int numSamples, double sampleRate) {
    // Calculate lag bounds
    int minLag = static_cast<int>(sampleRate / maxFreq_);
    int maxLag = static_cast<int>(sampleRate / minFreq_);
    maxLag = std::min(maxLag, numSamples / 2);

    if (maxLag <= minLag) {
        return 0;
    }

    // Resize buffers if needed
    if (static_cast<int>(diffBuffer_.size()) < maxLag + 1) {
        diffBuffer_.resize(maxLag + 1);
    }
    if (static_cast<int>(cmndfBuffer_.size()) < maxLag + 1) {
        cmndfBuffer_.resize(maxLag + 1);
    }

    // Calculate difference function
    calculateDifferenceFunction(samples, numSamples, maxLag, diffBuffer_);

    // Calculate cumulative mean normalized difference function
    calculateCMNDF(diffBuffer_, numSamples, cmndfBuffer_);

    // Find minimum below threshold
    int period = findMinimum(cmndfBuffer_, minLag, maxLag, threshold_);

    if (period > 0) {
        // Refine with parabolic interpolation
        float refinedPeriod = parabolicInterpolation(cmndfBuffer_, period);
        return static_cast<int>(std::round(refinedPeriod));
    }

    return 0;
}

void F0Extractor::calculateDifferenceFunction(const float* samples, int numSamples, int maxLag, std::vector<float>& diff) {
    // Initialize difference function
    std::fill(diff.begin(), diff.begin() + maxLag + 1, 0.0f);

    // Calculate autocorrelation-like difference function
    for (int lag = 0; lag <= maxLag; ++lag) {
        float sum = 0.0f;
        for (int j = 0; j < numSamples - maxLag; ++j) {
            float delta = samples[j] - samples[j + lag];
            sum += delta * delta;
        }
        diff[lag] = sum;
    }
}

void F0Extractor::calculateCMNDF(const std::vector<float>& diff, int numSamples, std::vector<float>& cmndf) {
    // Initialize CMNDF
    std::fill(cmndf.begin(), cmndf.begin() + diff.size(), 0.0f);

    if (diff.empty()) {
        return;
    }

    // First value is always 1.0 (by definition)
    cmndf[0] = 1.0f;

    // Calculate cumulative mean
    float cumulativeSum = 0.0f;
    for (size_t lag = 1; lag < diff.size(); ++lag) {
        cumulativeSum += diff[lag];

        if (cumulativeSum > 0.0f) {
            cmndf[lag] = diff[lag] * static_cast<float>(lag) / cumulativeSum;
        } else {
            cmndf[lag] = 1.0f;  // Default if division by zero
        }
    }
}

int F0Extractor::findMinimum(const std::vector<float>& cmndf, int minLag, int maxLag, float threshold) {
    // Search for first minimum below threshold
    for (int lag = minLag; lag <= maxLag && lag < static_cast<int>(cmndf.size()); ++lag) {
        if (cmndf[lag] < threshold) {
            // Found a candidate - check if it's a local minimum
            if (lag > minLag && lag < maxLag) {
                if (cmndf[lag] < cmndf[lag - 1] && cmndf[lag] < cmndf[lag + 1]) {
                    return lag;  // Local minimum found
                }
            } else if (lag == minLag && cmndf[lag] < cmndf[lag + 1]) {
                return lag;  // Edge case: minimum at start
            } else if (lag == maxLag && cmndf[lag] < cmndf[lag - 1]) {
                return lag;  // Edge case: minimum at end
            }
        }
    }

    // If no value below threshold, find absolute minimum
    float minValue = std::numeric_limits<float>::max();
    int bestLag = minLag;
    for (int lag = minLag; lag <= maxLag && lag < static_cast<int>(cmndf.size()); ++lag) {
        if (cmndf[lag] < minValue) {
            minValue = cmndf[lag];
            bestLag = lag;
        }
    }

    // Return minimum if it's reasonable (below 2x threshold)
    if (minValue < threshold * 2.0f) {
        return bestLag;
    }

    return 0;  // No reliable pitch detected
}

float F0Extractor::parabolicInterpolation(const std::vector<float>& cmndf, int lag) {
    if (lag <= 0 || lag >= static_cast<int>(cmndf.size()) - 1) {
        return static_cast<float>(lag);
    }

    // Parabolic interpolation around the minimum
    float y0 = cmndf[lag - 1];
    float y1 = cmndf[lag];
    float y2 = cmndf[lag + 1];

    // Calculate offset from integer lag
    float offset = (y2 - y0) / (2.0f * (2.0f * y1 - y0 - y2));

    // Clamp offset to reasonable range
    offset = std::clamp(offset, -0.5f, 0.5f);

    return static_cast<float>(lag) + offset;
}

} // namespace audio
} // namespace midikompanion
