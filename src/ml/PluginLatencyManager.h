#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <atomic>

namespace kelly {

/**
 * PluginLatencyManager - Manages plugin latency for ML inference.
 *
 * Tracks and reports total latency from:
 * - ML model inference time
 * - Lookahead buffer size
 * - Processing delays
 */
class PluginLatencyManager {
public:
    PluginLatencyManager(juce::AudioProcessor& processor)
        : audioProcessor_(processor)
        , currentLatency_(0)
        , mlLatency_(0)
        , lookaheadLatency_(0)
    {}

    /**
     * Set ML model latency in samples.
     * @param samples Latency in samples
     */
    void setMLLatency(int samples) {
        mlLatency_.store(samples);
        updateTotalLatency();
    }

    /**
     * Set lookahead buffer latency in samples.
     * @param samples Latency in samples
     */
    void setLookaheadLatency(int samples) {
        lookaheadLatency_.store(samples);
        updateTotalLatency();
    }

    /**
     * Get total latency in samples.
     */
    int getTotalLatency() const {
        return currentLatency_.load();
    }

    /**
     * Convert milliseconds to samples.
     * @param ms Milliseconds
     * @param sampleRate Sample rate in Hz
     * @return Samples
     */
    static int msToSamples(float ms, double sampleRate) {
        return static_cast<int>(ms * sampleRate / 1000.0);
    }

    /**
     * Convert samples to milliseconds.
     * @param samples Number of samples
     * @param sampleRate Sample rate in Hz
     * @return Milliseconds
     */
    static float samplesToMs(int samples, double sampleRate) {
        return static_cast<float>(samples * 1000.0 / sampleRate);
    }

    /**
     * Get ML latency in samples.
     */
    int getMLLatency() const {
        return mlLatency_.load();
    }

    /**
     * Get lookahead latency in samples.
     */
    int getLookaheadLatency() const {
        return lookaheadLatency_.load();
    }

private:
    /**
     * Update total latency and notify processor.
     */
    void updateTotalLatency() {
        int total = mlLatency_.load() + lookaheadLatency_.load();
        currentLatency_.store(total);
        audioProcessor_.setLatencySamples(total);
    }

    juce::AudioProcessor& audioProcessor_;
    std::atomic<int> currentLatency_;
    std::atomic<int> mlLatency_;
    std::atomic<int> lookaheadLatency_;
};

} // namespace kelly
