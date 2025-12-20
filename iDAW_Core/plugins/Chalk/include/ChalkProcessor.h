/**
 * ChalkProcessor.h - Plugin 009: "The Chalk"
 *
 * Profile: 'Lo-fi/Bitcrusher' with Dusty Texture UI
 * Priority: LOW
 *
 * Features:
 * - Bit depth reduction (1-16 bits)
 * - Sample rate reduction
 * - Analog-style noise (hiss, hum)
 * - Vinyl crackle simulation
 * - Low-pass filter for warmth
 * - Wow/flutter modulation
 * - Dusty chalkboard visualization
 */

#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <array>
#include <random>
#include <cmath>

namespace iDAW {

/**
 * Configuration for The Chalk
 */
struct ChalkConfig {
    static constexpr int MIN_BIT_DEPTH = 1;
    static constexpr int MAX_BIT_DEPTH = 16;
    static constexpr float MIN_SAMPLE_RATE_REDUCTION = 1.0f;    // No reduction
    static constexpr float MAX_SAMPLE_RATE_REDUCTION = 100.0f;  // Divide by 100
    static constexpr float MAX_NOISE_LEVEL = 0.5f;
    static constexpr float MAX_CRACKLE = 1.0f;
    static constexpr float MAX_WOW_DEPTH = 20.0f;   // cents
    static constexpr float MAX_FLUTTER_RATE = 8.0f; // Hz
};

/**
 * Lo-fi character presets
 */
enum class LofiPreset {
    CLEAN,          // Minimal degradation
    CASSETTE,       // Tape hiss, slight wow
    VINYL,          // Crackle, warmth
    TELEPHONE,      // Bandlimited, noise
    RADIO,          // AM radio character
    CHIPTUNE,       // 8-bit style
    CUSTOM
};

/**
 * Visual state for Dusty shader
 */
struct DustyVisualState {
    float chalkDensity = 0.5f;      // Amount of chalk dust
    float smearAmount = 0.0f;       // Smearing/degradation level
    float particleCount = 0.0f;     // Dust particles in air
    float eraserProgress = 0.0f;    // If erasing/cleaning
    std::array<float, 8> frequencyBins;  // Spectral display
    float noiseFloor = 0.0f;        // Visible noise level
};

/**
 * ChalkProcessor - Lo-fi/Bitcrusher Effect
 *
 * Algorithm:
 * 1. Apply sample rate reduction (sample and hold)
 * 2. Apply bit depth reduction (quantization)
 * 3. Add analog noise (hiss)
 * 4. Add vinyl crackle
 * 5. Apply low-pass filter for warmth
 * 6. Apply wow/flutter pitch modulation
 */
class ChalkProcessor : public juce::AudioProcessor {
public:
    ChalkProcessor();
    ~ChalkProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "The Chalk"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Parameters
    void setBitDepth(int bits);
    int getBitDepth() const { return m_bitDepth.load(); }

    void setSampleRateReduction(float factor);
    float getSampleRateReduction() const { return m_srReduction.load(); }

    void setNoiseLevel(float level);
    float getNoiseLevel() const { return m_noiseLevel.load(); }

    void setCrackleLevel(float level);
    float getCrackleLevel() const { return m_crackleLevel.load(); }

    void setWarmth(float warmth);
    float getWarmth() const { return m_warmth.load(); }

    void setWowDepth(float depthCents);
    float getWowDepth() const { return m_wowDepth.load(); }

    void setFlutterRate(float rateHz);
    float getFlutterRate() const { return m_flutterRate.load(); }

    void setMix(float mix);
    float getMix() const { return m_mix.load(); }

    void applyPreset(LofiPreset preset);

    // Ghost Hands
    void applyAISuggestion(const juce::String& suggestion);

    // Visual state
    DustyVisualState getVisualState() const;

private:
    float quantize(float sample, int bits);
    float generateNoise();
    float generateCrackle();
    float processLowpass(float sample, int channel);
    void updateLFO();

    // Sample and hold state (for SR reduction)
    std::array<float, 2> m_holdSample;
    std::array<float, 2> m_holdCounter;

    // Low-pass filter state (one-pole per channel)
    std::array<float, 2> m_lpfState;

    // Wow/flutter LFO
    float m_wowPhase = 0.0f;
    float m_flutterPhase = 0.0f;

    // Random generators
    std::mt19937 m_rng;
    std::uniform_real_distribution<float> m_noiseDist{-1.0f, 1.0f};
    std::uniform_real_distribution<float> m_crackleDist{0.0f, 1.0f};

    // Parameters
    std::atomic<int> m_bitDepth{16};
    std::atomic<float> m_srReduction{1.0f};
    std::atomic<float> m_noiseLevel{0.0f};
    std::atomic<float> m_crackleLevel{0.0f};
    std::atomic<float> m_warmth{0.0f};      // 0 = bright, 1 = warm
    std::atomic<float> m_wowDepth{0.0f};
    std::atomic<float> m_flutterRate{0.5f};
    std::atomic<float> m_mix{1.0f};

    // Visual state
    DustyVisualState m_visualState;
    mutable std::mutex m_visualMutex;

    double m_sampleRate = 44100.0;
    bool m_prepared = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChalkProcessor)
};

} // namespace iDAW
