/**
 * StencilProcessor.h - Plugin 008: "The Stencil"
 *
 * Profile: 'Sidechain/Ducking' with Cutout UI
 * Priority: LOW
 *
 * Features:
 * - Sidechain envelope follower
 * - Ducking/pumping effect
 * - Adjustable attack/release
 * - Depth and mix controls
 * - Optional internal sidechain (self-ducking rhythm)
 * - Cutout pattern visualization
 */

#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <array>
#include <cmath>

namespace iDAW {

/**
 * Configuration for The Stencil
 */
struct StencilConfig {
    static constexpr float MIN_ATTACK_MS = 0.1f;
    static constexpr float MAX_ATTACK_MS = 100.0f;
    static constexpr float MIN_RELEASE_MS = 10.0f;
    static constexpr float MAX_RELEASE_MS = 1000.0f;
    static constexpr float MIN_DEPTH = 0.0f;
    static constexpr float MAX_DEPTH = 1.0f;
    static constexpr float MIN_THRESHOLD_DB = -60.0f;
    static constexpr float MAX_THRESHOLD_DB = 0.0f;
    static constexpr float RMS_WINDOW_MS = 5.0f;
};

/**
 * Sidechain source type
 */
enum class SidechainSource {
    EXTERNAL,      // External sidechain input
    INTERNAL,      // Self-ducking with internal LFO
    MIDI_TRIGGER   // MIDI note triggers ducking
};

/**
 * Visual state for Cutout shader
 */
struct CutoutVisualState {
    float cutoutDepth = 0.0f;      // How deep the cutout is (0-1)
    float cutoutProgress = 0.0f;   // Animation progress
    float patternRotation = 0.0f;  // Pattern rotation angle
    float edgeSharpness = 0.5f;    // Sharp vs feathered edges
    float inputLevel = 0.0f;       // Sidechain input level
    float outputLevel = 0.0f;      // Main output level
};

/**
 * StencilProcessor - Sidechain Ducking Effect
 *
 * Algorithm:
 * 1. Analyze sidechain input (external, internal LFO, or MIDI)
 * 2. Calculate envelope with attack/release
 * 3. Apply depth-controlled ducking to main signal
 * 4. Mix dry/wet
 */
class StencilProcessor : public juce::AudioProcessor {
public:
    StencilProcessor();
    ~StencilProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "The Stencil"; }
    bool acceptsMidi() const override { return true; }  // For MIDI trigger mode
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
    void setAttack(float attackMs);
    float getAttack() const { return m_attackMs.load(); }

    void setRelease(float releaseMs);
    float getRelease() const { return m_releaseMs.load(); }

    void setDepth(float depth);
    float getDepth() const { return m_depth.load(); }

    void setThreshold(float thresholdDb);
    float getThreshold() const { return m_thresholdDb.load(); }

    void setMix(float mix);
    float getMix() const { return m_mix.load(); }

    void setSource(SidechainSource source);
    SidechainSource getSource() const { return m_source; }

    void setInternalRate(float rateHz);
    float getInternalRate() const { return m_internalRateHz.load(); }

    // Ghost Hands
    void applyAISuggestion(const juce::String& suggestion);

    // Visual state
    CutoutVisualState getVisualState() const;

private:
    float calculateEnvelope(float inputLevel);
    float processInternalLFO();
    void updateCoefficients();

    static float linearToDb(float linear) {
        return 20.0f * std::log10(std::max(linear, 1e-10f));
    }

    static float dbToLinear(float db) {
        return std::pow(10.0f, db / 20.0f);
    }

    // Envelope state
    float m_envelopeState = 0.0f;
    float m_attackCoeff = 0.0f;
    float m_releaseCoeff = 0.0f;

    // Internal LFO
    float m_lfoPhase = 0.0f;

    // MIDI trigger state
    bool m_midiTriggered = false;

    // Parameters
    std::atomic<float> m_attackMs{10.0f};
    std::atomic<float> m_releaseMs{100.0f};
    std::atomic<float> m_depth{1.0f};
    std::atomic<float> m_thresholdDb{-20.0f};
    std::atomic<float> m_mix{1.0f};
    std::atomic<float> m_internalRateHz{4.0f};  // 4 Hz = 1/16 @ 120 BPM
    SidechainSource m_source = SidechainSource::EXTERNAL;

    // Metering
    std::atomic<float> m_currentDucking{0.0f};
    std::atomic<float> m_inputLevel{-100.0f};
    std::atomic<float> m_outputLevel{-100.0f};

    // Visual state
    CutoutVisualState m_visualState;
    mutable std::mutex m_visualMutex;

    double m_sampleRate = 44100.0;
    bool m_prepared = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StencilProcessor)
};

} // namespace iDAW
