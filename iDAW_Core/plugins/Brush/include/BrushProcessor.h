/**
 * BrushProcessor.h - Plugin 010: "The Brush"
 *
 * Profile: 'Modulated Filter' with Paint Stroke UI
 * Priority: LOW
 *
 * Features:
 * - State variable filter (LP/HP/BP/Notch)
 * - LFO modulation with multiple waveforms
 * - Envelope follower modulation
 * - Resonance control
 * - Filter FM (audio-rate modulation)
 * - Paint stroke visualization
 */

#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <array>
#include <cmath>

namespace iDAW {

/**
 * Configuration for The Brush
 */
struct BrushConfig {
    static constexpr float MIN_CUTOFF_HZ = 20.0f;
    static constexpr float MAX_CUTOFF_HZ = 20000.0f;
    static constexpr float MIN_RESONANCE = 0.0f;
    static constexpr float MAX_RESONANCE = 1.0f;  // Self-oscillation at 1.0
    static constexpr float MAX_LFO_RATE_HZ = 20.0f;
    static constexpr float MAX_LFO_DEPTH = 1.0f;
    static constexpr float MAX_ENV_DEPTH = 1.0f;
};

/**
 * Filter types
 */
enum class FilterType {
    LOWPASS,
    HIGHPASS,
    BANDPASS,
    NOTCH
};

/**
 * LFO waveforms
 */
enum class LFOWaveform {
    SINE,
    TRIANGLE,
    SAW_UP,
    SAW_DOWN,
    SQUARE,
    RANDOM_HOLD  // Sample and hold
};

/**
 * Visual state for Brushstroke shader
 */
struct BrushstrokeVisualState {
    float strokePosition = 0.5f;     // Position along frequency spectrum
    float strokeWidth = 0.2f;        // Filter bandwidth
    float strokeIntensity = 0.0f;    // Resonance glow
    float strokeAngle = 0.0f;        // Paint direction (LFO phase)
    float wetness = 0.5f;            // Blend/drip effect
    float bristleSpread = 0.0f;      // Resonance spread
    std::array<float, 2> trailPos;   // Trailing paint positions
};

/**
 * BrushProcessor - Modulated Filter Effect
 *
 * Algorithm (State Variable Filter):
 * 1. Calculate base cutoff from parameter
 * 2. Add LFO modulation to cutoff
 * 3. Add envelope follower modulation
 * 4. Process through SVF (low/high/band/notch)
 * 5. Apply resonance
 * 6. Mix with dry signal
 */
class BrushProcessor : public juce::AudioProcessor {
public:
    BrushProcessor();
    ~BrushProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "The Brush"; }
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
    void setCutoff(float cutoffHz);
    float getCutoff() const { return m_cutoffHz.load(); }

    void setResonance(float resonance);
    float getResonance() const { return m_resonance.load(); }

    void setFilterType(FilterType type);
    FilterType getFilterType() const { return m_filterType; }

    void setLFORate(float rateHz);
    float getLFORate() const { return m_lfoRateHz.load(); }

    void setLFODepth(float depth);
    float getLFODepth() const { return m_lfoDepth.load(); }

    void setLFOWaveform(LFOWaveform waveform);
    LFOWaveform getLFOWaveform() const { return m_lfoWaveform; }

    void setEnvDepth(float depth);
    float getEnvDepth() const { return m_envDepth.load(); }

    void setEnvAttack(float attackMs);
    float getEnvAttack() const { return m_envAttackMs.load(); }

    void setEnvRelease(float releaseMs);
    float getEnvRelease() const { return m_envReleaseMs.load(); }

    void setMix(float mix);
    float getMix() const { return m_mix.load(); }

    // Ghost Hands
    void applyAISuggestion(const juce::String& suggestion);

    // Visual state
    BrushstrokeVisualState getVisualState() const;

private:
    // SVF processing
    struct SVFState {
        float low = 0.0f;
        float band = 0.0f;
        float high = 0.0f;
        float notch = 0.0f;
    };

    void processSVF(float input, float cutoff, float resonance, int channel);
    float getLFOValue();
    float getEnvelopeValue(float inputLevel);
    void updateLFO();

    // Filter state per channel
    std::array<SVFState, 2> m_svfState;

    // LFO state
    float m_lfoPhase = 0.0f;
    float m_lastRandomValue = 0.0f;
    float m_randomHoldCounter = 0.0f;

    // Envelope follower state
    float m_envelopeState = 0.0f;
    float m_envAttackCoeff = 0.0f;
    float m_envReleaseCoeff = 0.0f;

    // Parameters
    std::atomic<float> m_cutoffHz{1000.0f};
    std::atomic<float> m_resonance{0.0f};
    std::atomic<float> m_lfoRateHz{1.0f};
    std::atomic<float> m_lfoDepth{0.0f};
    std::atomic<float> m_envDepth{0.0f};
    std::atomic<float> m_envAttackMs{10.0f};
    std::atomic<float> m_envReleaseMs{100.0f};
    std::atomic<float> m_mix{1.0f};
    FilterType m_filterType = FilterType::LOWPASS;
    LFOWaveform m_lfoWaveform = LFOWaveform::SINE;

    // Metering
    std::atomic<float> m_currentCutoff{1000.0f};

    // Visual state
    BrushstrokeVisualState m_visualState;
    mutable std::mutex m_visualMutex;

    double m_sampleRate = 44100.0;
    bool m_prepared = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BrushProcessor)
};

} // namespace iDAW
