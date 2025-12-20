/**
 * StampProcessor.h - Plugin 011: "The Stamp"
 *
 * Profile: 'Stutter/Repeater' with Rubber Stamp UI
 * Priority: LOW
 *
 * Features:
 * - Audio buffer capture and repeat
 * - Adjustable repeat rate (sync to tempo)
 * - Pitch shifting on repeats
 * - Decay/feedback control
 * - Reverse mode
 * - Half-speed/double-speed modes
 * - Rubber stamp pattern visualization
 */

#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <array>
#include <vector>
#include <cmath>

namespace iDAW {

/**
 * Configuration for The Stamp
 */
struct StampConfig {
    static constexpr int MAX_BUFFER_SIZE = 96000;  // 2 seconds @ 48kHz
    static constexpr float MIN_REPEAT_RATE = 0.5f;  // Hz
    static constexpr float MAX_REPEAT_RATE = 32.0f; // Hz
    static constexpr float MAX_PITCH_SHIFT = 12.0f; // Semitones
    static constexpr float MAX_DECAY = 0.99f;
};

/**
 * Repeat mode
 */
enum class RepeatMode {
    NORMAL,      // Forward playback
    REVERSE,     // Backward playback
    PING_PONG,   // Alternating forward/backward
    RANDOM       // Random slice playback
};

/**
 * Sync note divisions
 */
enum class StampSync {
    FREE,           // Free running (Hz)
    WHOLE,          // 1/1
    HALF,           // 1/2
    QUARTER,        // 1/4
    EIGHTH,         // 1/8
    SIXTEENTH,      // 1/16
    THIRTY_SECOND,  // 1/32
    TRIPLET_QUARTER,// 1/4 triplet
    TRIPLET_EIGHTH  // 1/8 triplet
};

/**
 * Visual state for Rubber stamp shader
 */
struct RubberStampVisualState {
    float stampProgress = 0.0f;      // Current stamp animation (0-1)
    float inkIntensity = 1.0f;       // How much ink (decay level)
    float patternRotation = 0.0f;    // Pattern rotation
    int stampCount = 0;              // Number of stamps on page
    float pressureLevel = 0.0f;      // Stamp pressure visualization
    bool isStamping = false;         // Currently stamping
};

/**
 * StampProcessor - Stutter/Repeater Effect
 *
 * Algorithm:
 * 1. Continuously capture audio into circular buffer
 * 2. On trigger, lock capture and start playback
 * 3. Repeat captured slice at specified rate
 * 4. Apply pitch shift if enabled
 * 5. Apply decay to each repeat
 * 6. Mix with dry signal
 */
class StampProcessor : public juce::AudioProcessor {
public:
    StampProcessor();
    ~StampProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "The Stamp"; }
    bool acceptsMidi() const override { return true; }  // For MIDI trigger
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override;

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Parameters
    void setRepeatRate(float rateHz);
    float getRepeatRate() const { return m_repeatRateHz.load(); }

    void setSliceLength(float lengthMs);
    float getSliceLength() const { return m_sliceLengthMs.load(); }

    void setDecay(float decay);
    float getDecay() const { return m_decay.load(); }

    void setPitchShift(float semitones);
    float getPitchShift() const { return m_pitchShift.load(); }

    void setRepeatMode(RepeatMode mode);
    RepeatMode getRepeatMode() const { return m_repeatMode; }

    void setSync(StampSync sync);
    StampSync getSync() const { return m_sync; }

    void setMix(float mix);
    float getMix() const { return m_mix.load(); }

    void setActive(bool active);
    bool isActive() const { return m_isActive.load(); }

    // Manual trigger
    void trigger();
    void release();

    // Host sync
    void updateFromPlayHead(juce::AudioPlayHead* playHead);

    // Ghost Hands
    void applyAISuggestion(const juce::String& suggestion);

    // Visual state
    RubberStampVisualState getVisualState() const;

private:
    void captureToBuffer(const float* input, int channel, int numSamples);
    float readFromBuffer(int channel, float position);
    float getSyncedRepeatRate(double bpm);

    // Capture buffer (stereo)
    std::array<std::vector<float>, 2> m_captureBuffer;
    std::array<int, 2> m_writeIndex;
    int m_capturedLength = 0;
    bool m_capturing = true;

    // Playback state
    std::array<float, 2> m_playbackPosition;
    float m_repeatPhase = 0.0f;
    int m_repeatCount = 0;
    bool m_playingForward = true;
    float m_currentGain = 1.0f;

    // Random mode state
    std::mt19937 m_rng;
    float m_randomSliceStart = 0.0f;

    // Parameters
    std::atomic<float> m_repeatRateHz{4.0f};
    std::atomic<float> m_sliceLengthMs{125.0f};  // 1/8 note @ 120 BPM
    std::atomic<float> m_decay{0.8f};
    std::atomic<float> m_pitchShift{0.0f};
    std::atomic<float> m_mix{1.0f};
    std::atomic<bool> m_isActive{false};
    RepeatMode m_repeatMode = RepeatMode::NORMAL;
    StampSync m_sync = StampSync::FREE;

    // Host tempo
    double m_hostBPM = 120.0;

    // Visual state
    RubberStampVisualState m_visualState;
    mutable std::mutex m_visualMutex;

    double m_sampleRate = 44100.0;
    bool m_prepared = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StampProcessor)
};

} // namespace iDAW
