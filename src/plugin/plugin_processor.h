#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "../ml/ai_inference.h"
#include <atomic>
#include <future>

namespace kelly {

/**
 * @brief Parameter IDs for the plugin
 */
namespace ParameterID {
    constexpr const char* emotionTag = "emotionTag";
    constexpr const char* valence = "valence";
    constexpr const char* arousal = "arousal";
    constexpr const char* intensity = "intensity";
    constexpr const char* bars = "bars";
    constexpr const char* enableCloud = "enableCloud";
    constexpr const char* generationRate = "generationRate";
}

/**
 * @brief Main audio processor for Kelly Emotion Processor plugin
 * 
 * This processor generates MIDI based on emotional parameters using AI inference.
 * It supports both on-device (ONNX) and cloud-based generation.
 */
class PluginProcessor : public juce::AudioProcessor,
                         public juce::AudioProcessorValueTreeState::Listener {
public:
    PluginProcessor();
    ~PluginProcessor() override = default;

    //==============================================================================
    // AudioProcessor Interface
    //==============================================================================
    
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Kelly Emotion Processor"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return true; }
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;

    //==============================================================================
    // Parameter Access
    //==============================================================================
    
    /**
     * @brief Get the parameter tree state for UI binding
     */
    juce::AudioProcessorValueTreeState& getParameters() { return parameters_; }
    const juce::AudioProcessorValueTreeState& getParameters() const { return parameters_; }

    /**
     * @brief Get current emotion request based on parameters
     */
    ml::EmotionRequest getEmotionRequest() const;

    /**
     * @brief Check if cloud generation is enabled
     */
    bool isCloudEnabled() const;

    /**
     * @brief Get generation rate (how often to generate, in blocks)
     */
    int getGenerationRate() const;

private:
    //==============================================================================
    // Parameter Tree Setup
    //==============================================================================
    
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    
    //==============================================================================
    // Parameter Change Callback (AudioProcessorValueTreeState::Listener)
    //==============================================================================
    
    void parameterChanged(const juce::String& parameterID, float newValue) override;

    //==============================================================================
    // Member Variables
    //==============================================================================
    
    juce::AudioProcessorValueTreeState parameters_;
    
    double bpm_{120.0};
    double sampleRate_{44100.0};
    int blockSize_{512};

    ml::AIInferenceEngine aiEngine_;
    ml::AIInferenceEngine cloudEngine_;
    
    // Cached emotion request (updated from parameters)
    mutable ml::EmotionRequest emotionRequest_{"calm", 0.1f, 0.2f, 0.4f, 120, 1};
    mutable juce::CriticalSection emotionRequestLock_;

    std::future<std::vector<ml::MidiEvent>> cloudFuture_;
    std::atomic<bool> cloudInFlight_{false};
    std::atomic<int> cloudCooldown_{0};
    std::atomic<int> generationCounter_{0};
    juce::MidiBuffer cloudMidi_;
    juce::CriticalSection cloudLock_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

} // namespace kelly
