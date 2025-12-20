#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "plugin_processor.h"

namespace kelly {

/**
 * @brief Plugin editor UI with parameter controls
 */
class PluginEditor : public juce::AudioProcessorEditor,
                     public juce::Timer {
public:
    explicit PluginEditor(PluginProcessor&);
    ~PluginEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    
    void timerCallback() override;

private:
    PluginProcessor& processor;
    
    // UI Components
    juce::Label titleLabel_;
    juce::Label emotionLabel_;
    juce::ComboBox emotionComboBox_;
    
    juce::Label valenceLabel_;
    juce::Slider valenceSlider_;
    juce::Label valenceValueLabel_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> valenceAttachment_;
    
    juce::Label arousalLabel_;
    juce::Slider arousalSlider_;
    juce::Label arousalValueLabel_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> arousalAttachment_;
    
    juce::Label intensityLabel_;
    juce::Slider intensitySlider_;
    juce::Label intensityValueLabel_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> intensityAttachment_;
    
    juce::Label barsLabel_;
    juce::Slider barsSlider_;
    juce::Label barsValueLabel_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> barsAttachment_;
    
    juce::ToggleButton cloudToggle_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> cloudAttachment_;
    
    juce::Label generationRateLabel_;
    juce::Slider generationRateSlider_;
    juce::Label generationRateValueLabel_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> generationRateAttachment_;
    
    juce::Label statusLabel_;
    
    // Helper methods
    void setupSlider(juce::Slider& slider, juce::Label& label, 
                     const juce::String& parameterID,
                     const juce::String& labelText,
                     juce::NormalisableRange<float> range);
    void updateValueLabels();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

} // namespace kelly
