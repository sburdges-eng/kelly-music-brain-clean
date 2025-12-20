#pragma once

#include <memory>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include "plugin_processor.h"

namespace kelly {

/**
 * @brief Plugin editor UI with parameter controls, project management, and MIDI export
 */
class PluginEditor : public juce::AudioProcessorEditor,
                     public juce::Timer,
                     public juce::MenuBarModel {
public:
    explicit PluginEditor(PluginProcessor&);
    ~PluginEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    
    void timerCallback() override;

    //==============================================================================
    // MenuBarModel
    //==============================================================================
    juce::StringArray getMenuBarNames() override;
    juce::PopupMenu getMenuForIndex(int topLevelMenuIndex, const juce::String& menuName) override;
    void menuItemSelected(int menuItemID, int topLevelMenuIndex) override;

private:
    //==============================================================================
    // Menu Command IDs
    //==============================================================================
    enum CommandIDs {
        cmdNewProject = 1,
        cmdOpenProject,
        cmdSaveProject,
        cmdSaveProjectAs,
        cmdExportMidi,
        cmdExportMidiAs
    };

    //==============================================================================
    // Project Management
    //==============================================================================
    void newProject();
    void openProject();
    void saveProject();
    void saveProjectAs();
    void exportMidi();
    void exportMidiWithOptions();
    bool checkUnsavedChanges();

    //==============================================================================
    // Member Variables
    //==============================================================================
    PluginProcessor& processor;
    
    // Menu bar
    std::unique_ptr<juce::MenuBarComponent> menuBar_;
    
    // UI Components
    juce::Label titleLabel_;
    juce::Label emotionLabel_;
    juce::ComboBox emotionComboBox_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> emotionAttachment_;
    
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
    
    // ML Mode toggle for A/B testing
    juce::ToggleButton mlModeToggle_;
    juce::Label mlStatusLabel_;
    
    juce::Label generationRateLabel_;
    juce::Slider generationRateSlider_;
    juce::Label generationRateValueLabel_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> generationRateAttachment_;
    
    // Action buttons
    juce::TextButton saveButton_;
    juce::TextButton exportButton_;
    
    juce::Label statusLabel_;
    juce::Label projectNameLabel_;
    
    // Helper methods
    void setupSlider(juce::Slider& slider, juce::Label& label, 
                     const juce::String& parameterID,
                     const juce::String& labelText,
                     juce::NormalisableRange<float> range);
    void updateValueLabels();
    void updateProjectLabel();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

} // namespace kelly
