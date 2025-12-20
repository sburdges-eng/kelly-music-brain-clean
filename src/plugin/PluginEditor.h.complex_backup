#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "plugin/PluginProcessor.h"

namespace kelly {

/**
 * Kelly MIDI Companion Editor
 * 
 * Therapeutic MIDI generation through emotion mapping.
 * Uses valence, arousal, and intensity sliders controlled by emotion presets.
 */
class PluginEditor : public juce::AudioProcessorEditor,
                     public juce::Timer {
public:
    explicit PluginEditor(PluginProcessor& processor);
    ~PluginEditor() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    
    // Resizable interface for Logic Pro compatibility
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    void mouseUp(const juce::MouseEvent& e) override;
    
private:
    PluginProcessor& processor_;
    
    // Category and Style selectors (like Logic Session Player)
    juce::ComboBox categorySelector_;
    juce::Label categoryLabel_;
    juce::ComboBox styleSelector_;
    juce::Label styleLabel_;
    
    // Preset selector
    juce::ComboBox presetSelector_;
    juce::Label presetLabel_;
    
    // Emotion sliders
    juce::Slider valenceSlider_;
    juce::Label valenceLabel_;
    juce::AudioProcessorValueTreeState::SliderAttachment valenceAttachment_;
    
    juce::Slider arousalSlider_;
    juce::Label arousalLabel_;
    juce::AudioProcessorValueTreeState::SliderAttachment arousalAttachment_;
    
    juce::Slider intensitySlider_;
    juce::Label intensityLabel_;
    juce::AudioProcessorValueTreeState::SliderAttachment intensityAttachment_;
    
    // Fine-tuning controls (like Logic Session Player)
    juce::Slider complexitySlider_;
    juce::Label complexityLabel_;
    juce::AudioProcessorValueTreeState::SliderAttachment complexityAttachment_;
    
    juce::Slider feelSlider_;
    juce::Label feelLabel_;
    juce::AudioProcessorValueTreeState::SliderAttachment feelAttachment_;
    
    juce::Slider dynamicsSlider_;
    juce::Label dynamicsLabel_;
    juce::AudioProcessorValueTreeState::SliderAttachment dynamicsAttachment_;
    
    juce::Slider barsSlider_;
    juce::Label barsLabel_;
    juce::AudioProcessorValueTreeState::SliderAttachment barsAttachment_;
    
    // Control buttons
    juce::TextButton generateButton_;
    juce::TextButton playStopButton_;
    juce::TextButton exportButton_;
    
    // Status display
    juce::Label statusLabel_;
    juce::Label emotionDisplay_;
    
    // Emotion wheel (v2.0)
    std::unique_ptr<class EmotionWheel> emotionWheel_;
    juce::ToggleButton showWheelButton_;
    
    // Cassette view (v2.0) - optional visual wrapper
    std::unique_ptr<class CassetteView> cassetteView_;
    juce::ToggleButton showCassetteViewButton_;
    
    // Resize handle
    bool isResizing_ = false;
    juce::Point<int> resizeStartPos_;
    juce::Point<int> resizeStartSize_;
    static constexpr int MIN_WIDTH = 400;
    static constexpr int MIN_HEIGHT = 550;
    static constexpr int RESIZE_HANDLE_HEIGHT = 8;
    
    // Category/Style data structures
    struct Category {
        juce::String name;
        std::vector<juce::String> styles;
        std::vector<std::tuple<float, float, float>> styleParams; // (valence, arousal, intensity) for each style
    };
    
    static std::vector<Category> getCategories();
    
    void onGenerate();
    void onPlayStop();
    void onExport();
    void onPresetChanged();
    void onCategoryChanged();
    void onStyleChanged();
    void updateEmotionDisplay();
    void updatePlayStopButton();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

} // namespace kelly
