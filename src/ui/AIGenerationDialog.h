#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

/**
 * AI Generation Dialog
 * 
 * Allows user to request AI-generated MIDI tracks with:
 * - Variable number of tracks
 * - Input from A-side (theory), B-side (emotion), or both
 * - Variability options for different outputs each time
 */
class AIGenerationDialog : public juce::Component {
public:
    struct AIGenerationRequest {
        int numTracks = 4;
        bool useSideA = true;      // Use music theory settings
        bool useSideB = true;      // Use emotion settings
        float variability = 0.5f;  // 0.0 = consistent, 1.0 = very variable
        int barsPerTrack = 8;
        bool blendSides = true;    // Blend A and B side inputs
        juce::String apiKey;       // LLM API key for AI generation
    };
    
    AIGenerationDialog();
    ~AIGenerationDialog() override = default;
    
    static AIGenerationRequest showDialog(juce::Component* parent);
    
    AIGenerationRequest getRequest() const;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
private:
    juce::Slider numTracksSlider_;
    juce::Label numTracksLabel_;
    
    juce::ToggleButton useSideAToggle_;
    juce::Label useSideALabel_;
    juce::ToggleButton useSideBToggle_;
    juce::Label useSideBLabel_;
    juce::ToggleButton blendSidesToggle_;
    juce::Label blendSidesLabel_;
    
    juce::Slider variabilitySlider_;
    juce::Label variabilityLabel_;
    
    juce::Slider barsPerTrackSlider_;
    juce::Label barsPerTrackLabel_;
    
    juce::TextEditor apiKeyEditor_;
    juce::Label apiKeyLabel_;
    juce::ToggleButton saveApiKeyToggle_;
    juce::Label saveApiKeyLabel_;
    
    juce::TextButton generateButton_;
    juce::TextButton cancelButton_;
    
    bool wasCancelled_ = true;
    int modalResult_ = 0;
    mutable AIGenerationRequest cachedRequest_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AIGenerationDialog)
};

} // namespace kelly
