#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

/**
 * Generate Button - Custom button with animations for generating MIDI.
 * 
 * Currently a standard JUCE button in PluginEditor.
 * This will have custom animations in Phase 3.
 */
class GenerateButton : public juce::TextButton {
public:
    GenerateButton();
    ~GenerateButton() override = default;
    
    void paintButton(juce::Graphics& g, bool shouldDrawButtonAsHighlighted,
                     bool shouldDrawButtonAsDown) override;
    
    // Animation support (Phase 3)
    void startGenerateAnimation();
    void stopGenerateAnimation();
    
private:
    bool isAnimating_ = false;
    float animationProgress_ = 0.0f;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GenerateButton)
};

} // namespace kelly

