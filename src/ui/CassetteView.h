#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

/**
 * Cassette View - Main visual container for the cassette tape aesthetic.
 * 
 * Full v2.0 visual design with:
 * - Animated tape reels
 * - Realistic cassette body with texture
 * - Label area with custom text
 * - Tape window showing content
 */
class CassetteView : public juce::Component,
                     public juce::Timer {
public:
    CassetteView();
    ~CassetteView() override;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    
    /** Set the content component to display inside the cassette */
    void setContentComponent(juce::Component* component);
    
    /** Set label text */
    void setLabelText(const juce::String& text);
    
    /** Start/stop tape animation */
    void setTapeAnimating(bool animating);
    
    /** Set tape position (0.0 to 1.0) */
    void setTapePosition(float position);
    
private:
    juce::Component* contentComponent_ = nullptr;
    juce::String labelText_ = "KELLY MIDI COMPANION";
    bool isAnimating_ = false;
    float tapePosition_ = 0.0f;
    float animationPhase_ = 0.0f;
    
    void drawCassetteBody(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawTapeReels(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawTapeWindow(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawLabel(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CassetteView)
};

} // namespace kelly

