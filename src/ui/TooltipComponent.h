#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

/**
 * Tooltip Component - Provides helpful tooltips for UI elements
 * 
 * Based on UX research: Tooltips reduce cognitive load and improve discoverability
 */
class TooltipComponent : public juce::Component {
public:
    TooltipComponent();
    ~TooltipComponent() override = default;
    
    static void showTooltip(juce::Component* target, const juce::String& text, int timeoutMs = 3000);
    static void hideTooltip();
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    
private:
    juce::String tooltipText_;
    juce::Point<int> targetPosition_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TooltipComponent)
};

/**
 * Helper class to add tooltips to components
 */
class TooltipHelper {
public:
    static void setTooltip(juce::Component* component, const juce::String& tooltip) {
        if (component) {
            component->setHelpText(tooltip);
        }
    }
};

} // namespace kelly
