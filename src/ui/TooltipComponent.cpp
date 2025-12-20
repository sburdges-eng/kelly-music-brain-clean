#include "TooltipComponent.h"
#include "KellyLookAndFeel.h"

namespace kelly {

TooltipComponent::TooltipComponent() {
    setOpaque(false);
    setAlwaysOnTop(true);
    setInterceptsMouseClicks(false, false);
}

void TooltipComponent::showTooltip(juce::Component* target, const juce::String& text, int timeoutMs) {
    // JUCE's built-in tooltip system handles this
    if (target) {
        target->setHelpText(text);
    }
}

void TooltipComponent::hideTooltip() {
    // Handled by JUCE's tooltip system
}

void TooltipComponent::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();
    
    // Modern tooltip background
    g.setColour(KellyLookAndFeel::surfaceColor.withAlpha(0.95f));
    g.fillRoundedRectangle(bounds, 6.0f);
    
    // Border
    g.setColour(KellyLookAndFeel::borderColor);
    g.drawRoundedRectangle(bounds, 6.0f, 1.0f);
    
    // Text
    g.setColour(KellyLookAndFeel::textPrimary);
    g.setFont(juce::FontOptions(11.0f));
    g.drawText(tooltipText_, bounds.reduced(8.0f), juce::Justification::centredLeft, true);
}

void TooltipComponent::resized() {
    // Auto-sized based on text
}

} // namespace kelly
