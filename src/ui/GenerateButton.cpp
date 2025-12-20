#include "GenerateButton.h"
#include "KellyLookAndFeel.h"

namespace kelly {

GenerateButton::GenerateButton() : juce::TextButton("GENERATE") {
    setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF4A90E2));
    setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xFF357ABD));
    setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    setColour(juce::TextButton::textColourOnId, juce::Colours::white);
}

void GenerateButton::paintButton(juce::Graphics& g, bool shouldDrawButtonAsHighlighted,
                                  bool shouldDrawButtonAsDown) {
    auto bounds = getLocalBounds().toFloat();

    // Draw button background with rounded corners
    juce::Colour baseColour = findColour(juce::TextButton::buttonColourId);
    if (shouldDrawButtonAsDown || isDown()) {
        baseColour = findColour(juce::TextButton::buttonOnColourId);
    } else if (shouldDrawButtonAsHighlighted || isOver()) {
        baseColour = baseColour.brighter(0.2f);
    }

    // Show visual feedback when animating (generating)
    if (isAnimating_) {
        // Pulse effect - slightly brighter when animating
        baseColour = baseColour.brighter(0.15f);
    }

    g.setColour(baseColour);
    g.fillRoundedRectangle(bounds, 6.0f);

    // Draw border
    g.setColour(baseColour.darker(0.3f));
    g.drawRoundedRectangle(bounds, 6.0f, 2.0f);

    // Draw text
    juce::Colour textColour = findColour(juce::TextButton::textColourOffId);
    if (isAnimating_) {
        // Slightly dimmed text when animating
        textColour = textColour.withAlpha(0.8f);
    }
    g.setColour(textColour);
    g.setFont(16.0f);
    juce::String buttonText = isAnimating_ ? "Generating..." : getButtonText();
    g.drawText(buttonText, bounds, juce::Justification::centred);
}

void GenerateButton::startGenerateAnimation() {
    isAnimating_ = true;
    animationProgress_ = 0.0f;
    repaint();  // Trigger repaint to show animation state
}

void GenerateButton::stopGenerateAnimation() {
    isAnimating_ = false;
    animationProgress_ = 0.0f;
    repaint();  // Trigger repaint to show normal state
}

} // namespace kelly
