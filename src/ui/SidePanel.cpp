#include "SidePanel.h"
#include "KellyLookAndFeel.h"

namespace kelly {

SidePanel::SidePanel(Side side) : side_(side) {
    setOpaque(true);
    
    juce::String labelText = (side == Side::SideA) ? "Side A" : "Side B";
    label_.setText(labelText, juce::dontSendNotification);
    label_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(label_);
}

void SidePanel::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds();
    
    // Draw panel background
    juce::Colour bgColour = (side_ == Side::SideA) 
        ? KellyLookAndFeel::primaryColor.withAlpha(0.2f)
        : KellyLookAndFeel::secondaryColor.withAlpha(0.2f);
    
    g.fillAll(bgColour);
    
    // Draw border
    g.setColour(bgColour.darker(0.2f));
    g.drawRect(bounds, 1);
    
    // Label is drawn by the label component
}

void SidePanel::resized() {
    // Layout label
    auto labelArea = getLocalBounds().removeFromTop(25);
    label_.setBounds(labelArea);
    
    // Layout other child components
    auto contentArea = getLocalBounds();
    for (auto* child : getChildren()) {
        if (child != &label_) {
            child->setBounds(contentArea);
        }
    }
}

SideA SidePanel::getSideAState() const {
    if (side_ == Side::SideA) {
        return SideA{input_.getText().toStdString(), 
                     static_cast<float>(intensity_.getValue()), 
                     std::nullopt};
    }
    return SideA{"", 0.5f, std::nullopt};
}

SideB SidePanel::getSideBState() const {
    if (side_ == Side::SideB) {
        return SideB{input_.getText().toStdString(), 
                     static_cast<float>(intensity_.getValue()), 
                     std::nullopt};
    }
    return SideB{"", 0.5f, std::nullopt};
}

} // namespace kelly
