#include "CassetteView.h"
#include "KellyLookAndFeel.h"
#include <cmath>

namespace kelly {

CassetteView::CassetteView() {
    setOpaque(true);
    startTimer(16);  // ~60 FPS animation
}

CassetteView::~CassetteView() {
    stopTimer();
}

void CassetteView::timerCallback() {
    if (isAnimating_) {
        animationPhase_ += 0.02f;
        if (animationPhase_ > juce::MathConstants<float>::twoPi) {
            animationPhase_ -= juce::MathConstants<float>::twoPi;
        }
        repaint();
    }
}

void CassetteView::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds();
    
    // Draw cassette body with full v2.0 design
    drawCassetteBody(g, bounds);
    drawTapeReels(g, bounds);
    drawTapeWindow(g, bounds);
    drawLabel(g, bounds);
}

void CassetteView::drawCassetteBody(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    // Main cassette body with gradient
    juce::ColourGradient gradient(
        KellyLookAndFeel::surfaceColor.brighter(0.1f),
        bounds.getX(), bounds.getY(),
        KellyLookAndFeel::surfaceColor.darker(0.2f),
        bounds.getX(), bounds.getBottom(),
        false
    );
    g.setGradientFill(gradient);
    g.fillRoundedRectangle(bounds.toFloat(), 8.0f);
    
    // Outer border
    g.setColour(juce::Colours::black);
    g.drawRoundedRectangle(bounds.toFloat(), 8.0f, 2.0f);
    
    // Inner shadow for depth
    g.setColour(juce::Colours::black.withAlpha(0.3f));
    g.drawRoundedRectangle(bounds.reduced(1, 1).toFloat(), 7.0f, 1.0f);
}

void CassetteView::drawTapeReels(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    int reelSize = 35;
    int holeY = bounds.getCentreY();
    int leftHoleX = bounds.getWidth() / 4;
    int rightHoleX = 3 * bounds.getWidth() / 4;
    
    // Draw reel hubs with rotation animation
    float rotation = isAnimating_ ? animationPhase_ : 0.0f;
    
    // Left reel
    auto leftReel = juce::Rectangle<int>(leftHoleX - reelSize/2, holeY - reelSize/2, reelSize, reelSize);
    g.setColour(juce::Colours::darkgrey);
    g.fillEllipse(leftReel.toFloat());
    g.setColour(juce::Colours::black);
    g.drawEllipse(leftReel.toFloat(), 2.0f);
    
    // Reel spokes (rotating)
    g.saveState();
    g.addTransform(juce::AffineTransform::rotation(rotation, leftHoleX, holeY));
    g.setColour(juce::Colours::lightgrey);
    for (int i = 0; i < 6; ++i) {
        float angle = i * juce::MathConstants<float>::pi / 3.0f;
        float x1 = leftHoleX + std::cos(angle) * 8.0f;
        float y1 = holeY + std::sin(angle) * 8.0f;
        float x2 = leftHoleX + std::cos(angle) * 15.0f;
        float y2 = holeY + std::sin(angle) * 15.0f;
        g.drawLine(x1, y1, x2, y2, 1.5f);
    }
    g.restoreState();
    
    // Right reel (same but counter-rotating)
    auto rightReel = juce::Rectangle<int>(rightHoleX - reelSize/2, holeY - reelSize/2, reelSize, reelSize);
    g.setColour(juce::Colours::darkgrey);
    g.fillEllipse(rightReel.toFloat());
    g.setColour(juce::Colours::black);
    g.drawEllipse(rightReel.toFloat(), 2.0f);
    
    g.saveState();
    g.addTransform(juce::AffineTransform::rotation(-rotation, rightHoleX, holeY));
    g.setColour(juce::Colours::lightgrey);
    for (int i = 0; i < 6; ++i) {
        float angle = i * juce::MathConstants<float>::pi / 3.0f;
        float x1 = rightHoleX + std::cos(angle) * 8.0f;
        float y1 = holeY + std::sin(angle) * 8.0f;
        float x2 = rightHoleX + std::cos(angle) * 15.0f;
        float y2 = holeY + std::sin(angle) * 15.0f;
        g.drawLine(x1, y1, x2, y2, 1.5f);
    }
    g.restoreState();
}

void CassetteView::drawTapeWindow(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    // Draw tape window (transparent area showing content)
    auto windowArea = bounds.reduced(20, 50);
    windowArea.removeFromTop(30);  // Space for label
    windowArea.removeFromBottom(20);
    
    // Window frame
    g.setColour(juce::Colours::black);
    g.drawRoundedRectangle(windowArea.toFloat(), 4.0f, 2.0f);
    
    // Window glass effect
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.fillRoundedRectangle(windowArea.reduced(1, 1).toFloat(), 3.0f);
}

void CassetteView::drawLabel(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    // Draw label area
    auto labelArea = bounds.reduced(15, 25);
    labelArea.setHeight(25);
    labelArea.translate(0, 5);
    
    g.setColour(KellyLookAndFeel::textPrimary);
    g.fillRoundedRectangle(labelArea.toFloat(), 3.0f);
    
    // Label border
    g.setColour(juce::Colours::black);
    g.drawRoundedRectangle(labelArea.toFloat(), 3.0f, 1.0f);
    
    // Label text
    g.setColour(juce::Colours::black);
    g.setFont(12.0f);
    g.drawText(labelText_, labelArea, juce::Justification::centred);
}

void CassetteView::resized() {
    // Layout child components in tape window area
    if (contentComponent_) {
        auto bounds = getLocalBounds();
        auto contentArea = bounds.reduced(20, 50);
        contentArea.removeFromTop(55);  // Space for label and top margin
        contentArea.removeFromBottom(20);
        contentComponent_->setBounds(contentArea);
    }
}

void CassetteView::setContentComponent(juce::Component* component) {
    if (contentComponent_) {
        removeChildComponent(contentComponent_);
    }
    contentComponent_ = component;
    if (contentComponent_) {
        addAndMakeVisible(contentComponent_);
        resized();
    }
}

void CassetteView::setLabelText(const juce::String& text) {
    labelText_ = text;
    repaint();
}

void CassetteView::setTapeAnimating(bool animating) {
    isAnimating_ = animating;
}

void CassetteView::setTapePosition(float position) {
    tapePosition_ = juce::jlimit(0.0f, 1.0f, position);
    repaint();
}

} // namespace kelly
