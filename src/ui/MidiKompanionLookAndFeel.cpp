#include "KellyLookAndFeel.h"

namespace kelly {

// Research-based color palette optimized for productivity and creativity
// Based on studies: Blue (focus), Green (balance/creativity), Purple (imagination)

// Background colors - Dark gray instead of pure black for reduced eye strain
const juce::Colour KellyLookAndFeel::backgroundDark(0xFF121212);      // #121212 - Optimal dark mode background
const juce::Colour KellyLookAndFeel::backgroundLight(0xFF1E1E1E);     // #1E1E1E - Slightly lighter for depth
const juce::Colour KellyLookAndFeel::surfaceColor(0xFF2A2A2A);        // #2A2A2A - Elevated surfaces

// Primary colors based on research - Exact documented colors
const juce::Colour KellyLookAndFeel::primaryColor(0xFF3B82F6);        // Focus Blue #3B82F6 - Focus & Productivity (85% stress reduction)
const juce::Colour KellyLookAndFeel::secondaryColor(0xFFA855F7);      // Creative Purple #A855F7 - Creativity & Imagination (65% enhanced thinking)
const juce::Colour KellyLookAndFeel::accentColor(0xFF22C55E);        // Balance Green #22C55E - Balance & Creative Solutions (40% increase)

// Text colors - Off-white for reduced contrast strain
const juce::Colour KellyLookAndFeel::textPrimary(0xFFE8E8E8);         // #E8E8E8 - Off-white for readability
const juce::Colour KellyLookAndFeel::textSecondary(0xFFB8B8B8);       // #B8B8B8 - Secondary text

// UI elements
const juce::Colour KellyLookAndFeel::borderColor(0xFF404040);         // Subtle borders
const juce::Colour KellyLookAndFeel::successColor(0xFF22C55E);        // Balance Green #22C55E - Success states
const juce::Colour KellyLookAndFeel::warningColor(0xFFF59E0B);        // Amber - Warnings

// Glass aesthetic colors
const juce::Colour KellyLookAndFeel::glassBorder(0x20FFFFFF);          // Glass border with transparency
const juce::Colour KellyLookAndFeel::glassHighlight(0x08FFFFFF);      // Glass highlight
const juce::Colour KellyLookAndFeel::accentAlt(0xFFF472B6);           // Pink for Side B / secondary accent
const juce::Colour KellyLookAndFeel::accentTertiary(0xFF22D3EE);      // Cyan for tertiary accent

// Additional productivity-focused colors
static const juce::Colour focusBlue(0xFF3B82F6);      // Bright blue for focus areas
static const juce::Colour creativePurple(0xFFA855F7);  // Vibrant purple for creative zones
static const juce::Colour balanceGreen(0xFF22C55E);   // Fresh green for balance
static const juce::Colour energyCyan(0xFF06B6D4);     // Cyan for energy/action

KellyLookAndFeel::KellyLookAndFeel() {
    // Set research-optimized color scheme
    setColour(juce::ResizableWindow::backgroundColourId, backgroundDark);
    setColour(juce::TextButton::buttonColourId, primaryColor);
    setColour(juce::TextButton::textColourOffId, textPrimary);
    setColour(juce::TextButton::textColourOnId, textPrimary);
    setColour(juce::ComboBox::backgroundColourId, surfaceColor);
    setColour(juce::ComboBox::textColourId, textPrimary);
    setColour(juce::ComboBox::outlineColourId, borderColor);
    setColour(juce::Slider::backgroundColourId, surfaceColor);
    setColour(juce::Slider::thumbColourId, primaryColor);
    setColour(juce::Slider::trackColourId, borderColor);
    setColour(juce::Label::textColourId, textPrimary);
    setColour(juce::TextEditor::backgroundColourId, surfaceColor);
    setColour(juce::TextEditor::textColourId, textPrimary);
    setColour(juce::TextEditor::outlineColourId, borderColor);
    setColour(juce::ToggleButton::textColourId, textPrimary);
    setColour(juce::ToggleButton::tickColourId, primaryColor);
    setColour(juce::ToggleButton::tickDisabledColourId, textSecondary);
}

void KellyLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                            const juce::Colour& backgroundColour,
                                            bool shouldDrawButtonAsHighlighted,
                                            bool shouldDrawButtonAsDown) {
    auto bounds = button.getLocalBounds().toFloat().reduced(1.0f);
    auto cornerSize = 8.0f;
    
    juce::Colour bgColor = backgroundColour;
    
    if (shouldDrawButtonAsDown) {
        bgColor = primaryColor.darker(0.2f);
    }
    else if (shouldDrawButtonAsHighlighted) {
        bgColor = primaryColor.brighter(0.1f);
    }
    
    // Glass-style button with gradient
    if (button.getToggleState() || shouldDrawButtonAsDown) {
        juce::ColourGradient gradient(primaryColor, bounds.getX(), bounds.getY(),
                                       accentAlt, bounds.getRight(), bounds.getBottom(), true);
        g.setGradientFill(gradient);
    }
    else {
        g.setColour(surfaceColor.withAlpha(0.7f));  // Glass effect
    }
    
    g.fillRoundedRectangle(bounds, cornerSize);
    
    // Glass border
    if (shouldDrawButtonAsHighlighted || button.hasKeyboardFocus(false)) {
        g.setColour(primaryColor.withAlpha(0.5f));
        g.drawRoundedRectangle(bounds, cornerSize, 1.5f);
    }
    else {
        g.setColour(glassBorder);
        g.drawRoundedRectangle(bounds, cornerSize, 1.0f);
    }
    
    // Top highlight for glass effect
    juce::ColourGradient highlightGrad(glassHighlight, bounds.getX(), bounds.getY(),
                                        juce::Colours::transparentBlack, bounds.getX(), bounds.getY() + 20,
                                        false);
    g.setGradientFill(highlightGrad);
    g.fillRoundedRectangle(bounds.removeFromTop(20), cornerSize);
}

void KellyLookAndFeel::drawButtonText(juce::Graphics& g, juce::TextButton& button,
                                      bool shouldDrawButtonAsHighlighted,
                                      bool shouldDrawButtonAsDown) {
    auto bounds = button.getLocalBounds().toFloat();
    auto font = getTextButtonFont(button, button.getHeight());
    
    g.setFont(font);
    g.setColour(button.findColour(button.getToggleState() ? juce::TextButton::textColourOnId
                                                          : juce::TextButton::textColourOffId)
                .withMultipliedAlpha(button.isEnabled() ? 1.0f : 0.5f));
    
    g.drawText(button.getButtonText(), bounds, juce::Justification::centred, false);
}

void KellyLookAndFeel::drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                                        float sliderPos, float minSliderPos, float maxSliderPos,
                                        const juce::Slider::SliderStyle style, juce::Slider& slider) {
    if (style == juce::Slider::LinearHorizontal || style == juce::Slider::LinearVertical) {
        auto trackWidth = 6.0f;
        auto isHorizontal = style == juce::Slider::LinearHorizontal || style == juce::Slider::LinearBar;
        
        if (isHorizontal) {
            auto trackY = (float)y + (float)height * 0.5f - trackWidth * 0.5f;
            
            // Background track (glass)
            g.setColour(surfaceColor.withAlpha(0.6f));
            g.fillRoundedRectangle((float)x, trackY, (float)width, trackWidth, trackWidth * 0.5f);
            
            // Value track with gradient
            auto fillWidth = sliderPos - (float)x;
            if (fillWidth > 0) {
                juce::ColourGradient gradient(primaryColor, (float)x, trackY,
                                               accentAlt, sliderPos, trackY, false);
                g.setGradientFill(gradient);
                g.fillRoundedRectangle((float)x, trackY, fillWidth, trackWidth, trackWidth * 0.5f);
            }
            
            // Glass border
            g.setColour(glassBorder);
            g.drawRoundedRectangle((float)x, trackY, (float)width, trackWidth, trackWidth * 0.5f, 0.5f);
        }
        else {
            auto trackX = (float)x + (float)width * 0.5f - trackWidth * 0.5f;
            
            // Background track (glass)
            g.setColour(surfaceColor.withAlpha(0.6f));
            g.fillRoundedRectangle(trackX, (float)y, trackWidth, (float)height, trackWidth * 0.5f);
            
            // Value track
            auto fillHeight = (float)y + (float)height - sliderPos;
            if (fillHeight > 0) {
                juce::ColourGradient gradient(accentAlt, trackX, (float)y + (float)height,
                                               primaryColor, trackX, sliderPos, false);
                g.setGradientFill(gradient);
                g.fillRoundedRectangle(trackX, sliderPos, trackWidth, fillHeight, trackWidth * 0.5f);
            }
            
            // Glass border
            g.setColour(glassBorder);
            g.drawRoundedRectangle(trackX, (float)y, trackWidth, (float)height, trackWidth * 0.5f, 0.5f);
        }
        
        // Draw thumb
        drawLinearSliderThumb(g, x, y, width, height, sliderPos, minSliderPos, maxSliderPos, style, slider);
    } else {
        LookAndFeel_V4::drawLinearSlider(g, x, y, width, height, sliderPos, minSliderPos, maxSliderPos, style, slider);
    }
}

void KellyLookAndFeel::drawLinearSliderBackground(juce::Graphics& g, int x, int y, int width, int height,
                                                  float sliderPos, float minSliderPos, float maxSliderPos,
                                                  const juce::Slider::SliderStyle style, juce::Slider& slider) {
    auto trackBounds = slider.isHorizontal() 
        ? juce::Rectangle<float>(static_cast<float>(x), static_cast<float>(y + height * 0.4f),
                                 static_cast<float>(width), static_cast<float>(height * 0.2f))
        : juce::Rectangle<float>(static_cast<float>(x + width * 0.4f), static_cast<float>(y),
                                 static_cast<float>(width * 0.2f), static_cast<float>(height));
    
    drawModernSliderTrack(g, trackBounds, slider.findColour(juce::Slider::trackColourId));
}

void KellyLookAndFeel::drawLinearSliderThumb(juce::Graphics& g, int x, int y, int width, int height,
                                             float sliderPos, float minSliderPos, float maxSliderPos,
                                             const juce::Slider::SliderStyle style, juce::Slider& slider) {
    // Larger thumb for better visibility and easier interaction
    auto thumbSize = slider.isHorizontal() ? static_cast<float>(height) * 1.0f : static_cast<float>(width) * 1.0f;
    if (thumbSize < 16.0f) thumbSize = 16.0f;  // Minimum size for visibility
    auto thumbBounds = slider.isHorizontal()
        ? juce::Rectangle<float>(sliderPos - thumbSize * 0.5f, static_cast<float>(y) + (height - thumbSize) * 0.5f,
                                 thumbSize, thumbSize)
        : juce::Rectangle<float>(static_cast<float>(x) + (width - thumbSize) * 0.5f, sliderPos - thumbSize * 0.5f,
                                 thumbSize, thumbSize);
    
    bool isHighlighted = slider.isMouseOverOrDragging();
    
    // Color-code sliders based on function (productivity vs creativity)
    juce::Colour thumbColour = slider.findColour(juce::Slider::thumbColourId);
    
    // Use different colors for different slider types
    auto sliderName = slider.getName();
    if (sliderName.containsIgnoreCase("valence") || sliderName.containsIgnoreCase("arousal")) {
        thumbColour = creativePurple;  // Purple for emotion/creativity
    } else if (sliderName.containsIgnoreCase("complexity") || sliderName.containsIgnoreCase("dynamics")) {
        thumbColour = balanceGreen;     // Green for balance
    } else {
        thumbColour = focusBlue;       // Blue for focus/productivity
    }
    
    drawModernSliderThumb(g, thumbBounds, thumbColour, isHighlighted);
}

void KellyLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                                    int buttonX, int buttonY, int buttonW, int buttonH,
                                    juce::ComboBox& box) {
    auto bounds = juce::Rectangle<int>(0, 0, width, height).toFloat().reduced(1.0f);
    auto cornerSize = 6.0f;
    
    // Glass background
    g.setColour(surfaceColor.withAlpha(0.7f));
    g.fillRoundedRectangle(bounds, cornerSize);
    
    // Top highlight for glass effect
    juce::ColourGradient highlightGrad(glassHighlight, bounds.getX(), bounds.getY(),
                                        juce::Colours::transparentBlack, bounds.getX(), bounds.getY() + 30,
                                        false);
    g.setGradientFill(highlightGrad);
    g.fillRoundedRectangle(bounds.removeFromTop(30), cornerSize);
    
    // Border with focus state
    auto borderColour = box.hasKeyboardFocus(true) 
        ? primaryColor.withAlpha(0.8f)
        : glassBorder;
    g.setColour(borderColour);
    g.drawRoundedRectangle(bounds, cornerSize, 1.0f);
    
    // Draw text with better visibility
    auto textBounds = bounds.reduced(8.0f, 0.0f);
    textBounds.removeFromRight(static_cast<float>(buttonW));
    g.setColour(textPrimary);
    g.setFont(juce::FontOptions(14.0f).withStyle("Medium"));
    g.drawText(box.getText(), textBounds, juce::Justification::centredLeft, false);
    
    // Arrow with modern styling
    auto arrowZone = juce::Rectangle<float>(static_cast<float>(buttonX), static_cast<float>(buttonY),
                                             static_cast<float>(buttonW), static_cast<float>(buttonH));
    juce::Path arrow;
    arrow.addTriangle(arrowZone.getCentreX() - 4, arrowZone.getCentreY() - 2,
                      arrowZone.getCentreX() + 4, arrowZone.getCentreY() - 2,
                      arrowZone.getCentreX(), arrowZone.getCentreY() + 3);
    g.setColour(textSecondary);
    g.fillPath(arrow);
}

void KellyLookAndFeel::drawTextEditorOutline(juce::Graphics& g, int width, int height,
                                             juce::TextEditor& textEditor) {
    auto bounds = juce::Rectangle<int>(0, 0, width, height).toFloat().reduced(0.5f);
    auto cornerSize = 6.0f;
    
    if (textEditor.isEnabled()) {
        if (textEditor.hasKeyboardFocus(true)) {
            // Focus state with glass effect
            g.setColour(primaryColor.withAlpha(0.8f));
            g.drawRoundedRectangle(bounds, cornerSize, 1.5f);
            
            // Inner glow effect
            g.setColour(primaryColor.withAlpha(0.1f));
            g.fillRoundedRectangle(bounds.reduced(1.0f), cornerSize - 1.0f);
        } else {
            g.setColour(glassBorder);
            g.drawRoundedRectangle(bounds, cornerSize, 1.0f);
        }
    }
}

void KellyLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                                         bool shouldDrawButtonAsHighlighted,
                                         bool shouldDrawButtonAsDown) {
    auto bounds = button.getLocalBounds().toFloat();
    auto isOn = button.getToggleState();
    
    // Background with color psychology - Use exact documented color
    auto bgColour = isOn ? balanceGreen : surfaceColor;  // Balance Green #22C55E for active (balance/creativity)
    if (shouldDrawButtonAsHighlighted) {
        bgColour = bgColour.brighter(0.15f);
    }
    
    // Subtle gradient
    juce::ColourGradient gradient(
        bgColour.brighter(0.1f), bounds.getTopLeft(),
        bgColour.darker(0.1f), bounds.getBottomLeft(),
        false
    );
    g.setGradientFill(gradient);
    g.fillRoundedRectangle(bounds.reduced(1.0f), 6.0f);
    
    // Border
    g.setColour(borderColor);
    g.drawRoundedRectangle(bounds.reduced(1.0f), 6.0f, 1.0f);
    
    // Text
    g.setColour(button.findColour(juce::ToggleButton::textColourId)
                .withMultipliedAlpha(button.isEnabled() ? 1.0f : 0.5f));
    g.setFont(juce::FontOptions(12.0f));
    g.drawText(button.getButtonText(), bounds, juce::Justification::centred, false);
}

juce::Font KellyLookAndFeel::getTextButtonFont(juce::TextButton&, int buttonHeight) {
    return juce::FontOptions(static_cast<float>(buttonHeight) * 0.55f).withStyle("SemiBold");  // Larger
}

juce::Font KellyLookAndFeel::getLabelFont(juce::Label&) {
    return juce::FontOptions(14.0f).withStyle("Medium");  // Larger, more readable
}

juce::Font KellyLookAndFeel::getSliderPopupFont(juce::Slider&) {
    return juce::FontOptions(16.0f).withStyle("SemiBold");  // Large, bold for slider values
}

int KellyLookAndFeel::getSliderThumbRadius(juce::Slider&) {
    return 10;  // Larger thumb for better visibility and easier dragging
}

int KellyLookAndFeel::getTextButtonWidthToFitText(juce::TextButton& button, int height) {
    return getTextButtonFont(button, height).getStringWidth(button.getButtonText()) + height;
}

void KellyLookAndFeel::drawModernButton(juce::Graphics& g, const juce::Rectangle<float>& bounds,
                                        const juce::Colour& baseColour, bool isHighlighted, bool isDown) {
    auto colour = baseColour;
    
    if (isDown) {
        colour = colour.darker(0.2f);
    } else if (isHighlighted) {
        colour = colour.brighter(0.2f);
    }
    
    // Enhanced gradient for depth and engagement
    juce::ColourGradient gradient(
        colour.brighter(0.15f), bounds.getTopLeft(),
        colour.darker(0.15f), bounds.getBottomLeft(),
        false
    );
    g.setGradientFill(gradient);
    g.fillRoundedRectangle(bounds, 10.0f);
    
    // Enhanced shadow for depth perception
    g.setColour(juce::Colours::black.withAlpha(0.3f));
    g.fillRoundedRectangle(bounds.translated(0.0f, 2.0f), 10.0f);
    
    // Re-draw on top for proper layering
    g.setGradientFill(gradient);
    g.fillRoundedRectangle(bounds, 10.0f);
    
    // Subtle inner highlight for premium feel
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.fillRoundedRectangle(bounds.reduced(1.0f), 9.0f);
    
    // Border with glow effect when highlighted
    if (isHighlighted) {
        g.setColour(colour.brighter(0.4f).withAlpha(0.4f));
        g.drawRoundedRectangle(bounds, 10.0f, 2.0f);
    } else {
        g.setColour(colour.brighter(0.2f).withAlpha(0.3f));
        g.drawRoundedRectangle(bounds, 10.0f, 1.5f);
    }
}

void KellyLookAndFeel::drawModernSliderTrack(juce::Graphics& g, const juce::Rectangle<float>& trackBounds,
                                             const juce::Colour& trackColour) {
    // Glass-style track
    g.setColour(surfaceColor.withAlpha(0.6f));
    g.fillRoundedRectangle(trackBounds, trackBounds.getHeight() * 0.5f);
    
    // Glass border
    g.setColour(glassBorder);
    g.drawRoundedRectangle(trackBounds, trackBounds.getHeight() * 0.5f, 0.5f);
}

void KellyLookAndFeel::drawModernSliderThumb(juce::Graphics& g, const juce::Rectangle<float>& thumbBounds,
                                              const juce::Colour& thumbColour, bool isHighlighted) {
    auto colour = isHighlighted ? thumbColour.brighter(0.3f) : thumbColour;
    
    // Outer glow
    g.setColour(colour.withAlpha(0.3f));
    g.fillEllipse(thumbBounds.expanded(2.0f));
    
    // Main thumb with glass effect
    g.setColour(colour.withAlpha(0.9f));
    g.fillEllipse(thumbBounds);
    
    // Glass highlight
    juce::ColourGradient thumbGradient(
        juce::Colours::white.withAlpha(0.3f), thumbBounds.getTopLeft(),
        juce::Colours::transparentBlack, thumbBounds.getCentre(),
        true
    );
    g.setGradientFill(thumbGradient);
    g.fillEllipse(thumbBounds.reduced(thumbBounds.getWidth() * 0.2f, thumbBounds.getHeight() * 0.2f));
    
    // Inner dot
    g.setColour(colour);
    g.fillEllipse(thumbBounds.reduced(thumbBounds.getWidth() * 0.4f, thumbBounds.getHeight() * 0.4f));
    
    // Glass border
    g.setColour(colour.brighter(0.2f).withAlpha(0.6f));
    g.drawEllipse(thumbBounds, 1.0f);
}

} // namespace kelly
