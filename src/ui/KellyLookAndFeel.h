#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

/**
 * Kelly Look and Feel - Modern, sleek design system
 * 
 * Features:
 * - Modern color palette with gradients
 * - Smooth animations and transitions
 * - Clean typography
 * - Rounded corners and shadows
 * - Professional button styles
 */
class KellyLookAndFeel : public juce::LookAndFeel_V4 {
public:
    KellyLookAndFeel();
    ~KellyLookAndFeel() override = default;
    
    // Modern color palette
    static const juce::Colour backgroundDark;
    static const juce::Colour backgroundLight;
    static const juce::Colour surfaceColor;
    static const juce::Colour primaryColor;
    static const juce::Colour secondaryColor;
    static const juce::Colour accentColor;
    static const juce::Colour textPrimary;
    static const juce::Colour textSecondary;
    static const juce::Colour borderColor;
    static const juce::Colour successColor;
    static const juce::Colour warningColor;
    
    // Glass aesthetic colors
    static const juce::Colour glassBorder;
    static const juce::Colour glassHighlight;
    static const juce::Colour accentAlt;  // Pink for Side B
    static const juce::Colour accentTertiary;  // Cyan
    
    // Override drawing methods for modern styling
    void drawButtonBackground(juce::Graphics& g, juce::Button& button,
                              const juce::Colour& backgroundColour,
                              bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown) override;
    
    void drawButtonText(juce::Graphics& g, juce::TextButton& button,
                       bool shouldDrawButtonAsHighlighted,
                       bool shouldDrawButtonAsDown) override;
    
    void drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                         float sliderPos, float minSliderPos, float maxSliderPos,
                         const juce::Slider::SliderStyle style, juce::Slider& slider) override;
    
    void drawLinearSliderBackground(juce::Graphics& g, int x, int y, int width, int height,
                                   float sliderPos, float minSliderPos, float maxSliderPos,
                                   const juce::Slider::SliderStyle style, juce::Slider& slider) override;
    
    void drawLinearSliderThumb(juce::Graphics& g, int x, int y, int width, int height,
                               float sliderPos, float minSliderPos, float maxSliderPos,
                               const juce::Slider::SliderStyle style, juce::Slider& slider) override;
    
    void drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                     int buttonX, int buttonY, int buttonW, int buttonH,
                     juce::ComboBox& box) override;
    
    void drawTextEditorOutline(juce::Graphics& g, int width, int height,
                               juce::TextEditor& textEditor) override;
    
    void drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                          bool shouldDrawButtonAsHighlighted,
                          bool shouldDrawButtonAsDown) override;
    
    juce::Font getTextButtonFont(juce::TextButton&, int buttonHeight) override;
    juce::Font getLabelFont(juce::Label&) override;
    juce::Font getSliderPopupFont(juce::Slider&) override;
    int getSliderThumbRadius(juce::Slider&) override;
    
    int getTextButtonWidthToFitText(juce::TextButton& button, int height) override;
    
private:
    void drawModernButton(juce::Graphics& g, const juce::Rectangle<float>& bounds,
                         const juce::Colour& baseColour, bool isHighlighted, bool isDown);
    
    void drawModernSliderTrack(juce::Graphics& g, const juce::Rectangle<float>& trackBounds,
                              const juce::Colour& trackColour);
    
    void drawModernSliderThumb(juce::Graphics& g, const juce::Rectangle<float>& thumbBounds,
                               const juce::Colour& thumbColour, bool isHighlighted);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(KellyLookAndFeel)
};

} // namespace kelly
