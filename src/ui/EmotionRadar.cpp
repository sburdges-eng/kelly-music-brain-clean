/**
 * Emotion Radar Implementation
 */

#include "EmotionRadar.h"
#include "KellyLookAndFeel.h"
#include <cmath>

namespace kelly {

EmotionRadar::EmotionRadar() {
    setOpaque(false);
}

void EmotionRadar::setEmotion(float valence, float arousal, float intensity) {
    valence_ = juce::jlimit(-1.0f, 1.0f, valence);
    arousal_ = juce::jlimit(0.0f, 1.0f, arousal);
    intensity_ = juce::jlimit(0.0f, 1.0f, intensity);
    repaint();
}

void EmotionRadar::setTargetEmotion(float valence, float arousal, float intensity) {
    targetValence_ = juce::jlimit(-1.0f, 1.0f, valence);
    targetArousal_ = juce::jlimit(0.0f, 1.0f, arousal);
    targetIntensity_ = juce::jlimit(0.0f, 1.0f, intensity);
    hasTarget_ = true;
    repaint();
}

void EmotionRadar::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds();
    drawRadar(g, bounds);
    
    // Draw target if set
    if (hasTarget_) {
        drawPoint(g, targetValence_, targetArousal_, targetIntensity_,
                 KellyLookAndFeel::accentAlt, true);
    }
    
    // Draw current
    drawPoint(g, valence_, arousal_, intensity_,
             KellyLookAndFeel::accentColor, false);
}

void EmotionRadar::drawRadar(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    auto center = bounds.getCentre().toFloat();
    float radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) * 0.4f;
    
    // Background circle
    g.setColour(KellyLookAndFeel::surfaceColor);
    g.fillEllipse(center.x - radius, center.y - radius, radius * 2, radius * 2);
    
    // Grid circles
    g.setColour(KellyLookAndFeel::textSecondary.withAlpha(0.3f));
    for (int i = 1; i <= 3; ++i) {
        float r = radius * (i / 3.0f);
        g.drawEllipse(center.x - r, center.y - r, r * 2, r * 2, 1.0f);
    }
    
    // Axes
    g.setColour(KellyLookAndFeel::textSecondary.withAlpha(0.5f));
    g.drawLine(center.x - radius, center.y, center.x + radius, center.y, 1.0f);
    g.drawLine(center.x, center.y - radius, center.x, center.y + radius, 1.0f);
    
    // Labels
    g.setColour(KellyLookAndFeel::textSecondary);
    g.setFont(juce::Font(juce::FontOptions(10.0f)));
    g.drawText("Valence+", center.x + radius + 5, center.y - 8, 50, 16, juce::Justification::left);
    g.drawText("Valence-", center.x - radius - 55, center.y - 8, 50, 16, juce::Justification::right);
    g.drawText("Arousal+", center.x - 25, center.y - radius - 20, 50, 16, juce::Justification::centred);
    g.drawText("Arousal-", center.x - 25, center.y + radius + 4, 50, 16, juce::Justification::centred);
}

void EmotionRadar::drawPoint(juce::Graphics& g, float valence, float arousal, float intensity,
                             juce::Colour colour, bool isTarget) {
    auto bounds = getLocalBounds();
    auto point = emotionToPoint(valence, arousal, intensity, bounds);
    
    float pointSize = 8.0f + (intensity * 4.0f);
    if (isTarget) {
        pointSize += 2.0f;
        g.setColour(colour.withAlpha(0.5f));
        g.fillEllipse(point.x - pointSize - 2, point.y - pointSize - 2,
                     (pointSize + 2) * 2, (pointSize + 2) * 2);
    }
    
    g.setColour(colour);
    g.fillEllipse(point.x - pointSize, point.y - pointSize,
                 pointSize * 2, pointSize * 2);
    
    g.setColour(colour.brighter(0.5f));
    g.drawEllipse(point.x - pointSize, point.y - pointSize,
                 pointSize * 2, pointSize * 2, 1.5f);
}

juce::Point<float> EmotionRadar::emotionToPoint(float valence, float arousal, float intensity,
                                                const juce::Rectangle<int>& bounds) const {
    auto center = bounds.getCentre().toFloat();
    float radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) * 0.4f;
    
    // Convert to polar coordinates
    // Valence = angle (0 = right, positive = up, negative = down)
    // Arousal = distance from center (0 = center, 1 = edge)
    // Intensity = point size (handled in drawPoint)
    
    float angle = (valence + 1.0f) * juce::MathConstants<float>::pi;  // 0 to 2π
    float distance = arousal * radius;
    
    float x = center.x + std::cos(angle) * distance;
    float y = center.y - std::sin(angle) * distance;  // Negative because screen Y is inverted
    
    return juce::Point<float>(x, y);
}

void EmotionRadar::resized() {
    repaint();
}

} // namespace kelly
