/**
 * Emotion Radar Visualization
 * 
 * Visualizes valence/arousal/intensity in a radar/polar plot.
 */

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

/**
 * Emotion radar visualization component
 */
class EmotionRadar : public juce::Component {
public:
    EmotionRadar();
    ~EmotionRadar() override = default;

    /**
     * Set emotion values
     */
    void setEmotion(float valence, float arousal, float intensity);

    /**
     * Set target emotion (for comparison)
     */
    void setTargetEmotion(float valence, float arousal, float intensity);

    /**
     * Clear target
     */
    void clearTarget() { hasTarget_ = false; repaint(); }

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    float valence_ = 0.0f;
    float arousal_ = 0.5f;
    float intensity_ = 0.5f;
    
    float targetValence_ = 0.0f;
    float targetArousal_ = 0.5f;
    float targetIntensity_ = 0.5f;
    bool hasTarget_ = false;
    
    void drawRadar(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawPoint(juce::Graphics& g, float valence, float arousal, float intensity, 
                   juce::Colour colour, bool isTarget = false);
    juce::Point<float> emotionToPoint(float valence, float arousal, float intensity,
                                     const juce::Rectangle<int>& bounds) const;
};

} // namespace kelly
