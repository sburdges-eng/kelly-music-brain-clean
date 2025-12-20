#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "../common/Types.h"
#include "../engine/EmotionThesaurus.h"
#include <vector>
#include <functional>
#include <optional>

namespace kelly {

/**
 * Emotion Wheel Selector - Visual emotion selection component.
 * 
 * Displays emotions in a circular wheel organized by valence and arousal.
 * Users can click to select emotions visually.
 */
class EmotionWheel : public juce::Component {
public:
    EmotionWheel();
    ~EmotionWheel() override = default;
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseMove(const juce::MouseEvent& e) override;
    void mouseExit(const juce::MouseEvent& e) override;
    
    /** Set the thesaurus to use for emotion data.
     *  Non-owning reference - thesaurus must outlive this component.
     *  Lifetime is guaranteed by parent (PluginProcessor owns IntentPipeline which owns thesaurus).
     */
    void setThesaurus(const EmotionThesaurus& thesaurus);
    
    /** Set callback for when emotion is selected */
    void onEmotionSelected(std::function<void(const EmotionNode&)> callback);
    
    /** Get currently selected emotion */
    std::optional<EmotionNode> getSelectedEmotion() const;
    
    /** Set selected emotion by ID */
    void setSelectedEmotion(int emotionId);
    
private:
    // Non-owning observer - lifetime guaranteed by parent (IntentPipeline/PluginProcessor)
    const EmotionThesaurus* thesaurusRef_ = nullptr;
    std::function<void(const EmotionNode&)> onSelectedCallback_;
    std::optional<int> selectedEmotionId_;
    std::optional<int> hoveredEmotionId_;
    
    struct EmotionPosition {
        int emotionId;
        float angle;  // 0 to 2π
        float radius; // 0.0 to 1.0 (normalized)
        juce::Point<float> screenPos;
    };
    
    std::vector<EmotionPosition> emotionPositions_;
    
    void updateEmotionPositions();
    std::optional<int> getEmotionAtPoint(juce::Point<float> point) const;
    juce::Point<float> polarToCartesian(float angle, float radius, const juce::Rectangle<int>& bounds) const;
    void drawEmotionWheel(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawEmotionPoint(juce::Graphics& g, const EmotionPosition& pos, bool isSelected, bool isHovered);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(EmotionWheel)
};

} // namespace kelly
