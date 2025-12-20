#include "EmotionWheel.h"
#include "KellyLookAndFeel.h"
#include <cmath>
#include <algorithm>

namespace kelly {

EmotionWheel::EmotionWheel() {
    setOpaque(false);
}

void EmotionWheel::setThesaurus(const EmotionThesaurus& thesaurus) {
    thesaurusRef_ = &thesaurus;  // Non-owning reference
    updateEmotionPositions();
    repaint();
}

void EmotionWheel::onEmotionSelected(std::function<void(const EmotionNode&)> callback) {
    onSelectedCallback_ = callback;
}

std::optional<EmotionNode> EmotionWheel::getSelectedEmotion() const {
    if (selectedEmotionId_ && thesaurusRef_) {
        return thesaurusRef_->findById(*selectedEmotionId_);
    }
    return std::nullopt;
}

void EmotionWheel::setSelectedEmotion(int emotionId) {
    selectedEmotionId_ = emotionId;
    repaint();
}

void EmotionWheel::updateEmotionPositions() {
    emotionPositions_.clear();
    if (!thesaurusRef_) return;
    
    // Map emotions to polar coordinates
    // Angle = valence (0 = negative, π = positive)
    // Radius = arousal (0 = center/calm, 1 = edge/excited)
    
    for (const auto& [id, emotion] : thesaurusRef_->all()) {
        EmotionPosition pos;
        pos.emotionId = id;
        
        // Map valence (-1 to 1) to angle (0 to 2π)
        // Negative valence = left side (π to 2π), positive = right side (0 to π)
        pos.angle = (emotion.valence + 1.0f) * juce::MathConstants<float>::pi;
        
        // Map arousal (0 to 1) to radius (0.3 to 0.9, leaving center space)
        pos.radius = 0.3f + (emotion.arousal * 0.6f);
        
        emotionPositions_.push_back(pos);
    }
}

juce::Point<float> EmotionWheel::polarToCartesian(float angle, float radius, const juce::Rectangle<int>& bounds) const {
    float centerX = bounds.getCentreX();
    float centerY = bounds.getCentreY();
    float maxRadius = std::min(bounds.getWidth(), bounds.getHeight()) / 2.0f - 10.0f;
    
    float x = centerX + std::cos(angle) * radius * maxRadius;
    float y = centerY + std::sin(angle) * radius * maxRadius;
    
    return {x, y};
}

void EmotionWheel::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds();
    
    drawEmotionWheel(g, bounds);
    
    // Update screen positions and draw emotion points
    float centerX = bounds.getCentreX();
    float centerY = bounds.getCentreY();
    float maxRadius = std::min(bounds.getWidth(), bounds.getHeight()) / 2.0f - 10.0f;
    
    for (auto& pos : emotionPositions_) {
        pos.screenPos = polarToCartesian(pos.angle, pos.radius, bounds);
        bool isSelected = selectedEmotionId_ && *selectedEmotionId_ == pos.emotionId;
        bool isHovered = hoveredEmotionId_ && *hoveredEmotionId_ == pos.emotionId;
        drawEmotionPoint(g, pos, isSelected, isHovered);
    }
}

void EmotionWheel::drawEmotionWheel(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    float centerX = bounds.getCentreX();
    float centerY = bounds.getCentreY();
    float maxRadius = std::min(bounds.getWidth(), bounds.getHeight()) / 2.0f - 10.0f;
    
    // Draw background circle using documented surface color
    g.setColour(KellyLookAndFeel::surfaceColor.withAlpha(0.3f));
    g.fillEllipse(centerX - maxRadius, centerY - maxRadius, maxRadius * 2, maxRadius * 2);
    
    // Draw axes using documented text colors
    g.setColour(KellyLookAndFeel::textPrimary.withAlpha(0.5f));
    g.drawLine(centerX - maxRadius, centerY, centerX + maxRadius, centerY, 1.0f);  // Horizontal (valence)
    g.drawLine(centerX, centerY - maxRadius, centerX, centerY + maxRadius, 1.0f);  // Vertical (arousal)
    
    // Draw labels using documented text colors
    g.setFont(10.0f);
    g.setColour(KellyLookAndFeel::textPrimary);
    g.drawText("Negative", (int)(centerX - maxRadius - 40), (int)(centerY - 6), 40, 12, juce::Justification::centredRight);
    g.drawText("Positive", (int)(centerX + maxRadius), (int)(centerY - 6), 40, 12, juce::Justification::centredLeft);
    g.drawText("Calm", (int)(centerX - 20), (int)(centerY - maxRadius - 15), 40, 12, juce::Justification::centred);
    g.drawText("Excited", (int)(centerX - 20), (int)(centerY + maxRadius + 3), 40, 12, juce::Justification::centred);
}

void EmotionWheel::drawEmotionPoint(juce::Graphics& g, const EmotionPosition& pos, bool isSelected, bool isHovered) {
    float pointSize = isSelected ? 8.0f : (isHovered ? 6.0f : 4.0f);
    // Use documented colors: Creative Purple for selected, Focus Blue for hovered, textPrimary for normal
    juce::Colour pointColour = isSelected ? juce::Colour(0xFFA855F7)  // Creative Purple #A855F7
                                          : (isHovered ? juce::Colour(0xFF3B82F6)  // Focus Blue #3B82F6
                                                       : KellyLookAndFeel::textPrimary);
    
    g.setColour(pointColour);
    g.fillEllipse(pos.screenPos.x - pointSize/2, pos.screenPos.y - pointSize/2, pointSize, pointSize);
    
    // Draw emotion name on hover/select
    if (isSelected || isHovered) {
        if (thesaurusRef_) {
            auto emotion = thesaurusRef_->findById(pos.emotionId);
            if (emotion) {
                g.setColour(KellyLookAndFeel::textPrimary);
                g.setFont(9.0f);
                auto textBounds = juce::Rectangle<float>(pos.screenPos.x - 30, pos.screenPos.y - 20, 60, 12);
                g.drawText(emotion->name, textBounds, juce::Justification::centred);
            }
        }
    }
}

void EmotionWheel::resized() {
    updateEmotionPositions();
    repaint();
}

void EmotionWheel::mouseDown(const juce::MouseEvent& e) {
    auto emotionId = getEmotionAtPoint(e.position);
    if (emotionId && thesaurusRef_) {
        selectedEmotionId_ = emotionId;
        auto emotion = thesaurusRef_->findById(*emotionId);
        if (emotion && onSelectedCallback_) {
            onSelectedCallback_(*emotion);
        }
        repaint();
    }
}

void EmotionWheel::mouseMove(const juce::MouseEvent& e) {
    auto emotionId = getEmotionAtPoint(e.position);
    if (emotionId != hoveredEmotionId_) {
        hoveredEmotionId_ = emotionId;
        repaint();
    }
}

void EmotionWheel::mouseExit(const juce::MouseEvent&) {
    hoveredEmotionId_ = std::nullopt;
    repaint();
}

std::optional<int> EmotionWheel::getEmotionAtPoint(juce::Point<float> point) const {
    const float clickRadius = 10.0f;
    
    for (const auto& pos : emotionPositions_) {
        float distance = point.getDistanceFrom(pos.screenPos);
        if (distance < clickRadius) {
            return pos.emotionId;
        }
    }
    
    return std::nullopt;
}

} // namespace kelly
