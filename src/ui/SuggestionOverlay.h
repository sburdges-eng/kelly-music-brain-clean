#pragma once
/*
 * SuggestionOverlay.h - Intelligent Suggestion Display Component
 * ==============================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Bridge Layer: SuggestionBridge (Python suggestion engine)
 * - UI Layer: Used by EmotionWorkstation (displays intelligent suggestions)
 * - Engine Layer: Receives suggestions based on current musical state
 *
 * Purpose: Displays intelligent suggestions to the user as tooltips or in a side panel overlay.
 *          Part of Phase 3 of the "All-Knowing Interactive Musical Customization System".
 *
 * Features:
 * - Parameter suggestions
 * - Emotion suggestions
 * - Rule break suggestions
 * - Style suggestions
 * - Confidence scoring
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>
#include <string>
#include <memory>

namespace kelly {

/**
 * SuggestionOverlay - Displays intelligent suggestions to the user
 *
 * Shows suggestions as tooltips or in a side panel overlay.
 * Part of Phase 3 of the "All-Knowing Interactive Musical Customization System".
 */
class SuggestionOverlay : public juce::Component {
public:
    struct Suggestion {
        std::string title;
        std::string description;
        std::string explanation;
        std::string id;
        float confidence;  // 0.0 to 1.0
        std::string type;  // "parameter", "emotion", "rule_break", "style"
    };

    struct SuggestionCard {
        Suggestion suggestion;
        std::unique_ptr<juce::Label> titleLabel;
        std::unique_ptr<juce::Label> descriptionLabel;
        std::unique_ptr<juce::Label> confidenceLabel;
        std::unique_ptr<juce::Label> explanationLabel;
        std::unique_ptr<juce::Component> confidenceBar;
        std::unique_ptr<juce::TextButton> applyButton;
        std::unique_ptr<juce::TextButton> dismissButton;
        std::unique_ptr<juce::TextButton> expandButton;
        bool expanded = false;
    };

    std::function<void(const Suggestion&)> onSuggestionApplied;
    std::function<void(const std::string&)> onSuggestionDismissed;

    SuggestionOverlay();
    ~SuggestionOverlay() override = default;

    void setSuggestions(const std::vector<Suggestion>& suggestions);
    void clearSuggestions();
    void clear();
    void setVisible(bool shouldBeVisible) override;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    std::vector<Suggestion> suggestions_;
    std::vector<std::unique_ptr<SuggestionCard>> cards_;

    void rebuildCards();
    void layoutCards();
    juce::Colour getConfidenceColor(float confidence) const;
    std::string getConfidenceText(float confidence) const;
    void applyButtonClicked(int cardIndex);
    void dismissButtonClicked(int cardIndex);
    void expandButtonClicked(int cardIndex);
};

} // namespace kelly
