#pragma once
/*
 * NaturalLanguageEditor.h - Natural Language Feedback Input Component
 * ===================================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - UI Layer: Used by EmotionWorkstation (natural language feedback)
 * - Engine Layer: Processes natural language feedback to parameter changes
 * - Bridge Layer: May use Python NLP for interpreting natural language
 *
 * Purpose: UI for natural language feedback input with preview-before-apply functionality.
 *          Part of Phase 6 of the "All-Knowing Interactive Musical Customization System".
 *
 * Features:
 * - Text input for user feedback
 * - Preview of interpreted parameter changes
 * - Confidence scoring for interpretations
 * - Apply/cancel functionality
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>
#include <string>

namespace kelly {

/**
 * NaturalLanguageEditor - UI for natural language feedback input
 *
 * Part of Phase 6 of the "All-Knowing Interactive Musical Customization System".
 * Provides text input for user feedback with preview-before-apply functionality.
 */
class NaturalLanguageEditor : public juce::Component {
public:
    NaturalLanguageEditor();
    ~NaturalLanguageEditor() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Callbacks
    std::function<void(const juce::String& feedback)> onFeedbackSubmitted;
    std::function<void(const juce::String& feedback)> onFeedbackChanged;  // Real-time as user types

    juce::String getFeedbackText() const { return feedbackInput_.getText(); }
    void setFeedbackText(const juce::String& text) { feedbackInput_.setText(text); }
    void clear() { feedbackInput_.clear(); }

    // Show interpreted changes preview
    void showPreview(const std::map<std::string, float>& parameterChanges, float confidence);
    void clearPreview();

private:
    juce::TextEditor feedbackInput_;
    juce::Label feedbackLabel_;
    juce::TextButton applyButton_{"Apply Changes"};
    juce::TextButton cancelButton_{"Cancel"};

    // Preview display
    bool showPreview_ = false;
    std::map<std::string, float> previewChanges_;
    float previewConfidence_ = 0.0f;

    void setupComponents();
};

} // namespace kelly
