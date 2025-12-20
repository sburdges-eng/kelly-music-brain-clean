#include "NaturalLanguageEditor.h"

namespace kelly {

NaturalLanguageEditor::NaturalLanguageEditor() {
    setupComponents();
}

void NaturalLanguageEditor::setupComponents() {
    feedbackLabel_.setText("Describe what you'd like to change:", juce::dontSendNotification);
    feedbackLabel_.setFont(juce::Font(14.0f, juce::Font::bold));
    addAndMakeVisible(feedbackLabel_);

    feedbackInput_.setMultiLine(true);
    feedbackInput_.setReturnKeyStartsNewLine(true);
    feedbackInput_.setTextToShowWhenEmpty("e.g., 'make it more melancholic', 'bass line doesn't slap'",
                                         juce::Colours::grey);
    feedbackInput_.setFont(juce::Font(13.0f));
    feedbackInput_.onTextChange = [this]() {
        if (onFeedbackChanged) {
            onFeedbackChanged(feedbackInput_.getText());
        }
    };
    addAndMakeVisible(feedbackInput_);

    applyButton_.setButtonText("Apply Changes");
    applyButton_.onClick = [this]() {
        if (onFeedbackSubmitted && !feedbackInput_.getText().isEmpty()) {
            onFeedbackSubmitted(feedbackInput_.getText());
        }
    };
    addAndMakeVisible(applyButton_);

    cancelButton_.setButtonText("Cancel");
    cancelButton_.onClick = [this]() {
        clearPreview();
        feedbackInput_.clear();
    };
    addAndMakeVisible(cancelButton_);
}

void NaturalLanguageEditor::paint(juce::Graphics& g) {
    // Background
    g.fillAll(juce::Colour(0xff2a2a2a));

    // Preview overlay if active
    if (showPreview_ && !previewChanges_.empty()) {
        auto bounds = getLocalBounds();
        auto previewArea = bounds.removeFromBottom(100);

        g.setColour(juce::Colour(0x4000ff00).withAlpha(0.3f));
        g.fillRoundedRectangle(previewArea.toFloat(), 5.0f);

        g.setColour(juce::Colours::lightgreen);
        g.setFont(juce::Font(12.0f, juce::Font::bold));
        g.drawText("Preview Changes (Confidence: " + juce::String(previewConfidence_ * 100.0f, 1) + "%)",
                   previewArea.removeFromTop(20), juce::Justification::centredLeft);

        int y = previewArea.getY() + 5;
        g.setFont(juce::Font(11.0f));
        for (const auto& [param, value] : previewChanges_) {
            g.drawText(juce::String(param) + ": " + juce::String(value, 2),
                      previewArea.getX() + 10, y, 200, 15,
                      juce::Justification::centredLeft);
            y += 18;
        }
    }
}

void NaturalLanguageEditor::resized() {
    auto bounds = getLocalBounds().reduced(10);

    feedbackLabel_.setBounds(bounds.removeFromTop(20));
    bounds.removeFromTop(5);

    auto inputArea = bounds.removeFromTop(60);
    feedbackInput_.setBounds(inputArea);

    bounds.removeFromTop(10);

    auto buttonArea = bounds.removeFromTop(30);
    applyButton_.setBounds(buttonArea.removeFromLeft(120));
    buttonArea.removeFromLeft(10);
    cancelButton_.setBounds(buttonArea.removeFromLeft(120));
}

void NaturalLanguageEditor::showPreview(const std::map<std::string, float>& parameterChanges, float confidence) {
    previewChanges_ = parameterChanges;
    previewConfidence_ = confidence;
    showPreview_ = true;
    repaint();
}

void NaturalLanguageEditor::clearPreview() {
    showPreview_ = false;
    previewChanges_.clear();
    previewConfidence_ = 0.0f;
    repaint();
}

} // namespace kelly
