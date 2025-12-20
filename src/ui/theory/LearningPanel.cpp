#include "LearningPanel.h"

namespace kelly {

LearningPanel::LearningPanel(midikompanion::theory::MusicTheoryBrain* brain)
    : brain_(brain)
{
    setupComponents();
}

void LearningPanel::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff2a2a2a));
}

void LearningPanel::resized() {
    auto bounds = getLocalBounds();
    const int margin = 10;
    const int titleHeight = 30;
    const int controlHeight = 30;
    const int buttonWidth = 120;

    // Title at top
    conceptTitle_.setBounds(margin, margin, bounds.getWidth() - 2 * margin, titleHeight);

    // Style selector
    int controlTop = margin + titleHeight + margin;
    styleLabel_.setBounds(margin, controlTop, 150, controlHeight);
    styleSelector_.setBounds(margin + 150, controlTop, 200, controlHeight);

    // Buttons
    playExampleButton_.setBounds(bounds.getWidth() - buttonWidth - margin, controlTop,
                                 buttonWidth, controlHeight);
    nextExerciseButton_.setBounds(bounds.getWidth() - 2 * buttonWidth - 2 * margin, controlTop,
                                 buttonWidth, controlHeight);

    // Explanation display
    int displayTop = controlTop + controlHeight + margin;
    explanationDisplay_.setBounds(margin, displayTop,
                                  bounds.getWidth() - 2 * margin,
                                  bounds.getHeight() - displayTop - margin);
}

void LearningPanel::setupComponents() {
    // Title
    conceptTitle_.setFont(juce::Font(20.0f, juce::Font::bold));
    conceptTitle_.setColour(juce::Label::textColourId, juce::Colours::white);
    conceptTitle_.setText("Select a concept to learn", juce::dontSendNotification);
    addAndMakeVisible(conceptTitle_);

    // Style selector
    styleSelector_.addItem("Intuitive", 1);
    styleSelector_.addItem("Mathematical", 2);
    styleSelector_.addItem("Historical", 3);
    styleSelector_.addItem("Acoustic", 4);
    styleSelector_.setSelectedId(1);
    styleSelector_.onChange = [this] {
        int selected = styleSelector_.getSelectedId();
        if (selected == 1) currentStyle_ = midikompanion::theory::ExplanationType::Intuitive;
        else if (selected == 2) currentStyle_ = midikompanion::theory::ExplanationType::Mathematical;
        else if (selected == 3) currentStyle_ = midikompanion::theory::ExplanationType::Historical;
        else if (selected == 4) currentStyle_ = midikompanion::theory::ExplanationType::Acoustic;
        updateExplanationDisplay();
    };
    addAndMakeVisible(styleSelector_);
    addAndMakeVisible(styleLabel_);

    // Buttons
    playExampleButton_.onClick = [this] {
        // TODO: Play MIDI example for current concept
    };
    addAndMakeVisible(playExampleButton_);

    nextExerciseButton_.onClick = [this] {
        // TODO: Load next exercise
    };
    addAndMakeVisible(nextExerciseButton_);

    // Explanation display
    explanationDisplay_.setMultiLine(true);
    explanationDisplay_.setReadOnly(true);
    explanationDisplay_.setFont(juce::Font(14.0f));
    explanationDisplay_.setText("Select a concept from the Concepts tab to begin learning.");
    addAndMakeVisible(explanationDisplay_);
}

void LearningPanel::displayConcept(const std::string& conceptName) {
    currentConcept_ = conceptName;
    conceptTitle_.setText(juce::String(conceptName), juce::dontSendNotification);
    loadExplanation(conceptName);
}

void LearningPanel::displayExplanation(const std::string& text, midikompanion::theory::ExplanationType style) {
    juce::String displayText;
    displayText += "Style: " + juce::String(style == midikompanion::theory::ExplanationType::Intuitive ? "Intuitive" :
                                            style == midikompanion::theory::ExplanationType::Mathematical ? "Mathematical" :
                                            style == midikompanion::theory::ExplanationType::Historical ? "Historical" :
                                            style == midikompanion::theory::ExplanationType::Acoustic ? "Acoustic" : "Practical") + "\n\n";
    displayText += juce::String(text);
    explanationDisplay_.setText(displayText);
}

void LearningPanel::setMusicTheoryBrain(midikompanion::theory::MusicTheoryBrain* brain) {
    brain_ = brain;
    if (!currentConcept_.empty()) {
        loadExplanation(currentConcept_);
    }
}

void LearningPanel::setExplanationStyle(midikompanion::theory::ExplanationType style) {
    currentStyle_ = style;
    updateExplanationDisplay();
}

void LearningPanel::loadExplanation(const std::string& conceptName) {
    if (!brain_) {
        explanationDisplay_.setText("Error: MusicTheoryBrain not initialized.");
        return;
    }

    // Get explanation from KnowledgeGraph
    const auto& knowledge = brain_->getKnowledge();
    auto conceptNode = knowledge.getConcept(conceptName);

    if (!conceptNode.has_value()) {
        explanationDisplay_.setText("Concept not found: " + juce::String(conceptName));
        return;
    }

    // Find explanation with current style (explanations is a map)
    std::string explanationText = "No explanation available for this style.";

    // Access explanations as map
    auto it = conceptNode->explanations.find(currentStyle_);
    if (it != conceptNode->explanations.end()) {
        explanationText = it->second;
    } else {
        // If no match, use first available explanation
        if (!conceptNode->explanations.empty()) {
            explanationText = conceptNode->explanations.begin()->second;
        }
    }

    juce::String displayText;
    displayText += "Concept: " + juce::String(conceptName) + "\n\n";
    displayText += juce::String(explanationText);

    if (!conceptNode->examples.empty()) {
        displayText += "\n\nExamples:\n";
        for (const auto& example : conceptNode->examples) {
            displayText += "  - " + juce::String(example.description) + "\n";
        }
    }

    explanationDisplay_.setText(displayText);
}

void LearningPanel::updateExplanationDisplay() {
    if (!currentConcept_.empty()) {
        loadExplanation(currentConcept_);
    }
}

} // namespace kelly
