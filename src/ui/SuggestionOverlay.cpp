#include "SuggestionOverlay.h"
#include <algorithm>

namespace kelly {

SuggestionOverlay::SuggestionOverlay() {
    setOpaque(false);
    setInterceptsMouseClicks(true, true);
}

void SuggestionOverlay::paint(juce::Graphics& g) {
    // Semi-transparent background
    g.setColour(juce::Colour(0x80000000));
    g.fillAll();

    // Draw cards background
    g.setColour(juce::Colour(0xFF2A2A2A));
    for (const auto& card : cards_) {
        if (card) {
            auto bounds = card->titleLabel->getBounds().expanded(10, 10);
            g.fillRoundedRectangle(bounds.toFloat(), 8.0f);
            g.setColour(juce::Colour(0xFF404040));
            g.drawRoundedRectangle(bounds.toFloat(), 8.0f, 1.0f);
        }
    }
}

void SuggestionOverlay::resized() {
    layoutCards();
}

void SuggestionOverlay::setSuggestions(const std::vector<Suggestion>& suggestions) {
    suggestions_ = suggestions;
    rebuildCards();
    layoutCards();
    repaint();
}

void SuggestionOverlay::clear() {
    suggestions_.clear();
    cards_.clear();
    repaint();
}

void SuggestionOverlay::setVisible(bool shouldBeVisible) {
    Component::setVisible(shouldBeVisible);
    if (!shouldBeVisible) {
        clear();
    }
}

void SuggestionOverlay::rebuildCards() {
    cards_.clear();

    for (const auto& suggestion : suggestions_) {
        auto card = std::make_unique<SuggestionCard>();
        card->suggestion = suggestion;

        // Title label
        card->titleLabel = std::make_unique<juce::Label>();
        card->titleLabel->setText(suggestion.title, juce::dontSendNotification);
        juce::Font titleFont = juce::Font(juce::FontOptions().withHeight(16.0f));
        titleFont.setBold(true);
        card->titleLabel->setFont(titleFont);
        card->titleLabel->setColour(juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible(card->titleLabel.get());

        // Description label
        card->descriptionLabel = std::make_unique<juce::Label>();
        card->descriptionLabel->setText(suggestion.description, juce::dontSendNotification);
        juce::FontOptions descFontOptions = juce::FontOptions().withHeight(13.0f);
        card->descriptionLabel->setFont(juce::Font(descFontOptions));
        card->descriptionLabel->setColour(juce::Label::textColourId, juce::Colour(0xFFCCCCCC));
        addAndMakeVisible(card->descriptionLabel.get());

        // Confidence label
        card->confidenceLabel = std::make_unique<juce::Label>();
        card->confidenceLabel->setText(getConfidenceText(suggestion.confidence), juce::dontSendNotification);
        juce::FontOptions confFontOptions = juce::FontOptions().withHeight(11.0f);
        card->confidenceLabel->setFont(juce::Font(confFontOptions));
        card->confidenceLabel->setColour(juce::Label::textColourId, getConfidenceColor(suggestion.confidence));
        addAndMakeVisible(card->confidenceLabel.get());

        // Confidence bar (simple colored rectangle)
        card->confidenceBar = std::make_unique<juce::Component>();
        addAndMakeVisible(card->confidenceBar.get());

        // Apply button
        card->applyButton = std::make_unique<juce::TextButton>("Apply");
        card->applyButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF4CAF50));
        card->applyButton->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        int cardIndex = static_cast<int>(cards_.size());
        card->applyButton->onClick = [this, cardIndex] { applyButtonClicked(cardIndex); };
        addAndMakeVisible(card->applyButton.get());

        // Dismiss button
        card->dismissButton = std::make_unique<juce::TextButton>("×");
        card->dismissButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF666666));
        card->dismissButton->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        card->dismissButton->onClick = [this, cardIndex] { dismissButtonClicked(cardIndex); };
        addAndMakeVisible(card->dismissButton.get());

        // Expand button (for explanation)
        card->expandButton = std::make_unique<juce::TextButton>("...");
        card->expandButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0xFF555555));
        card->expandButton->setColour(juce::TextButton::textColourOffId, juce::Colour(0xFFCCCCCC));
        card->expandButton->onClick = [this, cardIndex] { expandButtonClicked(cardIndex); };
        addAndMakeVisible(card->expandButton.get());

        // Explanation label (initially hidden)
        card->explanationLabel = std::make_unique<juce::Label>();
        card->explanationLabel->setText(suggestion.explanation, juce::dontSendNotification);
        juce::FontOptions expFontOptions = juce::FontOptions().withHeight(12.0f);
        card->explanationLabel->setFont(juce::Font(expFontOptions));
        card->explanationLabel->setColour(juce::Label::textColourId, juce::Colour(0xFFAAAAAA));
        card->explanationLabel->setVisible(false);
        addAndMakeVisible(card->explanationLabel.get());

        cards_.push_back(std::move(card));
    }
}

void SuggestionOverlay::layoutCards() {
    const int margin = 10;
    const int cardHeight = 120;
    const int cardSpacing = 15;
    int y = margin;

    for (auto& card : cards_) {
        if (!card) continue;

        const int cardWidth = getWidth() - 2 * margin;
        const int cardX = margin;

        // Title
        card->titleLabel->setBounds(cardX + 15, y + 10, cardWidth - 100, 20);

        // Confidence label (top right)
        card->confidenceLabel->setBounds(cardX + cardWidth - 80, y + 10, 70, 20);

        // Description
        card->descriptionLabel->setBounds(cardX + 15, y + 35, cardWidth - 30, 20);

        // Confidence bar
        card->confidenceBar->setBounds(cardX + 15, y + 60, cardWidth - 30, 4);

        // Buttons
        const int buttonWidth = 60;
        const int buttonHeight = 30;
        card->applyButton->setBounds(cardX + 15, y + 70, buttonWidth, buttonHeight);
        card->expandButton->setBounds(cardX + 85, y + 70, 30, buttonHeight);
        card->dismissButton->setBounds(cardX + cardWidth - 35, y + 5, 30, 30);

        // Explanation (if expanded)
        if (card->expanded) {
            card->explanationLabel->setBounds(cardX + 15, y + 105, cardWidth - 30, 40);
            y += cardHeight + 40 + cardSpacing;
        } else {
            y += cardHeight + cardSpacing;
        }
    }
}

juce::Colour SuggestionOverlay::getConfidenceColor(float confidence) const {
    if (confidence >= 0.7f) {
        return juce::Colour(0xFF4CAF50);  // Green (high)
    } else if (confidence >= 0.4f) {
        return juce::Colour(0xFFFFC107);  // Yellow (medium)
    } else {
        return juce::Colour(0xFFFF9800);  // Orange (low)
    }
}

std::string SuggestionOverlay::getConfidenceText(float confidence) const {
    if (confidence >= 0.7f) {
        return "High";
    } else if (confidence >= 0.4f) {
        return "Medium";
    } else {
        return "Low";
    }
}

void SuggestionOverlay::applyButtonClicked(int cardIndex) {
    if (cardIndex >= 0 && cardIndex < static_cast<int>(cards_.size()) && cards_[cardIndex]) {
        if (onSuggestionApplied) {
            onSuggestionApplied(cards_[cardIndex]->suggestion);
        }
    }
}

void SuggestionOverlay::dismissButtonClicked(int cardIndex) {
    if (cardIndex >= 0 && cardIndex < static_cast<int>(cards_.size()) && cards_[cardIndex]) {
        const std::string suggestionId = cards_[cardIndex]->suggestion.id;
        if (onSuggestionDismissed) {
            onSuggestionDismissed(suggestionId);
        }
        // Remove card
        cards_.erase(cards_.begin() + cardIndex);
        layoutCards();
        repaint();
    }
}

void SuggestionOverlay::expandButtonClicked(int cardIndex) {
    if (cardIndex >= 0 && cardIndex < static_cast<int>(cards_.size()) && cards_[cardIndex]) {
        cards_[cardIndex]->expanded = !cards_[cardIndex]->expanded;
        cards_[cardIndex]->explanationLabel->setVisible(cards_[cardIndex]->expanded);
        layoutCards();
        repaint();
    }
}

} // namespace kelly
