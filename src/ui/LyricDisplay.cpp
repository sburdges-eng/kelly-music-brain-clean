#include "LyricDisplay.h"
#include "../voice/LyriSync.h"
#include <cmath>

namespace kelly {

LyricDisplay::LyricDisplay()
    : fontSize_(16.0f)
    , highlightSyllables_(true)
{
    setOpaque(true);
    startTimer(50); // Update at ~20 FPS for smooth highlighting
}

void LyricDisplay::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff1a1a1a)); // Dark background

    if (lyrics_.sections.empty()) {
        g.setColour(juce::Colours::grey);
        g.setFont(fontSize_);
        g.drawText("No lyrics available", getLocalBounds(),
                   juce::Justification::centred);
        return;
    }

    // Determine current highlighted line
    int currentLineIndex = getCurrentLineIndex();

    // Paint all sections
    int yOffset = 10;
    int lineHeight = static_cast<int>(fontSize_ * 2.5f);
    int lineSpacing = 10;

    for (const auto& section : lyrics_.sections) {
        // Section header
        g.setColour(juce::Colours::lightgrey);
        g.setFont(fontSize_ * 0.8f);

        juce::String sectionName;
        switch (section.type) {
            case LyricSectionType::Verse:
                sectionName = "Verse";
                break;
            case LyricSectionType::Chorus:
                sectionName = "Chorus";
                break;
            case LyricSectionType::Bridge:
                sectionName = "Bridge";
                break;
            case LyricSectionType::Intro:
                sectionName = "Intro";
                break;
            case LyricSectionType::Outro:
                sectionName = "Outro";
                break;
            default:
                sectionName = "Section";
                break;
        }

        g.drawText(sectionName, 10, yOffset, getWidth() - 20, 20,
                   juce::Justification::left);
        yOffset += 25;

        // Paint lines in section
        int lineIndex = 0;
        for (const auto& line : section.lines) {
            juce::Rectangle<int> lineBounds(10, yOffset, getWidth() - 20, lineHeight);
            bool isHighlighted = (lineIndex == currentLineIndex);

            paintLine(g, line, lineIndex, lineBounds, isHighlighted);

            yOffset += lineHeight + lineSpacing;
            lineIndex++;
        }

        // Section spacing
        yOffset += 20;
    }
}

void LyricDisplay::paintLine(juce::Graphics& g, const LyricLine& line, int lineIndex,
                             const juce::Rectangle<int>& bounds, bool isHighlighted) {
    juce::Colour textColor = isHighlighted ? highlightedColor_ : normalColor_;
    g.setColour(textColor);

    if (highlightSyllables_ && !line.syllables.empty()) {
        // Paint with syllable breakdown
        paintSyllables(g, line, bounds, getCurrentSyllableIndex(line));
    } else {
        // Paint as simple text
        g.setFont(fontSize_);
        g.drawText(line.text, bounds, juce::Justification::left);
    }
}

void LyricDisplay::paintSyllables(juce::Graphics& g, const LyricLine& line,
                                  const juce::Rectangle<int>& bounds, int highlightedSyllableIndex) {
    int xOffset = bounds.getX();
    int yPos = bounds.getY() + static_cast<int>(fontSize_ * 1.2f);

    for (size_t i = 0; i < line.syllables.size(); ++i) {
        const auto& syllable = line.syllables[i];

        // Determine color based on highlight
        juce::Colour syllableColor;
        if (static_cast<int>(i) == highlightedSyllableIndex) {
            syllableColor = highlightedColor_;
        } else if (syllable.stress == 2) {
            syllableColor = normalColor_;  // Primary stress = normal color
        } else if (syllable.stress == 1) {
            syllableColor = syllableColor_;  // Secondary stress = lighter
        } else {
            syllableColor = syllableColor_.withAlpha(0.7f);  // Unstressed = dimmed
        }

        g.setColour(syllableColor);
        g.setFont(fontSize_ * (syllable.stress == 2 ? 1.1f : 1.0f)); // Larger for stressed

        // Measure text width
        int textWidth = g.getCurrentFont().getStringWidth(syllable.text);

        // Draw syllable
        g.drawText(syllable.text, xOffset, yPos - static_cast<int>(fontSize_),
                   textWidth, static_cast<int>(fontSize_ * 2),
                   juce::Justification::left);

        // Draw stress mark (if stressed)
        if (syllable.stress > 0) {
            g.drawLine(static_cast<float>(xOffset), static_cast<float>(yPos + 2),
                      static_cast<float>(xOffset + textWidth), static_cast<float>(yPos + 2),
                      syllable.stress == 2 ? 2.0f : 1.0f);
        }

        xOffset += textWidth + 5; // Space between syllables
    }
}

void LyricDisplay::resized() {
    // Component layout if needed
}

void LyricDisplay::timerCallback() {
    repaint(); // Update highlighting based on current beat
}

void LyricDisplay::setLyrics(const LyricStructure& lyrics) {
    lyrics_ = lyrics;
    repaint();
}

void LyricDisplay::setSyncResult(const LyriSync::SyncResult& syncResult) {
    syncResult_ = syncResult;
}

void LyricDisplay::setCurrentBeat(double currentBeat) {
    currentBeat_ = currentBeat;
}

void LyricDisplay::clear() {
    lyrics_ = LyricStructure();
    syncResult_ = LyriSync::SyncResult();
    currentBeat_ = 0.0;
    repaint();
}

int LyricDisplay::getCurrentLineIndex() const {
    if (syncResult_.items.empty()) {
        return -1;
    }

    // Find which sync item corresponds to current beat
    int currentItemIndex = -1;
    for (size_t i = 0; i < syncResult_.items.size(); ++i) {
        const auto& item = syncResult_.items[i];
        if (currentBeat_ >= item.startBeat &&
            currentBeat_ < item.startBeat + item.duration) {
            currentItemIndex = static_cast<int>(i);
            break;
        }
    }

    // Map sync item index to line index
    // (This is simplified - in a full implementation, we'd track which item belongs to which line)
    if (currentItemIndex >= 0) {
        int lineCount = 0;
        for (const auto& section : lyrics_.sections) {
            if (currentItemIndex < lineCount + static_cast<int>(section.lines.size())) {
                return currentItemIndex - lineCount;
            }
            lineCount += static_cast<int>(section.lines.size());
        }
    }

    return -1;
}

int LyricDisplay::getCurrentSyllableIndex(const LyricLine& line) const {
    // Find which syllable is currently being sung
    // This is a simplified implementation
    // In a full version, we'd use syncResult_ to determine the exact syllable

    if (syncResult_.items.empty() || line.syllables.empty()) {
        return -1;
    }

    // For now, return -1 (no specific syllable highlight)
    // Full implementation would map sync items to syllables
    return -1;
}

} // namespace kelly
