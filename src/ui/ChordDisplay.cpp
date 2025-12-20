/**
 * Chord Display Implementation
 */

#include "ChordDisplay.h"
#include "KellyLookAndFeel.h"

namespace kelly {

ChordDisplay::ChordDisplay() {
    setOpaque(false);
}

void ChordDisplay::setChord(const juce::String& chordName, const std::vector<int>& notes) {
    chordName_ = chordName;
    chordNotes_ = notes;
    repaint();
}

void ChordDisplay::clear() {
    chordName_.clear();
    chordNotes_.clear();
    repaint();
}

void ChordDisplay::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds();
    
    // Background
    g.setColour(KellyLookAndFeel::surfaceColor.withAlpha(0.8f));
    g.fillRoundedRectangle(bounds.toFloat(), 8.0f);
    
    // Border
    g.setColour(KellyLookAndFeel::accentColor.withAlpha(0.3f));
    g.drawRoundedRectangle(bounds.toFloat().reduced(0.5f), 8.0f, 1.0f);
    
    if (chordName_.isEmpty()) return;
    
    drawChordName(g, bounds);
    drawChordNotes(g, bounds);
}

void ChordDisplay::drawChordName(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    g.setColour(KellyLookAndFeel::textPrimary);
    g.setFont(juce::Font(juce::FontOptions().withHeight(24.0f).withStyle("Bold")));
    auto mutableBounds = bounds;
    g.drawText(chordName_, mutableBounds.removeFromTop(30), juce::Justification::centred);
}

void ChordDisplay::drawChordNotes(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    if (chordNotes_.empty()) return;

    g.setColour(KellyLookAndFeel::textSecondary);
    g.setFont(juce::Font(juce::FontOptions(12.0f)));
    
    juce::String notesText;
    for (size_t i = 0; i < chordNotes_.size(); ++i) {
        if (i > 0) notesText += " ";
        notesText += noteNumberToName(chordNotes_[i]);
    }
    
    g.drawText(notesText, bounds, juce::Justification::centred);
}

juce::String ChordDisplay::noteNumberToName(int noteNumber) const {
    static const char* noteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    int octave = (noteNumber / 12) - 1;
    int note = noteNumber % 12;
    return juce::String(noteNames[note]) + juce::String(octave);
}

void ChordDisplay::resized() {
    repaint();
}

} // namespace kelly
