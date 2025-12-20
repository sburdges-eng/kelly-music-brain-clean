/**
 * Chord Display Component
 * 
 * Shows current chord name and notes.
 */

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

namespace kelly {

/**
 * Chord display component
 */
class ChordDisplay : public juce::Component {
public:
    ChordDisplay();
    ~ChordDisplay() override = default;

    /**
     * Set current chord
     */
    void setChord(const juce::String& chordName, const std::vector<int>& notes);

    /**
     * Clear display
     */
    void clear();

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    juce::String chordName_;
    std::vector<int> chordNotes_;
    
    void drawChordName(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    void drawChordNotes(juce::Graphics& g, const juce::Rectangle<int>& bounds);
    juce::String noteNumberToName(int noteNumber) const;
};

} // namespace kelly
