#pragma once
/*
 * VirtualKeyboard.h - Virtual Piano Keyboard Component
 * ===================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - UI Layer: Used by MusicTheoryWorkstation, LearningPanel
 * - MIDI Layer: Can output MIDI notes
 *
 * Purpose: Visual piano keyboard for interactive demonstrations
 *          and note/chord visualization.
 */

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>
#include <set>
#include <functional>

namespace kelly {

/**
 * VirtualKeyboard - Visual piano keyboard component
 *
 * Features:
 * - Visual piano keyboard (2-3 octaves)
 * - Click to play notes/chords
 * - Visual feedback for scale/chord patterns
 * - MIDI note output
 * - Highlight active notes from current scale/chord
 */
class VirtualKeyboard : public juce::Component {
public:
    VirtualKeyboard();
    ~VirtualKeyboard() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseUp(const juce::MouseEvent& e) override;

    /**
     * Highlight notes (for scale/chord visualization)
     */
    void highlightNotes(const std::vector<int>& midiNotes);

    /**
     * Clear highlights
     */
    void clearHighlights();

    /**
     * Set keyboard range (start note, number of octaves)
     */
    void setRange(int startNote, int numOctaves);

    // Callbacks
    std::function<void(int midiNote, bool isNoteOn)> onNotePlayed;

private:
    // Keyboard configuration
    int startNote_ = 60;  // C4
    int numOctaves_ = 2;
    std::set<int> highlightedNotes_;
    std::set<int> pressedNotes_;

    struct Key {
        int midiNote;
        bool isBlack;
        juce::Rectangle<float> bounds;
    };

    std::vector<Key> keys_;

    void setupKeys();
    int getMidiNoteAtPoint(juce::Point<float> point) const;
    void drawKey(juce::Graphics& g, const Key& key, bool isPressed, bool isHighlighted);
    juce::Rectangle<float> getKeyBounds(int midiNote) const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VirtualKeyboard)
};

} // namespace kelly
