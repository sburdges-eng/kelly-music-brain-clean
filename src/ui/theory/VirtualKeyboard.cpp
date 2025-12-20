#include "VirtualKeyboard.h"

namespace kelly {

VirtualKeyboard::VirtualKeyboard() {
    setRange(60, 2); // C4, 2 octaves
}

void VirtualKeyboard::paint(juce::Graphics& g) {
    g.fillAll(juce::Colour(0xff1a1a1a));

    // Draw keys
    for (const auto& key : keys_) {
        bool isPressed = pressedNotes_.find(key.midiNote) != pressedNotes_.end();
        bool isHighlighted = highlightedNotes_.find(key.midiNote) != highlightedNotes_.end();
        drawKey(g, key, isPressed, isHighlighted);
    }
}

void VirtualKeyboard::resized() {
    setupKeys();
}

void VirtualKeyboard::mouseDown(const juce::MouseEvent& e) {
    int note = getMidiNoteAtPoint(e.position);
    if (note >= 0) {
        pressedNotes_.insert(note);
        repaint();

        if (onNotePlayed) {
            onNotePlayed(note, true);
        }
    }
}

void VirtualKeyboard::mouseUp(const juce::MouseEvent& e) {
    int note = getMidiNoteAtPoint(e.position);
    if (note >= 0 && pressedNotes_.find(note) != pressedNotes_.end()) {
        pressedNotes_.erase(note);
        repaint();

        if (onNotePlayed) {
            onNotePlayed(note, false);
        }
    }
}

void VirtualKeyboard::highlightNotes(const std::vector<int>& midiNotes) {
    highlightedNotes_.clear();
    for (int note : midiNotes) {
        highlightedNotes_.insert(note);
    }
    repaint();
}

void VirtualKeyboard::clearHighlights() {
    highlightedNotes_.clear();
    repaint();
}

void VirtualKeyboard::setRange(int startNote, int numOctaves) {
    startNote_ = startNote;
    numOctaves_ = numOctaves;
    setupKeys();
    repaint();
}

void VirtualKeyboard::setupKeys() {
    keys_.clear();

    int endNote = startNote_ + (numOctaves_ * 12);
    float keyWidth = static_cast<float>(getWidth()) / (numOctaves_ * 7.0f);

    for (int note = startNote_; note < endNote; ++note) {
        int noteInOctave = note % 12;
        bool isBlack = (noteInOctave == 1 || noteInOctave == 3 ||
                       noteInOctave == 6 || noteInOctave == 8 || noteInOctave == 10);

        Key key;
        key.midiNote = note;
        key.isBlack = isBlack;
        key.bounds = getKeyBounds(note);
        keys_.push_back(key);
    }
}

int VirtualKeyboard::getMidiNoteAtPoint(juce::Point<float> point) const {
    for (const auto& key : keys_) {
        if (key.bounds.contains(point)) {
            return key.midiNote;
        }
    }
    return -1;
}

void VirtualKeyboard::drawKey(juce::Graphics& g, const Key& key, bool isPressed, bool isHighlighted) {
    juce::Colour keyColour;

    if (key.isBlack) {
        keyColour = isPressed ? juce::Colour(0xff333333) : juce::Colour(0xff000000);
        if (isHighlighted) {
            keyColour = keyColour.brighter(0.3f);
        }
    } else {
        keyColour = isPressed ? juce::Colour(0xffcccccc) : juce::Colours::white;
        if (isHighlighted) {
            keyColour = juce::Colour(0xffffcc00); // Yellow highlight
        }
    }

    g.setColour(keyColour);
    g.fillRoundedRectangle(key.bounds, 2.0f);

    g.setColour(juce::Colours::darkgrey);
    g.drawRoundedRectangle(key.bounds, 2.0f, 1.0f);
}

juce::Rectangle<float> VirtualKeyboard::getKeyBounds(int midiNote) const {
    int noteInOctave = midiNote % 12;
    int octave = (midiNote - startNote_) / 12;
    float keyWidth = static_cast<float>(getWidth()) / (numOctaves_ * 7.0f);
    float keyHeight = static_cast<float>(getHeight());

    // White key positions: C=0, D=1, E=2, F=3, G=4, A=5, B=6
    int whiteKeyPositions[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    int whiteKeyMap[] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6}; // MIDI note % 12 to white key

    if (noteInOctave == 1 || noteInOctave == 3 || noteInOctave == 6 ||
        noteInOctave == 8 || noteInOctave == 10) {
        // Black key
        float blackKeyWidth = keyWidth * 0.6f;
        float blackKeyHeight = keyHeight * 0.6f;
        float x = (octave * 7.0f + whiteKeyMap[noteInOctave] + 0.7f) * keyWidth;
        float y = 0.0f;
        return juce::Rectangle<float>(x, y, blackKeyWidth, blackKeyHeight);
    } else {
        // White key
        float x = (octave * 7.0f + whiteKeyMap[noteInOctave]) * keyWidth;
        float y = 0.0f;
        return juce::Rectangle<float>(x, y, keyWidth, keyHeight);
    }
}

} // namespace kelly
