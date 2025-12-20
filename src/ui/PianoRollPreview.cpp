/**
 * Piano Roll Preview Implementation
 */

#include "PianoRollPreview.h"
#include "../ui/KellyLookAndFeel.h"
#include <algorithm>
#include <cmath>

namespace kelly {

PianoRollPreview::PianoRollPreview() {
    setOpaque(false);
}

void PianoRollPreview::setMidiData(const GeneratedMidi& midi) {
    midiData_ = midi;
    
    // Auto-calculate time range
    double maxTime = 0.0;
    for (const auto& track : {midi.melody, midi.bass}) {
        for (const auto& note : track) {
            double endTime = note.startBeat + note.duration;
            maxTime = std::max(maxTime, endTime);
        }
    }
    
    if (maxTime > 0.0) {
        timeEnd_ = maxTime;
    }
    
    // Auto-calculate pitch range
    int minPitch = 127;
    int maxPitch = 0;
    for (const auto& track : {midi.melody, midi.bass}) {
        for (const auto& note : track) {
            minPitch = std::min(minPitch, note.pitch);
            maxPitch = std::max(maxPitch, note.pitch);
        }
    }
    
    if (minPitch < maxPitch) {
        pitchMin_ = std::max(0, minPitch - 5);
        pitchMax_ = std::min(127, maxPitch + 5);
    }
    
    repaint();
}

void PianoRollPreview::clear() {
    midiData_ = GeneratedMidi{};
    repaint();
}

void PianoRollPreview::setPlayheadPosition(float position) {
    playheadPosition_ = juce::jlimit(0.0f, 1.0f, position);
    repaint();
}

void PianoRollPreview::setTimeRange(double startBeat, double endBeat) {
    timeStart_ = startBeat;
    timeEnd_ = endBeat;
    repaint();
}

void PianoRollPreview::setPitchRange(int minPitch, int maxPitch) {
    pitchMin_ = minPitch;
    pitchMax_ = maxPitch;
    repaint();
}

void PianoRollPreview::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds();
    
    // Background
    g.setColour(KellyLookAndFeel::surfaceColor);
    g.fillRect(bounds);
    
    // Draw grid
    drawGrid(g, bounds);
    
    // Draw notes
    drawNotes(g, bounds);
    
    // Draw playhead
    drawPlayhead(g, bounds);
}

void PianoRollPreview::drawGrid(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    g.setColour(KellyLookAndFeel::textSecondary.withAlpha(0.2f));
    
    // Vertical lines (beats)
    double timeRange = timeEnd_ - timeStart_;
    int numBeats = static_cast<int>(std::ceil(timeRange));
    for (int i = 0; i <= numBeats; ++i) {
        double beat = timeStart_ + i;
        float x = static_cast<float>(timeToX(beat, bounds.getWidth()));
        g.drawVerticalLine(static_cast<int>(x), bounds.getY(), bounds.getBottom());
    }
    
    // Horizontal lines (octaves)
    int pitchRange = pitchMax_ - pitchMin_;
    int numOctaves = pitchRange / 12;
    for (int i = 0; i <= numOctaves; ++i) {
        int pitch = pitchMin_ + (i * 12);
        float y = static_cast<float>(pitchToY(pitch, bounds.getHeight()));
        g.drawHorizontalLine(static_cast<int>(y), bounds.getX(), bounds.getRight());
    }
}

void PianoRollPreview::drawNotes(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    struct NoteDisplay {
        int pitch;
        double start;
        double duration;
        juce::Colour colour;
    };
    
    std::vector<NoteDisplay> notes;
    
    // Collect all notes
    juce::Colour trackColours[] = {
        juce::Colour(0xFF6366F1),  // Melody - Indigo
        juce::Colour(0xFFF472B6),  // Bass - Pink
        juce::Colour(0xFF22D3EE),  // Harmony - Cyan
        juce::Colour(0xFF10B981),  // Texture - Green
    };
    
    int trackIndex = 0;
    for (const auto& track : {midiData_.melody, midiData_.bass}) {
        for (const auto& note : track) {
            if (note.startBeat >= timeStart_ && note.startBeat <= timeEnd_) {
                notes.push_back({
                    note.pitch,
                    note.startBeat,
                    note.duration,
                    trackColours[trackIndex].withAlpha(0.7f)
                });
            }
        }
        trackIndex++;
    }
    
    // Draw notes
    double timeRange = timeEnd_ - timeStart_;
    int pitchRange = pitchMax_ - pitchMin_;
    float noteHeight = bounds.getHeight() / static_cast<float>(pitchRange);
    
    for (const auto& note : notes) {
        float x = static_cast<float>(timeToX(note.start, bounds.getWidth()));
        float width = static_cast<float>(note.duration / timeRange * bounds.getWidth() * zoom_);
        float y = static_cast<float>(pitchToY(note.pitch, bounds.getHeight()));
        
        // Draw note rectangle
        g.setColour(note.colour);
        g.fillRect(x, y - noteHeight, width, noteHeight);
        
        // Draw note border
        g.setColour(note.colour.brighter(0.3f));
        g.drawRect(x, y - noteHeight, width, noteHeight, 1.0f);
    }
}

void PianoRollPreview::drawPlayhead(juce::Graphics& g, const juce::Rectangle<int>& bounds) {
    if (playheadPosition_ <= 0.0f) return;
    
    float x = bounds.getX() + playheadPosition_ * bounds.getWidth();
    
    g.setColour(KellyLookAndFeel::accentColor);
    g.drawLine(x, bounds.getY(), x, bounds.getBottom(), 2.0f);
    
    // Playhead indicator
    g.fillEllipse(x - 4, bounds.getY() - 4, 8, 8);
    g.fillEllipse(x - 4, bounds.getBottom() - 4, 8, 8);
}

int PianoRollPreview::pitchToY(int pitch, int height) const {
    int pitchRange = pitchMax_ - pitchMin_;
    if (pitchRange == 0) return height / 2;
    
    float normalized = static_cast<float>(pitch - pitchMin_) / static_cast<float>(pitchRange);
    return height - static_cast<int>(normalized * height);
}

double PianoRollPreview::timeToX(double time, int width) const {
    double timeRange = timeEnd_ - timeStart_;
    if (timeRange <= 0.0) return 0.0;
    
    double normalized = (time - timeStart_) / timeRange;
    return normalized * width;
}

void PianoRollPreview::resized() {
    repaint();
}

} // namespace kelly
