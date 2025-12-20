/**
 * Piano Roll Preview Component
 *
 * Mini preview of generated MIDI notes in piano roll format.
 */

#pragma once

#include "../common/Types.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

namespace kelly {

/**
 * Piano roll preview component
 */
class PianoRollPreview : public juce::Component {
public:
  PianoRollPreview();
  ~PianoRollPreview() override = default;

  /**
   * Set MIDI data to display
   */
  void setMidiData(const GeneratedMidi &midi);

  /**
   * Clear display
   */
  void clear();

  /**
   * Set playhead position (0.0 to 1.0)
   */
  void setPlayheadPosition(float position);

  /**
   * Set zoom level (1.0 = normal, 2.0 = 2x zoom)
   */
  void setZoom(float zoom) {
    zoom_ = zoom;
    repaint();
  }

  /**
   * Set time range to display (in beats)
   */
  void setTimeRange(double startBeat, double endBeat);

  /**
   * Set pitch range to display
   */
  void setPitchRange(int minPitch, int maxPitch);

  /**
   * Get current time range
   */
  double getTimeStart() const { return timeStart_; }
  double getTimeEnd() const { return timeEnd_; }

  /**
   * Get current zoom level
   */
  float getZoom() const { return zoom_; }

  /**
   * Get current pitch range
   */
  int getPitchMin() const { return pitchMin_; }
  int getPitchMax() const { return pitchMax_; }

  void paint(juce::Graphics &g) override;
  void resized() override;

protected:
  // Protected members for derived classes (like MidiEditor)
  GeneratedMidi midiData_;
  float playheadPosition_ = 0.0f;
  float zoom_ = 1.0f;
  double timeStart_ = 0.0;
  double timeEnd_ = 16.0;
  int pitchMin_ = 36; // C2
  int pitchMax_ = 96; // C7

private:
  void drawGrid(juce::Graphics &g, const juce::Rectangle<int> &bounds);
  void drawNotes(juce::Graphics &g, const juce::Rectangle<int> &bounds);
  void drawPlayhead(juce::Graphics &g, const juce::Rectangle<int> &bounds);
  int pitchToY(int pitch, int height) const;
  double timeToX(double time, int width) const;
};

} // namespace kelly
