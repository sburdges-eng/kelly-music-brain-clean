#include "InteractiveCustomizationPanel.h"
#include "KellyLookAndFeel.h"

namespace kelly {

InteractiveCustomizationPanel::InteractiveCustomizationPanel(
    EmotionWorkstation &workstation)
    : workstation_(workstation) {
  setupComponents();
}

void InteractiveCustomizationPanel::setupComponents() {
  // Setup MIDI editor
  midiEditor_.onMidiChanged = [this](const GeneratedMidi &midi) {
    if (onMidiEdited) {
      onMidiEdited(midi);
    }
  };
  addAndMakeVisible(midiEditor_);

  // Setup natural language editor
  naturalLanguageEditor_.onFeedbackSubmitted =
      [this](const juce::String &feedback) {
        if (onNaturalLanguageFeedback) {
          onNaturalLanguageFeedback(feedback);
        }
        // Note: Parameter changes would be applied via callback or separate
        // method For now, just forward the feedback
      };
  addAndMakeVisible(naturalLanguageEditor_);
}

void InteractiveCustomizationPanel::paint(juce::Graphics &g) {
  g.fillAll(juce::Colour(0xff1a1a1a));

  if (showPreferences_) {
    drawPreferenceOverlay(g);
  }

  if (showSuggestions_) {
    drawSuggestionOverlay(g);
  }
}

void InteractiveCustomizationPanel::resized() {
  auto bounds = getLocalBounds().reduced(5);

  // Layout based on view mode
  switch (viewMode_) {
  case ViewMode::Parameters:
    // Hide editing components, show only workstation
    midiEditor_.setVisible(false);
    naturalLanguageEditor_.setVisible(false);
    break;

  case ViewMode::Editing:
    // Show MIDI editor, hide natural language
    naturalLanguageEditor_.setVisible(false);
    midiEditor_.setBounds(bounds);
    break;

  case ViewMode::NaturalLanguage:
    // Show natural language, hide MIDI editor
    midiEditor_.setVisible(false);
    naturalLanguageEditor_.setBounds(bounds);
    break;

  case ViewMode::Combined:
  default:
    // Split: natural language at top, MIDI editor below
    auto nlBounds = bounds.removeFromTop(150);
    naturalLanguageEditor_.setBounds(nlBounds);
    bounds.removeFromTop(5);
    midiEditor_.setBounds(bounds);
    naturalLanguageEditor_.setVisible(true);
    midiEditor_.setVisible(true);
    break;
  }
}

void InteractiveCustomizationPanel::drawPreferenceOverlay(juce::Graphics &g) {
  // Draw preference visualization
  // - Highlight preferred emotion wheel regions
  // - Show parameter slider "sweet spots"
  // - Display learned editing patterns

  auto bounds = getLocalBounds();
  g.setColour(juce::Colour(0x40ffffff)); // Semi-transparent white

  // Draw preference regions on emotion wheel (if available)
  auto &emotionWheel = workstation_.getEmotionWheel();
  auto wheelBounds = emotionWheel.getBounds();
  if (wheelBounds.getWidth() > 0 && wheelBounds.getHeight() > 0) {
    // Draw preference highlights (simplified - would use actual preference
    // data)
    g.setColour(juce::Colour(
        0x3000ff00)); // Semi-transparent green for preferred regions
    g.fillEllipse(wheelBounds.reduced(20).toFloat());
  }

  // Draw parameter "sweet spots" on sliders (simplified visualization)
  // In full implementation, would use PreferenceTracker data to show preferred
  // ranges
  g.setColour(
      juce::Colour(0x40ffff00)); // Semi-transparent yellow for sweet spots
  // Would draw indicators on sliders showing preferred value ranges
}

void InteractiveCustomizationPanel::drawSuggestionOverlay(juce::Graphics &g) {
  // Draw suggestion overlay
  // - Show suggestions on emotion wheel
  // - Display suggestion cards
  // - Show confidence indicators

  auto bounds = getLocalBounds();

  // Draw suggestion indicators on emotion wheel
  auto &emotionWheel = workstation_.getEmotionWheel();

  // Note: SuggestionOverlay is not a member of EmotionWorkstation in current
  // version If suggestions are needed, they would be handled differently For
  // now, just draw emotion wheel without overlay
  if (false) { // Placeholder - would check for suggestions if available
               // Draw suggestion cards if available
  } else {
    // Draw placeholder suggestion indicators
    g.setColour(juce::Colour(0x40ff00ff)); // Semi-transparent magenta
    auto wheelBounds = emotionWheel.getBounds();
    if (wheelBounds.getWidth() > 0) {
      // Draw suggestion markers (simplified)
      g.fillEllipse(wheelBounds.getCentreX() - 5, wheelBounds.getCentreY() - 5,
                    10, 10);
    }
  }
}

} // namespace kelly
