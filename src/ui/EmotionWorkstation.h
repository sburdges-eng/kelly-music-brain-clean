#pragma once

#include "ChordDisplay.h"
#include "EmotionRadar.h"
#include "EmotionWheel.h"
#include "GenerateButton.h"
#include "KellyLookAndFeel.h"
#include "LyricDisplay.h"
#include "MusicTheoryPanel.h"
#include "PianoRollPreview.h"
#include "VocalControlPanel.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

/**
 * EmotionWorkstation - Unified emotional MIDI generation interface
 *
 * Replaces CassetteView + SidePanel with direct, functional layout.
 * Removes Side A/B metaphor while preserving all functionality.
 *
 * Layout:
 * ┌─────────────────────────────────────────────────┐
 * │ WOUND INPUT                                     │
 * │ [TextEditor: "Describe what you're feeling..."] │
 * ├─────────────────────┬───────────────────────────┤
 * │ EMOTION MAPPING     │ MUSICAL PARAMETERS        │
 * │                     │                           │
 * │ [EmotionWheel]      │ Valence    [-1.0 ━━● 1.0] │
 * │ [EmotionRadar]      │ Arousal    [0.0 ━━━● 1.0] │
 * │                     │ Intensity  [0.0 ━━━● 1.0] │
 * │                     │ Complexity [0.0 ━━━● 1.0] │
 * │                     │ Humanize   [0.0 ━━━● 1.0] │
 * │                     │ Feel       [-1.0 ━━● 1.0] │
 * │                     │ Dynamics   [0.0 ━━━● 1.0] │
 * │                     │ Bars       [4 ━━━━━● 32]  │
 * ├─────────────────────┴───────────────────────────┤
 * │ [ChordDisplay] [MusicTheoryPanel]               │
 * ├─────────────────────────────────────────────────┤
 * │ [LyricDisplay]                                  │
 * ├─────────────────────────────────────────────────┤
 * │ [VocalControlPanel]                             │
 * ├─────────────────────────────────────────────────┤
 * │ [PianoRollPreview]                              │
 * ├─────────────────────────────────────────────────┤
 * │ [Generate] [Preview] [Export to DAW] [Bypass]  │
 * └─────────────────────────────────────────────────┘
 */
class EmotionWorkstation : public juce::Component, public juce::Timer {
public:
  explicit EmotionWorkstation(juce::AudioProcessorValueTreeState &apvts);
  ~EmotionWorkstation() override = default;

  void paint(juce::Graphics &g) override;
  void resized() override;
  void timerCallback() override;

  // Access to wound input for processing
  juce::String getWoundText() const { return woundInput_.getText(); }
  void setWoundText(const juce::String &text) { woundInput_.setText(text); }

  // Access to emotion visualizations
  EmotionWheel &getEmotionWheel() { return emotionWheel_; }
  EmotionRadar &getEmotionRadar() { return emotionRadar_; }

  // Access to display components
  ChordDisplay &getChordDisplay() { return chordDisplay_; }
  MusicTheoryPanel &getMusicTheoryPanel() { return musicTheoryPanel_; }
  PianoRollPreview &getPianoRollPreview() { return pianoRollPreview_; }
  LyricDisplay &getLyricDisplay() { return lyricDisplay_; }
  VocalControlPanel &getVocalControlPanel() { return vocalControlPanel_; }

  // Access to action buttons
  GenerateButton &getGenerateButton() { return generateButton_; }

  // Access to APVTS
  juce::AudioProcessorValueTreeState &getAPVTS() { return apvts_; }

  // Callbacks
  std::function<void()> onGenerateClicked;
  std::function<void()> onPreviewClicked;
  std::function<void()> onExportClicked;
  std::function<void(const EmotionNode &emotion)> onEmotionSelected;

  // Project menu callbacks
  std::function<void()> onNewProject;
  std::function<void()> onOpenProject;
  std::function<void()> onSaveProject;
  std::function<void()> onSaveProjectAs;

private:
  juce::AudioProcessorValueTreeState &apvts_;

  // ========================================================================
  // WOUND INPUT
  // ========================================================================
  juce::TextEditor woundInput_;
  juce::Label woundLabel_{"", "What's on your heart?"};

  // ========================================================================
  // EMOTION MAPPING (LEFT PANEL)
  // ========================================================================
  EmotionWheel emotionWheel_;
  EmotionRadar emotionRadar_;

  // ========================================================================
  // MUSICAL PARAMETERS (RIGHT PANEL) - ALL 9 APVTS PARAMETERS
  // ========================================================================
  juce::Slider valenceSlider_;
  juce::Slider arousalSlider_;
  juce::Slider intensitySlider_;
  juce::Slider complexitySlider_;
  juce::Slider humanizeSlider_;
  juce::Slider feelSlider_;
  juce::Slider dynamicsSlider_;
  juce::Slider barsSlider_;
  juce::ToggleButton bypassButton_{"Bypass"};

  // Labels for accessibility
  juce::Label valenceLabel_{"", "Valence"};
  juce::Label arousalLabel_{"", "Arousal"};
  juce::Label intensityLabel_{"", "Intensity"};
  juce::Label complexityLabel_{"", "Complexity"};
  juce::Label humanizeLabel_{"", "Humanize"};
  juce::Label feelLabel_{"", "Feel"};
  juce::Label dynamicsLabel_{"", "Dynamics"};
  juce::Label barsLabel_{"", "Bars"};

  // APVTS attachments
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      valenceAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      arousalAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      intensityAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      complexityAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      humanizeAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      feelAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      dynamicsAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>
      barsAttachment_;
  std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment>
      bypassAttachment_;

  // ========================================================================
  // DISPLAY COMPONENTS
  // ========================================================================
  ChordDisplay chordDisplay_;
  MusicTheoryPanel musicTheoryPanel_;
  PianoRollPreview pianoRollPreview_;
  LyricDisplay lyricDisplay_;
  VocalControlPanel vocalControlPanel_;

  // ========================================================================
  // PROJECT MENU
  // ========================================================================
  juce::TextButton projectMenuButton_{"Project"};
  juce::PopupMenu projectMenu_;

  // ========================================================================
  // ACTIONS
  // ========================================================================
  GenerateButton generateButton_;
  juce::TextButton previewButton_{"Preview"};
  juce::TextButton exportButton_{"Export to DAW"};

  // ========================================================================
  // STYLING
  // ========================================================================
  KellyLookAndFeel lookAndFeel_;

  // ========================================================================
  // HELPER METHODS
  // ========================================================================
  void setupComponents();
  void setupSlider(juce::Slider &slider, juce::Label &label,
                   const juce::String &labelText);
  void setupButton(juce::Button &button, const juce::String &tooltip);
  void handleEmotionWheelSelection(const EmotionNode &emotion);
  void setupProjectMenu();
  void showProjectMenu();

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(EmotionWorkstation)
};

} // namespace kelly
