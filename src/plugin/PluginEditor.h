#pragma once

#include "common/Types.h"
#include "plugin/PluginProcessor.h"
#include "ui/EmotionWorkstation.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

namespace kelly {

/**
 * PluginEditor - UI Entry Point
 *
 * CRITICAL: This is the main UI entry point for the plugin. It serves as the
 * bridge between the PluginProcessor (audio/MIDI processing) and all UI
 * components.
 *
 * Responsibilities:
 * 1. Connects to PluginProcessor - receives processor reference and accesses:
 *    - AudioProcessorValueTreeState (APVTS) for parameter management
 *    - IntentPipeline for emotion processing and thesaurus access
 *    - MIDI generation and export functionality
 *
 * 2. Manages all UI components through EmotionWorkstation:
 *    - EmotionWheel: 216-node emotion selector
 *    - EmotionRadar: VAD visualization
 *    - Parameter sliders: Valence, Arousal, Intensity, Complexity, Humanize,
 * Feel, Dynamics, Bars
 *    - ChordDisplay: Current chord visualization
 *    - MusicTheoryPanel: Key, mode, tempo, instrument settings
 *    - PianoRollPreview: Generated MIDI visualization
 *    - Action buttons: Generate, Preview, Export to DAW, Bypass
 *    - Wound input: Text field for emotional description
 *
 * 3. Handles user interactions:
 *    - Generate: Processes wound text, generates MIDI, updates displays
 *    - Preview: Plays back generated MIDI
 *    - Export: Saves MIDI to file (standalone) or sends to DAW (plugin mode)
 *    - Emotion selection: Updates parameters and music theory based on selected
 * emotion
 *
 * 4. Window management:
 *    - Resizable with constraints (600x700 to 1200x1400)
 *    - Timer-based updates for real-time visualization
 *
 * Architecture:
 *   PluginProcessor::createEditor() -> new PluginEditor(*this)
 *   PluginEditor -> EmotionWorkstation -> All UI Components
 */
class PluginEditor : public juce::AudioProcessorEditor,
                     public juce::Timer,
                     public juce::AudioProcessorValueTreeState::Listener {
public:
  /**
   * Constructor - Creates the editor and connects to processor
   * @param processor Reference to PluginProcessor (lifetime managed by JUCE
   * framework)
   */
  explicit PluginEditor(PluginProcessor &processor);

  /**
   * Destructor - Cleans up UI components and stops timers
   */
  ~PluginEditor() override;

  // JUCE Component overrides
  void paint(juce::Graphics &g) override;
  void resized() override;

  // Timer callback for real-time updates
  void timerCallback() override;

  // AudioProcessorValueTreeState::Listener
  void parameterChanged(const juce::String &parameterID,
                        float newValue) override;

private:
  // Reference to processor (non-owning - lifetime guaranteed by JUCE framework)
  PluginProcessor &processor_;

  // Main unified workstation interface - manages all UI components
  std::unique_ptr<EmotionWorkstation> workstation_;

  // Track last pending MIDI state to detect changes
  bool lastPendingMidiState_ = false;

  // ========================================================================
  // USER ACTION CALLBACKS
  // ========================================================================

  /**
   * Called when Generate button is clicked
   * - Retrieves wound text from workstation
   * - Processes through IntentPipeline
   * - Generates MIDI via processor
   * - Updates piano roll and chord displays
   */
  void onGenerateClicked();

  /**
   * Called when Preview button is clicked
   * - Refreshes piano roll preview
   * - Resets playhead position
   */
  void onPreviewClicked();

  /**
   * Called when Export button is clicked
   * - Standalone mode: Opens file chooser, saves MIDI file
   * - Plugin mode: MIDI flows through automatically, shows info dialog
   */
  void onExportClicked();

  /**
   * Called when emotion is selected from EmotionWheel
   * - Updates music theory panel (mode, tempo) based on emotion
   * - EmotionWorkstation already handles slider and radar updates
   * @param emotion The selected emotion node
   */
  void onEmotionSelected(const EmotionNode &emotion);

  /**
   * Project menu handlers (Phase 0: v1.0 Critical Features)
   */
  void onNewProject();
  void onOpenProject();
  void onSaveProject();
  void onSaveProjectAs();

  // Current project file (for Save vs Save As)
  juce::File currentProjectFile_;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};

} // namespace kelly
