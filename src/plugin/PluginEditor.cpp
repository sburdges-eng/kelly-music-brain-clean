#include "plugin/PluginEditor.h"
#include "common/MusicConstants.h"
#include "midi/MidiExporter.h"
#include "project/ProjectManager.h"
#include <cmath>
#include <juce_audio_basics/juce_audio_basics.h>

namespace kelly {
using namespace MusicConstants;

PluginEditor::PluginEditor(PluginProcessor &p)
    : AudioProcessorEditor(&p), processor_(p) {

  // ========================================================================
  // CREATE AND CONFIGURE EMOTION WORKSTATION
  // ========================================================================
  // EmotionWorkstation manages all UI components:
  // - EmotionWheel, EmotionRadar
  // - Parameter sliders (Valence, Arousal, Intensity, etc.)
  // - ChordDisplay, MusicTheoryPanel, PianoRollPreview
  // - Action buttons (Generate, Preview, Export, Bypass)
  // - Wound input text field
  workstation_ = std::make_unique<EmotionWorkstation>(processor_.getAPVTS());

  // ========================================================================
  // CONNECT EMOTION WHEEL TO PROCESSOR'S INTENT PIPELINE
  // ========================================================================
  // The thesaurus provides 216 emotion nodes for the wheel
  // Non-owning reference - lifetime guaranteed by IntentPipeline in processor
  workstation_->getEmotionWheel().setThesaurus(
      processor_.getIntentPipeline().thesaurus());

  // ========================================================================
  // SET UP USER ACTION CALLBACKS
  // ========================================================================
  // These callbacks connect workstation UI events to processor functionality
  workstation_->onGenerateClicked = [this]() { onGenerateClicked(); };
  workstation_->onPreviewClicked = [this]() { onPreviewClicked(); };
  workstation_->onExportClicked = [this]() { onExportClicked(); };
  workstation_->onEmotionSelected = [this](const EmotionNode &emotion) {
    onEmotionSelected(emotion);
  };

  // Project menu callbacks
  workstation_->onNewProject = [this]() { onNewProject(); };
  workstation_->onOpenProject = [this]() { onOpenProject(); };
  workstation_->onSaveProject = [this]() { onSaveProject(); };
  workstation_->onSaveProjectAs = [this]() { onSaveProjectAs(); };

  // Make workstation visible and add to editor
  addAndMakeVisible(*workstation_);

  // ========================================================================
  // WINDOW CONFIGURATION
  // ========================================================================
  // Set default window size (medium)
  setSize(800, 900);

  // Make window resizable with constraints
  // User requirement: 400x300, 600x450, 800x600 (but we need more height for
  // all controls)
  setResizable(true, true);
  setResizeLimits(600, 700,    // Minimum: 600x700
                  1200, 1400); // Maximum: 1200x1400

  // ========================================================================
  // ADD PARAMETER LISTENER FOR REAL-TIME UPDATES
  // ========================================================================
  // Listen to parameter changes to update displays in real-time
  processor_.getAPVTS().addParameterListener(PluginProcessor::PARAM_VALENCE,
                                             this);
  processor_.getAPVTS().addParameterListener(PluginProcessor::PARAM_AROUSAL,
                                             this);
  processor_.getAPVTS().addParameterListener(PluginProcessor::PARAM_INTENSITY,
                                             this);
  processor_.getAPVTS().addParameterListener(PluginProcessor::PARAM_COMPLEXITY,
                                             this);
  processor_.getAPVTS().addParameterListener(PluginProcessor::PARAM_HUMANIZE,
                                             this);
  processor_.getAPVTS().addParameterListener(PluginProcessor::PARAM_FEEL, this);
  processor_.getAPVTS().addParameterListener(PluginProcessor::PARAM_DYNAMICS,
                                             this);
  processor_.getAPVTS().addParameterListener(PluginProcessor::PARAM_BARS, this);

  // ========================================================================
  // INITIALIZE DISPLAYS WITH CURRENT STATE
  // ========================================================================
  // Check for existing generated MIDI and load into displays
  if (processor_.hasPendingMidi()) {
    const auto &generatedMidi = processor_.getGeneratedMidi();
    if (!generatedMidi.chords.empty() || !generatedMidi.melody.empty()) {
      workstation_->getPianoRollPreview().setMidiData(generatedMidi);
      if (!generatedMidi.chords.empty()) {
        const auto &firstChord = generatedMidi.chords[0];
        workstation_->getChordDisplay().setChord(juce::String(firstChord.name),
                                                 firstChord.pitches);
      }
    }
  }

  // Initialize EmotionRadar with current parameter values
  float valence = *processor_.getAPVTS().getRawParameterValue(
      PluginProcessor::PARAM_VALENCE);
  float arousal = *processor_.getAPVTS().getRawParameterValue(
      PluginProcessor::PARAM_AROUSAL);
  float intensity = *processor_.getAPVTS().getRawParameterValue(
      PluginProcessor::PARAM_INTENSITY);
  workstation_->getEmotionRadar().setEmotion(valence, arousal, intensity);

  // ========================================================================
  // INITIALIZE MUSIC THEORY PANEL FROM PROCESSOR STATE
  // ========================================================================
  // Load saved music theory settings if available
  auto theorySettings = processor_.getMusicTheorySettings();
  if (!theorySettings.key.isEmpty()) {
    workstation_->getMusicTheoryPanel().setKey(theorySettings.key);
  }
  if (!theorySettings.mode.isEmpty()) {
    workstation_->getMusicTheoryPanel().setMode(theorySettings.mode);
  }
  if (theorySettings.tempoBpm > 0) {
    workstation_->getMusicTheoryPanel().setTempo(theorySettings.tempoBpm);
  }

  // ========================================================================
  // START UPDATE TIMER
  // ========================================================================
  // Timer updates UI components for real-time visualization
  // 100ms = 10 FPS for general updates (workstation has its own 30ms timer for
  // animations)
  startTimer(100);

  lastPendingMidiState_ = processor_.hasPendingMidi();
}

PluginEditor::~PluginEditor() {
  // Stop timer before destruction
  stopTimer();

  // Remove parameter listeners
  processor_.getAPVTS().removeParameterListener(PluginProcessor::PARAM_VALENCE,
                                                this);
  processor_.getAPVTS().removeParameterListener(PluginProcessor::PARAM_AROUSAL,
                                                this);
  processor_.getAPVTS().removeParameterListener(
      PluginProcessor::PARAM_INTENSITY, this);
  processor_.getAPVTS().removeParameterListener(
      PluginProcessor::PARAM_COMPLEXITY, this);
  processor_.getAPVTS().removeParameterListener(PluginProcessor::PARAM_HUMANIZE,
                                                this);
  processor_.getAPVTS().removeParameterListener(PluginProcessor::PARAM_FEEL,
                                                this);
  processor_.getAPVTS().removeParameterListener(PluginProcessor::PARAM_DYNAMICS,
                                                this);
  processor_.getAPVTS().removeParameterListener(PluginProcessor::PARAM_BARS,
                                                this);

  // Workstation will be automatically destroyed by unique_ptr
  // Remove it from component tree first
  removeChildComponent(workstation_.get());
  workstation_ = nullptr;
}

void PluginEditor::paint(juce::Graphics &g) {
  // Background is handled by EmotionWorkstation
  // This is a fallback in case workstation doesn't cover entire area
  g.fillAll(juce::Colour(0xFF1A1A1A));
}

void PluginEditor::resized() {
  // Give entire editor area to workstation
  // Workstation handles all internal layout of its components
  if (workstation_ != nullptr) {
    workstation_->setBounds(getLocalBounds());
  }
}

void PluginEditor::timerCallback() {
  // Update workstation for real-time visualization
  // Workstation has its own timerCallback for internal animations
  if (workstation_ == nullptr) {
    return;
  }

  workstation_->repaint();

  // ========================================================================
  // CHECK FOR NEW MIDI AND UPDATE DISPLAYS
  // ========================================================================
  bool currentPendingMidi = processor_.hasPendingMidi();
  if (currentPendingMidi != lastPendingMidiState_) {
    lastPendingMidiState_ = currentPendingMidi;

    if (currentPendingMidi) {
      // New MIDI is ready - update displays
      const auto &generatedMidi = processor_.getGeneratedMidi();
      workstation_->getPianoRollPreview().setMidiData(generatedMidi);

      // Update chord display with first chord or clear if empty
      if (!generatedMidi.chords.empty()) {
        const auto &firstChord = generatedMidi.chords[0];
        workstation_->getChordDisplay().setChord(juce::String(firstChord.name),
                                                 firstChord.pitches);
      } else {
        workstation_->getChordDisplay().clear();
      }

      // Force repaint of displays to show new data
      workstation_->getPianoRollPreview().repaint();
      workstation_->getChordDisplay().repaint();
    }
  }

  // ========================================================================
  // MONITOR GENERATION STATUS
  // ========================================================================
  // Check if generation has completed and update button state
  auto &generateButton = workstation_->getGenerateButton();
  if (processor_.isGenerating()) {
    // Generation in progress - ensure button shows loading state
    if (!generateButton.isEnabled()) {
      // Button already disabled and animating - good
    } else {
      // Button should be disabled during generation
      generateButton.setEnabled(false);
      generateButton.startGenerateAnimation();
    }
  } else {
    // Generation not in progress - ensure button is enabled
    if (!generateButton.isEnabled() && !generateButton.isDown()) {
      // Re-enable button if it was disabled for generation
      generateButton.setEnabled(true);
      generateButton.stopGenerateAnimation();
    }
  }

  // ========================================================================
  // PARAMETER CHANGE INDICATOR
  // ========================================================================
  // Check if parameters have changed and show visual feedback
  // This provides indication that regeneration is needed (manual regeneration
  // approach)
  if (processor_.hasParametersChanged()) {
    // Update Generate button appearance to indicate regeneration needed
    // The button can highlight or show a different state
    // Note: If GenerateButton has a setNeedsRegeneration() method, call it here
    // Otherwise, the button's normal state is sufficient (user will click when
    // ready)
    generateButton.repaint(); // Ensure button is repainted
  }

  // ========================================================================
  // REAL-TIME EMOTION RADAR UPDATE
  // ========================================================================
  // Update EmotionRadar with current parameter values for real-time
  // visualization Note: This is also updated via parameterChanged() callback,
  // but we ensure it's always in sync here as a backup
  float valence = *processor_.getAPVTS().getRawParameterValue(
      PluginProcessor::PARAM_VALENCE);
  float arousal = *processor_.getAPVTS().getRawParameterValue(
      PluginProcessor::PARAM_AROUSAL);
  float intensity = *processor_.getAPVTS().getRawParameterValue(
      PluginProcessor::PARAM_INTENSITY);
  workstation_->getEmotionRadar().setEmotion(valence, arousal, intensity);
}

void PluginEditor::onGenerateClicked() {
  // ========================================================================
  // CHECK IF ALREADY GENERATING
  // ========================================================================
  // Prevent multiple simultaneous generation requests
  if (processor_.isGenerating()) {
    juce::AlertWindow::showMessageBoxAsync(
        juce::AlertWindow::InfoIcon, "Generation In Progress",
        "MIDI generation is already in progress. Please wait for it to "
        "complete.",
        "OK");
    return;
  }

  // ========================================================================
  // SHOW GENERATION STATUS
  // ========================================================================
  // Start animation to show generation is in progress
  auto &generateButton = workstation_->getGenerateButton();
  generateButton.startGenerateAnimation();
  generateButton.setEnabled(false);

  // ========================================================================
  // GET USER INPUT AND VALIDATE
  // ========================================================================
  // Retrieve wound text from workstation's text input field
  juce::String woundText = workstation_->getWoundText().trim();

  // Validate wound text (optional - allow empty for parameter-only generation)
  // If wound text is provided, check it's not too long
  if (woundText.isNotEmpty() && woundText.length() > 1000) {
    generateButton.stopGenerateAnimation();
    generateButton.setEnabled(true);
    juce::AlertWindow::showMessageBoxAsync(
        juce::AlertWindow::WarningIcon, "Invalid Input",
        "Wound text is too long. Please limit to 1000 characters.", "OK");
    return;
  }

  // ========================================================================
  // GET MUSIC THEORY PANEL SETTINGS
  // ========================================================================
  // Read settings from MusicTheoryPanel to pass to generation
  auto &theoryPanel = workstation_->getMusicTheoryPanel();
  auto theorySettings = theoryPanel.getSettings();

  // Pass key, mode, and tempo to processor
  // These will override emotion-derived parameters if set
  processor_.setMusicTheorySettings(theorySettings.key, theorySettings.mode,
                                    theorySettings.tempoBpm);

  // ========================================================================
  // PROCESS THROUGH INTENT PIPELINE
  // ========================================================================
  // Set the wound description on the processor
  // This will be processed through IntentPipeline to extract emotion
  processor_.setWoundDescription(woundText);

  // If wound text was provided, process it to update emotion parameters
  // This provides immediate feedback that the wound was processed
  if (woundText.isNotEmpty()) {
    // Process wound through IntentPipeline to get emotion
    // This is done synchronously here for immediate feedback
    // The actual generation will use the same processing
    try {
      Wound wound;
      wound.description = woundText.toStdString();
      wound.intensity = *processor_.getAPVTS().getRawParameterValue(
          PluginProcessor::PARAM_INTENSITY);
      wound.source = "user_input";

      // Process wound to get emotion (this updates parameters)
      // Note: This is a preview - actual generation will use the same
      // processing
      auto &intentPipeline = processor_.getIntentPipeline();
      IntentResult previewIntent = intentPipeline.process(wound);

      // Update EmotionRadar with processed emotion for immediate feedback
      workstation_->getEmotionRadar().setEmotion(
          previewIntent.emotion.valence, previewIntent.emotion.arousal,
          previewIntent.emotion.intensity);
      workstation_->getEmotionRadar().repaint();
    } catch (const std::exception &e) {
      // If processing fails, continue with generation anyway
      // The error will be caught during actual generation
      juce::ignoreUnused(e);
    }
  }

  // ========================================================================
  // GENERATE MIDI
  // ========================================================================
  // Processor will:
  // 1. Process wound through IntentPipeline to get emotion
  // 2. Use current parameter values (from APVTS sliders)
  // 3. Generate MIDI via MidiGenerator
  try {
    processor_.generateMidi();

    // Check if generation completed successfully
    // Note: generateMidi() is synchronous, so it should complete immediately
    // But we check isGenerating() to ensure it finished
    int checkCount = 0;
    const int maxChecks =
        100; // Safety limit (shouldn't be needed for sync generation)
    while (processor_.isGenerating() && checkCount < maxChecks) {
      juce::Thread::getCurrentThread()->sleep(10); // Small delay
      checkCount++;
    }

    // Update displays immediately if MIDI is ready
    if (processor_.hasPendingMidi()) {
      const auto &generatedMidi = processor_.getGeneratedMidi();

      // Update piano roll preview with generated MIDI
      workstation_->getPianoRollPreview().setMidiData(generatedMidi);
      workstation_->getPianoRollPreview().repaint();

      // Update chord display if chords are available
      if (!generatedMidi.chords.empty()) {
        const auto &firstChord = generatedMidi.chords[0];
        workstation_->getChordDisplay().setChord(juce::String(firstChord.name),
                                                 firstChord.pitches);
      } else {
        // Clear chord display if no chords
        workstation_->getChordDisplay().clear();
      }
      workstation_->getChordDisplay().repaint();

      // Update EmotionRadar with processed emotion if available
      // The emotion should have been processed through IntentPipeline
      float valence = *processor_.getAPVTS().getRawParameterValue(
          PluginProcessor::PARAM_VALENCE);
      float arousal = *processor_.getAPVTS().getRawParameterValue(
          PluginProcessor::PARAM_AROUSAL);
      float intensity = *processor_.getAPVTS().getRawParameterValue(
          PluginProcessor::PARAM_INTENSITY);
      workstation_->getEmotionRadar().setEmotion(valence, arousal, intensity);
      workstation_->getEmotionRadar().repaint();

      // Update MusicTheoryPanel to reflect the generated MIDI's key/mode/tempo
      // This provides feedback that the settings were used
      auto &theoryPanel = workstation_->getMusicTheoryPanel();
      auto currentSettings = processor_.getMusicTheorySettings();
      if (!currentSettings.key.isEmpty()) {
        theoryPanel.setKey(currentSettings.key);
      }
      if (!currentSettings.mode.isEmpty()) {
        theoryPanel.setMode(currentSettings.mode);
      }
      if (currentSettings.tempoBpm > 0) {
        theoryPanel.setTempo(currentSettings.tempoBpm);
      }
    } else {
      // Generation completed but no MIDI was generated
      // This might indicate an error or empty result
      juce::AlertWindow::showMessageBoxAsync(
          juce::AlertWindow::WarningIcon, "Generation Complete",
          "MIDI generation completed, but no MIDI data was produced.\n"
          "This may indicate an issue with the generation parameters.\n\n"
          "Try adjusting:\n"
          "- Valence, Arousal, or Intensity sliders\n"
          "- Complexity or Humanize parameters\n"
          "- Music Theory Panel settings (key, mode, tempo)",
          "OK");
    }
  } catch (const std::exception &e) {
    // Error handling - show user feedback
    juce::AlertWindow::showMessageBoxAsync(
        juce::AlertWindow::WarningIcon, "Generation Failed",
        "An error occurred while generating MIDI:\n" + juce::String(e.what()),
        "OK");
  } catch (...) {
    // Catch-all for unknown errors
    juce::AlertWindow::showMessageBoxAsync(
        juce::AlertWindow::WarningIcon, "Generation Failed",
        "An unknown error occurred while generating MIDI.", "OK");
  }

  // ========================================================================
  // RESET GENERATION STATUS
  // ========================================================================
  // Stop animation and re-enable button
  generateButton.stopGenerateAnimation();
  generateButton.setEnabled(true);
}

void PluginEditor::onPreviewClicked() {
  // ========================================================================
  // PREVIEW GENERATED MIDI
  // ========================================================================
  // Preview functionality - play back generated MIDI
  // In standalone mode: Could trigger internal playback
  // In plugin mode: DAW handles playback automatically

  // Refresh the piano roll preview with current generated MIDI
  const auto &generatedMidi = processor_.getGeneratedMidi();
  workstation_->getPianoRollPreview().setMidiData(generatedMidi);

  // Reset playhead to beginning
  workstation_->getPianoRollPreview().setPlayheadPosition(0.0f);
}

void PluginEditor::onExportClicked() {
  // ========================================================================
  // EXPORT MIDI TO DAW OR FILE
  // ========================================================================
  // In plugin mode: MIDI flows through automatically via processBlock()
  // In standalone mode: Save to a MIDI file

  // Check if we're in standalone mode
  bool isStandalone =
      (processor_.wrapperType == juce::AudioProcessor::wrapperType_Standalone);

  if (isStandalone) {
    // ====================================================================
    // STANDALONE MODE: SAVE TO MIDI FILE
    // ====================================================================

    // Ensure MIDI is generated before export
    processor_.generateMidi();

    // Get the generated MIDI
    const auto &generatedMidi = processor_.getGeneratedMidi();

    // Validate that we have MIDI data to export
    if (generatedMidi.chords.empty() && generatedMidi.melody.empty() &&
        generatedMidi.bass.empty()) {
      juce::AlertWindow::showMessageBoxAsync(
          juce::AlertWindow::WarningIcon, "No MIDI to Export",
          "Please generate MIDI first by clicking the Generate button.", "OK");
      return;
    }

    // Create file chooser for saving MIDI file
    auto chooser = std::make_shared<juce::FileChooser>(
        "Save MIDI File",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
        "*.mid");

    // Launch async file chooser
    chooser->launchAsync(
        juce::FileBrowserComponent::saveMode |
            juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser &fc) {
          auto file = fc.getResult();
          if (file != juce::File{}) {
            // Ensure file has .mid extension
            if (!file.hasFileExtension("mid")) {
              file = file.withFileExtension("mid");
            }

            // Use MidiExporter for comprehensive export
            using namespace midikompanion;
            MidiExporter exporter;

            // Configure export options
            MidiExporter::ExportOptions options;
            options.format =
                MidiExporter::Format::SMF_Type1; // Multi-track format
            options.includeVocals = false;       // No vocals in v1.0
            options.includeLyrics = false;       // No lyrics in v1.0
            options.includeExpression = true;    // Include CC events
            options.ticksPerQuarterNote = 960;   // Standard MIDI resolution

            // Export MIDI file
            bool success = exporter.exportToFile(
                file, processor_.getGeneratedMidi(), options);

            if (success) {
              juce::AlertWindow::showMessageBoxAsync(
                  juce::AlertWindow::InfoIcon, "Export Complete",
                  "MIDI file saved successfully:\n" + file.getFullPathName(),
                  "OK");
            } else {
              juce::String errorMsg = exporter.getLastError();
              if (errorMsg.isEmpty()) {
                errorMsg = "Unknown error occurred";
              }
              juce::AlertWindow::showMessageBoxAsync(
                  juce::AlertWindow::WarningIcon, "Export Failed",
                  "Could not save MIDI file:\n" + file.getFullPathName() +
                      "\n\n" + errorMsg,
                  "OK");
            }
          }
        });
  } else {
    // ====================================================================
    // PLUGIN MODE: MIDI FLOWS THROUGH AUTOMATICALLY
    // ====================================================================
    // In plugin mode, MIDI is automatically sent to DAW via processBlock()
    // Generate MIDI to ensure it's ready
    processor_.generateMidi();

    // Inform user that MIDI is available in DAW
    juce::AlertWindow::showMessageBoxAsync(
        juce::AlertWindow::InfoIcon, "Export to DAW",
        "MIDI is automatically sent to your DAW.\n"
        "The generated MIDI will be available in your DAW's MIDI track.",
        "OK");
  }
}

void PluginEditor::onEmotionSelected(const EmotionNode &emotion) {
  // ========================================================================
  // EMOTION SELECTED FROM WHEEL
  // ========================================================================
  // Note: EmotionWorkstation already handles:
  // - Updating parameter sliders (valence, arousal, intensity)
  // - Updating emotion radar visualization
  //
  // Here we update processor and additional UI elements based on emotion
  // selection

  // ========================================================================
  // UPDATE PROCESSOR WITH SELECTED EMOTION ID
  // ========================================================================
  // This ensures generateMidi() uses the exact emotion from thesaurus
  // rather than finding nearest by VAD coordinates
  processor_.setSelectedEmotionId(emotion.id);

  // ========================================================================
  // UPDATE MUSIC THEORY PANEL
  // ========================================================================
  auto &theoryPanel = workstation_->getMusicTheoryPanel();

  // Get thesaurus for emotion-based suggestions
  const auto &thesaurus = processor_.getIntentPipeline().thesaurus();

  // Use EmotionThesaurus to suggest mode (more accurate than simple valence
  // check)
  std::string suggestedMode = thesaurus.suggestMode(emotion);
  theoryPanel.setMode(juce::String(suggestedMode));

  // Set key based on emotion characteristics
  // Negative valence: flat keys (Eb, Bb, F) for darker emotions
  // Positive valence: sharp keys (G, D, A) for brighter emotions
  // High intensity: more extreme keys
  juce::String suggestedKey;
  if (emotion.valence < VALENCE_NEGATIVE) {
    // Very negative: darkest keys
    if (emotion.intensity > 0.7f) {
      suggestedKey = "Eb"; // Darkest
    } else {
      suggestedKey = "Bb"; // Dark but not extreme
    }
  } else if (emotion.valence < VALENCE_NEUTRAL) {
    // Slightly negative: minor keys
    suggestedKey = emotion.intensity > 0.6f ? "Dm" : "Am";
  } else if (emotion.valence < VALENCE_POSITIVE) {
    // Neutral: C major/minor
    suggestedKey = "C";
  } else if (emotion.valence < VALENCE_VERY_POSITIVE) {
    // Positive: bright keys
    suggestedKey = emotion.intensity > 0.6f ? "G" : "C";
  } else {
    // Very positive: brightest keys
    suggestedKey = emotion.intensity > 0.7f ? "A" : "D";
  }
  theoryPanel.setKey(suggestedKey);

  // Adjust tempo using EmotionThesaurus tempo modifier
  // This provides more nuanced tempo suggestions than simple arousal mapping
  float tempoModifier = thesaurus.suggestTempoModifier(emotion);
  int baseTempo = TEMPO_MODERATE;
  int newTempo = static_cast<int>(baseTempo * tempoModifier);
  newTempo = std::clamp(newTempo, TEMPO_MIN, TEMPO_MAX);
  theoryPanel.setTempo(newTempo);

  // Instrument selection based on emotion category
  // Note: MusicTheoryPanel doesn't expose instrument setters directly,
  // but we can suggest appropriate instruments based on emotion category:
  // - Joy/Surprise: Bright instruments (Piano, Flute, Trumpet)
  // - Sadness: Warm, mellow instruments (Cello, Viola, Pad Warm)
  // - Anger/Fear: Aggressive instruments (Distortion Guitar, Brass, Synth Bass)
}

//==============================================================================
// Project Menu Handlers (Phase 0: v1.0 Critical Features)
//==============================================================================

void PluginEditor::onNewProject() {
  // Reset to default state
  currentProjectFile_ = juce::File();

  // Clear wound text
  workstation_->setWoundText("");

  // Reset parameters to defaults (can be done via APVTS)
  // Note: This is a simple reset - full reset would restore all defaults
  juce::AlertWindow::showMessageBoxAsync(
      juce::AlertWindow::InfoIcon, "New Project",
      "Project reset. You can now start a new project.", "OK");
}

void PluginEditor::onOpenProject() {
  using namespace midikompanion;

  auto chooser = std::make_shared<juce::FileChooser>(
      "Open Project",
      juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
      midikompanion::ProjectManager::getProjectFilePattern());

  chooser->launchAsync(
      juce::FileBrowserComponent::openMode |
          juce::FileBrowserComponent::canSelectFiles,
      [this, chooser](const juce::FileChooser &fc) {
        auto file = fc.getResult();
        if (file != juce::File{}) {
          bool success = processor_.loadProject(file);
          if (success) {
            currentProjectFile_ = file;

            // Update UI with loaded state
            workstation_->setWoundText(
                processor_.getPluginState()
                    .saveState(processor_.getAPVTS(),
                               processor_.getPluginState()
                                   .saveState(processor_.getAPVTS(), "",
                                              std::nullopt, {})
                                   .getProperty("woundDescription")
                                   .toString(),
                               std::nullopt, {})
                    .getProperty("woundDescription")
                    .toString());

            // Refresh displays
            if (processor_.hasPendingMidi()) {
              const auto &generatedMidi = processor_.getGeneratedMidi();
              workstation_->getPianoRollPreview().setMidiData(generatedMidi);
            }

            juce::AlertWindow::showMessageBoxAsync(
                juce::AlertWindow::InfoIcon, "Project Loaded",
                "Project loaded successfully:\n" + file.getFileName(), "OK");
          } else {
            juce::String errorMsg = processor_.getProjectError();
            if (errorMsg.isEmpty()) {
              errorMsg = "Unknown error occurred";
            }
            juce::AlertWindow::showMessageBoxAsync(
                juce::AlertWindow::WarningIcon, "Load Failed",
                "Could not load project:\n" + file.getFullPathName() + "\n\n" +
                    errorMsg,
                "OK");
          }
        }
      });
}

void PluginEditor::onSaveProject() {
  if (currentProjectFile_.existsAsFile()) {
    // Save to current file
    bool success = processor_.saveCurrentProject(currentProjectFile_);
    if (success) {
      juce::AlertWindow::showMessageBoxAsync(
          juce::AlertWindow::InfoIcon, "Project Saved",
          "Project saved successfully:\n" + currentProjectFile_.getFileName(),
          "OK");
    } else {
      juce::String errorMsg = processor_.getProjectError();
      if (errorMsg.isEmpty()) {
        errorMsg = "Unknown error occurred";
      }
      juce::AlertWindow::showMessageBoxAsync(
          juce::AlertWindow::WarningIcon, "Save Failed",
          "Could not save project:\n" + currentProjectFile_.getFullPathName() +
              "\n\n" + errorMsg,
          "OK");
    }
  } else {
    // No current file, use Save As
    onSaveProjectAs();
  }
}

void PluginEditor::onSaveProjectAs() {
  using namespace midikompanion;

  auto chooser = std::make_shared<juce::FileChooser>(
      "Save Project As",
      juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
      midikompanion::ProjectManager::getProjectFilePattern());

  chooser->launchAsync(
      juce::FileBrowserComponent::saveMode |
          juce::FileBrowserComponent::canSelectFiles,
      [this, chooser](const juce::FileChooser &fc) {
        auto file = fc.getResult();
        if (file != juce::File{}) {
          // Ensure file has correct extension
          if (!file.hasFileExtension(
                  midikompanion::ProjectManager::getProjectFileExtension()
                      .substring(1))) {
            file = file.withFileExtension(
                midikompanion::ProjectManager::getProjectFileExtension());
          }

          bool success = processor_.saveCurrentProject(file);
          if (success) {
            currentProjectFile_ = file;
            juce::AlertWindow::showMessageBoxAsync(
                juce::AlertWindow::InfoIcon, "Project Saved",
                "Project saved successfully:\n" + file.getFullPathName(), "OK");
          } else {
            juce::String errorMsg = processor_.getProjectError();
            if (errorMsg.isEmpty()) {
              errorMsg = "Unknown error occurred";
            }
            juce::AlertWindow::showMessageBoxAsync(
                juce::AlertWindow::WarningIcon, "Save Failed",
                "Could not save project:\n" + file.getFullPathName() + "\n\n" +
                    errorMsg,
                "OK");
          }
        }
      });
  // - Trust/Anticipation: Balanced instruments (String Ensemble, Acoustic
  // Guitar)
  // - Disgust: Harsh, edgy instruments (Synth Brass, Overdriven Guitar)
  //
  // For now, we update mode/key/tempo which are the primary musical parameters.
  // Instrument selection could be added via MusicTheoryPanel setter methods if
  // needed.
}

void PluginEditor::parameterChanged(const juce::String &parameterID,
                                    float newValue) {
  // Called from message thread when parameters change (including automation)
  // Update displays in real-time based on parameter changes

  if (workstation_ == nullptr) {
    return;
  }

  // ========================================================================
  // UPDATE EMOTION RADAR FOR VAD PARAMETERS
  // ========================================================================
  // Update EmotionRadar when VAD parameters change for real-time visualization
  if (parameterID == PluginProcessor::PARAM_VALENCE ||
      parameterID == PluginProcessor::PARAM_AROUSAL ||
      parameterID == PluginProcessor::PARAM_INTENSITY) {

    float valence = *processor_.getAPVTS().getRawParameterValue(
        PluginProcessor::PARAM_VALENCE);
    float arousal = *processor_.getAPVTS().getRawParameterValue(
        PluginProcessor::PARAM_AROUSAL);
    float intensity = *processor_.getAPVTS().getRawParameterValue(
        PluginProcessor::PARAM_INTENSITY);

    workstation_->getEmotionRadar().setEmotion(valence, arousal, intensity);
    workstation_->getEmotionRadar().repaint(); // Force immediate visual update

    // Also update EmotionWheel selection if it's close to a node
    // This provides visual feedback that parameters match an emotion
    auto &emotionWheel = workstation_->getEmotionWheel();
    const auto &thesaurus = processor_.getIntentPipeline().thesaurus();
    EmotionNode nearestEmotion =
        thesaurus.findNearest(valence, arousal, intensity);

    // If the nearest emotion is very close (within threshold), highlight it
    float distance =
        std::sqrt(std::pow(valence - nearestEmotion.valence, 2.0f) +
                  std::pow(arousal - nearestEmotion.arousal, 2.0f) +
                  std::pow(intensity - nearestEmotion.intensity, 2.0f));
    if (distance < EMOTION_WHEEL_AUTO_SELECT_THRESHOLD) {
      emotionWheel.setSelectedEmotion(nearestEmotion.id);
    }
  }

  // ========================================================================
  // HANDLE OTHER PARAMETER CHANGES
  // ========================================================================
  // Other parameter changes (complexity, humanize, feel, dynamics, bars)
  // don't directly affect the visual displays in real-time, but they will be
  // used in the next generation. The UI will show visual feedback that
  // regeneration is needed via hasParametersChanged() check in timerCallback().
  //
  // Note: PianoRollPreview and ChordDisplay are only updated when new MIDI
  // is generated, not when parameters change. This is intentional - we use
  // manual regeneration (user clicks Generate) rather than auto-regeneration.
  //
  // However, we can provide subtle visual feedback that parameters have changed
  // by ensuring the Generate button shows it needs to be clicked again.

  juce::ignoreUnused(newValue);
}

} // namespace kelly
