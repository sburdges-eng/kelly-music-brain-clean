#pragma once

#include "engine/IntentPipeline.h"
#include <juce_audio_processors/juce_audio_processors.h>
// Note: KellyBrain.h is included via MLBridge.h (below)
// We don't include it directly to avoid type conflicts between Types.h and
// KellyTypes.h KellyBrain.h includes KellyTypes.h, while IntentPipeline.h
// includes Types.h
#include "common/Types.h"
#include "midi/MidiBuilder.h"
#include "midi/MidiGenerator.h"
#include "ml/InferenceThreadManager.h"
#include "ml/MLFeatureExtractor.h"
#include "ml/MultiModelProcessor.h"
#include "ml/PluginLatencyManager.h"
#include "plugin/PluginState.h"
// Forward declare KellyBrain and MLIntentPipeline to avoid type conflicts
// (KellyBrain.h includes KellyTypes.h, while PluginProcessor uses Types.h)
// Include these headers only in PluginProcessor.cpp where conversions are
// handled
namespace kelly {
class KellyBrain;
class MLIntentPipeline;
} // namespace kelly
using KellyBrain = kelly::KellyBrain;
using MLIntentPipeline = kelly::MLIntentPipeline;
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>

namespace kelly {

class PluginProcessor : public juce::AudioProcessor,
                        public juce::AudioProcessorValueTreeState::Listener,
                        public juce::AsyncUpdater,
                        public juce::Timer {
public:
  PluginProcessor();
  ~PluginProcessor() override;

  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override;

  bool isBusesLayoutSupported(const BusesLayout &layouts) const override;

  void processBlock(juce::AudioBuffer<float> &, juce::MidiBuffer &) override;

  juce::AudioProcessorEditor *createEditor() override;
  bool hasEditor() const override { return true; }

  const juce::String getName() const override { return JucePlugin_Name; }

  bool acceptsMidi() const override { return true; }
  bool producesMidi() const override { return true; }
  bool isMidiEffect() const override { return true; }
  double getTailLengthSeconds() const override { return 0.0; }

  int getNumPrograms() override { return 1; }
  int getCurrentProgram() override { return 0; }
  void setCurrentProgram(int) override {}
  const juce::String getProgramName(int) override { return {}; }
  void changeProgramName(int, const juce::String &) override {}

  void getStateInformation(juce::MemoryBlock &destData) override;
  void setStateInformation(const void *data, int sizeInBytes) override;

  // Parameter version (for JUCE ParameterID)
  static constexpr int PARAM_VERSION = 1;

  // Parameter IDs
  static constexpr const char *PARAM_VALENCE = "valence";
  static constexpr const char *PARAM_AROUSAL = "arousal";
  static constexpr const char *PARAM_INTENSITY = "intensity";
  static constexpr const char *PARAM_COMPLEXITY = "complexity";
  static constexpr const char *PARAM_HUMANIZE = "humanize";
  static constexpr const char *PARAM_FEEL = "feel";
  static constexpr const char *PARAM_DYNAMICS = "dynamics";
  static constexpr const char *PARAM_BARS = "bars";
  static constexpr const char *PARAM_BYPASS = "bypass";

  // Access APVTS
  juce::AudioProcessorValueTreeState &getAPVTS() { return apvts_; }

  // Access plugin state manager
  PluginState &getPluginState() { return pluginState_; }
  const PluginState &getPluginState() const { return pluginState_; }

  // Access intent pipeline for UI
  // NOTE: Returns non-owning reference. The thesaurus is thread-safe, but
  // process()/processJourney() should only be called from UI thread. For audio
  // thread access, use try_lock pattern. IntentPipeline owns EmotionThesaurus,
  // which is passed as non-owning pointer to VADSystem/VADCalculator. Lifetime
  // is guaranteed: IntentPipeline outlives all components that reference
  // thesaurus.
  IntentPipeline &getIntentPipeline() { return intentPipeline_; }
  const IntentPipeline &getIntentPipeline() const { return intentPipeline_; }

  // Generate MIDI from current parameters
  void generateMidi();

  // Get generated MIDI for export/drag
  // NOTE: Thread-safe - called from UI thread, can safely lock
  const GeneratedMidi &getGeneratedMidi() const {
    std::lock_guard<std::mutex> lock(midiMutex_);
    return generatedMidi_;
  }

  // Check if new MIDI is ready
  bool hasPendingMidi() const { return hasPendingMidi_.load(); }
  void clearPendingMidi() { hasPendingMidi_.store(false); }

  // Check if generation is in progress
  bool isGenerating() const { return isGenerating_.load(); }

  // Check if parameters have changed (for UI to show regeneration needed
  // indicator)
  bool hasParametersChanged() const { return parametersChanged_.load(); }

  // Get last processed emotion from wound text (for UI feedback)
  // Returns the emotion that was detected/used in the last generation
  std::optional<EmotionNode> getLastProcessedEmotion() const;

  // Set wound description from UI
  void setWoundDescription(const juce::String &description);

  // ML Inference methods
  void enableMLInference(bool enable);
  bool isMLInferenceEnabled() const { return mlInferenceEnabled_.load(); }

  // Set custom inference function (for RTNeural or other ML backends)
  // Note: InferenceFunction type removed - using callback interface instead
  // void setInferenceFunction(InferenceThreadManager::InferenceFunction fn);

  // Multi-Model ML access
  Kelly::ML::MultiModelProcessor &getMultiModelProcessor() {
    return multiModelProcessor_;
  }
  const Kelly::ML::MultiModelProcessor &getMultiModelProcessor() const {
    return multiModelProcessor_;
  }

  // Enable/disable individual ML models
  void setModelEnabled(Kelly::ML::ModelType type, bool enabled);
  bool isModelEnabled(Kelly::ML::ModelType type) const;

  // Set selected emotion ID from wheel
  void setSelectedEmotionId(int emotionId);

  // Set music theory settings from UI
  // These settings can override emotion-derived parameters
  void setMusicTheorySettings(const juce::String &key, const juce::String &mode,
                              int tempoBpm);

  // Get current music theory settings
  struct MusicTheorySettings {
    juce::String key = "C";
    juce::String mode = "Ionian";
    int tempoBpm = 120;
  };
  MusicTheorySettings getMusicTheorySettings() const;

  //==============================================================================
  // Project Save/Load (Phase 0: v1.0 Critical Features)
  //==============================================================================

  /**
   * Save current project to file.
   * Saves complete project state including plugin parameters, generated MIDI,
   * vocal notes, lyrics, and emotion selections.
   *
   * @param file Target file to save to
   * @return true if successful
   */
  bool saveCurrentProject(const juce::File &file);

  /**
   * Load project from file.
   * Restores complete project state including plugin parameters, generated
   * MIDI, vocal notes, lyrics, and emotion selections.
   *
   * @param file Source file to load from
   * @return true if successful
   */
  bool loadProject(const juce::File &file);

  /**
   * Get last error message from project save/load operation.
   *
   * @return Error message string, empty if no error
   */
  juce::String getProjectError() const { return projectError_; }

  // KellyBrain API methods (optional high-level API)
  void enableKellyBrainAPI(bool enable);
  bool isKellyBrainAPIEnabled() const { return useKellyBrainAPI_.load(); }
  kelly::KellyBrain &getKellyBrain();
  const kelly::KellyBrain &getKellyBrain() const;

  // MLBridge methods (audio-driven generation)
  void enableMLBridge(bool enable);
  bool isMLBridgeEnabled() const { return useMLBridge_.load(); }
  kelly::MLIntentPipeline &getMLBridge();
  const kelly::MLIntentPipeline &getMLBridge() const;

  // AudioProcessorValueTreeState::Listener
  void parameterChanged(const juce::String &parameterID,
                        float newValue) override;

  // AsyncUpdater
  void handleAsyncUpdate() override;

  // Timer
  void timerCallback() override;

private:
  //==============================================================================
  // Thread Safety Architecture
  //==============================================================================
  //
  // Audio Thread (processBlock):
  //   - MUST NEVER BLOCK - use try_lock, skip if lock unavailable
  //   - Can read APVTS parameters atomically via getRawParameterValue()
  //   - Can read atomic flags (hasPendingMidi_, isGenerating_)
  //   - Can access generatedMidi_ with try_lock (skip if unavailable)
  //
  // Message/UI Thread:
  //   - Can block on locks (std::lock_guard)
  //   - Can call generateMidi(), setWoundDescription(), etc.
  //   - Can access IntentPipeline (which may do heavy processing)
  //
  // Parameter Automation:
  //   - APVTS parameters are thread-safe and lock-free
  //   - parameterChanged() is called from message thread
  //   - Audio thread reads via getRawParameterValue() (atomic)
  //
  //==============================================================================

  juce::AudioProcessorValueTreeState apvts_;
  PluginState pluginState_;

  IntentPipeline intentPipeline_; // Default pipeline (always available for
                                  // backward compatibility)
  MidiGenerator midiGenerator_;
  MidiBuilder midiBuilder_;

  // Optional high-level APIs (using pointers to avoid including headers with
  // conflicting types)
  std::unique_ptr<kelly::KellyBrain>
      kellyBrain_; // High-level API wrapper (optional)
  std::unique_ptr<kelly::MLIntentPipeline>
      mlBridge_; // ML-to-Intent bridge (optional)
  std::atomic<bool> useKellyBrainAPI_{
      false}; // Use KellyBrain instead of IntentPipeline
  std::atomic<bool> useMLBridge_{
      false}; // Use MLBridge for audio-driven generation

  GeneratedMidi generatedMidi_;
  juce::MidiBuffer outputBuffer_;

  std::atomic<bool> hasPendingMidi_{false};
  std::atomic<bool> isGenerating_{false};

  // Parameter change tracking for manual regeneration
  // User requested manual regeneration only (no auto-regeneration during
  // playback)
  std::atomic<bool> parametersChanged_{false};

  // Real-time parameter regeneration support (kept for potential future use)
  std::atomic<bool> isHostPlaying_{false};
  std::atomic<bool> regenerationPending_{false};
  std::atomic<bool> autoRegenerateEnabled_{
      false}; // Disabled by default for manual regeneration
  static constexpr int DEBOUNCE_DELAY_MS = 200; // 200ms debounce delay

  // Protected by intentMutex_ - accessed from UI/message thread only
  // Read operations must hold lock, write operations already protected
  juce::String woundDescription_;
  std::optional<int> selectedEmotionId_;
  MusicTheorySettings musicTheorySettings_;
  std::optional<EmotionNode>
      lastProcessedEmotion_; // Last emotion processed from wound text

  mutable std::mutex midiMutex_; // Protects generatedMidi_ and outputBuffer_
                                 // Audio thread: use try_lock (never block)
                                 // UI thread: use lock_guard (can block)

  mutable std::mutex
      intentMutex_; // Protects intentPipeline_, woundDescription_,
                    // selectedEmotionId_ Audio thread: use try_lock (never
                    // block) UI thread: use lock_guard (can block)

  double currentSampleRate_ = 44100.0;
  int currentBlockSize_ = 512;
  double playheadPosition_ = 0.0;
  double lastPpqPosition_ = 0.0; // Track last playhead position for scheduling

  // ML Inference components
  PluginLatencyManager latencyManager_;
  InferenceThreadManager inferenceManager_;
  MLFeatureExtractor featureExtractor_;
  Kelly::ML::MultiModelProcessor multiModelProcessor_;
  std::unique_ptr<Kelly::ML::AsyncMLPipeline> asyncMLPipeline_;
  std::atomic<bool> mlInferenceEnabled_{false};

  // Lookahead buffer for ML inference
  static constexpr int ML_LOOKAHEAD_MS = 20;
  juce::AudioBuffer<float> lookaheadBuffer_;
  int lookaheadWritePos_ = 0;
  int lookaheadReadPos_ = 0;
  int lookaheadSamples_ = 0;

  // Current emotion state from ML inference
  std::atomic<float> mlValence_{0.0f};
  std::atomic<float> mlArousal_{0.0f};
  int64_t sampleCounter_ = 0;

  // Feature extraction and emotion application
  std::array<float, 128> extractFeatures(const juce::AudioBuffer<float> &buffer,
                                         int startPos);
  void applyEmotionVector(const std::array<float, 64> &emotionVector);

  // Project management
  juce::String projectError_; // Last error from save/load operations

  static juce::AudioProcessorValueTreeState::ParameterLayout
  createParameterLayout();

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

} // namespace kelly
