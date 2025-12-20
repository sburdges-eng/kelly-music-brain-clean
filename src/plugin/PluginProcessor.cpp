#include "plugin/PluginProcessor.h"
// PluginProcessor.h includes Types.h (via IntentPipeline.h)
// Before including KellyBrain.h (which includes KellyTypes.h), create aliases
// Note: Types.h is already included, so we can't prevent redefinition errors
// Instead, we'll handle conversions between the two type systems
namespace kelly {
// Create aliases for KellyTypes before including KellyBrain.h
// These will be shadowed when KellyTypes.h is included, but we can use them for
// conversions
namespace KellyTypesAlias {
// Forward declare what we need for conversions
struct IntentResult;
struct EmotionNode;
struct Wound;
} // namespace KellyTypesAlias
} // namespace kelly
// Include KellyBrain and MLBridge here - they use KellyTypes.h
#include "engine/KellyBrain.h"
#include "ml/MLBridge.h"
// Now create conversion helpers between Types.h and KellyTypes.h
namespace kelly {
namespace {
// Convert KellyTypes::IntentResult to Types::IntentResult
// Note: Since Types.h includes KellyTypes.h, they're the same types
// This function ensures emotion field is synced with sourceWound.primaryEmotion
IntentResult convertKellyTypesToLegacy(const IntentResult &kellyResult) {
  IntentResult result = kellyResult;
  // Sync emotion field with sourceWound.primaryEmotion
  result.emotion = kellyResult.sourceWound.primaryEmotion;
  // Set tempo multiplier from tempoBpm
  result.tempo = static_cast<float>(kellyResult.tempoBpm) / 120.0f;
  return result;
}

// Convert Types::EmotionNode to KellyTypes::EmotionNode (for Wound conversion)
// Note: Since Types.h includes KellyTypes.h, they're the same types
EmotionNode convertLegacyToKellyTypesEmotionNode(const EmotionNode &legacy) {
  return legacy; // Types are the same, just return copy
}

// Convert Types::EmotionNode to KellyTypes::EmotionNode (for IntentResult
// conversion) Note: Since Types.h includes KellyTypes.h, they're the same types
EmotionNode convertKellyTypesToLegacyEmotionNode(const EmotionNode &kelly) {
  return kelly; // Types are the same, just return copy
}

// Convert Types::Wound to KellyTypes::Wound
// Note: Since Types.h includes KellyTypes.h, they're the same types
Wound convertLegacyToKellyTypesWound(const Wound &legacy) {
  Wound kelly = legacy;
  // Ensure urgency is set from intensity if not already set
  if (kelly.urgency == 0.0f && kelly.intensity > 0.0f) {
    kelly.urgency = kelly.intensity;
  }
  // Use description as expression if expression is empty
  if (kelly.expression.empty() && !kelly.description.empty()) {
    kelly.expression = kelly.description;
  }
  return kelly;
}
} // namespace
} // namespace kelly
#include "common/MusicConstants.h"
#include "common/PathResolver.h"
#include "plugin/PluginEditor.h"
#include "project/ProjectManager.h"

using namespace kelly::MusicConstants;
using namespace midikompanion;

namespace kelly {

PluginProcessor::PluginProcessor()
    : AudioProcessor(
          BusesProperties()
              .withInput("Input", juce::AudioChannelSet::stereo(), true)
              .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts_(*this, nullptr, "KellyParameters", createParameterLayout()),
      latencyManager_(*this),
      kellyBrain_(std::make_unique<kelly::KellyBrain>()),
      mlBridge_(std::make_unique<kelly::MLIntentPipeline>()) {
  // Add parameter listeners for automation support
  apvts_.addParameterListener(PARAM_VALENCE, this);
  apvts_.addParameterListener(PARAM_AROUSAL, this);
  apvts_.addParameterListener(PARAM_INTENSITY, this);
  apvts_.addParameterListener(PARAM_COMPLEXITY, this);
  apvts_.addParameterListener(PARAM_HUMANIZE, this);
  apvts_.addParameterListener(PARAM_FEEL, this);
  apvts_.addParameterListener(PARAM_DYNAMICS, this);
  apvts_.addParameterListener(PARAM_BARS, this);
  apvts_.addParameterListener(PARAM_BYPASS, this);
}

PluginProcessor::~PluginProcessor() {
  // Stop timer before destruction
  stopTimer();

  // Cancel any pending async updates
  cancelPendingUpdate();

  // Remove parameter listeners
  apvts_.removeParameterListener(PARAM_VALENCE, this);
  apvts_.removeParameterListener(PARAM_AROUSAL, this);
  apvts_.removeParameterListener(PARAM_INTENSITY, this);
  apvts_.removeParameterListener(PARAM_COMPLEXITY, this);
  apvts_.removeParameterListener(PARAM_HUMANIZE, this);
  apvts_.removeParameterListener(PARAM_FEEL, this);
  apvts_.removeParameterListener(PARAM_DYNAMICS, this);
  apvts_.removeParameterListener(PARAM_BARS, this);
  apvts_.removeParameterListener(PARAM_BYPASS, this);
}

juce::AudioProcessorValueTreeState::ParameterLayout
PluginProcessor::createParameterLayout() {
  std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

  // Emotion parameters
  params.push_back(std::make_unique<juce::AudioParameterFloat>(
      juce::ParameterID{PARAM_VALENCE, PARAM_VERSION}, "Valence",
      juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f), 0.0f,
      juce::AudioParameterFloatAttributes().withLabel("Valence (-/+)")));

  params.push_back(std::make_unique<juce::AudioParameterFloat>(
      juce::ParameterID{PARAM_AROUSAL, PARAM_VERSION}, "Arousal",
      juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), AROUSAL_MODERATE,
      juce::AudioParameterFloatAttributes().withLabel("Arousal")));

  params.push_back(std::make_unique<juce::AudioParameterFloat>(
      juce::ParameterID{PARAM_INTENSITY, PARAM_VERSION}, "Intensity",
      juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), INTENSITY_MODERATE,
      juce::AudioParameterFloatAttributes().withLabel("Intensity")));

  // Generation parameters
  params.push_back(std::make_unique<juce::AudioParameterFloat>(
      juce::ParameterID{PARAM_COMPLEXITY, PARAM_VERSION}, "Complexity",
      juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), RULE_BREAK_MODERATE,
      juce::AudioParameterFloatAttributes().withLabel("Complexity")));

  params.push_back(std::make_unique<juce::AudioParameterFloat>(
      juce::ParameterID{PARAM_HUMANIZE, PARAM_VERSION}, "Humanize",
      juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.4f,
      juce::AudioParameterFloatAttributes().withLabel("Humanize")));

  params.push_back(std::make_unique<juce::AudioParameterFloat>(
      juce::ParameterID{PARAM_FEEL, PARAM_VERSION}, "Feel",
      juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f), 0.0f,
      juce::AudioParameterFloatAttributes().withLabel("Feel (Pull/Push)")));

  params.push_back(std::make_unique<juce::AudioParameterFloat>(
      juce::ParameterID{PARAM_DYNAMICS, PARAM_VERSION}, "Dynamics",
      juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
      MIDI_VELOCITY_MEDIUM / static_cast<float>(MIDI_VELOCITY_MAX),
      juce::AudioParameterFloatAttributes().withLabel("Dynamics")));

  params.push_back(std::make_unique<juce::AudioParameterInt>(
      juce::ParameterID{PARAM_BARS, PARAM_VERSION}, "Bars", 4, 32, 8,
      juce::AudioParameterIntAttributes().withLabel("Bars")));

  params.push_back(std::make_unique<juce::AudioParameterBool>(
      juce::ParameterID{PARAM_BYPASS, PARAM_VERSION}, "Bypass", false));

  return {params.begin(), params.end()};
}

void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
  currentSampleRate_ = sampleRate;
  currentBlockSize_ = samplesPerBlock;
  playheadPosition_ = 0.0;
  sampleCounter_ = 0;

  // Calculate lookahead in samples for ML inference
  lookaheadSamples_ =
      PluginLatencyManager::msToSamples(ML_LOOKAHEAD_MS, sampleRate);
  latencyManager_.setLookaheadLatency(lookaheadSamples_);

  // Allocate lookahead buffer (stereo, with extra space for safety)
  lookaheadBuffer_.setSize(2, lookaheadSamples_ + samplesPerBlock);
  lookaheadBuffer_.clear();
  lookaheadWritePos_ = 0;
  lookaheadReadPos_ = 0;

  // Set default ML latency (will be updated when model is loaded)
  latencyManager_.setMLLatency(1024); // Default: ~23ms at 44.1kHz

  // Initialize multi-model processor
  if (!multiModelProcessor_.isInitialized()) {
    auto modelsDir =
        juce::File::getSpecialLocation(juce::File::currentApplicationFile)
            .getParentDirectory()
            .getChildFile("models");

    // Fallback to Resources folder if models/ doesn't exist
    if (!modelsDir.isDirectory()) {
      modelsDir =
          juce::File::getSpecialLocation(juce::File::currentApplicationFile)
              .getChildFile("Resources/models");
    }

    multiModelProcessor_.initialize(modelsDir);

    // Create async pipeline for non-blocking inference
    asyncMLPipeline_ =
        std::make_unique<Kelly::ML::AsyncMLPipeline>(multiModelProcessor_);
  }
}

void PluginProcessor::releaseResources() {
  inferenceManager_.stop();

  // Stop async ML pipeline
  if (asyncMLPipeline_) {
    asyncMLPipeline_->stop();
  }
}

bool PluginProcessor::isBusesLayoutSupported(const BusesLayout &layouts) const {
  // MIDI effect - accept any layout
  juce::ignoreUnused(layouts);
  return true;
}

void PluginProcessor::processBlock(juce::AudioBuffer<float> &buffer,
                                   juce::MidiBuffer &midiMessages) {
  juce::ScopedNoDenormals noDenormals;

  const int numSamples = buffer.getNumSamples();
  const int numChannels = buffer.getNumChannels();

  // ML Inference processing (if enabled and we have audio input)
  if (mlInferenceEnabled_.load() && numChannels > 0) {
    // Write input to lookahead buffer
    for (int ch = 0; ch < numChannels && ch < lookaheadBuffer_.getNumChannels();
         ++ch) {
      const float *src = buffer.getReadPointer(ch);
      float *dst = lookaheadBuffer_.getWritePointer(ch);

      for (int i = 0; i < numSamples; ++i) {
        int writeIdx =
            (lookaheadWritePos_ + i) % lookaheadBuffer_.getNumSamples();
        dst[writeIdx] = src[i];
      }
    }

    // Extract features from lookahead buffer for ML inference
    std::array<float, 128> features =
        extractFeatures(lookaheadBuffer_, lookaheadReadPos_);

    // Submit inference request (non-blocking)
    InferenceRequest request;
    request.features = features;
    request.timestamp = sampleCounter_;
    inferenceManager_.submitRequest(request);

    // Get completed results (non-blocking)
    InferenceResult result;
    while (inferenceManager_.getResult(result)) {
      applyEmotionVector(result.emotionVector);
    }

    // Read delayed audio from lookahead buffer
    for (int ch = 0; ch < numChannels && ch < lookaheadBuffer_.getNumChannels();
         ++ch) {
      const float *src = lookaheadBuffer_.getReadPointer(ch);
      float *dst = buffer.getWritePointer(ch);

      for (int i = 0; i < numSamples; ++i) {
        int readIdx =
            (lookaheadReadPos_ + i) % lookaheadBuffer_.getNumSamples();
        dst[i] = src[readIdx];
      }
    }

    // Update positions
    lookaheadWritePos_ =
        (lookaheadWritePos_ + numSamples) % lookaheadBuffer_.getNumSamples();
    lookaheadReadPos_ =
        (lookaheadReadPos_ + numSamples) % lookaheadBuffer_.getNumSamples();
  } else {
    // Clear audio (we're a MIDI effect, no ML processing)
    buffer.clear();
  }

  sampleCounter_ += numSamples;

  // Check bypass - use atomic-safe parameter read
  auto *bypassParam = apvts_.getRawParameterValue(PARAM_BYPASS);
  if (bypassParam && *bypassParam > MusicConstants::RULE_BREAK_MODERATE) {
    return;
  }

  // Get playhead position from host
  auto *playHead = getPlayHead();
  if (!playHead) {
    return; // No playhead info available
  }

  auto position = playHead->getPosition();
  if (!position) {
    return; // No position info
  }

  // Get timing information
  float bpm = static_cast<float>(position->getBpm().orFallback(
      static_cast<double>(MusicConstants::TEMPO_DEFAULT)));
  double ppqPosition = position->getPpqPosition().orFallback(0.0);
  bool isPlaying = position->getIsPlaying();

  // Track playback state for real-time regeneration
  isHostPlaying_.store(isPlaying);

  // Calculate samples per beat and quarter note
  double samplesPerBeat = (currentSampleRate_ * 60.0) / bpm;
  double samplesPerQuarterNote = samplesPerBeat / BEATS_PER_QUARTER_NOTE;

  // numSamples already defined earlier in function - don't redeclare

  // CRITICAL: Audio thread must NEVER block. Use try_lock instead of
  // lock_guard. If we can't acquire the lock immediately, skip this block to
  // avoid audio glitches.
  if (hasPendingMidi_.load() && isPlaying) {
    std::unique_lock<std::mutex> lock(midiMutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      // Couldn't acquire lock - skip this block to avoid blocking audio thread
      // The MIDI will be scheduled on the next block
      return;
    }

    // Calculate current position in beats (PPQ is quarter notes, so divide by
    // BEATS_PER_QUARTER_NOTE)
    double currentBeat = ppqPosition / BEATS_PER_QUARTER_NOTE;
    double beatsPerBlock = numSamples / samplesPerBeat;
    double blockStartBeat = currentBeat;
    double blockEndBeat = currentBeat + beatsPerBlock;

    // Helper lambda to schedule a note on/off event
    auto scheduleNoteEvent = [&](int channel, int pitch, int velocity,
                                 double beatPosition, bool isNoteOn) {
      if (beatPosition >= blockStartBeat && beatPosition < blockEndBeat) {
        int sampleOffset =
            static_cast<int>((beatPosition - blockStartBeat) * samplesPerBeat);
        sampleOffset = juce::jlimit(0, numSamples - 1, sampleOffset);

        if (isNoteOn) {
          midiMessages.addEvent(
              juce::MidiMessage::noteOn(channel, pitch,
                                        static_cast<juce::uint8>(velocity)),
              sampleOffset);
        } else {
          midiMessages.addEvent(juce::MidiMessage::noteOff(channel, pitch),
                                sampleOffset);
        }
      }
    };

    // Schedule chord notes (channel 1)
    for (const auto &chord : generatedMidi_.chords) {
      double chordStartBeat = chord.startBeat;
      double chordEndBeat = chord.startBeat + chord.duration;

      // Check if chord overlaps with this block
      if (chordEndBeat > blockStartBeat && chordStartBeat < blockEndBeat) {
        // Note on: if chord starts in this block
        if (chordStartBeat >= blockStartBeat && chordStartBeat < blockEndBeat) {
          int sampleOffset = static_cast<int>(
              (chordStartBeat - blockStartBeat) * samplesPerBeat);
          sampleOffset = juce::jlimit(0, numSamples - 1, sampleOffset);

          for (int pitch : chord.pitches) {
            midiMessages.addEvent(
                juce::MidiMessage::noteOn(
                    MIDI_CHANNEL_CHORDS + 1, pitch,
                    static_cast<juce::uint8>(MIDI_VELOCITY_MEDIUM)),
                sampleOffset);
          }
        }

        // Note off: if chord ends in this block
        if (chordEndBeat >= blockStartBeat && chordEndBeat < blockEndBeat) {
          int sampleOffset = static_cast<int>((chordEndBeat - blockStartBeat) *
                                              samplesPerBeat);
          sampleOffset = juce::jlimit(0, numSamples - 1, sampleOffset);

          for (int pitch : chord.pitches) {
            midiMessages.addEvent(
                juce::MidiMessage::noteOff(MIDI_CHANNEL_CHORDS + 1, pitch),
                sampleOffset);
          }
        }
      }
    }

    // Schedule melody notes (channel 2)
    // Convert ticks to beats (480 ticks per quarter note)
    constexpr double TICKS_PER_BEAT = 480.0;
    for (const auto &note : generatedMidi_.melody) {
      double noteStartBeat = note.startTick / TICKS_PER_BEAT;
      double noteEndBeat =
          (note.startTick + note.durationTicks) / TICKS_PER_BEAT;

      if (noteEndBeat > blockStartBeat && noteStartBeat < blockEndBeat) {
        scheduleNoteEvent(MIDI_CHANNEL_MELODY + 1, note.pitch, note.velocity,
                          noteStartBeat, true);
        scheduleNoteEvent(MIDI_CHANNEL_MELODY + 1, note.pitch, 0, noteEndBeat,
                          false);
      }
    }

    // Schedule bass notes (channel 3)
    for (const auto &note : generatedMidi_.bass) {
      double noteStartBeat = note.startTick / TICKS_PER_BEAT;
      double noteEndBeat =
          (note.startTick + note.durationTicks) / TICKS_PER_BEAT;

      if (noteEndBeat > blockStartBeat && noteStartBeat < blockEndBeat) {
        scheduleNoteEvent(MIDI_CHANNEL_BASS + 1, note.pitch, note.velocity,
                          noteStartBeat, true);
        scheduleNoteEvent(MIDI_CHANNEL_BASS + 1, note.pitch, 0, noteEndBeat,
                          false);
      }
    }

    // Schedule counter-melody notes (channel 4)
    for (const auto &note : generatedMidi_.counterMelody) {
      double noteStartBeat = note.startTick / TICKS_PER_BEAT;
      double noteEndBeat =
          (note.startTick + note.durationTicks) / TICKS_PER_BEAT;

      if (noteEndBeat > blockStartBeat && noteStartBeat < blockEndBeat) {
        scheduleNoteEvent(MIDI_CHANNEL_COUNTER_MELODY + 1, note.pitch,
                          note.velocity, noteStartBeat, true);
        scheduleNoteEvent(MIDI_CHANNEL_COUNTER_MELODY + 1, note.pitch, 0,
                          noteEndBeat, false);
      }
    }

    // Schedule pad notes (channel 5)
    for (const auto &note : generatedMidi_.pad) {
      double noteStartBeat = note.startTick / TICKS_PER_BEAT;
      double noteEndBeat =
          (note.startTick + note.durationTicks) / TICKS_PER_BEAT;

      if (noteEndBeat > blockStartBeat && noteStartBeat < blockEndBeat) {
        scheduleNoteEvent(MIDI_CHANNEL_PAD + 1, note.pitch, note.velocity,
                          noteStartBeat, true);
        scheduleNoteEvent(MIDI_CHANNEL_PAD + 1, note.pitch, 0, noteEndBeat,
                          false);
      }
    }

    // Schedule string notes (channel 6)
    for (const auto &note : generatedMidi_.strings) {
      double noteStartBeat = note.startTick / TICKS_PER_BEAT;
      double noteEndBeat =
          (note.startTick + note.durationTicks) / TICKS_PER_BEAT;

      if (noteEndBeat > blockStartBeat && noteStartBeat < blockEndBeat) {
        scheduleNoteEvent(MIDI_CHANNEL_STRINGS + 1, note.pitch, note.velocity,
                          noteStartBeat, true);
        scheduleNoteEvent(MIDI_CHANNEL_STRINGS + 1, note.pitch, 0, noteEndBeat,
                          false);
      }
    }

    // Schedule fill notes (channel 7)
    for (const auto &note : generatedMidi_.fills) {
      double noteStartBeat = note.startTick / TICKS_PER_BEAT;
      double noteEndBeat =
          (note.startTick + note.durationTicks) / TICKS_PER_BEAT;

      if (noteEndBeat > blockStartBeat && noteStartBeat < blockEndBeat) {
        scheduleNoteEvent(MIDI_CHANNEL_FILLS + 1, note.pitch, note.velocity,
                          noteStartBeat, true);
        scheduleNoteEvent(MIDI_CHANNEL_FILLS + 1, note.pitch, 0, noteEndBeat,
                          false);
      }
    }

    // Mark as processed (only after successful scheduling)
    hasPendingMidi_.store(false);
  }

  // Update last position (atomic-safe, no lock needed)
  lastPpqPosition_ = ppqPosition;
}

juce::AudioProcessorEditor *PluginProcessor::createEditor() {
  return new PluginEditor(*this);
}

void PluginProcessor::getStateInformation(juce::MemoryBlock &destData) {
  // Called from message thread - safe to access all members
  std::lock_guard<std::mutex> lock(intentMutex_);

  // Create default cassette state (can be extended later)
  CassetteState cassetteState;
  cassetteState.sideA.description = woundDescription_.toStdString();
  cassetteState.sideA.intensity = *apvts_.getRawParameterValue(PARAM_INTENSITY);
  if (selectedEmotionId_) {
    cassetteState.sideA.emotionId = *selectedEmotionId_;
  }
  cassetteState.sideB.description = "musical expression";
  cassetteState.sideB.intensity = cassetteState.sideA.intensity;
  cassetteState.isFlipped = false;

  // Use PluginState to save complete state
  auto state = pluginState_.saveState(apvts_, woundDescription_,
                                      selectedEmotionId_, cassetteState);

  std::unique_ptr<juce::XmlElement> xml(state.createXml());
  copyXmlToBinary(*xml, destData);
}

void PluginProcessor::setStateInformation(const void *data, int sizeInBytes) {
  // Called from message thread - safe to access all members
  std::unique_ptr<juce::XmlElement> xmlState(
      getXmlFromBinary(data, sizeInBytes));

  if (xmlState && xmlState->hasTagName(apvts_.state.getType())) {
    auto state = juce::ValueTree::fromXml(*xmlState);

    // Use PluginState to load complete state
    std::lock_guard<std::mutex> lock(intentMutex_);
    CassetteState cassetteState;
    pluginState_.loadState(apvts_, woundDescription_, selectedEmotionId_,
                           cassetteState, state);

    // Note: cassetteState is loaded but not stored in PluginProcessor yet
    // This can be extended if needed for UI display
  }
}

void PluginProcessor::generateMidi() {
  if (isGenerating_.exchange(true)) {
    return; // Already generating
  }

  // Get parameters
  float valence = *apvts_.getRawParameterValue(PARAM_VALENCE);
  float arousal = *apvts_.getRawParameterValue(PARAM_AROUSAL);
  float intensity = *apvts_.getRawParameterValue(PARAM_INTENSITY);
  float complexity = *apvts_.getRawParameterValue(PARAM_COMPLEXITY);
  float humanize = *apvts_.getRawParameterValue(PARAM_HUMANIZE);
  float feel = *apvts_.getRawParameterValue(PARAM_FEEL);
  float dynamics = *apvts_.getRawParameterValue(PARAM_DYNAMICS);
  int bars = static_cast<int>(*apvts_.getRawParameterValue(PARAM_BARS));

  // Build wound from description or use emotion coordinates
  // Read protected state while holding lock
  Wound wound;
  juce::String woundDesc;
  std::optional<int> emotionId;
  {
    std::lock_guard<std::mutex> lock(intentMutex_);
    woundDesc = woundDescription_;
    emotionId = selectedEmotionId_;
  }

  if (woundDesc.isNotEmpty()) {
    wound.description = woundDesc.toStdString();
    wound.intensity = intensity;
    wound.source = "user_input";
  } else {
    wound.description = "emotional state";
    wound.intensity = intensity;
    wound.source = "parameters";
  }

  // Get music theory settings (may override emotion-derived parameters)
  MusicTheorySettings theorySettings;
  {
    std::lock_guard<std::mutex> lock(intentMutex_);
    theorySettings = musicTheorySettings_;
  }

  // Process through intent pipeline (UI thread - can block)
  IntentResult intent;
  EmotionNode processedEmotion;
  {
    std::lock_guard<std::mutex> lock(intentMutex_);

    // Option 1: Use KellyBrain API (if enabled)
    if (useKellyBrainAPI_.load() && kellyBrain_) {
      // Convert Types::Wound to KellyTypes::Wound
      ::kelly::Wound kellyWound = convertLegacyToKellyTypesWound(wound);

      if (emotionId) {
        // Use selected emotion from wheel
        auto emotionOpt =
            kellyBrain_->pipeline().thesaurus().findById(*emotionId);
        if (emotionOpt.has_value()) {
          processedEmotion = *emotionOpt;
          // Set emotion in wound
          kellyWound.primaryEmotion =
              convertLegacyToKellyTypesEmotionNode(processedEmotion);

          ::kelly::SideA sideA;
          sideA.description = wound.description;
          sideA.intensity = intensity;
          sideA.emotionId = *emotionId;

          ::kelly::SideB sideB;
          sideB.description = "musical expression";
          sideB.intensity = intensity;
          sideB.emotionId = *emotionId;

          // Use fromJourney() method on KellyBrain, not
          // pipeline().processJourney()
          ::kelly::IntentResult kellyIntent =
              kellyBrain_->fromJourney(sideA, sideB);
          // Convert KellyTypes::IntentResult to Types::IntentResult
          intent = convertKellyTypesToLegacy(kellyIntent);
          processedEmotion = intent.emotion;
        } else {
          ::kelly::IntentResult kellyIntent =
              kellyBrain_->fromWound(kellyWound);
          intent = convertKellyTypesToLegacy(kellyIntent);
          processedEmotion = intent.emotion;
        }
      } else if (woundDesc.isNotEmpty()) {
        ::kelly::IntentResult kellyIntent = kellyBrain_->fromWound(kellyWound);
        intent = convertKellyTypesToLegacy(kellyIntent);
        processedEmotion = intent.emotion;
      } else {
        EmotionNode nearestEmotion =
            kellyBrain_->pipeline().thesaurus().findNearest(valence, arousal,
                                                            intensity);
        processedEmotion = nearestEmotion;

        ::kelly::SideA sideA;
        sideA.description = wound.description;
        sideA.intensity = intensity;
        sideA.emotionId = nearestEmotion.id;

        ::kelly::SideB sideB;
        sideB.description = "musical expression";
        sideB.intensity = intensity;
        sideB.emotionId = nearestEmotion.id;

        ::kelly::IntentResult kellyIntent =
            kellyBrain_->fromJourney(sideA, sideB);
        intent = convertKellyTypesToLegacy(kellyIntent);
        processedEmotion = intent.emotion;
      }
    }
    // Option 2: Use existing IntentPipeline (default, backward compatible)
    else {
      if (emotionId) {
        // Use selected emotion from wheel
        auto emotionOpt = intentPipeline_.thesaurus().findById(*emotionId);
        if (emotionOpt.has_value()) {
          // Emotion ID found in thesaurus - use it for journey
          processedEmotion = *emotionOpt;

          SideA sideA;
          sideA.description = wound.description;
          sideA.intensity = intensity;
          sideA.emotionId = *emotionId;

          SideB sideB;
          sideB.description = "musical expression";
          sideB.intensity = intensity;
          sideB.emotionId = *emotionId;

          intent = intentPipeline_.processJourney(sideA, sideB);
        } else {
          // Emotion ID not found - fallback to processing wound description
          // This handles cases where emotion ID might be invalid or thesaurus
          // not fully loaded
          intent = intentPipeline_.process(wound);
          processedEmotion = intent.emotion; // Store the processed emotion
        }
      } else if (woundDesc.isNotEmpty()) {
        // Process wound text to detect emotion
        intent = intentPipeline_.process(wound);
        processedEmotion =
            intent.emotion; // Store the detected emotion from wound text

        // Note: We don't automatically update APVTS parameters here because:
        // 1. User may have manually set parameters they want to keep
        // 2. The UI will update EmotionWheel/Radar to show detected emotion
        // 3. Parameters can be updated by the UI if user wants to match
        // detected emotion
      } else {
        // Use valence/arousal/intensity coordinates
        EmotionNode nearestEmotion = intentPipeline_.thesaurus().findNearest(
            valence, arousal, intensity);
        processedEmotion = nearestEmotion;

        SideA sideA;
        sideA.description = wound.description;
        sideA.intensity = intensity;
        sideA.emotionId = nearestEmotion.id;

        SideB sideB;
        sideB.description = "musical expression";
        sideB.intensity = intensity;
        sideB.emotionId = nearestEmotion.id;

        intent = intentPipeline_.processJourney(sideA, sideB);
      }
    }

    // Store the processed emotion for UI access
    lastProcessedEmotion_ = processedEmotion;
  }

  // Override mode and tempo from MusicTheoryPanel if set
  // Note: Full integration may require API changes to
  // IntentPipeline/MidiGenerator For now, we store the settings and they can be
  // used in future enhancements
  if (!theorySettings.mode.isEmpty()) {
    intent.mode = theorySettings.mode.toStdString();
  }
  if (theorySettings.tempoBpm > 0) {
    // Convert BPM to tempo modifier (relative to base tempo)
    // Base tempo is typically 120 BPM, so modifier = newBpm / 120.0
    float baseTempo = 120.0f;
    intent.tempo = static_cast<float>(theorySettings.tempoBpm) / baseTempo;
  }

  // Generate MIDI
  {
    std::lock_guard<std::mutex> lock(midiMutex_);
    generatedMidi_ = midiGenerator_.generate(intent, bars, complexity, humanize,
                                             feel, dynamics);
  }

  // Clear parameter change flag after successful generation
  // This allows UI to detect when regeneration is needed
  parametersChanged_.store(false);

  hasPendingMidi_.store(true);
  isGenerating_.store(false);
}

void PluginProcessor::setWoundDescription(const juce::String &description) {
  // Called from UI thread - safe to lock
  std::lock_guard<std::mutex> lock(intentMutex_);
  woundDescription_ = description;
}

void PluginProcessor::setSelectedEmotionId(int emotionId) {
  // Called from UI thread - safe to lock
  std::lock_guard<std::mutex> lock(intentMutex_);
  selectedEmotionId_ = emotionId;
}

void PluginProcessor::setMusicTheorySettings(const juce::String &key,
                                             const juce::String &mode,
                                             int tempoBpm) {
  // Called from UI thread - safe to lock
  std::lock_guard<std::mutex> lock(intentMutex_);
  musicTheorySettings_.key = key;
  musicTheorySettings_.mode = mode;
  musicTheorySettings_.tempoBpm = tempoBpm;
}

PluginProcessor::MusicTheorySettings
PluginProcessor::getMusicTheorySettings() const {
  // Called from UI thread - safe to lock
  std::lock_guard<std::mutex> lock(intentMutex_);
  return musicTheorySettings_;
}

std::optional<EmotionNode> PluginProcessor::getLastProcessedEmotion() const {
  // Called from UI thread - safe to lock
  std::lock_guard<std::mutex> lock(intentMutex_);
  return lastProcessedEmotion_;
}

void PluginProcessor::enableMLInference(bool enable) {
  mlInferenceEnabled_.store(enable);

  if (enable) {
    // Try to load default model if available
    juce::File modelFile =
        juce::File::getSpecialLocation(juce::File::currentApplicationFile)
            .getChildFile("Resources/emotion_model.json");

    if (modelFile.existsAsFile()) {
      inferenceManager_.start(modelFile);

      // Estimate model latency based on file size
      auto fileSize = modelFile.getSize();
      int estimatedLatency = 512; // Default
      if (fileSize > 10 * 1024 * 1024) {
        estimatedLatency = 2048; // Large model
      } else if (fileSize > 1 * 1024 * 1024) {
        estimatedLatency = 1024; // Medium model
      }

      latencyManager_.setMLLatency(estimatedLatency);
    } else {
      juce::Logger::writeToLog(
          "ML Inference enabled but model file not found: " +
          modelFile.getFullPathName());
    }
  } else {
    inferenceManager_.stop();
  }
}

void PluginProcessor::enableKellyBrainAPI(bool enable) {
  useKellyBrainAPI_.store(enable);

  if (enable) {
    // Create KellyBrain instance if it doesn't exist
    if (!kellyBrain_) {
      kellyBrain_ = std::make_unique<kelly::KellyBrain>();

      // Initialize with data directory
      auto dataDir =
          juce::File::getSpecialLocation(juce::File::currentApplicationFile)
              .getParentDirectory()
              .getChildFile("data");

      // Fallback to Resources folder if data/ doesn't exist
      if (!dataDir.isDirectory()) {
        dataDir =
            juce::File::getSpecialLocation(juce::File::currentApplicationFile)
                .getChildFile("Resources/data");
      }

      if (dataDir.isDirectory()) {
        kellyBrain_->initialize(dataDir.getFullPathName().toStdString());
      } else {
        juce::Logger::writeToLog(
            "KellyBrain enabled but data directory not found: " +
            dataDir.getFullPathName());
      }
    }
  }
  // Note: We don't destroy kellyBrain_ when disabled to preserve state
  // It can be reused if re-enabled
}

std::array<float, 128>
PluginProcessor::extractFeatures(const juce::AudioBuffer<float> &buffer,
                                 int startPos) {
  return featureExtractor_.extractFeatures(buffer, startPos);
}

void PluginProcessor::applyEmotionVector(
    const std::array<float, 64> &emotionVector) {
  // Map emotion vector to valence and arousal
  // The first 32 dimensions could represent valence-related features
  // The last 32 dimensions could represent arousal-related features

  // Simple mapping: average first half for valence, second half for arousal
  float valenceSum = 0.0f;
  float arousalSum = 0.0f;

  for (size_t i = 0; i < 32; ++i) {
    valenceSum += emotionVector[i];
    arousalSum += emotionVector[i + 32];
  }

  // Normalize to [-1, 1] for valence, [0, 1] for arousal
  float valence = std::tanh(valenceSum / 32.0f); // Tanh maps to [-1, 1]
  float arousal =
      (std::tanh(arousalSum / 32.0f) + 1.0f) * 0.5f; // Map to [0, 1]

  // Update atomic emotion state
  mlValence_.store(valence);
  mlArousal_.store(arousal);

  // Optionally blend with UI parameters
  // For now, we just store the ML-derived values
  // The synthesis can use these values directly or blend with UI parameters
}

void PluginProcessor::handleAsyncUpdate() {
  // Called from message thread after triggerAsyncUpdate()
  // This allows regeneration to happen off the parameterChanged callback thread

  // Check if regeneration is still needed and host is still playing
  if (!regenerationPending_.load()) {
    return; // Regeneration was cancelled or already processed
  }

  if (!isHostPlaying_.load()) {
    // Host stopped playing, cancel regeneration
    regenerationPending_.store(false);
    return;
  }

  // Clear pending flag before generating (to prevent duplicate triggers)
  regenerationPending_.store(false);

  // Trigger MIDI regeneration
  // generateMidi() is thread-safe and can be called from message thread
  generateMidi();
}

void PluginProcessor::timerCallback() {
  // Called from message thread when debounce timer expires
  // This implements the debounce delay to prevent excessive regenerations

  // Stop the timer (it will be restarted if another parameter change occurs)
  stopTimer();

  // Check if regeneration is still pending and host is still playing
  if (!regenerationPending_.load()) {
    return; // Regeneration was cancelled
  }

  if (!isHostPlaying_.load()) {
    // Host stopped playing, cancel regeneration
    regenerationPending_.store(false);
    return;
  }

  // Trigger async update to regenerate MIDI in background
  // This ensures we don't block the message thread
  triggerAsyncUpdate();
}

void PluginProcessor::parameterChanged(const juce::String &parameterID,
                                       float newValue) {
  // This is called from the message thread when parameters change (including
  // automation) Parameters are already stored in APVTS atomically, so they're
  // safe to read from audio thread

  juce::ignoreUnused(newValue);

  // Manual regeneration approach: Track parameter changes but don't
  // auto-regenerate User requested manual regeneration only (no
  // auto-regeneration during playback) The UI will show visual feedback when
  // regeneration is needed

  // Bypass parameter does not require regeneration (it's a control parameter)
  if (parameterID == PARAM_BYPASS) {
    return; // Bypass doesn't require regeneration
  }

  // Mark that parameters have changed (for UI to show regeneration needed
  // indicator)
  parametersChanged_.store(true);

  // Optional: Keep auto-regeneration code for future use (currently disabled)
  if (autoRegenerateEnabled_.load()) {
    // Real-time regeneration: Only regenerate when host is playing and
    // auto-regeneration is enabled
    if (!isHostPlaying_.load()) {
      return; // Host is not playing, skip regeneration
    }

    // Mark that regeneration is pending
    regenerationPending_.store(true);

    // Start or reset debounce timer
    // If timer is already running, this will reset it (restart from 0)
    startTimer(DEBOUNCE_DELAY_MS);
  }
}

void PluginProcessor::setModelEnabled(Kelly::ML::ModelType type, bool enabled) {
  multiModelProcessor_.setModelEnabled(type, enabled);
}

bool PluginProcessor::isModelEnabled(Kelly::ML::ModelType type) const {
  return multiModelProcessor_.isModelEnabled(type);
}

void PluginProcessor::enableMLBridge(bool enable) {
  useMLBridge_.store(enable);

  if (enable) {
    // Create MLIntentPipeline instance if it doesn't exist
    if (!mlBridge_) {
      mlBridge_ = std::make_unique<kelly::MLIntentPipeline>();

      // Initialize with models and data directories
      auto modelsDir =
          juce::File::getSpecialLocation(juce::File::currentApplicationFile)
              .getParentDirectory()
              .getChildFile("models");

      // Fallback to Resources folder if models/ doesn't exist
      if (!modelsDir.isDirectory()) {
        modelsDir =
            juce::File::getSpecialLocation(juce::File::currentApplicationFile)
                .getChildFile("Resources/models");
      }

      auto dataDir =
          juce::File::getSpecialLocation(juce::File::currentApplicationFile)
              .getParentDirectory()
              .getChildFile("data");

      // Fallback to Resources folder if data/ doesn't exist
      if (!dataDir.isDirectory()) {
        dataDir =
            juce::File::getSpecialLocation(juce::File::currentApplicationFile)
                .getChildFile("Resources/data");
      }

      std::string modelsPath = modelsDir.isDirectory()
                                   ? modelsDir.getFullPathName().toStdString()
                                   : "./models";
      std::string dataPath =
          dataDir.isDirectory() ? dataDir.getFullPathName().toStdString() : "";

      if (!mlBridge_->initialize(modelsPath, dataPath)) {
        juce::Logger::writeToLog("MLBridge initialization failed");
      }
    }
  }
  // Note: We don't destroy mlBridge_ when disabled to preserve state
  // It can be reused if re-enabled
}

// Implement accessor methods
kelly::KellyBrain &PluginProcessor::getKellyBrain() {
  if (!kellyBrain_) {
    // Initialize if not already created
    enableKellyBrainAPI(true);
  }
  return *kellyBrain_;
}

const kelly::KellyBrain &PluginProcessor::getKellyBrain() const {
  if (!kellyBrain_) {
    // For const version, we can't modify, so throw or return a default
    // For now, we'll just return (but this should be handled better)
    static kelly::KellyBrain defaultBrain;
    return defaultBrain;
  }
  return *kellyBrain_;
}

kelly::MLIntentPipeline &PluginProcessor::getMLBridge() { return *mlBridge_; }

const kelly::MLIntentPipeline &PluginProcessor::getMLBridge() const {
  return *mlBridge_;
}

//==============================================================================
// Project Save/Load (Phase 0: v1.0 Critical Features)
//==============================================================================

bool PluginProcessor::saveCurrentProject(const juce::File &file) {
  projectError_.clear();

  // Create ProjectManager instance
  ProjectManager projectManager;

  // Gather current plugin state
  std::lock_guard<std::mutex> intentLock(intentMutex_);

  // Create preset from current state
  PluginState::Preset preset = pluginState_.createPresetFromState(
      file.getFileNameWithoutExtension(), "Project save", "User", apvts_,
      woundDescription_, selectedEmotionId_,
      pluginState_.loadEmotionSettings(
          apvts_.copyState()) // Get current CassetteState
  );

  // Get generated MIDI (thread-safe)
  GeneratedMidi generatedMidi;
  {
    std::lock_guard<std::mutex> midiLock(midiMutex_);
    generatedMidi = generatedMidi_;
  }

  // For v1.0, vocal notes and lyrics are empty (can be added in v1.1)
  std::vector<MidiNote> vocalNotes;
  std::vector<juce::String> lyrics;

  // Gather emotion selections
  std::vector<int> selectedEmotionIds;
  if (selectedEmotionId_.has_value()) {
    selectedEmotionIds.push_back(*selectedEmotionId_);
  }

  // Save project
  bool success = projectManager.saveProject(
      file, preset, generatedMidi, vocalNotes, lyrics, selectedEmotionIds,
      selectedEmotionId_);

  if (!success) {
    projectError_ = projectManager.getLastError();
    return false;
  }

  return true;
}

bool PluginProcessor::loadProject(const juce::File &file) {
  projectError_.clear();

  // Create ProjectManager instance
  ProjectManager projectManager;

  // Load project data
  PluginState::Preset preset;
  GeneratedMidi generatedMidi;
  std::vector<MidiNote> vocalNotes;
  std::vector<juce::String> lyrics;
  std::vector<int> selectedEmotionIds;
  std::optional<int> primaryEmotionId;

  bool success =
      projectManager.loadProject(file, preset, generatedMidi, vocalNotes,
                                 lyrics, selectedEmotionIds, primaryEmotionId);

  if (!success) {
    projectError_ = projectManager.getLastError();
    return false;
  }

  // Restore plugin state
  {
    std::lock_guard<std::mutex> intentLock(intentMutex_);

    // Get current cassette state (will be replaced by preset)
    CassetteState cassetteState = preset.cassetteState;

    // Apply preset to restore all parameters and state
    pluginState_.applyPreset(preset, apvts_, woundDescription_,
                             selectedEmotionId_, cassetteState);

    // Save the restored cassette state back to APVTS
    auto state = apvts_.copyState();
    pluginState_.saveEmotionSettings(state, cassetteState);
    apvts_.replaceState(state);
  }

  // Restore generated MIDI (thread-safe)
  {
    std::lock_guard<std::mutex> midiLock(midiMutex_);
    generatedMidi_ = generatedMidi;
    hasPendingMidi_.store(true);
  }

  // For v1.0, vocal notes and lyrics are not restored (can be added in v1.1)
  // They would be restored here if we had VoiceSynthesizer integration

  // Trigger async update to refresh UI
  triggerAsyncUpdate();

  return true;
}

} // namespace kelly

juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter() {
  return new kelly::PluginProcessor();
}
