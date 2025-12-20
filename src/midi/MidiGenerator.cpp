#include "midi/MidiGenerator.h"
#include "common/MusicConstants.h"
#include "engine/EmotionMusicMapper.h"
#include "engines/ArrangementEngine.h"
#include <algorithm>
#include <cmath>
#include <juce_core/juce_core.h>
#include <numeric>
#include <sstream>

namespace kelly {
using namespace MusicConstants;

MidiGenerator::MidiGenerator() : rng_(std::random_device{}()) {}

GeneratedMidi MidiGenerator::generate(const IntentResult &intent, int bars,
                                      float complexity, float humanize,
                                      float feel, float dynamics) {
  GeneratedMidi result;

  // Extract musical parameters from intent
  // IntentResult from Types.h has: mode, tempo (modifier), dynamicRange, etc.
  // We need to calculate actual tempo from modifier
  std::string mode = intent.mode;
  float tempoModifier = intent.tempo; // 0.5 to 2.0
  using namespace MusicConstants;
  int baseTempo = TEMPO_MODERATE; // Use named constant instead of magic number
  int tempoBpm = static_cast<int>(baseTempo * tempoModifier);
  std::string key = "C"; // Default, can be derived from emotion if needed

  // Use EmotionMusicMapper to get more detailed parameters if available
  auto musicalParams = EmotionMusicMapper::mapEmotion(intent.emotion);
  tempoBpm = musicalParams.tempo; // Override with calculated tempo
  if (!musicalParams.detailedMode.empty()) {
    mode = musicalParams.detailedMode;
  }

  result.bpm = static_cast<float>(tempoBpm);
  result.lengthInBeats = bars * BEATS_PER_BAR;

  // Generate arrangement structure (section metadata)
  // This informs the generation process with section types and energy levels
  std::optional<ArrangementOutput> arrangementOpt;
  ArrangementOutput *arrangementPtr = nullptr;
  if (bars >= 8) { // Only generate arrangement for longer pieces
    ArrangementOutput arrangement = generateArrangement(intent, bars);
    arrangementOpt = arrangement;
    // Store in result (as pointer for optional/nullable)
    // Note: This assumes the arrangement will be used during generation
    // For proper lifetime management, consider storing in a member variable or
    // changing GeneratedMidi to use std::optional instead of pointer
    static thread_local ArrangementOutput
        storedArrangement; // Thread-local storage for pointer
    storedArrangement = arrangement;
    arrangementPtr = &storedArrangement;
  }

  // Determine which layers to generate based on complexity and emotion
  // If arrangement exists, we can use it to inform layer selection per section
  LayerFlags layers = determineLayers(intent, complexity, bars, arrangementOpt);

  // Store pointer in result after layers are determined
  result.arrangement = arrangementPtr;

  // ========================================================================
  // PHASE 1: Generate harmonic foundation
  // ========================================================================

  // Generate chord progression
  result.chords = generateChords(intent, bars);

  // Convert chords to string format for engines that need it
  std::vector<std::string> chordStrings = chordsToStrings(result.chords);

  // ========================================================================
  // PHASE 2: Generate melodic layers
  // ========================================================================

  // Generate primary melody
  if (layers.melody) {
    result.melody = generateMelody(result.chords, intent, complexity, dynamics);
  }

  // Generate bass
  if (layers.bass) {
    result.bass = generateBass(result.chords, intent, complexity, dynamics);
  }

  // Generate counter melody (if complexity is high)
  if (layers.counterMelody && !result.melody.empty()) {
    result.counterMelody =
        generateCounterMelody(result.melody, intent, complexity);
  }

  // ========================================================================
  // PHASE 3: Generate textural layers
  // ========================================================================

  // Generate pads
  if (layers.pads) {
    result.pad = generatePads(result.chords, intent, complexity);
  }

  // Generate strings
  if (layers.strings) {
    result.strings = generateStrings(result.chords, intent, complexity);
  }

  // ========================================================================
  // PHASE 4: Generate rhythmic layers
  // ========================================================================

  // Generate rhythm (convert DrumHits to MidiNotes if needed)
  if (layers.rhythm) {
    result.rhythm = generateRhythm(intent, bars, complexity, tempoBpm);
  }

  // Generate drum groove
  if (layers.drumGroove) {
    result.drumGroove = generateDrumGroove(intent, bars, complexity, tempoBpm);
  }

  // Generate fills
  if (layers.fills) {
    result.fills = generateFills(intent, bars, complexity);
  }

  // Generate transitions
  if (layers.transitions) {
    result.transitions = generateTransitions(intent, bars, tempoBpm);
  }

  // ========================================================================
  // PHASE 5: Apply expression and processing
  // ========================================================================

  // Apply dynamics
  applyDynamics(result, intent, dynamics);

  // Apply tension curve
  applyTension(result, intent, bars);

  // Apply rule breaks (intentional violations for emotional authenticity)
  applyRuleBreaks(result, intent);

  // Apply variations (post-processing to add interest)
  if (layers.variations) {
    applyVariations(result, intent, complexity);
  }

  // Apply groove and humanization (last, as it affects timing)
  applyGrooveAndHumanize(result, humanize, intent.emotion, feel);

  return result;
}

// ============================================================================
// Chord Generation
// ============================================================================

std::vector<Chord> MidiGenerator::generateChords(const IntentResult &intent,
                                                 int bars) {
  return chordGen_.generate(intent, bars);
}

// ============================================================================
// Melody Generation
// ============================================================================

std::vector<MidiNote>
MidiGenerator::generateMelody(const std::vector<Chord> &chords,
                              const IntentResult &intent, float complexity,
                              float dynamics) {
  // Get parameters from intent and emotion mapping
  std::string mode = intent.mode;
  float tempoModifier = intent.tempo;
  int tempoBpm = static_cast<int>(TEMPO_FAST * tempoModifier);
  std::string key = "C";

  auto musicalParams = EmotionMusicMapper::mapEmotion(intent.emotion);
  tempoBpm = musicalParams.tempo;
  if (!musicalParams.detailedMode.empty())
    mode = musicalParams.detailedMode;
  if (!musicalParams.mode.empty())
    key = musicalParams.mode;

  // Use MelodyEngine to generate melody
  MelodyConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.key = key;
  config.mode = mode;
  config.bars =
      static_cast<int>(chords.size() / static_cast<double>(BEATS_PER_BAR) +
                       0.5); // Approximate bars from chords
  config.tempoBpm = tempoBpm;
  config.seed = rng_();

  MelodyOutput output = melodyEngine_.generate(config);

  // Convert MelodyNote to MidiNote
  std::vector<MidiNote> melody;
  for (const auto &note : output.notes) {
    melody.push_back(melodyNoteToMidi(note, tempoBpm));
  }

  // Apply dynamics scaling
  for (auto &note : melody) {
    note.velocity = std::clamp(static_cast<int>(note.velocity * dynamics),
                               MIDI_VELOCITY_MIN, MIDI_VELOCITY_MAX);
  }

  return melody;
}

// ============================================================================
// Bass Generation
// ============================================================================

std::vector<MidiNote>
MidiGenerator::generateBass(const std::vector<Chord> &chords,
                            const IntentResult &intent, float complexity,
                            float dynamics) {
  // Get parameters from intent and emotion mapping
  float tempoModifier = intent.tempo;
  int tempoBpm = static_cast<int>(TEMPO_FAST * tempoModifier);
  std::string key = "C";

  auto musicalParams = EmotionMusicMapper::mapEmotion(intent.emotion);
  tempoBpm = musicalParams.tempo;
  if (!musicalParams.mode.empty())
    key = musicalParams.mode;

  // Convert chords to string format
  std::vector<std::string> chordStrings = chordsToStrings(chords);

  // Use BassEngine to generate bass
  BassConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.chordProgression = chordStrings;
  config.key = key;
  config.bars = static_cast<int>(chords.size() / BEATS_PER_BAR + 0.5);
  config.tempoBpm = tempoBpm;
  config.seed = rng_();

  BassOutput output = bassEngine_.generate(config);

  // Convert BassNote to MidiNote
  std::vector<MidiNote> bass;
  for (const auto &note : output.notes) {
    bass.push_back(bassNoteToMidi(note, tempoBpm));
  }

  // Apply dynamics scaling (bass slightly louder)
  for (auto &note : bass) {
    note.velocity = std::clamp(
        static_cast<int>(note.velocity * dynamics * BASS_VELOCITY_MULTIPLIER),
        MIDI_VELOCITY_DEFAULT, MIDI_VELOCITY_MAX);
  }

  return bass;
}

// ============================================================================
// Pad Generation
// ============================================================================

std::vector<MidiNote>
MidiGenerator::generatePads(const std::vector<Chord> &chords,
                            const IntentResult &intent, float complexity) {
  // Get parameters from intent and emotion mapping
  float tempoModifier = intent.tempo;
  int tempoBpm = static_cast<int>(TEMPO_FAST * tempoModifier);
  std::string key = "C";

  auto musicalParams = EmotionMusicMapper::mapEmotion(intent.emotion);
  tempoBpm = musicalParams.tempo;
  if (!musicalParams.mode.empty())
    key = musicalParams.mode;

  // Convert chords to string format
  std::vector<std::string> chordStrings = chordsToStrings(chords);

  // Use PadEngine to generate pads
  PadConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.chordProgression = chordStrings;
  config.key = key;
  config.bars = static_cast<int>(chords.size() / BEATS_PER_BAR + 0.5);
  config.tempoBpm = tempoBpm;
  config.seed = rng_();

  PadOutput output = padEngine_.generate(config);

  // Convert PadNote to MidiNote
  std::vector<MidiNote> pads;
  for (const auto &note : output.notes) {
    pads.push_back(padNoteToMidi(note, tempoBpm));
  }

  return pads;
}

// ============================================================================
// String Generation
// ============================================================================

std::vector<MidiNote>
MidiGenerator::generateStrings(const std::vector<Chord> &chords,
                               const IntentResult &intent, float complexity) {
  // Get parameters from intent and emotion mapping
  float tempoModifier = intent.tempo;
  int tempoBpm = static_cast<int>(TEMPO_FAST * tempoModifier);
  std::string key = "C";

  auto musicalParams = EmotionMusicMapper::mapEmotion(intent.emotion);
  tempoBpm = musicalParams.tempo;
  if (!musicalParams.mode.empty())
    key = musicalParams.mode;

  // Convert chords to string format
  std::vector<std::string> chordStrings = chordsToStrings(chords);

  // Use StringEngine to generate strings
  StringConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.chordProgression = chordStrings;
  config.key = key;
  config.bars = static_cast<int>(chords.size() / BEATS_PER_BAR + 0.5);
  config.tempoBpm = tempoBpm;
  config.seed = rng_();

  StringOutput output = stringEngine_.generate(config);

  // Convert StringNote to MidiNote
  std::vector<MidiNote> strings;
  for (const auto &note : output.notes) {
    strings.push_back(stringNoteToMidi(note, tempoBpm));
  }

  return strings;
}

// ============================================================================
// Counter Melody Generation
// ============================================================================

std::vector<MidiNote>
MidiGenerator::generateCounterMelody(const std::vector<MidiNote> &primaryMelody,
                                     const IntentResult &intent,
                                     float complexity) {
  // Get parameters from intent and emotion mapping
  std::string mode = intent.mode;
  std::string key = "C";

  auto musicalParams = EmotionMusicMapper::mapEmotion(intent.emotion);
  if (!musicalParams.detailedMode.empty())
    mode = musicalParams.detailedMode;
  if (!musicalParams.mode.empty())
    key = musicalParams.mode;

  // Use CounterMelodyEngine to generate counter melody
  CounterMelodyConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.primaryMelody = primaryMelody;
  config.key = key;
  config.mode = mode;
  config.density = complexity;
  config.seed = rng_();

  CounterMelodyOutput output = counterMelodyEngine_.generate(config);

  // Convert CounterMelodyNote to MidiNote
  std::vector<MidiNote> counterMelody;
  float tempoModifier = intent.tempo;
  int tempoBpm = static_cast<int>(TEMPO_FAST * tempoModifier);
  auto musicalParams2 = EmotionMusicMapper::mapEmotion(intent.emotion);
  tempoBpm = musicalParams2.tempo;
  for (const auto &note : output.notes) {
    counterMelody.push_back(counterMelodyNoteToMidi(note, tempoBpm));
  }

  return counterMelody;
}

// ============================================================================
// Rhythm and Fills Generation
// ============================================================================

std::vector<MidiNote> MidiGenerator::generateRhythm(const IntentResult &intent,
                                                    int bars, float complexity,
                                                    int tempoBpm) {
  // Create RhythmConfig from intent parameters
  RhythmConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.bars = bars;
  config.tempoBpm = tempoBpm;
  config.timeSignature = {static_cast<int>(BEATS_PER_BAR), 4}; // Default to 4/4
  config.seed = rng_();

  // Adjust density based on complexity
  if (complexity > INTENSITY_VERY_HIGH) {
    config.densityOverride = PatternDensity::Busy;
  } else if (complexity > AROUSAL_LOW) {
    config.densityOverride = PatternDensity::Moderate;
  } else {
    config.densityOverride = PatternDensity::Sparse;
  }

  // Call RhythmEngine to generate rhythm pattern
  RhythmOutput output = rhythmEngine_.generate(config);

  // Convert DrumHits to MidiNotes
  std::vector<MidiNote> rhythm;
  for (const auto &hit : output.hits) {
    MidiNote midiNote = drumHitToMidi(hit, tempoBpm);
    rhythm.push_back(midiNote);
  }

  return rhythm;
}

std::vector<MidiNote>
MidiGenerator::generateDrumGroove(const IntentResult &intent, int bars,
                                  float complexity, int tempoBpm) {
  // Create GrooveConfig from intent parameters
  GrooveConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.bars = bars;
  config.tempoBpm = tempoBpm;
  config.beatsPerBar = static_cast<int>(BEATS_PER_BAR); // Default to 4/4
  config.humanization = intent.humanization;            // From IntentResult
  config.seed = rng_();

  // Adjust groove style based on emotion and complexity
  if (intent.emotion.arousal > AROUSAL_HIGH) {
    config.styleOverride = GrooveType::Syncopated;
  } else if (intent.emotion.valence > VALENCE_POSITIVE) {
    config.styleOverride = GrooveType::Swing;
  }

  // Call DrumGrooveEngine to generate groove pattern
  GrooveOutput output = drumGrooveEngine_.generate(config);

  // Convert GrooveHits to MidiNotes
  std::vector<MidiNote> drumGroove;
  for (const auto &hit : output.hits) {
    MidiNote midiNote = grooveHitToMidi(hit, tempoBpm);
    drumGroove.push_back(midiNote);
  }

  return drumGroove;
}

std::vector<MidiNote>
MidiGenerator::generateTransitions(const IntentResult &intent, int bars,
                                   int tempoBpm) {
  std::vector<MidiNote> transitions;

  // Determine transition points (e.g., every 4 bars, at end)
  // For simplicity, generate transitions at section boundaries
  int transitionBars = 2; // Duration of each transition
  std::vector<int> transitionPoints;

  // Add transition before final section (if bars > 4)
  if (bars > 4) {
    transitionPoints.push_back(bars - 2);
  }

  // Add transitions every 8 bars (if long enough)
  for (int i = 8; i < bars; i += 8) {
    transitionPoints.push_back(i);
  }

  // Generate transition at each point
  for (int bar : transitionPoints) {
    // Determine transition type based on emotion
    TransitionType transitionType = TransitionType::Crossfade;
    if (intent.emotion.arousal > AROUSAL_HIGH) {
      transitionType = TransitionType::Build;
    } else if (intent.emotion.arousal < AROUSAL_LOW) {
      transitionType = TransitionType::Breakdown;
    }

    // Create TransitionConfig
    TransitionConfig config;
    config.emotion = getEmotionName(intent.emotion);
    config.type = transitionType;
    config.durationBars = transitionBars;
    config.tempoBpm = tempoBpm;
    config.intensity = intent.emotion.intensity;
    config.seed = rng_();

    // Generate transition
    TransitionOutput output = transitionEngine_.generate(config);

    // Convert TransitionNotes to MidiNotes and adjust start position
    int startTicks = beatsToTicks(bar * BEATS_PER_BAR, tempoBpm);
    for (const auto &note : output.notes) {
      MidiNote midiNote = transitionNoteToMidi(note, tempoBpm);
      // Adjust start position to account for transition bar offset
      midiNote.startBeat += bar * BEATS_PER_BAR;
      transitions.push_back(midiNote);
    }
  }

  return transitions;
}

std::vector<MidiNote> MidiGenerator::generateFills(const IntentResult &intent,
                                                   int bars, float complexity) {
  // Get parameters from intent and emotion mapping
  float tempoModifier = intent.tempo;
  int tempoBpm = static_cast<int>(TEMPO_FAST * tempoModifier);

  auto musicalParams = EmotionMusicMapper::mapEmotion(intent.emotion);
  tempoBpm = musicalParams.tempo;

  // Generate fills at strategic points (every 4 bars or at end)
  std::vector<MidiNote> fills;

  // Generate fill at the end
  FillConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.length = FillLength::Full;
  config.startTick = beatsToTicks((bars - 1) * BEATS_PER_BAR, tempoBpm);
  config.tempoBpm = tempoBpm;
  config.seed = rng_();

  FillOutput output = fillEngine_.generate(config);

  // Convert DrumHits to MidiNotes (for fills, we might want to keep as drum
  // hits) For now, we'll convert them
  for (const auto &hit : output.hits) {
    fills.push_back(drumHitToMidi(hit, tempoBpm));
  }

  return fills;
}

// ============================================================================
// Arrangement Generation
// ============================================================================

ArrangementOutput MidiGenerator::generateArrangement(const IntentResult &intent,
                                                     int bars) {
  // Create ArrangementConfig from intent parameters
  ArrangementConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.targetBars = bars;
  config.seed = rng_();

  // Determine emotional arc from emotion valence
  // Positive valence = ascending arc, negative = descending
  if (intent.emotion.valence > VALENCE_POSITIVE) {
    config.emotionalArc = 0.7f; // Ascending
  } else if (intent.emotion.valence < VALENCE_NEGATIVE) {
    config.emotionalArc = 0.3f; // Descending
  } else {
    config.emotionalArc = 0.5f; // Cyclical
  }

  // Include intro/outro based on length
  config.includeIntro = bars >= 8;
  config.includeOutro = bars >= 8;

  // Generate arrangement structure
  ArrangementOutput output = arrangementEngine_.generate(config);

  return output;
}

// ============================================================================
// Dynamics Application
// ============================================================================

void MidiGenerator::applyDynamics(GeneratedMidi &midi,
                                  const IntentResult &intent, float dynamics) {
  // Apply dynamics to all layers
  DynamicsConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.expressiveness = intent.dynamicRange; // From IntentResult
  config.applyAccents = true;

  // Apply to melody
  if (!midi.melody.empty()) {
    config.notes = midi.melody;
    config.totalTicks =
        beatsToTicks(midi.lengthInBeats, static_cast<int>(midi.bpm));
    DynamicsOutput output = dynamicsEngine_.apply(config);
    midi.melody = output.notes;
  }

  // Apply to bass
  if (!midi.bass.empty()) {
    config.notes = midi.bass;
    DynamicsOutput output = dynamicsEngine_.apply(config);
    midi.bass = output.notes;
  }

  // Apply to counter melody
  if (!midi.counterMelody.empty()) {
    config.notes = midi.counterMelody;
    DynamicsOutput output = dynamicsEngine_.apply(config);
    midi.counterMelody = output.notes;
  }
}

// ============================================================================
// Tension Application
// ============================================================================

void MidiGenerator::applyTension(GeneratedMidi &midi,
                                 const IntentResult &intent, int bars) {
  // Get parameters from intent and emotion mapping
  float tempoModifier = intent.tempo;
  int tempoBpm = static_cast<int>(TEMPO_FAST * tempoModifier);

  auto musicalParams = EmotionMusicMapper::mapEmotion(intent.emotion);
  tempoBpm = musicalParams.tempo;

  // Generate tension curve
  std::vector<std::string> chordStrings = chordsToStrings(midi.chords);

  TensionConfig config;
  config.emotion = getEmotionName(intent.emotion);
  config.chordProgression = chordStrings;
  config.bars = bars;
  config.tempoBpm = tempoBpm;
  config.maxTension = musicalParams.resonance; // From MusicalParameters
  config.seed = rng_();

  TensionOutput output = tensionEngine_.generate(config);

  // Apply tension to chords (add tension notes based on curve)
  // This is a simplified application - in a full implementation,
  // we would modify chord pitches based on tension points
  for (size_t i = 0; i < midi.chords.size() && i < output.tensionCurve.size();
       ++i) {
    float tension = output.tensionCurve[i];
    if (tension > INTENSITY_MODERATE && !midi.chords[i].pitches.empty()) {
      // Add tension notes (simplified - just add a semitone above root)
      int tensionNote = midi.chords[i].pitches[0] + INTERVAL_MINOR_SECOND;
      if (std::find(midi.chords[i].pitches.begin(),
                    midi.chords[i].pitches.end(),
                    tensionNote) == midi.chords[i].pitches.end()) {
        midi.chords[i].pitches.push_back(tensionNote);
      }
    }
  }
}

// ============================================================================
// Rule Breaks Application
// ============================================================================

void MidiGenerator::applyRuleBreaks(GeneratedMidi &midi,
                                    const IntentResult &intent) {
  for (const auto &ruleBreak : intent.ruleBreaks) {
    // RuleBreak from Types.h has: type, severity (method), description, reason
    // Use severity to determine intensity of rule break application
    float severity = ruleBreak.severity();

    switch (ruleBreak.type) {
    case RuleBreakType::ModalMixture:
    case RuleBreakType::HarmonicAmbiguity: {
      // Apply dissonance to chords based on severity
      if (severity > RULE_BREAK_LOW) {
        for (auto &chord : midi.chords) {
          if (!chord.pitches.empty() && severity > RULE_BREAK_MODERATE) {
            // Add cluster notes or dissonant intervals
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            if (dist(rng_) < severity) {
              // Add a semitone cluster (dissonant interval)
              int clusterNote = chord.pitches[0] + INTERVAL_MINOR_SECOND;
              if (std::find(chord.pitches.begin(), chord.pitches.end(),
                            clusterNote) == chord.pitches.end()) {
                chord.pitches.push_back(clusterNote);
              }
            }
          }
        }
      }
      break;
    }

    case RuleBreakType::CrossRhythm: {
      // Apply syncopation based on severity
      if (severity > RULE_BREAK_LOW) {
        float syncAmount = severity * SYNCOPATION_MAX_SHIFT;
        for (auto &note : midi.melody) {
          std::uniform_real_distribution<float> dist(-syncAmount, syncAmount);
          note.startBeat += dist(rng_);
        }
        // Also apply to bass
        for (auto &note : midi.bass) {
          std::uniform_real_distribution<float> dist(
              -syncAmount * BASS_HUMANIZE_MULTIPLIER,
              syncAmount * BASS_HUMANIZE_MULTIPLIER);
          note.startBeat += dist(rng_);
        }
      }
      break;
    }

    case RuleBreakType::DynamicContrast: {
      // Apply extreme dynamic range based on severity
      int minVel = static_cast<int>(
          MIDI_VELOCITY_SOFT - DYNAMICS_RULE_BREAK_VELOCITY_ADJUSTMENT +
          (1.0f - severity) *
              (MIDI_VELOCITY_SOFT - DYNAMICS_RULE_BREAK_VELOCITY_ADJUSTMENT));
      int maxVel = static_cast<int>(
          MIDI_VELOCITY_MEDIUM + DYNAMICS_RULE_BREAK_VELOCITY_ADJUSTMENT +
          severity * (MIDI_VELOCITY_MAX - MIDI_VELOCITY_MEDIUM -
                      DYNAMICS_RULE_BREAK_VELOCITY_ADJUSTMENT));
      minVel = std::clamp(minVel, MIDI_VELOCITY_MIN, MIDI_VELOCITY_MAX);
      maxVel = std::clamp(maxVel, MIDI_VELOCITY_MIN, MIDI_VELOCITY_MAX);
      for (auto &note : midi.melody) {
        std::uniform_int_distribution<int> velDist(minVel, maxVel);
        note.velocity = velDist(rng_);
      }
      break;
    }

    case RuleBreakType::ParallelMotion:
    case RuleBreakType::RegisterShift: {
      // Apply melodic rule breaks (wide leaps, chromaticism)
      if (severity > RULE_BREAK_MODERATE) {
        for (auto &note : midi.melody) {
          // Add chromaticism occasionally
          std::uniform_real_distribution<float> dist(0.0f, 1.0f);
          if (dist(rng_) < severity * CHROMATICISM_PROBABILITY_FACTOR) {
            note.pitch +=
                (dist(rng_) < 0.5f ? -INTERVAL_MINOR_SECOND
                                   : INTERVAL_MINOR_SECOND); // Chromatic shift
            note.pitch = std::clamp(note.pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
          }
        }
      }
      break;
    }

    case RuleBreakType::UnresolvedTension: {
      // Apply structural breaks (rests, fragmentation)
      if (severity > RULE_BREAK_LOW) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float restProb = severity * REST_PROBABILITY_FACTOR;
        for (auto &note : midi.melody) {
          if (dist(rng_) < restProb) {
            // Remove note by setting duration to 0 (or skip in playback)
            note.duration = 0.0;
          }
        }
      }
      break;
    }
    }
  }
}

// ============================================================================
// Variations Application
// ============================================================================

void MidiGenerator::applyVariations(GeneratedMidi &midi,
                                    const IntentResult &intent,
                                    float complexity) {
  // Apply variations to melody, bass, and counter melody
  std::string key = "C";
  std::string mode = intent.mode;

  auto musicalParams = EmotionMusicMapper::mapEmotion(intent.emotion);
  if (!musicalParams.mode.empty())
    key = musicalParams.mode;
  if (!musicalParams.detailedMode.empty())
    mode = musicalParams.detailedMode;

  // Apply variations to melody if it exists
  if (!midi.melody.empty()) {
    VariationConfig config;
    config.emotion = getEmotionName(intent.emotion);
    config.source = midi.melody;
    config.key = key;
    config.mode = mode;
    config.intensity = complexity;
    config.preserveContour = true;
    config.preserveRhythm = false;
    config.seed = rng_();

    VariationOutput output = variationEngine_.generate(config);

    // Blend original with variation based on complexity
    // Higher complexity = more variation
    float blendAmount = complexity * MAX_VARIATION_BLEND;
    if (blendAmount > VARIATION_BLEND_THRESHOLD && !output.notes.empty()) {
      // For simplicity, replace with variation if intensity is high enough
      // In a full implementation, we'd blend notes
      if (complexity > VARIATION_REPLACE_THRESHOLD) {
        midi.melody = output.notes;
      }
    }
  }

  // Apply variations to bass (lighter touch)
  if (!midi.bass.empty() && complexity > BASS_VARIATION_THRESHOLD) {
    VariationConfig config;
    config.emotion = getEmotionName(intent.emotion);
    config.source = midi.bass;
    config.key = key;
    config.mode = mode;
    config.preserveContour = true;
    config.preserveRhythm = true; // Preserve rhythm for bass
    config.intensity = complexity * BASS_VARIATION_INTENSITY_MULTIPLIER;
    config.seed = rng_();

    VariationOutput output = variationEngine_.generate(config);
    if (complexity > BASS_VARIATION_REPLACE_THRESHOLD &&
        !output.notes.empty()) {
      midi.bass = output.notes;
    }
  }

  // Apply variations to counter melody (if exists)
  if (!midi.counterMelody.empty() &&
      complexity > COUNTER_MELODY_VARIATION_THRESHOLD) {
    VariationConfig config;
    config.emotion = getEmotionName(intent.emotion);
    config.source = midi.counterMelody;
    config.key = key;
    config.mode = mode;
    config.intensity =
        complexity * COUNTER_MELODY_VARIATION_INTENSITY_MULTIPLIER;
    config.preserveContour = true;
    config.preserveRhythm = false;
    config.seed = rng_();

    VariationOutput output = variationEngine_.generate(config);
    if (complexity > COUNTER_MELODY_VARIATION_REPLACE_THRESHOLD &&
        !output.notes.empty()) {
      midi.counterMelody = output.notes;
    }
  }
}

// ============================================================================
// Groove and Humanization
// ============================================================================

void MidiGenerator::applyGrooveAndHumanize(GeneratedMidi &midi, float humanize,
                                           const EmotionNode &emotion,
                                           float feel) {
  if (humanize < MIN_HUMANIZE_THRESHOLD &&
      std::abs(feel) < MIN_HUMANIZE_THRESHOLD)
    return;

  // Determine groove type from emotion
  GrooveType grooveType = GrooveType::Straight;
  if (emotion.arousal > AROUSAL_HIGH)
    grooveType = GrooveType::Syncopated;
  else if (emotion.arousal < AROUSAL_LOW)
    grooveType = GrooveType::Shuffle;
  else if (emotion.valence > VALENCE_POSITIVE)
    grooveType = GrooveType::Swing;

  // Apply emotion-based timing
  midi.melody = grooveEngine_.applyEmotionTiming(midi.melody, emotion);
  midi.bass = grooveEngine_.applyEmotionTiming(midi.bass, emotion);
  if (!midi.counterMelody.empty()) {
    midi.counterMelody =
        grooveEngine_.applyEmotionTiming(midi.counterMelody, emotion);
  }

  // Apply groove
  midi.melody = grooveEngine_.applyGroove(midi.melody, grooveType, humanize);
  midi.bass = grooveEngine_.applyGroove(midi.bass, grooveType,
                                        humanize * BASS_HUMANIZE_MULTIPLIER);
  if (!midi.counterMelody.empty()) {
    midi.counterMelody = grooveEngine_.applyGroove(
        midi.counterMelody, grooveType,
        humanize * COUNTER_MELODY_HUMANIZE_MULTIPLIER);
  }

  // Apply feel (pull/push) using GrooveEngine's timing feel
  if (std::abs(feel) > MIN_HUMANIZE_THRESHOLD) {
    midi.melody = grooveEngine_.applyTimingFeel(midi.melody, feel, 1.0f);
    midi.bass = grooveEngine_.applyTimingFeel(midi.bass, feel,
                                              BASS_HUMANIZE_MULTIPLIER);
    if (!midi.counterMelody.empty()) {
      midi.counterMelody = grooveEngine_.applyTimingFeel(
          midi.counterMelody, feel, COUNTER_MELODY_HUMANIZE_MULTIPLIER);
    }
  }
}

// ============================================================================
// Helper Methods - Conversions
// ============================================================================

std::vector<std::string>
MidiGenerator::chordsToStrings(const std::vector<Chord> &chords) {
  std::vector<std::string> result;
  for (const auto &chord : chords) {
    result.push_back(chord.name);
  }
  return result;
}

double MidiGenerator::ticksToBeats(int ticks, int tempoBpm) {
  juce::ignoreUnused(
      tempoBpm); // Tempo doesn't affect ticks-to-beats conversion
  return static_cast<double>(ticks) / MIDI_PPQ;
}

int MidiGenerator::beatsToTicks(double beats, int tempoBpm) {
  juce::ignoreUnused(
      tempoBpm); // Tempo doesn't affect beats-to-ticks conversion
  return static_cast<int>(beats * MIDI_PPQ);
}

MidiNote MidiGenerator::melodyNoteToMidi(const MelodyNote &note, int tempoBpm) {
  MidiNote midiNote;
  midiNote.pitch = note.pitch;
  midiNote.velocity = note.velocity;
  midiNote.startBeat = ticksToBeats(note.startTick, tempoBpm);
  midiNote.duration = ticksToBeats(note.durationTicks, tempoBpm);
  return midiNote;
}

MidiNote MidiGenerator::bassNoteToMidi(const BassNote &note, int tempoBpm) {
  MidiNote midiNote;
  midiNote.pitch = note.pitch;
  midiNote.velocity = note.velocity;
  midiNote.startBeat = ticksToBeats(note.startTick, tempoBpm);
  midiNote.duration = ticksToBeats(note.durationTicks, tempoBpm);
  return midiNote;
}

MidiNote MidiGenerator::padNoteToMidi(const PadNote &note, int tempoBpm) {
  MidiNote midiNote;
  midiNote.pitch = note.pitch;
  midiNote.velocity = note.velocity;
  midiNote.startBeat = ticksToBeats(note.startTick, tempoBpm);
  midiNote.duration = ticksToBeats(note.durationTicks, tempoBpm);
  return midiNote;
}

MidiNote MidiGenerator::stringNoteToMidi(const StringNote &note, int tempoBpm) {
  MidiNote midiNote;
  midiNote.pitch = note.pitch;
  midiNote.velocity = note.velocity;
  midiNote.startBeat = ticksToBeats(note.startTick, tempoBpm);
  midiNote.duration = ticksToBeats(note.durationTicks, tempoBpm);
  return midiNote;
}

MidiNote MidiGenerator::counterMelodyNoteToMidi(const CounterMelodyNote &note,
                                                int tempoBpm) {
  MidiNote midiNote;
  midiNote.pitch = note.pitch;
  midiNote.velocity = note.velocity;
  midiNote.startBeat = ticksToBeats(note.startTick, tempoBpm);
  midiNote.duration = ticksToBeats(note.durationTicks, tempoBpm);
  return midiNote;
}

MidiNote MidiGenerator::drumHitToMidi(const DrumHit &hit, int tempoBpm) {
  MidiNote midiNote;
  midiNote.pitch = hit.note; // Drum note number
  midiNote.velocity = hit.velocity;
  midiNote.startBeat = ticksToBeats(hit.startTick + hit.timingOffset, tempoBpm);
  midiNote.duration = ticksToBeats(hit.duration, tempoBpm);
  return midiNote;
}

MidiNote MidiGenerator::grooveHitToMidi(const GrooveHit &hit, int tempoBpm) {
  MidiNote midiNote;
  midiNote.pitch = hit.pitch; // Drum note number from GrooveHit
  midiNote.velocity = hit.velocity;
  midiNote.startBeat = ticksToBeats(hit.tick, tempoBpm);
  midiNote.duration = ticksToBeats(hit.durationTicks, tempoBpm);
  return midiNote;
}

MidiNote MidiGenerator::transitionNoteToMidi(const TransitionNote &note,
                                             int tempoBpm) {
  MidiNote midiNote;
  midiNote.pitch = note.pitch;
  midiNote.velocity = note.velocity;
  midiNote.startBeat = ticksToBeats(note.startTick, tempoBpm);
  midiNote.duration = ticksToBeats(note.durationTicks, tempoBpm);
  return midiNote;
}

// ============================================================================
// Helper Methods - Utilities
// ============================================================================

std::string MidiGenerator::getEmotionName(const EmotionNode &emotion) {
  // Convert emotion name to lowercase for engine compatibility
  std::string name = emotion.name;
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  return name;
}

MidiGenerator::LayerFlags MidiGenerator::determineLayers(
    const IntentResult &intent, float complexity, int bars,
    const std::optional<ArrangementOutput> &arrangement) {
  LayerFlags flags;

  // Always generate melody and bass
  flags.melody = true;
  flags.bass = true;

  // If arrangement exists, use it to inform layer selection
  if (arrangement.has_value()) {
    const auto &arr = arrangement.value();

    // Check if arrangement has high-energy sections (Chorus, Drop, Build)
    bool hasHighEnergySections = false;
    for (const auto &section : arr.sections) {
      if (section.energy > INTENSITY_HIGH) {
        hasHighEnergySections = true;
        break;
      }
    }

    // Pads: use arrangement sections or default logic
    flags.pads = complexity > PADS_COMPLEXITY_THRESHOLD ||
                 intent.emotion.arousal < AROUSAL_LOW;

    // Strings: high complexity, high intensity, or high-energy sections
    flags.strings = complexity > INTENSITY_HIGH ||
                    intent.emotion.intensity > INTENSITY_VERY_HIGH ||
                    hasHighEnergySections;

    // Counter melody: high complexity or chorus sections
    flags.counterMelody = complexity > INTENSITY_VERY_HIGH;

    // Rhythm and drum groove: medium complexity, high arousal, or high-energy
    // sections
    flags.rhythm = complexity > RULE_BREAK_MODERATE ||
                   intent.emotion.arousal > AROUSAL_HIGH ||
                   hasHighEnergySections;
    flags.drumGroove = complexity > RULE_BREAK_MODERATE ||
                       intent.emotion.arousal > AROUSAL_HIGH ||
                       hasHighEnergySections;

    // Fills: medium complexity or sections that typically have fills
    flags.fills = complexity > RULE_BREAK_MODERATE;

    // Transitions: enabled when arrangement has multiple sections
    flags.transitions = arr.sections.size() > 1;

    // Variations: high complexity
    flags.variations = complexity > INTENSITY_VERY_HIGH;
  } else {
    // Default logic when no arrangement
    flags.pads = complexity > PADS_COMPLEXITY_THRESHOLD ||
                 intent.emotion.arousal < AROUSAL_LOW;
    flags.strings = complexity > INTENSITY_HIGH ||
                    intent.emotion.intensity > INTENSITY_VERY_HIGH;
    flags.counterMelody = complexity > INTENSITY_VERY_HIGH;
    flags.rhythm = complexity > RULE_BREAK_MODERATE ||
                   intent.emotion.arousal > AROUSAL_HIGH;
    flags.drumGroove = complexity > RULE_BREAK_MODERATE ||
                       intent.emotion.arousal > AROUSAL_HIGH;
    flags.fills = complexity > RULE_BREAK_MODERATE;
    flags.transitions = bars > 8;
    flags.variations = complexity > INTENSITY_VERY_HIGH;
  }

  return flags;
}

} // namespace kelly
