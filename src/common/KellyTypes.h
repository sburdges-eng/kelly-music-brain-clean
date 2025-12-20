#pragma once
/*
 * KellyTypes.h - Unified Type Definitions
 * ========================================
 * Single source of truth for all Kelly types.
 * Resolves conflicts between Types.h, midi_pipeline.h, emotion_engine.h
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace kelly {

// =============================================================================
// FORWARD DECLARATIONS
// =============================================================================
struct MidiNote;
struct EmotionNode;
struct Wound;
struct IntentResult;
struct ArrangementOutput; // Forward declaration (defined in
                          // engines/ArrangementEngine.h)

// =============================================================================
// CONSTANTS
// =============================================================================
constexpr int TICKS_PER_BEAT = 480;
constexpr int TICKS_PER_QUARTER = 480;
constexpr int DRUM_CHANNEL = 9;

// =============================================================================
// ENUMS
// =============================================================================

enum class EmotionCategory : uint8_t {
  Joy = 0,
  Sadness,
  Anger,
  Fear,
  Surprise,
  Disgust,
  Trust,
  Anticipation,
  COUNT
};

enum class RuleBreakType : uint8_t {
  None = 0,
  ModalMixture,      // Borrowed chords for grief
  ParallelMotion,    // Defiance of voice leading rules
  UnresolvedTension, // Anxiety through non-resolution
  CrossRhythm,       // Disorientation
  RegisterShift,     // Vulnerability through exposure
  DynamicContrast,   // Emotional whiplash
  HarmonicAmbiguity, // Uncertain emotional state
  COUNT
};

// =============================================================================
// MIDI STRUCTURES (Unified - resolves midi_pipeline.h conflict)
// =============================================================================

struct MidiNote {
  int pitch = 60;          // MIDI note number 0-127
  int startTick = 0;       // Start time in ticks
  int durationTicks = 480; // Duration in ticks
  int velocity = 100;      // Velocity 0-127
  int channel = 0;         // MIDI channel 0-15

  // Compatibility methods for existing Types.h MidiNote structure
  uint8_t note() const { return static_cast<uint8_t>(pitch); }
  uint32_t time() const { return static_cast<uint32_t>(startTick); }

  // Compatibility fields for existing Types.h MidiNote structure
  // These fields are computed from startTick/durationTicks when accessed
  mutable double startBeat = 0.0; // Position in beats
  mutable double duration = 0.0;  // Length in beats
};

struct MidiEvent {
  int tick = 0;
  int type = 0x90; // 0x90 = note on, 0x80 = note off
  int channel = 0;
  int data1 = 0;
  int data2 = 0;
};

struct Chord {
  std::string symbol; // "Am7", "Fmaj7"
  std::string root = "C";
  std::string quality = "major";
  std::vector<int> intervals; // Semitones from root
  std::vector<int> pitches;   // Actual MIDI pitches
  int bass = -1;              // For slash chords
  std::string romanNumeral;   // "vi", "IV"

  // Compatibility fields for existing Types.h Chord structure
  std::string name; // Alias for symbol
  double startBeat = 0.0;
  double duration = 0.0;
};

struct GeneratedMidi {
  std::vector<MidiNote> notes;
  std::vector<Chord> chords;
  int tempoBpm = 120;
  int bars = 4;
  std::string key = "C";
  std::string mode = "major";
  std::map<std::string, std::string> metadata;

  // Compatibility fields for existing GeneratedMidi structure (Types.h)
  std::vector<MidiNote> melody;
  std::vector<MidiNote> bass;
  std::vector<MidiNote> counterMelody;
  std::vector<MidiNote> pad;
  std::vector<MidiNote> strings;
  std::vector<MidiNote> fills;
  std::vector<MidiNote> rhythm;
  std::vector<MidiNote> drumGroove;
  std::vector<MidiNote> transitions;
  double lengthInBeats = 0.0;
  float bpm = 120.0f;

  // Use pointer to avoid requiring complete type definition (avoids circular
  // dependency)
  ArrangementOutput *arrangement =
      nullptr; // Section structure metadata (optional, null if not set)
};

// =============================================================================
// TIME SIGNATURES
// =============================================================================

struct TimeSignature {
  int numerator = 4;
  int denominator = 4;
};

struct KeySignature {
  std::string root = "C";
  std::string mode = "major";
  int sharpsFlats = 0;
};

// =============================================================================
// EMOTION SYSTEM (Unified - resolves emotion_engine.h conflict)
// =============================================================================

struct MusicalAttributes {
  float tempoModifier = 1.0f; // 0.5 = half speed, 2.0 = double
  std::string mode = "minor";
  float dynamics = 0.5f;     // 0-1
  float articulation = 0.5f; // Staccato (0) to Legato (1)
  float density = 0.5f;      // Sparse (0) to Dense (1)
  float dissonance = 0.0f;   // Consonant (0) to Dissonant (1)
  std::vector<RuleBreakType> suggestedRuleBreaks;
};

struct EmotionNode {
  int id = 0;
  std::string name;
  std::string category; // String for serialization
  EmotionCategory categoryEnum = EmotionCategory::Joy;

  // Dimensional model
  float valence = 0.0f;   // Negative (-1) to Positive (+1)
  float arousal = 0.5f;   // Calm (0) to Excited (1)
  float dominance = 0.5f; // Submissive (0) to Dominant (1)
  float intensity = 0.5f; // How strong the emotion is

  // Relationships
  std::vector<int> relatedEmotions;
  std::vector<std::string> synonyms;

  // Musical mapping
  MusicalAttributes musicalAttributes;

  // 216-node thesaurus position (6x6x6)
  int layerIndex = 0;  // Base emotion (0-5)
  int subIndex = 0;    // Sub-emotion (0-5)
  int subSubIndex = 0; // Sub-sub-emotion (0-5)

  // ML Enhancement (Phase 1: ML Infrastructure & Node Enhancement)
  // Optional ML embeddings from EmotionRecognizer model
  std::optional<std::vector<float>>
      mlEmbedding;                         // 64-dim from EmotionRecognizer
  std::optional<float> mlConfidence;       // Model confidence score (0.0-1.0)
  std::map<std::string, float> mlFeatures; // Additional ML-derived features
};

struct EmotionState {
  std::string primary;
  float intensity = 0.5f;
  std::string secondary;
  float secondaryIntensity = 0.0f;
  std::map<std::string, float> dimensions;
};

// =============================================================================
// INTENT SYSTEM (Wound → Music)
// =============================================================================

struct Wound {
  std::string description; // "Finding Kelly sleeping forever"
  std::string desire;      // "To honor her memory through music"
  float urgency = 0.5f;    // How pressing this feeling is
  std::string expression;  // "Misdirection piece - sounds like love"
  EmotionNode primaryEmotion;
  EmotionNode secondaryEmotion;

  // Compatibility fields for existing code
  std::string source; // Source identifier (e.g., "user_input", "text_input")
  float intensity =
      0.5f; // Alias for urgency (for compatibility with existing code)
};

struct RuleBreak {
  RuleBreakType type = RuleBreakType::None;
  std::string description;
  std::string justification; // Emotional reason for breaking rule
  float intensity = 0.5f;    // How strongly to apply

  // Compatibility fields for existing Types.h RuleBreak structure
  float &severity() { return intensity; }
  const float &severity() const { return intensity; }
  std::string &reason() { return justification; }
  const std::string &reason() const { return justification; }
};

struct IntentResult {
  // Musical parameters derived from wound
  std::string key = "F";
  std::string mode = "major"; // Major for misdirection
  int tempoBpm = 82;
  TimeSignature timeSignature = {4, 4};

  // Harmonic choices
  std::vector<std::string> chordProgression; // ["F", "C", "Am", "Dm"]
  std::vector<RuleBreak> ruleBreaks;

  // Melodic guidance
  float melodicRange = 0.6f;    // Octave span (0-1)
  float leapProbability = 0.3f; // Chance of interval > 3rd
  bool allowChromaticism = false;

  // Compatibility field for existing Types.h IntentResult
  bool allowDissonance = false; // Maps to allowChromaticism

  // Rhythmic guidance
  float swingAmount = 0.0f;
  float syncopationLevel = 0.3f;
  float humanization = 0.15f; // Timing imperfection

  // Dynamics
  float baseVelocity = 0.6f;
  float dynamicRange = 0.4f;

  // Production notes
  std::vector<std::string> productionNotes;
  std::string narrativeArc;

  // Source tracking
  Wound sourceWound;
  float confidence = 0.8f;

  // Compatibility fields for existing IntentResult structure (Types.h)
  // These map to the unified structure fields above
  Wound &wound() { return sourceWound; }
  const Wound &wound() const { return sourceWound; }
  EmotionNode emotion; // Emotion associated with intent (should be synced with
                       // sourceWound.primaryEmotion)
  float tempo = 1.0f;  // BPM modifier (0.5 to 2.0) - computed from tempoBpm
};

// =============================================================================
// GROOVE/RHYTHM
// =============================================================================

enum class GrooveType {
  Straight,
  Swing,
  Syncopated,
  Halftime,
  Shuffle,
  Human,
  Broken,
  DoubleTime,
  FourOnFloor,
  Trap
};

struct GrooveTemplate {
  GrooveType type;
  std::string name;
  int numerator;
  int denominator;
  float swingAmount;                          // 0.0 to 1.0
  std::vector<std::pair<float, int>> pattern; // (beat position, velocity)
};

struct GroovePattern {
  std::string name;
  std::vector<bool> kickPattern; // 16 steps
  std::vector<bool> snarePattern;
  std::vector<float> hatPattern; // 0-1 probability
  float swingAmount = 0.0f;
  float humanize = 0.0f;
};

struct DrumHit {
  std::string type; // Drum type name (e.g., "kick", "snare", "tom", "hat")
  int note;         // MIDI drum note (General MIDI percussion map)
  int startTick;    // Tick position (base position, also accessible as 'tick')
  int duration;     // Ticks (usually short for drums)
  int velocity;     // 0-127
  bool isGhost = false; // Ghost note (very quiet)
  int timingOffset = 0; // Humanization offset in ticks

  // Getter for backwards compatibility (some code uses .tick instead of
  // .startTick)
  inline int &getTick() { return startTick; }
  inline const int &getTick() const { return startTick; }
};

// =============================================================================
// SIDE A / SIDE B (Cassette Metaphor)
// =============================================================================

struct SideA {
  std::string description; // "Where you are" - current emotional state
  float intensity;
  std::optional<int> emotionId; // If user selected from wheel
};

struct SideB {
  std::string description; // "Where you want to go" - desired state
  float intensity;
  std::optional<int> emotionId;
};

struct CassetteState {
  SideA sideA;
  SideB sideB;
  bool isFlipped; // Which side is currently "active"
};

// =============================================================================
// GENERATION CONFIG
// =============================================================================

struct GenerationConfig {
  int bars = 8;
  int tempoBpm = 120;
  TimeSignature timeSignature = {4, 4};
  KeySignature key;
  std::string emotion = "neutral";
  std::string genre;
  int seed = -1;

  float humanization = 0.1f;
  float expressiveness = 0.5f;
  float complexity = 0.5f;
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

inline int ticksPerBar(const TimeSignature &ts) {
  return ts.numerator * TICKS_PER_BEAT * 4 / ts.denominator;
}

inline int msToTicks(int ms, int tempoBpm) {
  return static_cast<int>(ms * TICKS_PER_BEAT * tempoBpm / 60000.0);
}

inline int ticksToMs(int ticks, int tempoBpm) {
  return static_cast<int>(ticks * 60000.0 / (TICKS_PER_BEAT * tempoBpm));
}

inline int midiToFrequency(int midiNote) {
  return static_cast<int>(440.0 * std::pow(2.0, (midiNote - 69) / 12.0));
}

inline std::string midiNoteToName(int note) {
  static const char *names[] = {"C",  "C#", "D",  "D#", "E",  "F",
                                "F#", "G",  "G#", "A",  "A#", "B"};
  int octave = (note / 12) - 1;
  return std::string(names[note % 12]) + std::to_string(octave);
}

inline int noteNameToMidi(const std::string &name) {
  static const std::map<std::string, int> noteMap = {
      {"C", 0},  {"C#", 1}, {"Db", 1},  {"D", 2},   {"D#", 3}, {"Eb", 3},
      {"E", 4},  {"F", 5},  {"F#", 6},  {"Gb", 6},  {"G", 7},  {"G#", 8},
      {"Ab", 8}, {"A", 9},  {"A#", 10}, {"Bb", 10}, {"B", 11}};

  if (name.empty())
    return 60;

  size_t i = 1;
  if (name.size() > 1 && (name[1] == '#' || name[1] == 'b'))
    i = 2;

  std::string notePart = name.substr(0, i);
  int octave = 4;
  if (i < name.size()) {
    octave = std::stoi(name.substr(i));
  }

  auto it = noteMap.find(notePart);
  if (it == noteMap.end())
    return 60;

  return (octave + 1) * 12 + it->second;
}

inline const char *categoryToString(EmotionCategory cat) {
  static const char *names[] = {"Joy",      "Sadness", "Anger", "Fear",
                                "Surprise", "Disgust", "Trust", "Anticipation"};
  return names[static_cast<int>(cat)];
}

inline const char *ruleBreakToString(RuleBreakType type) {
  static const char *names[] = {"None",
                                "Harmony",
                                "Rhythm",
                                "Dynamics",
                                "Melody",
                                "Form",
                                "ModalMixture",
                                "ParallelMotion",
                                "UnresolvedTension",
                                "CrossRhythm",
                                "RegisterShift",
                                "DynamicContrast",
                                "HarmonicAmbiguity"};
  size_t idx = static_cast<size_t>(type);
  if (idx < sizeof(names) / sizeof(names[0])) {
    return names[idx];
  }
  return "Unknown";
}

} // namespace kelly
