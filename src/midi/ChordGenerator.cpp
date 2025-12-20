#include "midi/ChordGenerator.h"
#include "common/MusicConstants.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace kelly {
using namespace MusicConstants;

ChordGenerator::ChordGenerator()
    : rng_(std::random_device{}()),
      voiceLeadingEngine_(std::make_unique<VoiceLeadingEngine>()) {
  std::lock_guard<std::mutex> lock(mutex_);
  initializeProgressionFamilies();
  initializeTemplates();
}

// =============================================================================
// PROGRESSION FAMILY INITIALIZATION
// =============================================================================

void ChordGenerator::initializeProgressionFamilies() {
  // Universal progressions
  families_.push_back({"I-IV-V-I",
                       "universal",
                       {INTERVAL_UNISON, INTERVAL_PERFECT_FOURTH,
                        INTERVAL_PERFECT_FIFTH, INTERVAL_UNISON},
                       {"I", "IV", "V", "I"},
                       {0.5f, 1.0f}, // Positive valence
                       {0.4f, 1.0f}, // Moderate to high arousal
                       "Classic resolution, strong and stable",
                       false});

  families_.push_back({"I-V-vi-IV",
                       "universal",
                       {INTERVAL_UNISON, INTERVAL_PERFECT_FIFTH,
                        INTERVAL_MAJOR_SIXTH, INTERVAL_PERFECT_FOURTH},
                       {"I", "V", "vi", "IV"},
                       {0.2f, 0.8f}, // Wide valence range
                       {0.3f, 0.8f}, // Moderate arousal
                       "Uplifting yet emotional",
                       true});

  families_.push_back({"I-vi-IV-V",
                       "universal",
                       {INTERVAL_UNISON, INTERVAL_MAJOR_SIXTH,
                        INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH},
                       {"I", "vi", "IV", "V"},
                       {0.3f, 0.7f}, // Positive
                       {0.2f, 0.6f}, // Lower arousal
                       "Nostalgic, classic",
                       false});

  families_.push_back({"vi-IV-I-V",
                       "universal",
                       {INTERVAL_MAJOR_SIXTH, INTERVAL_PERFECT_FOURTH,
                        INTERVAL_UNISON, INTERVAL_PERFECT_FIFTH},
                       {"vi", "IV", "I", "V"},
                       {-0.2f, 0.5f}, // Can be bittersweet
                       {0.3f, 0.7f},  // Moderate
                       "Emotional, introspective",
                       true});

  // Jazz progressions
  families_.push_back({
      "ii-V-I",
      "jazz",
      {INTERVAL_MAJOR_SECOND, INTERVAL_PERFECT_FIFTH, INTERVAL_UNISON},
      {"ii", "V", "I"},
      {0.4f, 0.9f}, // Sophisticated, positive
      {0.4f, 0.8f}, // Moderate to high
      "Sophisticated resolution",
      true // Always use 7ths in jazz
  });

  families_.push_back({"I-vi-ii-V",
                       "jazz",
                       {INTERVAL_UNISON, INTERVAL_MAJOR_SIXTH,
                        INTERVAL_MAJOR_SECOND, INTERVAL_PERFECT_FIFTH},
                       {"I", "vi", "ii", "V"},
                       {0.3f, 0.8f},
                       {0.5f, 0.9f},
                       "Bouncy, classic jazz",
                       true});

  families_.push_back({"iii-vi-ii-V",
                       "jazz",
                       {INTERVAL_MAJOR_THIRD, INTERVAL_MAJOR_SIXTH,
                        INTERVAL_MAJOR_SECOND, INTERVAL_PERFECT_FIFTH},
                       {"iii", "vi", "ii", "V"},
                       {0.5f, 0.9f},
                       {0.4f, 0.7f},
                       "Elegant, flowing",
                       true});

  // Minor progressions
  families_.push_back({"i-bVI-bIII-bVII",
                       "minor",
                       {INTERVAL_UNISON, INTERVAL_MINOR_SIXTH,
                        INTERVAL_MINOR_THIRD, INTERVAL_MINOR_SEVENTH},
                       {"i", "bVI", "bIII", "bVII"},
                       {-0.8f, -0.2f}, // Negative valence
                       {0.4f, 0.9f},   // Variable arousal
                       "Epic minor, cinematic",
                       false});

  families_.push_back({"i-bVII-bVI-bVII",
                       "minor",
                       {INTERVAL_UNISON, INTERVAL_MINOR_SEVENTH,
                        INTERVAL_MINOR_SIXTH, INTERVAL_MINOR_SEVENTH},
                       {"i", "bVII", "bVI", "bVII"},
                       {-0.7f, -0.1f},
                       {0.3f, 0.7f},
                       "Andalusian cadence, dramatic",
                       false});

  families_.push_back({"i-iv-v-i",
                       "minor",
                       {INTERVAL_UNISON, INTERVAL_PERFECT_FOURTH,
                        INTERVAL_PERFECT_FIFTH, INTERVAL_UNISON},
                       {"i", "iv", "v", "i"},
                       {-0.6f, -0.2f},
                       {0.2f, 0.6f},
                       "Natural minor resolution",
                       false});

  // Rock progressions
  families_.push_back(
      {"I-bVII-IV",
       "rock",
       {INTERVAL_UNISON, INTERVAL_MINOR_SEVENTH, INTERVAL_PERFECT_FOURTH},
       {"I", "bVII", "IV"},
       {0.2f, 0.8f},
       {0.6f, 1.0f}, // High arousal
       "Classic rock swagger",
       false});

  families_.push_back(
      {"I-bIII-IV",
       "rock",
       {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FOURTH},
       {"I", "bIII", "IV"},
       {-0.3f, 0.5f},
       {0.7f, 1.0f},
       "Dark power",
       false});

  // Blues progressions
  families_.push_back({
      "12-bar-blues",
      "blues",
      {INTERVAL_UNISON, INTERVAL_UNISON, INTERVAL_UNISON, INTERVAL_UNISON,
       INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FOURTH, INTERVAL_UNISON,
       INTERVAL_UNISON, INTERVAL_PERFECT_FIFTH, INTERVAL_PERFECT_FOURTH,
       INTERVAL_UNISON, INTERVAL_PERFECT_FIFTH},
      {"I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"},
      {-0.3f, 0.7f}, // Wide range
      {0.4f, 0.9f},
      "Gritty, soulful, foundational",
      true // Blues uses 7ths
  });

  // Modal progressions
  families_.push_back({"dorian-vamp",
                       "modal",
                       {INTERVAL_UNISON, INTERVAL_PERFECT_FOURTH},
                       {"i", "IV"},
                       {-0.2f, 0.6f},
                       {0.3f, 0.7f},
                       "Groovy, funky",
                       false});

  families_.push_back({"mixolydian-vamp",
                       "modal",
                       {INTERVAL_UNISON, INTERVAL_MINOR_SEVENTH},
                       {"I", "bVII"},
                       {0.1f, 0.7f},
                       {0.4f, 0.8f},
                       "Open, floating",
                       false});

  // Grief/Sadness specific
  families_.push_back({"grief-descent",
                       "minor",
                       {INTERVAL_UNISON, INTERVAL_MINOR_SIXTH,
                        INTERVAL_MINOR_THIRD, INTERVAL_MINOR_SEVENTH},
                       {"i", "bVI", "bIII", "bVII"},
                       {-1.0f, -0.5f},
                       {0.0f, 0.4f},
                       "Grief descent, melancholy",
                       false});

  families_.push_back({"melancholy-cycle",
                       "minor",
                       {INTERVAL_UNISON, INTERVAL_MINOR_SEVENTH,
                        INTERVAL_MINOR_SIXTH, INTERVAL_MINOR_SEVENTH},
                       {"i", "bVII", "bVI", "bVII"},
                       {-0.7f, -0.2f},
                       {0.1f, 0.5f},
                       "Melancholy cycle",
                       false});

  // Build map for quick lookup
  for (const auto &family : families_) {
    familyMap_[family.name] = family;
  }
}

void ChordGenerator::initializeTemplates() {
  // Legacy templates for backward compatibility
  templates_.push_back(
      {"Sad i-VI-III-VII", {1, 6, 3, 7}, {-1.0f, -0.3f}, {0.0f, 0.4f}});
  templates_.push_back(
      {"Grief i-iv-i-V", {1, 4, 1, 5}, {-1.0f, -0.5f}, {0.0f, 0.3f}});
  templates_.push_back(
      {"Melancholy i-VII-VI-VII", {1, 7, 6, 7}, {-0.7f, -0.2f}, {0.1f, 0.5f}});
  templates_.push_back(
      {"Rage i-bII-i-V", {1, 2, 1, 5}, {-1.0f, -0.5f}, {0.7f, 1.0f}});
  templates_.push_back(
      {"Hope I-V-vi-IV", {1, 5, 6, 4}, {0.2f, 0.7f}, {0.3f, 0.7f}});
  templates_.push_back(
      {"Joy I-IV-V-I", {1, 4, 5, 1}, {0.5f, 1.0f}, {0.6f, 1.0f}});
  templates_.push_back(
      {"Peace I-vi-IV-V", {1, 6, 4, 5}, {0.3f, 0.7f}, {0.0f, 0.3f}});
}

// =============================================================================
// MAIN GENERATION
// =============================================================================

std::vector<Chord> ChordGenerator::generate(const IntentResult &intent,
                                            int bars) {
  std::lock_guard<std::mutex> lock(mutex_);

  float valence = intent.emotion.valence;
  float arousal = intent.emotion.arousal;
  float intensity = intent.emotion.intensity;
  std::string mode = intent.mode;

  // Select progression family based on emotion
  const ProgressionFamily *family =
      selectFamilyForEmotion(valence, arousal, mode);

  // If no family matches, use template fallback
  if (!family) {
    // Fallback to template system
    const ProgressionTemplate *best = nullptr;
    float bestScore = -999.0f;

    for (const auto &tmpl : templates_) {
      if (valence >= tmpl.valenceRange[0] && valence <= tmpl.valenceRange[1] &&
          arousal >= tmpl.arousalRange[0] && arousal <= tmpl.arousalRange[1]) {

        float vCenter = (tmpl.valenceRange[0] + tmpl.valenceRange[1]) / 2.0f;
        float aCenter = (tmpl.arousalRange[0] + tmpl.arousalRange[1]) / 2.0f;
        float score =
            1.0f - (std::abs(valence - vCenter) + std::abs(arousal - aCenter));

        if (score > bestScore) {
          bestScore = score;
          best = &tmpl;
        }
      }
    }

    if (!best) {
      best = &templates_[1]; // Default to grief
    }

    // Convert template to family format
    ProgressionFamily fallbackFamily;
    fallbackFamily.degrees = best->degrees;
    fallbackFamily.useExtensions =
        intensity > CHORD_EXTENSION_INTENSITY_THRESHOLD;
    family = &fallbackFamily;
  }

  // Select root note
  int rootNote = selectRootNote(valence, intensity);

  // Generate chords
  std::vector<Chord> chords;
  double beatsPerChord = (bars * BEATS_PER_BAR) / family->degrees.size();
  double currentBeat = 0.0;

  bool addExtensions =
      family->useExtensions || intensity > CHORD_EXTENSION_INTENSITY_THRESHOLD;

  for (size_t i = 0; i < family->degrees.size(); ++i) {
    int degree = family->degrees[i];
    Chord chord = buildChord(degree, mode, rootNote, currentBeat, beatsPerChord,
                             addExtensions);

    // Generate chord name
    if (i < family->romanNumerals.size()) {
      chord.name = family->romanNumerals[i];
    }

    chords.push_back(chord);
    currentBeat += beatsPerChord;
  }

  // Loop progression if needed
  if (shouldLoopProgression(bars, family->degrees)) {
    int loops =
        static_cast<int>(std::ceil((bars * BEATS_PER_BAR) / currentBeat)) - 1;
    std::vector<Chord> loopedChords = chords;
    for (int i = 0; i < loops; ++i) {
      for (const auto &chord : chords) {
        Chord loopedChord = chord;
        loopedChord.startBeat += currentBeat * (i + 1);
        loopedChords.push_back(loopedChord);
      }
    }
    chords = loopedChords;
  }

  // Apply voice leading
  VoiceLeadingStyle style = VoiceLeadingStyle::Smooth;
  for (const auto &rb : intent.ruleBreaks) {
    if (rb.type == RuleBreakType::ModalMixture ||
        rb.type == RuleBreakType::HarmonicAmbiguity) {
      if (rb.severity() > RULE_BREAK_MODERATE) { // Use severity as threshold
                                                 // for parallel motion
        style = VoiceLeadingStyle::Free;
      }
    }
  }
  applyVoiceLeadingToProgression(chords, style);

  // Apply rule-break modifications
  for (const auto &rb : intent.ruleBreaks) {
    if (rb.type == RuleBreakType::ModalMixture ||
        rb.type == RuleBreakType::HarmonicAmbiguity) {
      if (rb.severity() > RULE_BREAK_MODERATE) {
        applyDissonance(chords, rb.severity());
      }
      if (rb.severity() > RULE_BREAK_HIGH) {
        addChromaticism(chords, rb.severity());
      }
      if (rb.severity() >
          INTENSITY_HIGH) { // Use severity as threshold for dissonance
        applyModalInterchange(chords, mode, intensity);
      }
    }
  }

  return chords;
}

std::vector<Chord> ChordGenerator::generateProgression(const std::string &mode,
                                                       int rootNote, int bars,
                                                       bool allowDissonance,
                                                       float intensity) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Create synthetic intent
  IntentResult intent;
  intent.mode = mode;
  intent.allowDissonance = allowDissonance;
  intent.emotion.intensity = intensity;
  intent.emotion.valence = (mode == "major" || mode == "lydian")
                               ? VALENCE_POSITIVE
                               : VALENCE_SLIGHTLY_NEGATIVE;
  intent.emotion.arousal = intensity;

  if (allowDissonance) {
    RuleBreak rb;
    rb.type = RuleBreakType::HarmonicAmbiguity;
    rb.intensity = intensity;
    rb.description = "User requested dissonance";
    rb.justification = "Direct parameter";
    intent.ruleBreaks.push_back(rb);
  }

  return generate(intent, bars);
}

std::vector<Chord>
ChordGenerator::generateFromFamily(const std::string &familyName,
                                   const std::string &mode, int rootNote,
                                   int bars, float intensity) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = familyMap_.find(familyName);
  if (it == familyMap_.end()) {
    // Default to I-IV-V-I
    it = familyMap_.find("I-IV-V-I");
  }

  const ProgressionFamily &family = it->second;
  std::vector<Chord> chords;
  double beatsPerChord = (bars * BEATS_PER_BAR) / family.degrees.size();
  double currentBeat = 0.0;

  bool addExtensions =
      family.useExtensions || intensity > CHORD_EXTENSION_INTENSITY_THRESHOLD;

  for (size_t i = 0; i < family.degrees.size(); ++i) {
    int degree = family.degrees[i];
    Chord chord = buildChord(degree, mode, rootNote, currentBeat, beatsPerChord,
                             addExtensions);

    if (i < family.romanNumerals.size()) {
      chord.name = family.romanNumerals[i];
    }

    chords.push_back(chord);
    currentBeat += beatsPerChord;
  }

  // Apply voice leading
  applyVoiceLeadingToProgression(chords, VoiceLeadingStyle::Smooth);

  return chords;
}

// =============================================================================
// PROGRESSION SELECTION
// =============================================================================

const ChordGenerator::ProgressionFamily *
ChordGenerator::selectFamilyForEmotion(float valence, float arousal,
                                       const std::string &mode) const {
  const ProgressionFamily *best = nullptr;
  float bestScore = -999.0f;

  for (const auto &family : families_) {
    float score = scoreFamilyForEmotion(family, valence, arousal);

    // Bonus for mode match
    if (mode == "minor" || mode == "aeolian") {
      if (family.category == "minor" || family.category == "blues") {
        score += 0.3f;
      }
    } else if (mode == "major" || mode == "ionian") {
      if (family.category == "universal" || family.category == "jazz") {
        score += 0.3f;
      }
    }

    if (score > bestScore) {
      bestScore = score;
      best = &family;
    }
  }

  return best;
}

float ChordGenerator::scoreFamilyForEmotion(const ProgressionFamily &family,
                                            float valence,
                                            float arousal) const {
  // Check if emotion fits range
  if (valence < family.valenceRange[0] || valence > family.valenceRange[1] ||
      arousal < family.arousalRange[0] || arousal > family.arousalRange[1]) {
    return -999.0f; // Doesn't match
  }

  // Score by how centered we are in the range
  float vCenter = (family.valenceRange[0] + family.valenceRange[1]) / 2.0f;
  float aCenter = (family.arousalRange[0] + family.arousalRange[1]) / 2.0f;

  float vDist = std::abs(valence - vCenter);
  float aDist = std::abs(arousal - aCenter);

  float vRange = family.valenceRange[1] - family.valenceRange[0];
  float aRange = family.arousalRange[1] - family.arousalRange[0];

  float score = 1.0f - (vDist / vRange + aDist / aRange) / 2.0f;

  // Add some randomness for variety (thread-safe: rng_ is protected by mutex in
  // calling function)
  std::uniform_real_distribution<float> noise(-0.1f, 0.1f);
  score += noise(rng_);

  return score;
}

// =============================================================================
// CHORD BUILDING
// =============================================================================

std::vector<int>
ChordGenerator::getScaleIntervals(const std::string &mode) const {
  if (mode == "major" || mode == "ionian")
    return {INTERVAL_UNISON,        INTERVAL_MAJOR_SECOND,
            INTERVAL_MAJOR_THIRD,   INTERVAL_PERFECT_FOURTH,
            INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
            INTERVAL_MAJOR_SEVENTH};
  if (mode == "minor" || mode == "aeolian")
    return {INTERVAL_UNISON,        INTERVAL_MAJOR_SECOND,
            INTERVAL_MINOR_THIRD,   INTERVAL_PERFECT_FOURTH,
            INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SIXTH,
            INTERVAL_MINOR_SEVENTH};
  if (mode == "dorian")
    return {INTERVAL_UNISON,        INTERVAL_MAJOR_SECOND,
            INTERVAL_MINOR_THIRD,   INTERVAL_PERFECT_FOURTH,
            INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
            INTERVAL_MINOR_SEVENTH};
  if (mode == "phrygian")
    return {INTERVAL_UNISON,        INTERVAL_MINOR_SECOND,
            INTERVAL_MINOR_THIRD,   INTERVAL_PERFECT_FOURTH,
            INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SIXTH,
            INTERVAL_MINOR_SEVENTH};
  if (mode == "lydian")
    return {INTERVAL_UNISON,       INTERVAL_MAJOR_SECOND,  INTERVAL_MAJOR_THIRD,
            INTERVAL_TRITONE,      INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
            INTERVAL_MAJOR_SEVENTH};
  if (mode == "mixolydian")
    return {INTERVAL_UNISON,        INTERVAL_MAJOR_SECOND,
            INTERVAL_MAJOR_THIRD,   INTERVAL_PERFECT_FOURTH,
            INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
            INTERVAL_MINOR_SEVENTH};
  if (mode == "locrian")
    return {INTERVAL_UNISON,       INTERVAL_MINOR_SECOND,
            INTERVAL_MINOR_THIRD,  INTERVAL_PERFECT_FOURTH,
            INTERVAL_TRITONE,      INTERVAL_MINOR_SIXTH,
            INTERVAL_MINOR_SEVENTH};

  return {INTERVAL_UNISON,         INTERVAL_MAJOR_SECOND,  INTERVAL_MINOR_THIRD,
          INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SIXTH,
          INTERVAL_MINOR_SEVENTH}; // Default to natural minor
}

int ChordGenerator::degreeToMidiNote(int degree, const std::string &mode,
                                     int rootNote, int octave) const {
  auto scale = getScaleIntervals(mode);

  // Handle negative degrees (flat chords)
  bool isFlat = false;
  if (degree < 0) {
    isFlat = true;
    degree = std::abs(degree);
  }

  // Convert degree to scale index (0-6)
  int degreeIndex = (degree - 1) % 7;
  if (degreeIndex < 0)
    degreeIndex += 7;

  int semitones = scale[degreeIndex];

  // Apply flat if needed
  if (isFlat) {
    semitones -= 1;
  }

  return rootNote + semitones + (octave * INTERVAL_OCTAVE);
}

Chord ChordGenerator::buildChord(int degree, const std::string &mode,
                                 int rootNote, double startBeat,
                                 double duration, bool addExtension,
                                 int inversion) {
  auto scale = getScaleIntervals(mode);

  // Handle negative degrees (flat chords like bVII)
  bool isFlat = false;
  int absDegree = std::abs(degree);
  if (degree < 0) {
    isFlat = true;
  }

  // Convert degree to scale index (0-6)
  int degreeIndex = (absDegree - 1) % 7;
  if (degreeIndex < 0)
    degreeIndex += 7;

  int chordRoot = rootNote + scale[degreeIndex];
  if (isFlat) {
    chordRoot -= 1; // Flat the root
  }

  // Build triad: root, third, fifth
  int thirdIndex = (degreeIndex + 2) % 7;
  int fifthIndex = (degreeIndex + 4) % 7;

  int third = chordRoot + scale[thirdIndex] - scale[degreeIndex];
  int fifth = chordRoot + scale[fifthIndex] - scale[degreeIndex];

  // Handle octave wrapping
  while (third < chordRoot)
    third += INTERVAL_OCTAVE;
  while (fifth < third)
    fifth += INTERVAL_OCTAVE;

  Chord chord;
  chord.pitches = {chordRoot, third, fifth};
  chord.startBeat = startBeat;
  chord.duration = duration;

  // Add extensions (use moderate intensity threshold for extensions)
  if (addExtension) {
    addExtensions(chord, degree, mode, INTENSITY_MODERATE);
  }

  // Apply inversion
  if (inversion > 0 && !chord.pitches.empty()) {
    for (int i = 0; i < inversion && chord.pitches.size() > 1; ++i) {
      int bottom = chord.pitches[0];
      chord.pitches.erase(chord.pitches.begin());
      chord.pitches.push_back(bottom + INTERVAL_OCTAVE);
    }
  }

  // Generate chord name (12 chromatic notes per octave)
  static const char *noteNames[] = {"C",  "C#", "D",  "D#", "E",  "F",
                                    "F#", "G",  "G#", "A",  "A#", "B"};
  constexpr int NOTES_PER_OCTAVE = INTERVAL_OCTAVE;
  chord.name = noteNames[chordRoot % NOTES_PER_OCTAVE];

  int thirdInterval = third - chordRoot;
  if (thirdInterval == INTERVAL_MINOR_THIRD)
    chord.name += "m"; // Minor third
  if (addExtension)
    chord.name += "7";

  return chord;
}

void ChordGenerator::addExtensions(Chord &chord, int degree,
                                   const std::string &mode, float intensity) {
  if (chord.pitches.empty())
    return;

  auto scale = getScaleIntervals(mode);
  int root = chord.pitches[0];

  // Add 7th
  int degreeIndex = (std::abs(degree) - 1) % 7;
  int seventhIndex = (degreeIndex + 6) % 7;
  int seventh = root + scale[seventhIndex] - scale[degreeIndex];
  while (seventh < chord.pitches.back())
    seventh += INTERVAL_OCTAVE;
  chord.pitches.push_back(seventh);

  // Add 9th for high intensity
  if (intensity > INTENSITY_HIGH) {
    int ninthIndex = (degreeIndex + 1) % 7;
    int ninth = root + scale[ninthIndex] - scale[degreeIndex] + INTERVAL_OCTAVE;
    while (ninth < chord.pitches.back())
      ninth += INTERVAL_OCTAVE;
    chord.pitches.push_back(ninth);
  }
}

// =============================================================================
// VOICE LEADING
// =============================================================================

std::vector<Chord>
ChordGenerator::applyVoiceLeading(const std::vector<Chord> &chords,
                                  VoiceLeadingStyle style) {
  std::vector<Chord> voicedChords = chords;
  applyVoiceLeadingToProgression(voicedChords, style);
  return voicedChords;
}

void ChordGenerator::applyVoiceLeadingToProgression(std::vector<Chord> &chords,
                                                    VoiceLeadingStyle style) {
  if (chords.size() < 2)
    return;

  VoiceLeadingConfig config;
  config.style = style;
  config.avoidParallelFifths = (style != VoiceLeadingStyle::Free);
  config.avoidParallelOctaves = (style != VoiceLeadingStyle::Free);
  config.maxVoiceMovement = (style == VoiceLeadingStyle::Smooth)
                                ? INTERVAL_MAJOR_THIRD // 4 semitones
                                : INTERVAL_OCTAVE;     // 12 semitones
  config.preferStepwiseMotion = (style == VoiceLeadingStyle::Smooth);

  voiceLeadingEngine_->setConfig(config);

  // Convert chords to voicings (4-part harmony: bass + 3 upper voices)
  constexpr int NUM_VOICES = 4;
  constexpr int DEFAULT_BASS = MIDI_C2; // C2 = 36

  std::vector<std::vector<int>> voicings;
  for (const auto &chord : chords) {
    int bass = chord.pitches.empty() ? DEFAULT_BASS : chord.pitches[0];
    voicings.push_back(chordToVoicing(chord, bass, NUM_VOICES));
  }

  // Apply voice leading
  std::vector<std::vector<int>> voicedVoicings =
      voiceLeadingEngine_->voiceProgression(
          voicings, voicings[0].empty() ? DEFAULT_BASS : voicings[0][0]);

  // Update chords with new voicings
  for (size_t i = 0; i < chords.size() && i < voicedVoicings.size(); ++i) {
    chords[i].pitches = voicedVoicings[i];
  }
}

std::vector<int> ChordGenerator::chordToVoicing(const Chord &chord,
                                                int bassNote,
                                                int numVoices) const {
  if (chord.pitches.empty())
    return {};

  std::vector<int> voicing;
  voicing.push_back(bassNote);

  // Add other chord tones in ascending order
  std::vector<int> sortedPitches = chord.pitches;
  std::sort(sortedPitches.begin(), sortedPitches.end());

  constexpr int MIN_VOICE_SPACING =
      INTERVAL_MINOR_THIRD; // Minimum spacing between voices

  for (int pitch : sortedPitches) {
    if (pitch > bassNote && voicing.size() < static_cast<size_t>(numVoices)) {
      // Place in appropriate octave (minimum spacing above previous voice)
      while (pitch < voicing.back() + MIN_VOICE_SPACING) {
        pitch += INTERVAL_OCTAVE;
      }
      voicing.push_back(pitch);
    }
  }

  // Fill remaining voices if needed
  while (voicing.size() < static_cast<size_t>(numVoices) &&
         !sortedPitches.empty()) {
    int lastPitch = voicing.back();
    int nextPitch = sortedPitches[voicing.size() % sortedPitches.size()];
    while (nextPitch <= lastPitch)
      nextPitch += INTERVAL_OCTAVE;
    voicing.push_back(nextPitch);
  }

  return voicing;
}

// =============================================================================
// RULE BREAKS & MODIFICATIONS
// =============================================================================

void ChordGenerator::applyDissonance(std::vector<Chord> &chords,
                                     float severity) {
  std::uniform_real_distribution<float> chance(0.0f, 1.0f);

  for (auto &chord : chords) {
    if (chord.pitches.empty())
      continue;

    if (chance(rng_) < severity * DISSONANCE_APPLICATION_FACTOR) {
      int root = chord.pitches[0];

      if (chance(rng_) < 0.5f) {
        // Minor 2nd above root
        chord.pitches.push_back(root + INTERVAL_MINOR_SECOND);
        chord.name += "(add b9)";
      } else {
        // Tritone
        chord.pitches.push_back(root + INTERVAL_TRITONE);
        chord.name += "(add #4)";
      }
    }
  }
}

void ChordGenerator::addChromaticism(std::vector<Chord> &chords,
                                     float severity) {
  if (chords.size() < 2)
    return;

  std::uniform_real_distribution<float> chance(0.0f, 1.0f);
  std::vector<Chord> newChords;

  for (size_t i = 0; i < chords.size(); ++i) {
    newChords.push_back(chords[i]);

    // Maybe insert a chromatic passing chord
    if (i < chords.size() - 1 &&
        chance(rng_) < severity * CHROMATIC_PASSING_PROBABILITY) {
      Chord passing = chords[i];

      // Shift all notes up or down by half step toward next chord
      int nextRoot =
          chords[i + 1].pitches.empty() ? 0 : chords[i + 1].pitches[0];
      int currentRoot = chords[i].pitches.empty() ? 0 : chords[i].pitches[0];
      int direction = (nextRoot > currentRoot) ? 1 : -1;

      for (auto &pitch : passing.pitches) {
        pitch += direction;
      }

      // Shorten the original and add passing chord
      newChords.back().duration *= DURATION_MULTIPLIER_75_PERCENT;
      passing.startBeat =
          newChords.back().startBeat + newChords.back().duration;
      passing.duration = chords[i].duration * DURATION_MULTIPLIER_25_PERCENT;
      passing.name = "pass";

      newChords.push_back(passing);
    }
  }

  chords = newChords;
}

void ChordGenerator::applyModalInterchange(std::vector<Chord> &chords,
                                           const std::string &baseMode,
                                           float intensity) {
  if (chords.empty())
    return;

  std::uniform_real_distribution<float> chance(0.0f, 1.0f);

  for (auto &chord : chords) {
    if (chance(rng_) < intensity * MODAL_INTERCHANGE_PROBABILITY) {
      // Borrow chord from parallel mode
      if (baseMode == "major" || baseMode == "ionian") {
        // Borrow from minor
        if (!chord.pitches.empty() && chord.pitches.size() >= 2) {
          int third = chord.pitches[1] - chord.pitches[0];
          if (third == INTERVAL_MAJOR_THIRD) {         // Major third
            chord.pitches[1] -= INTERVAL_MINOR_SECOND; // Make minor
            chord.name += "(borrowed)";
          }
        }
      } else if (baseMode == "minor" || baseMode == "aeolian") {
        // Borrow from major
        if (!chord.pitches.empty() && chord.pitches.size() >= 2) {
          int third = chord.pitches[1] - chord.pitches[0];
          if (third == INTERVAL_MINOR_THIRD) {         // Minor third
            chord.pitches[1] += INTERVAL_MINOR_SECOND; // Make major
            chord.name += "(borrowed)";
          }
        }
      }
    }
  }
}

// =============================================================================
// UTILITIES
// =============================================================================

int ChordGenerator::selectRootNote(float valence, float intensity) const {
  if (valence < VALENCE_NEGATIVE) {
    return ROOT_NOTE_DARK; // A2 - darker
  } else if (valence > VALENCE_POSITIVE) {
    return ROOT_NOTE_BRIGHT; // E3 - brighter
  }
  return ROOT_NOTE_DEFAULT; // C3
}

std::string ChordGenerator::generateChordName(const std::vector<int> &pitches,
                                              int rootNote) const {
  if (pitches.empty())
    return "N.C.";

  static const char *noteNames[] = {"C",  "C#", "D",  "D#", "E",  "F",
                                    "F#", "G",  "G#", "A",  "A#", "B"};
  constexpr int NOTES_PER_OCTAVE = INTERVAL_OCTAVE;

  int root = pitches[0] % NOTES_PER_OCTAVE;
  std::string name = noteNames[root];

  if (pitches.size() >= 2) {
    int third = pitches[1] - pitches[0];
    if (third == INTERVAL_MINOR_THIRD)
      name += "m"; // Minor
  }

  if (pitches.size() >= 3) {
    int fifth = pitches[2] - pitches[0];
    if (fifth == INTERVAL_TRITONE)
      name += "b5"; // Diminished
  }

  return name;
}

bool ChordGenerator::shouldLoopProgression(
    int bars, const std::vector<int> &degrees) const {
  // Loop if progression is shorter than requested bars
  double progressionLength = degrees.size() * BEATS_PER_BAR;
  return (bars * BEATS_PER_BAR) > progressionLength;
}

} // namespace kelly
