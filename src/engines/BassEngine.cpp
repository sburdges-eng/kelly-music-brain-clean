#include "BassEngine.h"
#include "../common/MusicConstants.h"
#include "../midi/InstrumentSelector.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace kelly {
using namespace MusicConstants;

namespace {
// Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition
const std::vector<std::string> CHROMATIC = {"C",  "C#", "D",  "D#", "E",  "F",
                                            "F#", "G",  "G#", "A",  "A#", "B"};

const std::map<std::string, std::vector<int>> CHORD_INTERVALS = {
    {"maj", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH}},
    {"", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH}},
    {"min", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FIFTH}},
    {"m", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FIFTH}},
    {"dim", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_TRITONE}},
    {"aug", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_MINOR_SIXTH}},
    {"7",
     {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH,
      INTERVAL_MINOR_SEVENTH}},
    {"maj7",
     {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH,
      INTERVAL_MAJOR_SEVENTH}},
    {"min7",
     {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FIFTH,
      INTERVAL_MINOR_SEVENTH}},
    {"m7",
     {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FIFTH,
      INTERVAL_MINOR_SEVENTH}},
    {"sus2", {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_PERFECT_FIFTH}},
    {"sus4",
     {INTERVAL_UNISON, INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH}},
};

// Removed static RNG - each generation creates its own from seed

int noteNameToMidi(const std::string &name, int octave) {
  for (size_t i = 0; i < CHROMATIC.size(); ++i) {
    if (CHROMATIC[i] == name) {
      return static_cast<int>(i) + (octave + 1) * INTERVAL_OCTAVE;
    }
  }
  return MIDI_C2; // Default C2
}
} // namespace

BassEngine::BassEngine() { initializeProfiles(); }

void BassEngine::initializeProfiles() {
  // Grief - breathing, sparse, sustained
  profiles_["grief"] = {
      {BassPattern::Breathing, BassPattern::Pedal, BassPattern::RootOnly},
      BassRegister::Low,
      {50, 85},
      0.9f,
      0.1f,
      GM::ELECTRIC_BASS_FINGER};

  // Sadness - descending, walking
  profiles_["sadness"] = {
      {BassPattern::RootFifth, BassPattern::Walking, BassPattern::Breathing},
      BassRegister::Low,
      {55, 90},
      0.85f,
      0.15f,
      GM::FRETLESS_BASS};

  // Hope - rising, arpeggiated
  profiles_["hope"] = {
      {BassPattern::Arpeggiated, BassPattern::Climbing, BassPattern::RootFifth},
      BassRegister::Low,
      {60, 95},
      0.75f,
      0.2f,
      GM::ACOUSTIC_BASS};

  // Joy - active, bouncy
  profiles_["joy"] = {
      {BassPattern::Arpeggiated, BassPattern::Walking, BassPattern::Pulsing},
      BassRegister::Mid,
      {70, 100},
      0.6f,
      0.25f,
      GM::SLAP_BASS_1};

  // Anger/Rage - driving, aggressive
  profiles_["anger"] = {
      {BassPattern::Driving, BassPattern::Syncopated, BassPattern::Pulsing},
      BassRegister::Sub,
      {90, 120},
      0.5f,
      0.35f,
      GM::SYNTH_BASS_2};
  profiles_["rage"] = profiles_["anger"];

  // Fear - unstable, syncopated
  profiles_["fear"] = {
      {BassPattern::Syncopated, BassPattern::Ghost, BassPattern::Pedal},
      BassRegister::Low,
      {55, 85},
      0.65f,
      0.3f,
      GM::ELECTRIC_BASS_FINGER};

  // Anxiety - restless, pulsing
  profiles_["anxiety"] = {
      {BassPattern::Pulsing, BassPattern::Syncopated, BassPattern::Driving},
      BassRegister::Low,
      {65, 95},
      0.55f,
      0.4f,
      GM::ELECTRIC_BASS_PICK};

  // Peace - sustained, simple
  profiles_["peace"] = {
      {BassPattern::RootOnly, BassPattern::Pedal, BassPattern::Breathing},
      BassRegister::Low,
      {45, 75},
      0.95f,
      0.05f,
      GM::ACOUSTIC_BASS};

  // Love - warm, walking
  profiles_["love"] = {
      {BassPattern::Walking, BassPattern::RootFifth, BassPattern::Arpeggiated},
      BassRegister::Low,
      {55, 85},
      0.8f,
      0.15f,
      GM::FRETLESS_BASS};

  // Neutral
  profiles_["neutral"] = {
      {BassPattern::RootFifth, BassPattern::Walking, BassPattern::Arpeggiated},
      BassRegister::Low,
      {60, 90},
      0.7f,
      0.2f,
      GM::ELECTRIC_BASS_FINGER};
}

int BassEngine::getRegisterOctave(BassRegister reg) {
  switch (reg) {
  case BassRegister::Sub:
    return 1;
  case BassRegister::Low:
    return 2;
  case BassRegister::Mid:
    return 3;
  case BassRegister::Flexible:
    return 2; // Default to low for flexible
  default:
    return 2;
  }
}

BassEngine::ChordTones BassEngine::parseChord(const std::string &chord,
                                              int octave) {
  std::string root;
  std::string quality = "";

  if (chord.length() >= 2 && (chord[1] == '#' || chord[1] == 'b')) {
    root = chord.substr(0, 2);
    quality = chord.substr(2);
  } else if (!chord.empty()) {
    root = chord.substr(0, 1);
    quality = chord.substr(1);
  } else {
    root = "C";
  }

  int rootMidi = noteNameToMidi(root, octave);

  auto it = CHORD_INTERVALS.find(quality);
  const auto &intervals =
      (it != CHORD_INTERVALS.end()) ? it->second : CHORD_INTERVALS.at("maj");

  ChordTones tones;
  tones.root = rootMidi;
  tones.third =
      rootMidi + (intervals.size() > 1 ? intervals[1] : INTERVAL_MAJOR_THIRD);
  tones.fifth =
      rootMidi + (intervals.size() > 2 ? intervals[2] : INTERVAL_PERFECT_FIFTH);
  tones.seventh =
      rootMidi + (intervals.size() > 3 ? intervals[3] : INTERVAL_MINOR_SEVENTH);

  return tones;
}

std::vector<BassNote>
BassEngine::generateRootOnly(const ChordTones &tones, int barTicks,
                             const BassEmotionProfile &profile,
                             std::mt19937 &rng) {
  std::vector<BassNote> notes;
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);

  int duration = static_cast<int>(barTicks * profile.noteLengthRatio) -
                 MIN_NOTE_DURATION_OFFSET;
  notes.push_back({tones.root, 0, duration, velDist(rng),
                   BassArticulation::Sustained, "root"});

  return notes;
}

std::vector<BassNote>
BassEngine::generateRootFifth(const ChordTones &tones, int barTicks,
                              const BassEmotionProfile &profile,
                              std::mt19937 &rng) {
  std::vector<BassNote> notes;
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);

  int halfBar = barTicks / 2;
  int duration = static_cast<int>(halfBar * profile.noteLengthRatio) -
                 MIN_NOTE_DURATION_OFFSET;

  notes.push_back({tones.root, 0, duration, velDist(rng),
                   BassArticulation::Sustained, "root"});
  notes.push_back({tones.fifth, halfBar, duration, velDist(rng),
                   BassArticulation::Sustained, "fifth"});

  return notes;
}

std::vector<BassNote>
BassEngine::generateWalking(const ChordTones &tones, int barTicks,
                            const BassEmotionProfile &profile,
                            std::mt19937 &rng) {
  std::vector<BassNote> notes;
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);
  std::uniform_int_distribution<int> stepDist(1, 3);

  int beatTicks = TICKS_PER_BEAT;
  int numBeats = barTicks / beatTicks;
  constexpr int WALKING_NOTE_TAIL_OFFSET = 10;
  int duration = static_cast<int>(beatTicks * profile.noteLengthRatio) -
                 WALKING_NOTE_TAIL_OFFSET;

  int currentPitch = tones.root;
  constexpr int WALKING_PITCH_RANGE_BELOW = 5;
  constexpr int WALKING_PITCH_RANGE_ABOVE = 7;
  for (int i = 0; i < numBeats; ++i) {
    notes.push_back({currentPitch, i * beatTicks, duration, velDist(rng),
                     BassArticulation::Sustained, "walk"});

    // Step toward next chord tone
    int step = stepDist(rng);
    if (rng() % 2 == 0)
      step = -step;
    currentPitch =
        std::clamp(currentPitch + step, tones.root - WALKING_PITCH_RANGE_BELOW,
                   tones.root + WALKING_PITCH_RANGE_ABOVE);
  }

  return notes;
}

std::vector<BassNote>
BassEngine::generatePedal(const ChordTones &tones, int barTicks,
                          const BassEmotionProfile &profile,
                          std::mt19937 &rng) {
  std::vector<BassNote> notes;
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);

  // Single sustained note
  constexpr int PEDAL_NOTE_TAIL_OFFSET = 40;
  int duration = barTicks - PEDAL_NOTE_TAIL_OFFSET;
  notes.push_back({tones.root, 0, duration, velDist(rng),
                   BassArticulation::Sustained, "pedal"});

  return notes;
}

std::vector<BassNote>
BassEngine::generateArpeggiated(const ChordTones &tones, int barTicks,
                                const BassEmotionProfile &profile,
                                std::mt19937 &rng) {
  std::vector<BassNote> notes;
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);

  std::vector<int> arpPitches = {tones.root, tones.third, tones.fifth,
                                 tones.third};
  int noteLen = barTicks / 4;
  constexpr int ARPEGGIATED_NOTE_TAIL_OFFSET = 10;
  int duration = static_cast<int>(noteLen * profile.noteLengthRatio) -
                 ARPEGGIATED_NOTE_TAIL_OFFSET;

  std::vector<std::string> functions = {"root", "third", "fifth", "third"};
  for (size_t i = 0; i < arpPitches.size(); ++i) {
    notes.push_back({arpPitches[i], static_cast<int>(i * noteLen), duration,
                     velDist(rng), BassArticulation::Sustained,
                     functions[static_cast<size_t>(i % functions.size())]});
  }

  return notes;
}

std::vector<BassNote>
BassEngine::generateSyncopated(const ChordTones &tones, int barTicks,
                               const BassEmotionProfile &profile,
                               std::mt19937 &rng) {
  std::vector<BassNote> notes;
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);

  int eighth = TICKS_PER_BEAT / 2;
  int duration = static_cast<int>(eighth * profile.noteLengthRatio);

  // Syncopated positions: off-beats
  constexpr int SYNCOPATED_POSITION_1 = 3;
  constexpr int SYNCOPATED_POSITION_2 = 5;
  constexpr int SYNCOPATED_POSITION_3 = 6;
  std::vector<int> positions = {0, eighth * SYNCOPATED_POSITION_1,
                                eighth * SYNCOPATED_POSITION_2,
                                eighth * SYNCOPATED_POSITION_3};
  std::vector<int> pitches = {tones.root, tones.root, tones.fifth, tones.root};

  for (size_t i = 0; i < positions.size(); ++i) {
    if (positions[i] < barTicks) {
      notes.push_back({pitches[i], positions[i], duration, velDist(rng),
                       BassArticulation::Staccato, "synco"});
    }
  }

  return notes;
}

std::vector<BassNote>
BassEngine::generateDriving(const ChordTones &tones, int barTicks,
                            const BassEmotionProfile &profile,
                            std::mt19937 &rng) {
  std::vector<BassNote> notes;
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);

  int eighth = TICKS_PER_BEAT / 2;
  int numNotes = barTicks / eighth;
  int duration = static_cast<int>(eighth * profile.noteLengthRatio) -
                 MIN_NOTE_DURATION_OFFSET;

  for (int i = 0; i < numNotes; ++i) {
    int pitch = (i % 4 == 2) ? tones.fifth : tones.root;
    notes.push_back({pitch, i * eighth, duration, velDist(rng),
                     BassArticulation::Staccato, "drive"});
  }

  return notes;
}

std::vector<BassNote>
BassEngine::generatePulsing(const ChordTones &tones, int barTicks,
                            const BassEmotionProfile &profile,
                            std::mt19937 &rng) {
  std::vector<BassNote> notes;
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);

  int sixteenth = TICKS_PER_BEAT / 4;
  int duration = sixteenth - MIN_NOTE_DURATION_OFFSET;

  // Pulse pattern: hit-rest-hit-rest
  for (int i = 0; i < barTicks / sixteenth; i += 2) {
    notes.push_back({tones.root, i * sixteenth, duration, velDist(rng),
                     BassArticulation::Staccato, "pulse"});
  }

  return notes;
}

std::vector<BassNote>
BassEngine::generateBreathing(const ChordTones &tones, int barTicks,
                              const BassEmotionProfile &profile,
                              std::mt19937 &rng) {
  std::vector<BassNote> notes;
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);

  // Long note, rest, shorter note
  constexpr float BREATHING_FIRST_DURATION_RATIO = 0.6f;
  constexpr float BREATHING_SECOND_START_RATIO = 0.75f;
  constexpr float BREATHING_SECOND_DURATION_RATIO = 0.2f;
  int firstDuration =
      static_cast<int>(barTicks * BREATHING_FIRST_DURATION_RATIO);
  int secondStart = static_cast<int>(barTicks * BREATHING_SECOND_START_RATIO);
  int secondDuration =
      static_cast<int>(barTicks * BREATHING_SECOND_DURATION_RATIO);

  notes.push_back({tones.root, 0, firstDuration, velDist(rng),
                   BassArticulation::Sustained, "root"});
  notes.push_back({tones.fifth, secondStart, secondDuration,
                   velDist(rng) - VELOCITY_GHOST_REDUCTION,
                   BassArticulation::Sustained, "fifth"});

  return notes;
}

BassOutput
BassEngine::generate(const std::string &emotion,
                     const std::vector<std::string> &chordProgression,
                     const std::string &key, int bars, int tempoBpm) {
  BassConfig config;
  config.emotion = emotion;
  config.chordProgression = chordProgression;
  config.key = key;
  config.bars = bars;
  config.tempoBpm = tempoBpm;
  return generate(config);
}

BassOutput BassEngine::generate(const BassConfig &config) {
  // Create RNG from seed - each generation gets its own RNG
  unsigned int seed = config.seed >= 0
                          ? static_cast<unsigned int>(config.seed)
                          : static_cast<unsigned int>(std::random_device{}());
  std::mt19937 rng(seed);

  std::string emotionLower = config.emotion;
  std::transform(emotionLower.begin(), emotionLower.end(), emotionLower.begin(),
                 ::tolower);

  auto it = profiles_.find(emotionLower);
  const auto &profile =
      (it != profiles_.end()) ? it->second : profiles_["neutral"];

  // Determine pattern and register
  BassPattern pattern = config.patternOverride.value_or(
      profile.patterns[rng() % profile.patterns.size()]);
  BassRegister reg =
      config.registerOverride.value_or(profile.preferredRegister);
  int octave = getRegisterOctave(reg);

  int beatsPerBar = config.timeSignature.first;
  int barTicks = beatsPerBar * TICKS_PER_BEAT;
  int totalTicks = config.bars * barTicks;

  // Use chords or generate from key
  std::vector<std::string> chords = config.chordProgression;
  if (chords.empty()) {
    chords = {config.key, config.key}; // Simple tonic
  }

  // Extend chords to cover all bars
  while (static_cast<int>(chords.size()) < config.bars) {
    for (size_t i = 0; i < config.chordProgression.size() &&
                       static_cast<int>(chords.size()) < config.bars;
         ++i) {
      chords.push_back(config.chordProgression[i]);
    }
  }

  std::vector<BassNote> allNotes;

  for (int bar = 0; bar < config.bars; ++bar) {
    ChordTones tones = parseChord(
        chords[static_cast<size_t>(bar % static_cast<int>(chords.size()))],
        octave);
    int barStart = bar * barTicks;

    std::vector<BassNote> barNotes;

    switch (pattern) {
    case BassPattern::RootOnly:
      barNotes = generateRootOnly(tones, barTicks, profile, rng);
      break;
    case BassPattern::RootFifth:
      barNotes = generateRootFifth(tones, barTicks, profile, rng);
      break;
    case BassPattern::Walking:
      barNotes = generateWalking(tones, barTicks, profile, rng);
      break;
    case BassPattern::Pedal:
      barNotes = generatePedal(tones, barTicks, profile, rng);
      break;
    case BassPattern::Arpeggiated:
      barNotes = generateArpeggiated(tones, barTicks, profile, rng);
      break;
    case BassPattern::Syncopated:
      barNotes = generateSyncopated(tones, barTicks, profile, rng);
      break;
    case BassPattern::Driving:
      barNotes = generateDriving(tones, barTicks, profile, rng);
      break;
    case BassPattern::Pulsing:
      barNotes = generatePulsing(tones, barTicks, profile, rng);
      break;
    case BassPattern::Breathing:
      barNotes = generateBreathing(tones, barTicks, profile, rng);
      break;
    case BassPattern::Descending:
      // Use root-only pattern for descending
      barNotes = generateRootOnly(tones, barTicks, profile, rng);
      break;
    case BassPattern::Climbing:
      // Use walking pattern for climbing
      barNotes = generateWalking(tones, barTicks, profile, rng);
      break;
    case BassPattern::Ghost:
      // Use breathing pattern for ghost notes
      barNotes = generateBreathing(tones, barTicks, profile, rng);
      break;
    }

    // Offset notes to bar position
    for (auto &note : barNotes) {
      note.startTick += barStart;
      allNotes.push_back(note);
    }
  }

  return {allNotes, emotionLower,         pattern,
          reg,      profile.gmInstrument, totalTicks};
}

BassOutput
BassEngine::generateForSection(const std::string &emotion,
                               const std::vector<std::string> &chordProgression,
                               const std::string &sectionType,
                               const std::string &key, int bars, int tempoBpm) {
  BassConfig config;
  config.emotion = emotion;
  config.chordProgression = chordProgression;
  config.key = key;
  config.bars = bars;
  config.tempoBpm = tempoBpm;

  // Section-specific patterns
  if (sectionType == "verse") {
    config.patternOverride = BassPattern::RootFifth;
  } else if (sectionType == "chorus") {
    config.patternOverride = BassPattern::Driving;
  } else if (sectionType == "bridge") {
    config.patternOverride = BassPattern::Walking;
  } else if (sectionType == "intro" || sectionType == "outro") {
    config.patternOverride = BassPattern::Pedal;
  }

  return generate(config);
}

} // namespace kelly
