#include "MelodyEngine.h"
#include "../common/MusicConstants.h"
#include "../midi/InstrumentSelector.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace kelly {
using namespace MusicConstants;

namespace {
// Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition

const std::map<std::string, std::vector<int>> SCALE_INTERVALS = {
    {"major",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MAJOR_THIRD,
      INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
      INTERVAL_MAJOR_SEVENTH}},
    {"ionian",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MAJOR_THIRD,
      INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
      INTERVAL_MAJOR_SEVENTH}},
    {"minor",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MINOR_THIRD,
      INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SIXTH,
      INTERVAL_MINOR_SEVENTH}},
    {"aeolian",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MINOR_THIRD,
      INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SIXTH,
      INTERVAL_MINOR_SEVENTH}},
    {"dorian",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MINOR_THIRD,
      INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
      INTERVAL_MINOR_SEVENTH}},
    {"phrygian",
     {INTERVAL_UNISON, INTERVAL_MINOR_SECOND, INTERVAL_MINOR_THIRD,
      INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SIXTH,
      INTERVAL_MINOR_SEVENTH}},
    {"lydian",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MAJOR_THIRD,
      INTERVAL_TRITONE, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
      INTERVAL_MAJOR_SEVENTH}},
    {"mixolydian",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MAJOR_THIRD,
      INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
      INTERVAL_MINOR_SEVENTH}},
    {"locrian",
     {INTERVAL_UNISON, INTERVAL_MINOR_SECOND, INTERVAL_MINOR_THIRD,
      INTERVAL_PERFECT_FOURTH, INTERVAL_TRITONE, INTERVAL_MINOR_SIXTH,
      INTERVAL_MINOR_SEVENTH}},
    {"pentatonic_major",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MAJOR_THIRD,
      INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH}},
    {"pentatonic_minor",
     {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FOURTH,
      INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SEVENTH}},
    {"blues",
     {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FOURTH,
      INTERVAL_TRITONE, INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SEVENTH}},
};

const std::vector<std::string> CHROMATIC = {"C",  "C#", "D",  "D#", "E",  "F",
                                            "F#", "G",  "G#", "A",  "A#", "B"};

int noteNameToMidi(const std::string &name, int octave = 4) {
  std::string noteName = name;
  if (noteName.length() > 1 &&
      (noteName.back() == 'm' || noteName.back() == 'M')) {
    noteName = noteName.substr(0, noteName.length() - 1);
  }
  for (size_t i = 0; i < CHROMATIC.size(); ++i) {
    if (CHROMATIC[i] == noteName) {
      return static_cast<int>(i) + (octave + 1) * INTERVAL_OCTAVE;
    }
  }
  return MIDI_C4; // Default C4
}

// Removed static RNG - each generation creates its own from seed
} // namespace

MelodyEngine::MelodyEngine() { initializeProfiles(); }

void MelodyEngine::initializeProfiles() {
  // Grief - descending, sparse, vulnerable
  profiles_["grief"] = {
      {ContourType::Descending, ContourType::SpiralDown, ContourType::Collapse},
      RhythmDensity::Sparse,
      {40, 75},
      0.8f,
      0.3f,
      GM::ACOUSTIC_GRAND_PIANO,
      {55, 75},
      0.3f};

  // Sadness - descending/static, moderate
  profiles_["sadness"] = {
      {ContourType::Descending, ContourType::Static, ContourType::InverseArch},
      RhythmDensity::Sparse,
      {45, 80},
      0.75f,
      0.25f,
      GM::ELECTRIC_PIANO_1,
      {55, 78},
      0.35f};

  // Hope - ascending with hesitation
  profiles_["hope"] = {
      {ContourType::Ascending, ContourType::SpiralUp, ContourType::Arch},
      RhythmDensity::Moderate,
      {55, 90},
      0.6f,
      0.15f,
      GM::FLUTE,
      {65, 88},
      0.4f};

  // Joy - ascending, active
  profiles_["joy"] = {
      {ContourType::Ascending, ContourType::Arch, ContourType::Wave},
      RhythmDensity::Moderate,
      {70, 105},
      0.5f,
      0.1f,
      GM::BRIGHT_ACOUSTIC_PIANO,
      {60, 84},
      0.5f};

  // Anger/Rage - jagged, aggressive
  profiles_["anger"] = {
      {ContourType::Jagged, ContourType::Ascending, ContourType::Collapse},
      RhythmDensity::Dense,
      {85, 120},
      0.3f,
      0.05f,
      GM::DISTORTION_GUITAR,
      {48, 78},
      0.7f};
  profiles_["rage"] = profiles_["anger"];

  // Fear - jagged, erratic
  profiles_["fear"] = {
      {ContourType::Jagged, ContourType::SpiralDown, ContourType::Collapse},
      RhythmDensity::Moderate,
      {50, 90},
      0.4f,
      0.2f,
      GM::TREMOLO_STRINGS,
      {50, 80},
      0.6f};

  // Anxiety - restless, oscillating
  profiles_["anxiety"] = {
      {ContourType::Wave, ContourType::Jagged, ContourType::Static},
      RhythmDensity::Dense,
      {55, 85},
      0.35f,
      0.1f,
      GM::ELECTRIC_PIANO_2,
      {58, 82},
      0.55f};

  // Peace - static/gentle arch
  profiles_["peace"] = {
      {ContourType::Static, ContourType::Arch, ContourType::Wave},
      RhythmDensity::Sparse,
      {45, 70},
      0.85f,
      0.25f,
      GM::ACOUSTIC_GUITAR_NYLON,
      {55, 75},
      0.25f};

  // Love - ascending, warm
  profiles_["love"] = {
      {ContourType::Arch, ContourType::Ascending, ContourType::Wave},
      RhythmDensity::Moderate,
      {50, 85},
      0.7f,
      0.15f,
      GM::VIOLIN,
      {60, 84},
      0.35f};

  // Nostalgia - descending arch
  profiles_["nostalgia"] = {
      {ContourType::Arch, ContourType::Descending, ContourType::SpiralDown},
      RhythmDensity::Sparse,
      {45, 75},
      0.75f,
      0.2f,
      GM::MUSIC_BOX,
      {65, 88},
      0.3f};

  // Neutral
  profiles_["neutral"] = {
      {ContourType::Static, ContourType::Wave, ContourType::Arch},
      RhythmDensity::Moderate,
      {60, 90},
      0.6f,
      0.15f,
      GM::ACOUSTIC_GRAND_PIANO,
      {55, 80},
      0.4f};
}

std::vector<int> MelodyEngine::getScalePitches(const std::string &key,
                                               const std::string &mode,
                                               int octave) {
  int rootMidi = noteNameToMidi(key, octave);
  auto it = SCALE_INTERVALS.find(mode);
  const auto &intervals =
      (it != SCALE_INTERVALS.end()) ? it->second : SCALE_INTERVALS.at("major");

  std::vector<int> pitches;
  for (int interval : intervals) {
    pitches.push_back(rootMidi + interval);
  }
  // Add next octave
  for (int interval : intervals) {
    pitches.push_back(rootMidi + interval + INTERVAL_OCTAVE);
  }
  return pitches;
}

std::vector<int> MelodyEngine::generateContour(ContourType contour,
                                               int numNotes, int startPitch,
                                               int range, std::mt19937 &rng) {
  std::vector<int> pitches(static_cast<size_t>(numNotes));
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  switch (contour) {
  case ContourType::Ascending:
    for (int i = 0; i < numNotes; ++i) {
      float progress = static_cast<float>(i) / std::max(1, numNotes - 1);
      pitches[static_cast<size_t>(i)] =
          startPitch + static_cast<int>(progress * range);
    }
    break;

  case ContourType::Descending:
    for (int i = 0; i < numNotes; ++i) {
      float progress = static_cast<float>(i) / std::max(1, numNotes - 1);
      pitches[static_cast<size_t>(i)] =
          startPitch + range - static_cast<int>(progress * range);
    }
    break;

  case ContourType::Arch:
    for (int i = 0; i < numNotes; ++i) {
      float progress = static_cast<float>(i) / std::max(1, numNotes - 1);
      float archValue = static_cast<float>(std::sin(progress * M_PI));
      pitches[static_cast<size_t>(i)] =
          startPitch + static_cast<int>(archValue * range);
    }
    break;

  case ContourType::InverseArch:
    for (int i = 0; i < numNotes; ++i) {
      float progress = static_cast<float>(i) / std::max(1, numNotes - 1);
      float archValue = static_cast<float>(-std::sin(progress * M_PI));
      pitches[static_cast<size_t>(i)] =
          startPitch + range / 2 + static_cast<int>(archValue * range / 2);
    }
    break;

  case ContourType::Static:
    for (int i = 0; i < numNotes; ++i) {
      constexpr int STATIC_VARIANCE_SEMITONES = 4;
      int offset =
          static_cast<int>((dist(rng) - 0.5f) * STATIC_VARIANCE_SEMITONES);
      pitches[static_cast<size_t>(i)] = startPitch + range / 2 + offset;
    }
    break;

  case ContourType::Wave:
    for (int i = 0; i < numNotes; ++i) {
      float progress = static_cast<float>(i) / std::max(1, numNotes - 1);
      float waveValue = static_cast<float>(std::sin(progress * M_PI * 3));
      pitches[static_cast<size_t>(i)] =
          startPitch + range / 2 + static_cast<int>(waveValue * range / 3);
    }
    break;

  case ContourType::SpiralDown:
    for (size_t i = 0; i < static_cast<size_t>(numNotes); ++i) {
      float progress = static_cast<float>(i) / std::max(1, numNotes - 1);
      float descent = progress * static_cast<float>(range);
      float oscillation = static_cast<float>(
          std::sin(progress * M_PI * 4) * (static_cast<float>(range) / 6.0f) *
          (1.0f - progress));
      pitches[i] = startPitch + range - static_cast<int>(descent) +
                   static_cast<int>(oscillation);
    }
    break;

  case ContourType::SpiralUp:
    for (size_t i = 0; i < static_cast<size_t>(numNotes); ++i) {
      float progress = static_cast<float>(i) / std::max(1, numNotes - 1);
      float ascent = progress * static_cast<float>(range);
      float oscillation = static_cast<float>(
          std::sin(progress * M_PI * 4) * (static_cast<float>(range) / 6.0f) *
          (1.0f - progress));
      pitches[i] =
          startPitch + static_cast<int>(ascent) + static_cast<int>(oscillation);
    }
    break;

  case ContourType::Jagged:
    pitches[0] = startPitch + range / 2;
    for (size_t i = 1; i < static_cast<size_t>(numNotes); ++i) {
      constexpr float JAGGED_JUMP_FACTOR = 0.6f;
      int jump = static_cast<int>(
          (dist(rng) - 0.5f) * static_cast<float>(range) * JAGGED_JUMP_FACTOR);
      pitches[i] =
          std::clamp(pitches[i - 1] + jump, startPitch, startPitch + range);
    }
    break;

  case ContourType::Collapse:
    for (size_t i = 0; i < static_cast<size_t>(numNotes); ++i) {
      float progress = static_cast<float>(i) / std::max(1, numNotes - 1);
      constexpr float COLLAPSE_START_THRESHOLD = 0.7f;
      constexpr float COLLAPSE_DESCENT_FACTOR = 0.3f;
      if (progress < COLLAPSE_START_THRESHOLD) {
        pitches[i] = startPitch + range -
                     static_cast<int>(progress * static_cast<float>(range) *
                                      COLLAPSE_DESCENT_FACTOR);
      } else {
        float collapse = (progress - COLLAPSE_START_THRESHOLD) /
                         (1.0f - COLLAPSE_START_THRESHOLD);
        pitches[i] = startPitch + static_cast<int>(static_cast<float>(range) *
                                                   COLLAPSE_START_THRESHOLD *
                                                   (1.0f - collapse));
      }
    }
    break;
  }

  return pitches;
}

int MelodyEngine::snapToScale(int pitch, const std::vector<int> &scale) const {
  int bestPitch = pitch;
  int minDist = INTERVAL_OCTAVE;

  for (int scalePitch : scale) {
    int dist = std::abs(pitch - scalePitch);
    if (dist < minDist) {
      minDist = dist;
      bestPitch = scalePitch;
    }
  }
  return bestPitch;
}

MelodyOutput MelodyEngine::generate(const std::string &emotion,
                                    const std::string &key,
                                    const std::string &mode, int bars,
                                    int tempoBpm) {
  MelodyConfig config;
  config.emotion = emotion;
  config.key = key;
  config.mode = mode;
  config.bars = bars;
  config.tempoBpm = tempoBpm;
  return generate(config);
}

MelodyOutput MelodyEngine::generate(const MelodyConfig &config) {
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

  // Determine contour and density
  ContourType contour = config.contourOverride.value_or(
      profile.preferredContours[rng() % profile.preferredContours.size()]);
  RhythmDensity density =
      config.densityOverride.value_or(profile.preferredDensity);

  // Calculate notes per bar based on density
  int notesPerBar;
  switch (density) {
  case RhythmDensity::Sparse:
    notesPerBar = 2 + rng() % 3;
    break;
  case RhythmDensity::Moderate:
    notesPerBar = 4 + rng() % 5;
    break;
  case RhythmDensity::Dense:
    notesPerBar = 8 + rng() % 9;
    break;
  case RhythmDensity::Frantic:
    notesPerBar = 16 + rng() % 8;
    break;
  }

  int totalNotes = notesPerBar * config.bars;
  int totalTicks = config.bars * config.beatsPerBar * TICKS_PER_BEAT;

  // Get scale
  int baseOctave =
      (profile.registerRange.first + profile.registerRange.second) / 2 /
          INTERVAL_OCTAVE -
      1;
  auto scale = getScalePitches(config.key, config.mode, baseOctave);

  // Generate contour
  int range = profile.registerRange.second - profile.registerRange.first;
  auto contourPitches = generateContour(
      contour, totalNotes, profile.registerRange.first, range, rng);

  // Generate notes
  std::vector<MelodyNote> notes;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first,
                                             profile.velocityRange.second);

  int ticksPerNote = totalTicks / std::max(1, totalNotes);

  // Check if we have chord progression
  bool useChords = !config.chordProgression.empty();
  float chordPreference = config.chordTonePreference;

  for (int i = 0; i < totalNotes; ++i) {
    // Rest probability
    if (dist(rng) < profile.restProbability) {
      continue;
    }

    int startTick = i * ticksPerNote;
    int pitch = contourPitches[static_cast<size_t>(i)];

    // If we have chords, align with chord tones
    if (useChords) {
      int chordIndex =
          getChordAtTick(startTick, totalTicks, config.chordProgression);
      if (chordIndex >= 0 &&
          static_cast<size_t>(chordIndex) < config.chordProgression.size()) {
        auto chordTones = getChordTones(
            config.chordProgression[static_cast<size_t>(chordIndex)]);
        pitch =
            snapToChordOrScale(pitch, chordTones, scale, chordPreference, rng);
      } else {
        pitch = snapToScale(pitch, scale);
      }
    } else {
      pitch = snapToScale(pitch, scale);
    }

    // Duration based on legato ratio
    int baseDuration = ticksPerNote;
    constexpr float DURATION_VARIANCE = 0.2f;
    int duration = static_cast<int>(
        baseDuration * (profile.legatoRatio + dist(rng) * DURATION_VARIANCE));
    duration = std::min(duration, baseDuration - MIN_NOTE_DURATION_OFFSET);

    // Velocity with variance
    int velocity = velDist(rng);

    // Articulation
    MelodyArticulation art = MelodyArticulation::Legato;
    float artRoll = dist(rng);
    constexpr float STACCATO_THRESHOLD_OFFSET = 0.3f;
    if (artRoll > profile.legatoRatio + STACCATO_THRESHOLD_OFFSET) {
      art = MelodyArticulation::Staccato;
      duration = std::max(MIN_STACCATO_DURATION, duration / 2);
    } else if (artRoll > profile.legatoRatio) {
      art = MelodyArticulation::Accent;
      velocity = std::min(MIDI_VELOCITY_MAX, velocity + VELOCITY_ACCENT_BOOST);
    }

    // Timing humanization
    int timingOffset = static_cast<int>((dist(rng) - 0.5f) *
                                        static_cast<float>(TIMING_OFFSET_MAX));

    notes.push_back({pitch, startTick, duration, velocity, art, timingOffset});
  }

  return {notes,   emotionLower,         contour,
          density, profile.gmInstrument, totalTicks};
}

MelodyOutput MelodyEngine::generateWithChords(
    const std::string &emotion, const std::vector<Chord> &chordProgression,
    const std::string &key, const std::string &mode, int tempoBpm,
    float chordTonePreference) {
  MelodyConfig config;
  config.emotion = emotion;
  config.key = key;
  config.mode = mode;
  config.chordProgression = chordProgression;
  config.chordTonePreference = chordTonePreference;

  // Estimate bars from chord progression
  if (!chordProgression.empty()) {
    // Assume each chord is roughly 1 beat, calculate bars
    double totalBeats = 0.0;
    for (const auto &chord : chordProgression) {
      totalBeats += chord.duration;
    }
    config.bars = static_cast<int>(std::ceil(totalBeats / BEATS_PER_BAR));
  } else {
    config.bars = 4;
  }

  config.tempoBpm = tempoBpm;
  return generate(config);
}

MelodyOutput MelodyEngine::generateForSection(const std::string &emotion,
                                              const std::string &sectionType,
                                              const std::string &key, int bars,
                                              int tempoBpm) {
  MelodyConfig config;
  config.emotion = emotion;
  config.key = key;
  config.bars = bars;
  config.tempoBpm = tempoBpm;

  // Section-specific adjustments
  if (sectionType == "intro" || sectionType == "outro") {
    config.densityOverride = RhythmDensity::Sparse;
  } else if (sectionType == "chorus") {
    config.densityOverride = RhythmDensity::Moderate;
  } else if (sectionType == "bridge") {
    config.contourOverride = ContourType::Arch;
  }

  return generate(config);
}

int MelodyEngine::getChordAtTick(int tick, int totalTicks,
                                 const std::vector<Chord> &chords) const {
  if (chords.empty())
    return -1;

  // Convert tick to beat position
  double beatPosition = (static_cast<double>(tick) / TICKS_PER_BEAT) *
                        BEATS_PER_BAR; // Assuming 4/4

  double currentBeat = 0.0;
  for (size_t i = 0; i < chords.size(); ++i) {
    double chordEnd = currentBeat + chords[i].duration;
    if (beatPosition >= currentBeat && beatPosition < chordEnd) {
      return static_cast<int>(i);
    }
    currentBeat = chordEnd;
  }

  // If past all chords, return last chord
  return static_cast<int>(chords.size() - 1);
}

std::vector<int> MelodyEngine::getChordTones(const Chord &chord) const {
  std::vector<int> tones;
  for (int pitch : chord.pitches) {
    // Add chord tones in multiple octaves for better matching
    for (int octave = 0; octave <= 2; ++octave) {
      int tone = pitch + (octave * INTERVAL_OCTAVE);
      if (tone >= MIDI_A0 &&
          tone <= MIDI_PITCH_REASONABLE_MAX) { // Reasonable MIDI range
        tones.push_back(tone);
      }
    }
  }
  return tones;
}

int MelodyEngine::snapToChordOrScale(int pitch,
                                     const std::vector<int> &chordTones,
                                     const std::vector<int> &scaleTones,
                                     float chordPreference,
                                     std::mt19937 &rng) const {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Decide whether to use chord tone or scale tone
  bool useChordTone = dist(rng) < chordPreference;

  if (useChordTone && !chordTones.empty()) {
    // Find closest chord tone
    int bestPitch = pitch;
    int minDist = INTERVAL_OCTAVE;
    for (int chordTone : chordTones) {
      int dist = std::abs(pitch - chordTone);
      if (dist < minDist) {
        minDist = dist;
        bestPitch = chordTone;
      }
    }
    return bestPitch;
  } else {
    // Fall back to scale
    return snapToScale(pitch, scaleTones);
  }
}

} // namespace kelly
