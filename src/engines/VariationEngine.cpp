#include "VariationEngine.h"
#include "../common/MusicConstants.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace kelly {
using namespace MusicConstants;

namespace {
// Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition
// Removed static RNG - each generation creates its own from seed

const std::map<std::string, std::vector<int>> SCALE_INTERVALS = {
    {"major",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MAJOR_THIRD,
      INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH,
      INTERVAL_MAJOR_SEVENTH}},
    {"minor",
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
    {"pentatonic",
     {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_MAJOR_THIRD,
      INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SIXTH}},
    {"blues",
     {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FOURTH,
      INTERVAL_TRITONE, INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SEVENTH}},
};

const std::vector<std::string> CHROMATIC = {"C",  "C#", "D",  "D#", "E",  "F",
                                            "F#", "G",  "G#", "A",  "A#", "B"};

int noteNameToSemitone(const std::string &name) {
  for (size_t i = 0; i < CHROMATIC.size(); ++i) {
    if (CHROMATIC[i] == name)
      return static_cast<int>(i);
  }
  return 0;
}
} // namespace

VariationEngine::VariationEngine() { initializeProfiles(); }

void VariationEngine::initializeProfiles() {
  // Grief - simplification, fragmentation
  profiles_["grief"] = {{VariationType::Simplification,
                         VariationType::Fragmentation,
                         VariationType::Reduction},
                        0.1f,
                        0.6f,
                        0.7f,
                        {OrnamentType::Slide, OrnamentType::Bend}};

  // Sadness - reduction, melodic variation
  profiles_["sadness"] = {{VariationType::Simplification,
                           VariationType::Melodic, VariationType::Displacement},
                          0.15f,
                          0.5f,
                          0.8f,
                          {OrnamentType::Slide, OrnamentType::Appoggiatura}};

  // Hope - extension, embellishment
  profiles_["hope"] = {
      {VariationType::Extension, VariationType::Embellishment,
       VariationType::Melodic},
      0.4f,
      0.2f,
      1.1f,
      {OrnamentType::Turn, OrnamentType::Mordent, OrnamentType::GraceNote}};

  // Joy - ornamentation, rhythmic
  profiles_["joy"] = {{VariationType::Ornamentation, VariationType::Rhythmic,
                       VariationType::Embellishment},
                      0.6f,
                      0.1f,
                      1.2f,
                      {OrnamentType::Trill, OrnamentType::Turn,
                       OrnamentType::Mordent, OrnamentType::Acciaccatura}};

  // Anger - rhythmic, displacement
  profiles_["anger"] = {{VariationType::Rhythmic, VariationType::Displacement,
                         VariationType::Fragmentation},
                        0.3f,
                        0.2f,
                        1.3f,
                        {OrnamentType::Acciaccatura, OrnamentType::Bend}};
  profiles_["rage"] = profiles_["anger"];

  // Fear - fragmentation, diminution
  profiles_["fear"] = {{VariationType::Fragmentation, VariationType::Diminution,
                        VariationType::Displacement},
                       0.2f,
                       0.3f,
                       0.9f,
                       {OrnamentType::Vibrato, OrnamentType::Slide}};

  // Anxiety - rhythmic, diminution
  profiles_["anxiety"] = {{VariationType::Rhythmic, VariationType::Diminution,
                           VariationType::Displacement},
                          0.25f,
                          0.25f,
                          1.1f,
                          {OrnamentType::Mordent, OrnamentType::Acciaccatura}};

  // Peace - simplification, augmentation
  profiles_["peace"] = {{VariationType::Simplification,
                         VariationType::Augmentation, VariationType::Reduction},
                        0.1f,
                        0.7f,
                        0.6f,
                        {OrnamentType::Slide}};

  // Love - embellishment, extension
  profiles_["love"] = {{VariationType::Embellishment, VariationType::Extension,
                        VariationType::Melodic},
                       0.35f,
                       0.15f,
                       1.0f,
                       {OrnamentType::Appoggiatura, OrnamentType::Turn,
                        OrnamentType::GraceNote}};

  // Neutral
  profiles_["neutral"] = {{VariationType::Melodic, VariationType::Rhythmic},
                          0.25f,
                          0.25f,
                          1.0f,
                          {OrnamentType::Mordent, OrnamentType::Turn}};
}

std::vector<int> VariationEngine::getScalePitches(const std::string &key,
                                                  const std::string &mode) {
  int root = noteNameToSemitone(key);
  auto it = SCALE_INTERVALS.find(mode);
  const auto &intervals =
      (it != SCALE_INTERVALS.end()) ? it->second : SCALE_INTERVALS.at("major");

  std::vector<int> pitches;
  for (int octave = 0; octave < 10; ++octave) {
    for (int interval : intervals) {
      pitches.push_back(root + interval + octave * 12);
    }
  }
  return pitches;
}

int VariationEngine::snapToScale(int pitch, const std::vector<int> &scale) {
  int best = pitch;
  int minDist = 12;
  for (int s : scale) {
    int dist = std::abs(pitch - s);
    if (dist < minDist) {
      minDist = dist;
      best = s;
    }
  }
  return best;
}

float VariationEngine::calculateSimilarity(const std::vector<MidiNote> &a,
                                           const std::vector<MidiNote> &b) {
  if (a.empty() || b.empty())
    return 0.0f;

  // Compare pitch contours
  float pitchSim = 0.0f;
  size_t minSize = std::min(a.size(), b.size());
  for (size_t i = 0; i < minSize; ++i) {
    int diff = std::abs(a[i].pitch - b[i].pitch);
    pitchSim += 1.0f - std::min(1.0f, diff / 12.0f);
  }
  pitchSim /= minSize;

  // Compare note count
  float countSim = 1.0f - std::abs(static_cast<float>(a.size()) - b.size()) /
                              std::max(a.size(), b.size());

  return (pitchSim * 0.7f + countSim * 0.3f);
}

std::vector<MidiNote>
VariationEngine::ornament(const std::vector<MidiNote> &source,
                          OrnamentType type, std::mt19937 &rng) {
  std::vector<MidiNote> result;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (const auto &note : source) {
    (void)rng; // Mark as intentionally unused
    switch (type) {
    case OrnamentType::Trill: {
      int trillInterval = 2;
      int noteDurationTicks = static_cast<int>(note.duration * TICKS_PER_BEAT);
      int trillCount = noteDurationTicks / (TICKS_PER_BEAT / 8);
      int trillDur = noteDurationTicks / std::max(1, trillCount);
      double trillDurBeats = static_cast<double>(trillDur) / TICKS_PER_BEAT;
      for (int i = 0; i < trillCount; ++i) {
        int pitch = (i % 2 == 0) ? note.pitch : note.pitch + trillInterval;
        double startBeat = note.startBeat + (i * trillDurBeats);
        double durBeats = static_cast<double>(trillDur - 10) / TICKS_PER_BEAT;
        MidiNote newNote;
        newNote.pitch = pitch;
        newNote.velocity = note.velocity;
        newNote.startBeat = static_cast<float>(startBeat);
        newNote.duration = static_cast<float>(durBeats);
        result.push_back(newNote);
      }
      break;
    }

    case OrnamentType::Mordent: {
      double mordentDurBeats =
          static_cast<double>(TICKS_PER_BEAT / 16) / TICKS_PER_BEAT;
      MidiNote n1, n2, n3;
      n1.pitch = note.pitch;
      n1.velocity = note.velocity;
      n1.startBeat = static_cast<float>(note.startBeat);
      n1.duration = static_cast<float>(mordentDurBeats);
      n2.pitch = note.pitch + 2;
      n2.velocity = note.velocity - 10;
      n2.startBeat = static_cast<float>(note.startBeat + mordentDurBeats);
      n2.duration = static_cast<float>(mordentDurBeats);
      n3.pitch = note.pitch;
      n3.velocity = note.velocity;
      n3.startBeat = static_cast<float>(note.startBeat + mordentDurBeats * 2);
      n3.duration = static_cast<float>(note.duration - mordentDurBeats * 2);
      result.push_back(n1);
      result.push_back(n2);
      result.push_back(n3);
      break;
    }

    case OrnamentType::Turn: {
      double turnDurBeats =
          static_cast<double>(TICKS_PER_BEAT / 8) / TICKS_PER_BEAT;
      MidiNote t1, t2, t3, t4;
      t1.pitch = note.pitch + 2;
      t1.velocity = note.velocity - 5;
      t1.startBeat = static_cast<float>(note.startBeat);
      t1.duration = static_cast<float>(turnDurBeats);
      t2.pitch = note.pitch;
      t2.velocity = note.velocity;
      t2.startBeat = static_cast<float>(note.startBeat + turnDurBeats);
      t2.duration = static_cast<float>(turnDurBeats);
      t3.pitch = note.pitch - 1;
      t3.velocity = note.velocity - 5;
      t3.startBeat = static_cast<float>(note.startBeat + turnDurBeats * 2);
      t3.duration = static_cast<float>(turnDurBeats);
      t4.pitch = note.pitch;
      t4.velocity = note.velocity;
      t4.startBeat = static_cast<float>(note.startBeat + turnDurBeats * 3);
      t4.duration = static_cast<float>(note.duration - turnDurBeats * 3);
      result.push_back(t1);
      result.push_back(t2);
      result.push_back(t3);
      result.push_back(t4);
      break;
    }

    case OrnamentType::Appoggiatura: {
      double appoggDurBeats = note.duration / 4.0;
      MidiNote a1, a2;
      a1.pitch = note.pitch + 2;
      a1.velocity = note.velocity;
      a1.startBeat = static_cast<float>(note.startBeat);
      a1.duration = static_cast<float>(appoggDurBeats);
      a2.pitch = note.pitch;
      a2.velocity = note.velocity;
      a2.startBeat = static_cast<float>(note.startBeat + appoggDurBeats);
      a2.duration = static_cast<float>(note.duration - appoggDurBeats);
      result.push_back(a1);
      result.push_back(a2);
      break;
    }

    case OrnamentType::Acciaccatura: {
      double graceDurBeats =
          static_cast<double>(TICKS_PER_BEAT / 32) / TICKS_PER_BEAT;
      MidiNote g1, g2;
      g1.pitch = note.pitch - 1;
      g1.velocity = note.velocity - 15;
      g1.startBeat = static_cast<float>(note.startBeat);
      g1.duration = static_cast<float>(graceDurBeats);
      g2.pitch = note.pitch;
      g2.velocity = note.velocity;
      g2.startBeat = static_cast<float>(note.startBeat + graceDurBeats);
      g2.duration = static_cast<float>(note.duration);
      result.push_back(g1);
      result.push_back(g2);
      break;
    }

    case OrnamentType::GraceNote: {
      double graceDurBeats =
          static_cast<double>(TICKS_PER_BEAT / 16) / TICKS_PER_BEAT;
      MidiNote gr;
      gr.pitch = note.pitch - 2;
      gr.velocity = note.velocity - 10;
      gr.startBeat = static_cast<float>(note.startBeat - graceDurBeats);
      gr.duration = static_cast<float>(graceDurBeats);
      result.push_back(gr);
      result.push_back(note);
      break;
    }

    default:
      result.push_back(note);
    }
  }

  return result;
}

std::vector<MidiNote>
VariationEngine::simplify(const std::vector<MidiNote> &source, float amount,
                          std::mt19937 &rng) {
  if (source.empty())
    return source;

  std::vector<MidiNote> result;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  // Keep only important notes based on position and duration
  for (size_t i = 0; i < source.size(); ++i) {
    int noteTick = static_cast<int>(source[i].startBeat * TICKS_PER_BEAT);
    bool isDownbeat = (noteTick % TICKS_PER_BEAT) < 20;
    bool isLong = source[i].duration > 0.5; // More than half a beat
    bool isFirst = (i == 0);
    bool isLast = (i == source.size() - 1);

    float keepProb = 1.0f - amount;
    if (isDownbeat)
      keepProb += 0.3f;
    if (isLong)
      keepProb += 0.2f;
    if (isFirst || isLast)
      keepProb += 0.4f;

    if (dist(rng) < keepProb) {
      result.push_back(source[i]);
    }
  }

  // Ensure we keep at least some notes
  if (result.empty() && !source.empty()) {
    result.push_back(source[0]);
    if (source.size() > 1)
      result.push_back(source.back());
  }

  return result;
}

std::vector<MidiNote>
VariationEngine::rhythmicVariation(const std::vector<MidiNote> &source,
                                   float amount, std::mt19937 &rng) {
  std::vector<MidiNote> result;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::uniform_int_distribution<int> offsetDist(-TICKS_PER_BEAT / 4,
                                                TICKS_PER_BEAT / 4);

  for (auto note : source) {
    if (dist(rng) < amount) {
      double offsetBeats =
          static_cast<double>(offsetDist(rng) * amount) / TICKS_PER_BEAT;
      note.startBeat = std::max(0.0, note.startBeat + offsetBeats);

      // Optionally change duration
      if (dist(rng) < amount * 0.5f) {
        float durMult = 0.5f + dist(rng);
        note.duration =
            std::max(static_cast<double>(TICKS_PER_BEAT / 8) / TICKS_PER_BEAT,
                     note.duration * durMult);
      }
    }
    result.push_back(note);
  }

  // Sort by start time
  std::sort(result.begin(), result.end(),
            [](const MidiNote &a, const MidiNote &b) {
              return a.startBeat < b.startBeat;
            });

  return result;
}

std::vector<MidiNote>
VariationEngine::transpose(const std::vector<MidiNote> &source, int semitones) {
  std::vector<MidiNote> result;
  for (auto note : source) {
    note.pitch = std::clamp(note.pitch + semitones, 0, 127);
    result.push_back(note);
  }
  return result;
}

std::vector<MidiNote>
VariationEngine::invert(const std::vector<MidiNote> &source, int axis) {
  if (source.empty())
    return source;

  int axisNote = (axis < 0) ? source[0].pitch : axis;
  std::vector<MidiNote> result;

  for (auto note : source) {
    int interval = note.pitch - axisNote;
    note.pitch = std::clamp(axisNote - interval, 0, 127);
    result.push_back(note);
  }

  return result;
}

std::vector<MidiNote>
VariationEngine::retrograde(const std::vector<MidiNote> &source) {
  if (source.empty())
    return source;

  std::vector<MidiNote> result = source;
  std::reverse(result.begin(), result.end());

  // Recalculate timing
  double totalDuration = source.back().startBeat + source.back().duration;
  for (auto &note : result) {
    note.startBeat = totalDuration - note.startBeat - note.duration;
  }

  std::sort(result.begin(), result.end(),
            [](const MidiNote &a, const MidiNote &b) {
              return a.startBeat < b.startBeat;
            });

  return result;
}

std::vector<MidiNote>
VariationEngine::augment(const std::vector<MidiNote> &source, float factor) {
  std::vector<MidiNote> result;
  for (auto note : source) {
    note.startBeat = note.startBeat * factor;
    note.duration = note.duration * factor;
    result.push_back(note);
  }
  return result;
}

std::vector<MidiNote>
VariationEngine::diminish(const std::vector<MidiNote> &source, float factor) {
  return augment(source, 1.0f / factor);
}

std::vector<MidiNote>
VariationEngine::fragment(const std::vector<MidiNote> &source, int fragments,
                          std::mt19937 &rng) {
  if (source.empty() || fragments <= 0)
    return source;

  std::vector<MidiNote> result;
  int notesPerFragment = static_cast<int>(source.size()) / fragments;
  if (notesPerFragment < 1)
    notesPerFragment = 1;

  // Pick random fragments
  for (int f = 0; f < fragments; ++f) {
    int startIdx = rng() % source.size();
    double totalDuration = source.back().startBeat + source.back().duration;
    for (int i = 0;
         i < notesPerFragment && startIdx + i < static_cast<int>(source.size());
         ++i) {
      auto note = source[startIdx + i];
      double offset = f * totalDuration / fragments;
      note.startBeat = offset + (note.startBeat - source[startIdx].startBeat);
      result.push_back(note);
    }
  }

  std::sort(result.begin(), result.end(),
            [](const MidiNote &a, const MidiNote &b) {
              return a.startBeat < b.startBeat;
            });

  return result;
}

std::vector<MidiNote>
VariationEngine::extend(const std::vector<MidiNote> &source, int bars) {
  if (source.empty() || bars <= 0)
    return source;

  double sourceDuration = source.back().startBeat + source.back().duration;
  std::vector<MidiNote> result = source;

  for (int b = 1; b <= bars; ++b) {
    for (const auto &note : source) {
      MidiNote extended = note;
      extended.startBeat += b * sourceDuration;
      result.push_back(extended);
    }
  }

  return result;
}

std::vector<MidiNote>
VariationEngine::displace(const std::vector<MidiNote> &source, int ticks) {
  std::vector<MidiNote> result;
  double offsetBeats = static_cast<double>(ticks) / TICKS_PER_BEAT;
  for (auto note : source) {
    note.startBeat = std::max(0.0, note.startBeat + offsetBeats);
    result.push_back(note);
  }
  return result;
}

VariationOutput VariationEngine::generate(const std::vector<MidiNote> &source,
                                          const std::string &emotion,
                                          float intensity) {
  VariationConfig config;
  config.source = source;
  config.emotion = emotion;
  config.intensity = intensity;
  return generate(config);
}

VariationOutput VariationEngine::generate(const VariationConfig &config) {
  // Create RNG from seed - each generation gets its own RNG
  unsigned int seed = config.seed >= 0
                          ? static_cast<unsigned int>(config.seed)
                          : static_cast<unsigned int>(std::random_device{}());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::string emotionLower = config.emotion;
  std::transform(emotionLower.begin(), emotionLower.end(), emotionLower.begin(),
                 ::tolower);

  auto it = profiles_.find(emotionLower);
  const auto &profile =
      (it != profiles_.end()) ? it->second : profiles_["neutral"];

  VariationType type = config.typeOverride.value_or(
      profile.preferredTypes[rng() % profile.preferredTypes.size()]);

  float adjustedIntensity = config.intensity * profile.intensityMultiplier;

  std::vector<MidiNote> result;
  std::string description;

  switch (type) {
  case VariationType::Ornamentation:
    if (!profile.ornaments.empty()) {
      OrnamentType orn = profile.ornaments[rng() % profile.ornaments.size()];
      result = ornament(config.source, orn, rng);
      description = "Applied ornaments";
    }
    break;

  case VariationType::Simplification:
  case VariationType::Reduction:
    result = simplify(config.source, adjustedIntensity, rng);
    description = "Simplified melody";
    break;

  case VariationType::Rhythmic:
    result = rhythmicVariation(config.source, adjustedIntensity, rng);
    description = "Rhythmic variation";
    break;

  case VariationType::Transposition: {
    int semis = (rng() % 7) - 3; // -3 to +3
    result = transpose(config.source, semis);
    description = "Transposed by " + std::to_string(semis) + " semitones";
  } break;

  case VariationType::Inversion:
    result = invert(config.source, -1);
    description = "Inverted";
    break;

  case VariationType::Retrograde:
    result = retrograde(config.source);
    description = "Retrograde";
    break;

  case VariationType::Augmentation:
    result = augment(config.source, 1.0f + adjustedIntensity);
    description = "Augmented";
    break;

  case VariationType::Diminution:
    result = diminish(config.source, 1.0f + adjustedIntensity);
    description = "Diminished";
    break;

  case VariationType::Fragmentation:
    result = fragment(config.source,
                      2 + static_cast<int>(adjustedIntensity * 4), rng);
    description = "Fragmented";
    break;

  case VariationType::Extension:
    result = extend(config.source, 1 + static_cast<int>(adjustedIntensity * 2));
    description = "Extended";
    break;

  case VariationType::Displacement:
    result = displace(config.source,
                      static_cast<int>(TICKS_PER_BEAT * adjustedIntensity));
    description = "Displaced";
    break;

  case VariationType::Melodic:
  case VariationType::Embellishment:
  default:
    // Combine simplification with ornamentation
    result = config.source;
    if (dist(rng) < profile.simplifyProbability) {
      result = simplify(result, adjustedIntensity * 0.3f, rng);
    }
    if (dist(rng) < profile.ornamentProbability && !profile.ornaments.empty()) {
      OrnamentType orn = profile.ornaments[rng() % profile.ornaments.size()];
      result = ornament(result, orn, rng);
    }
    description = "Melodic variation with embellishments";
    break;
  }

  float similarity = calculateSimilarity(config.source, result);

  return {result, type, emotionLower, similarity, description};
}

} // namespace kelly
