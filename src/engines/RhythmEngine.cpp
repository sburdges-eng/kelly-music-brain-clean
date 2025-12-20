#include "RhythmEngine.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace kelly {

namespace {
// Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition

// Removed static RNG - each generation creates its own from seed
} // namespace

RhythmEngine::RhythmEngine() {
  initializeProfiles();
  initializeGenreModifiers();
}

void RhythmEngine::initializeProfiles() {
  // Grief - minimal, hesitant
  profiles_["grief"] = {
      {GrooveType::Human, GrooveType::Halftime, GrooveType::Broken},
      PatternDensity::Sparse,
      AccentPattern::None,
      {1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, // kick
      {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, // snare
      0.3f,
      0.15f,
      0.1f,
      0.0f,
      {35, 70},
      0.08f,
      0.2f};

  // Sadness - slow, sparse
  profiles_["sadness"] = {
      {GrooveType::Human, GrooveType::Straight, GrooveType::Halftime},
      PatternDensity::Sparse,
      AccentPattern::Backbeat,
      {1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
      0.4f,
      0.2f,
      0.15f,
      0.0f,
      {40, 75},
      0.06f,
      0.15f};

  // Hope - building, organic
  profiles_["hope"] = {
      {GrooveType::Straight, GrooveType::Human, GrooveType::Shuffle},
      PatternDensity::Moderate,
      AccentPattern::Backbeat,
      {1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
      0.6f,
      0.25f,
      0.2f,
      0.1f,
      {55, 90},
      0.04f,
      0.1f};

  // Joy - energetic, bouncy
  profiles_["joy"] = {
      {GrooveType::Straight, GrooveType::Shuffle, GrooveType::Swing},
      PatternDensity::Moderate,
      AccentPattern::AllBeats,
      {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
      {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
      0.8f,
      0.3f,
      0.25f,
      0.15f,
      {70, 105},
      0.03f,
      0.05f};

  // Anger/Rage - aggressive, driving
  profiles_["anger"] = {
      {GrooveType::Straight, GrooveType::DoubleTime, GrooveType::FourOnFloor},
      PatternDensity::Busy,
      AccentPattern::AllBeats,
      {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
      {0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1},
      0.9f,
      0.15f,
      0.3f,
      0.0f,
      {90, 127},
      0.02f,
      0.02f};
  profiles_["rage"] = profiles_["anger"];

  // Fear - erratic, tense
  profiles_["fear"] = {
      {GrooveType::Broken, GrooveType::Human, GrooveType::Trap},
      PatternDensity::Moderate,
      AccentPattern::Offbeat,
      {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0},
      {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
      0.5f,
      0.35f,
      0.15f,
      0.0f,
      {45, 85},
      0.07f,
      0.25f};

  // Anxiety - restless, unstable
  profiles_["anxiety"] = {
      {GrooveType::Broken, GrooveType::Trap, GrooveType::Human},
      PatternDensity::Busy,
      AccentPattern::Offbeat,
      {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0},
      {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0},
      0.75f,
      0.4f,
      0.2f,
      0.0f,
      {50, 90},
      0.05f,
      0.15f};

  // Peace - gentle, spacious
  profiles_["peace"] = {
      {GrooveType::Human, GrooveType::Swing, GrooveType::Shuffle},
      PatternDensity::Minimal,
      AccentPattern::None,
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
      0.2f,
      0.1f,
      0.05f,
      0.2f,
      {30, 60},
      0.06f,
      0.3f};

  // Neutral
  profiles_["neutral"] = {{GrooveType::Straight, GrooveType::Human},
                          PatternDensity::Moderate,
                          AccentPattern::Backbeat,
                          {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                          {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                          0.7f,
                          0.2f,
                          0.2f,
                          0.0f,
                          {60, 95},
                          0.02f,
                          0.05f};
}

void RhythmEngine::initializeGenreModifiers() {
  genreModifiers_["lo-fi"] = {{"velocity_mult", 0.7f},
                              {"timing_humanize_mult", 2.0f},
                              {"hat_probability_mult", 0.6f}};
  genreModifiers_["rock"] = {{"velocity_mult", 1.1f},
                             {"timing_humanize_mult", 0.8f},
                             {"hat_probability_mult", 1.0f}};
  genreModifiers_["electronic"] = {{"velocity_mult", 1.0f},
                                   {"timing_humanize_mult", 0.3f},
                                   {"hat_probability_mult", 1.2f}};
  genreModifiers_["jazz"] = {{"velocity_mult", 0.85f},
                             {"timing_humanize_mult", 1.5f},
                             {"hat_probability_mult", 0.8f},
                             {"swing_override", 0.3f}};
  genreModifiers_["hip-hop"] = {{"velocity_mult", 1.0f},
                                {"timing_humanize_mult", 1.0f},
                                {"hat_probability_mult", 1.1f}};
  genreModifiers_["ambient"] = {{"velocity_mult", 0.5f},
                                {"timing_humanize_mult", 2.5f},
                                {"hat_probability_mult", 0.3f}};
}

int RhythmEngine::applySwing(int tick, int beatTicks, float swingAmount) {
  if (swingAmount <= 0.0f)
    return tick;

  int posInBeat = tick % beatTicks;
  int eighth = beatTicks / 2;

  if (posInBeat >= eighth && posInBeat < beatTicks) {
    int offset = static_cast<int>(eighth * swingAmount);
    return tick + offset;
  }
  return tick;
}

std::vector<std::tuple<int, std::string, int>> RhythmEngine::generateHatPattern(
    int barTicks, const RhythmEmotionProfile &profile, GrooveType groove) {
  std::vector<std::tuple<int, std::string, int>> hits;
  // Create unique RNG for this function
  unsigned int seed = static_cast<unsigned int>(barTicks) ^
                      static_cast<unsigned int>(groove) ^ 0xDDDD0000;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::uniform_int_distribution<int> velDist(profile.velocityRange.first - 20,
                                             profile.velocityRange.second - 10);

  int eighth = TICKS_PER_BEAT / 2;
  int numEighths = barTicks / eighth;

  for (int i = 0; i < numEighths; ++i) {
    if (dist(rng) < profile.hatProbability) {
      int tick = i * eighth;
      std::string hat = (i % 4 == 3) ? "open_hat" : "closed_hat";
      int velocity = velDist(rng);

      // Accent on beats
      if (i % 2 == 0)
        velocity += 10;
      velocity = std::clamp(velocity, 1, 127);

      hits.push_back({tick, hat, velocity});
    }
  }

  return hits;
}

std::vector<std::tuple<int, std::string, int>>
RhythmEngine::generateTrapHats(int barTicks,
                               const RhythmEmotionProfile &profile) {
  (void)profile; // Mark as intentionally unused
  std::vector<std::tuple<int, std::string, int>> hits;
  // Create unique RNG for this function
  unsigned int seed = static_cast<unsigned int>(barTicks) ^ 0xBBBB0000;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::uniform_int_distribution<int> velDist(40, 100);

  int sixteenth = TICKS_PER_BEAT / 4;
  int thirtysecond = sixteenth / 2;

  // Base 16th note pattern
  for (int i = 0; i < barTicks / sixteenth; ++i) {
    if (dist(rng) < 0.8f) {
      hits.push_back({i * sixteenth, "closed_hat", velDist(rng)});
    }

    // Trap rolls on certain beats
    if ((i == 6 || i == 14) && dist(rng) < 0.6f) {
      for (int j = 0; j < 4; ++j) {
        int rollTick = i * sixteenth + j * thirtysecond;
        int vel = 60 + j * 15;
        hits.push_back({rollTick, "closed_hat", vel});
      }
    }
  }

  return hits;
}

std::vector<std::tuple<int, std::string, int>>
RhythmEngine::generateFill(int startTick, int durationTicks) {
  std::vector<std::tuple<int, std::string, int>> hits;
  // Create unique RNG for this function
  unsigned int seed = static_cast<unsigned int>(startTick) ^
                      static_cast<unsigned int>(durationTicks) ^ 0xEEEE0000;
  std::mt19937 rng(seed);

  int sixteenth = TICKS_PER_BEAT / 4;
  int numNotes = durationTicks / sixteenth;

  std::vector<std::string> fillInstruments = {"snare", "high_tom", "mid_tom",
                                              "low_tom"};

  for (int i = 0; i < numNotes; ++i) {
    int tick = startTick + i * sixteenth;
    std::string instr = fillInstruments[rng() % fillInstruments.size()];
    int velocity = 70 + static_cast<int>(40.0f * i / numNotes);
    hits.push_back({tick, instr, std::min(127, velocity)});
  }

  // Crash at end
  hits.push_back({startTick + durationTicks, "crash", 100});

  return hits;
}

RhythmOutput RhythmEngine::generate(const std::string &emotion, int bars,
                                    int tempoBpm, const std::string &genre) {
  RhythmConfig config;
  config.emotion = emotion;
  config.bars = bars;
  config.tempoBpm = tempoBpm;
  config.genre = genre;
  return generate(config);
}

RhythmOutput RhythmEngine::generate(const RhythmConfig &config) {
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
  auto profile = (it != profiles_.end()) ? it->second : profiles_["neutral"];

  // Apply genre modifiers
  float velocityMult = 1.0f;
  float timingMult = 1.0f;
  float hatMult = 1.0f;

  if (!config.genre.empty()) {
    auto genreIt = genreModifiers_.find(config.genre);
    if (genreIt != genreModifiers_.end()) {
      auto &mods = genreIt->second;
      if (mods.count("velocity_mult"))
        velocityMult = mods.at("velocity_mult");
      if (mods.count("timing_humanize_mult"))
        timingMult = mods.at("timing_humanize_mult");
      if (mods.count("hat_probability_mult"))
        hatMult = mods.at("hat_probability_mult");
      if (mods.count("swing_override"))
        profile.swingAmount = mods.at("swing_override");
    }
  }

  profile.hatProbability *= hatMult;
  profile.timingHumanize *= timingMult;

  GrooveType groove = config.grooveOverride.value_or(
      profile.grooves[rng() % profile.grooves.size()]);
  PatternDensity density = config.densityOverride.value_or(profile.density);

  int beatsPerBar = config.timeSignature.first;
  int ticksPerBar = beatsPerBar * TICKS_PER_BEAT;
  int totalTicks = config.bars * ticksPerBar;
  int sixteenth = TICKS_PER_BEAT / 4;

  std::vector<DrumHit> allHits;
  std::set<std::string> instrumentsUsed;

  std::uniform_int_distribution<int> velDist(
      static_cast<int>(profile.velocityRange.first * velocityMult),
      static_cast<int>(profile.velocityRange.second * velocityMult));

  for (int bar = 0; bar < config.bars; ++bar) {
    int barStart = bar * ticksPerBar;

    // Check for fill
    bool isFillBar =
        config.includeFills &&
        (std::find(config.fillOnBar.begin(), config.fillOnBar.end(), bar) !=
             config.fillOnBar.end() ||
         (bar == config.bars - 1 && dist(rng) < profile.fillProbability));

    if (isFillBar) {
      auto fillHits =
          generateFill(barStart + ticksPerBar - TICKS_PER_BEAT, TICKS_PER_BEAT);
      for (auto &[tick, instr, vel] : fillHits) {
        int midiNote = GMDrum::SNARE;
        if (instr == "high_tom")
          midiNote = GMDrum::HIGH_TOM;
        else if (instr == "mid_tom")
          midiNote = GMDrum::MID_TOM;
        else if (instr == "low_tom")
          midiNote = GMDrum::LOW_TOM;
        else if (instr == "crash")
          midiNote = GMDrum::CRASH;

        allHits.push_back({instr, midiNote, tick, sixteenth, vel, false, 0});
        instrumentsUsed.insert(instr);
      }
    }

    // Kick pattern
    for (size_t i = 0; i < profile.kickPattern.size(); ++i) {
      if (profile.kickPattern[i] && dist(rng) > profile.dropProbability) {
        int tick = barStart + static_cast<int>(i) * sixteenth;
        tick = applySwing(tick, TICKS_PER_BEAT, profile.swingAmount);

        int timingOffset = static_cast<int>(
            (dist(rng) - 0.5f) * profile.timingHumanize * TICKS_PER_BEAT);
        int velocity = velDist(rng);
        if (i == 0)
          velocity = std::min(127, velocity + 15);

        allHits.push_back({"kick", GMDrum::KICK, tick, sixteenth, velocity,
                           false, timingOffset});
        instrumentsUsed.insert("kick");
      }
    }

    // Snare pattern
    for (size_t i = 0; i < profile.snarePattern.size(); ++i) {
      if (profile.snarePattern[i] && dist(rng) > profile.dropProbability) {
        int tick = barStart + static_cast<int>(i) * sixteenth;
        tick = applySwing(tick, TICKS_PER_BEAT, profile.swingAmount);

        int timingOffset = static_cast<int>(
            (dist(rng) - 0.5f) * profile.timingHumanize * TICKS_PER_BEAT);
        int velocity = velDist(rng);
        if (i == 4 || i == 12)
          velocity = std::min(127, velocity + 10);

        allHits.push_back({"snare", GMDrum::SNARE, tick, sixteenth, velocity,
                           false, timingOffset});
        instrumentsUsed.insert("snare");
      }
    }

    // Hi-hats
    std::vector<std::tuple<int, std::string, int>> hatHits;
    if (groove == GrooveType::Trap) {
      hatHits = generateTrapHats(ticksPerBar, profile);
    } else {
      hatHits = generateHatPattern(ticksPerBar, profile, groove);
    }

    for (auto &[tick, instr, vel] : hatHits) {
      int actualTick = barStart + tick;
      actualTick = applySwing(actualTick, TICKS_PER_BEAT, profile.swingAmount);
      int timingOffset = static_cast<int>(
          (dist(rng) - 0.5f) * profile.timingHumanize * TICKS_PER_BEAT * 0.5f);

      int midiNote =
          (instr == "open_hat") ? GMDrum::OPEN_HAT : GMDrum::CLOSED_HAT;

      allHits.push_back({instr, midiNote, actualTick, TICKS_PER_BEAT / 8, vel,
                         false, timingOffset});
      instrumentsUsed.insert(instr);
    }

    // Ghost notes
    if (dist(rng) < profile.ghostNoteProbability) {
      std::vector<int> ghostPositions = {2, 6, 10, 14};
      int ghostPos = ghostPositions[rng() % ghostPositions.size()];
      int tick = barStart + ghostPos * sixteenth;

      allHits.push_back({"snare", GMDrum::SNARE, tick, sixteenth,
                         static_cast<int>(25 + dist(rng) * 20), true, 0});
    }
  }

  // Sort by time
  std::sort(
      allHits.begin(), allHits.end(), [](const DrumHit &a, const DrumHit &b) {
        return (a.startTick + a.timingOffset) < (b.startTick + b.timingOffset);
      });

  RhythmOutput output;
  output.hits = allHits;
  output.config = config;
  output.grooveUsed = groove;
  output.densityUsed = density;
  output.totalTicks = totalTicks;
  output.instrumentsUsed = instrumentsUsed;

  return output;
}

RhythmOutput RhythmEngine::generateIntro(const std::string &emotion, int bars,
                                         int tempo, const std::string &genre) {
  RhythmConfig config;
  config.emotion = emotion;
  config.bars = bars;
  config.tempoBpm = tempo;
  config.genre = genre;
  config.densityOverride = PatternDensity::Minimal;
  config.includeFills = false;
  return generate(config);
}

RhythmOutput RhythmEngine::generateBuildup(const std::string &emotion, int bars,
                                           int tempo,
                                           const std::string &genre) {
  RhythmConfig config;
  config.emotion = emotion;
  config.bars = bars;
  config.tempoBpm = tempo;
  config.genre = genre;
  config.densityOverride = PatternDensity::Busy;
  config.includeFills = true;
  config.fillOnBar = {bars - 1};
  return generate(config);
}

} // namespace kelly
