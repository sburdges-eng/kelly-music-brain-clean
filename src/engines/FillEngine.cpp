#include "FillEngine.h"
#include "../midi/InstrumentSelector.h"
#include "../common/MusicConstants.h"
#include <algorithm>
#include <random>
#include <cmath>

namespace kelly {
using namespace MusicConstants;

namespace {
    // Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition
    // Removed static RNG - each generation creates its own from seed
}

FillEngine::FillEngine() {
    initializeProfiles();
}

void FillEngine::initializeProfiles() {
    // Grief - subtle, sparse fills
    profiles_["grief"] = {
        {FillType::GhostNotes, FillType::SnareRoll, FillType::Flam},
        FillIntensity::Subtle,
        {30, 65},
        0.4f, false, true
    };

    // Sadness - restrained fills
    profiles_["sadness"] = {
        {FillType::TomRoll, FillType::GhostNotes, FillType::Linear},
        FillIntensity::Subtle,
        {35, 70},
        0.35f, false, true
    };

    // Hope - building fills
    profiles_["hope"] = {
        {FillType::Buildup, FillType::TomRoll, FillType::Linear},
        FillIntensity::Moderate,
        {50, 90},
        0.25f, false, false
    };

    // Joy - energetic fills
    profiles_["joy"] = {
        {FillType::SixteenthRush, FillType::TomRoll, FillType::Linear, FillType::Triplet},
        FillIntensity::Intense,
        {70, 110},
        0.2f, true, false
    };

    // Anger - aggressive fills
    profiles_["anger"] = {
        {FillType::SixteenthRush, FillType::ThirtySecondRoll, FillType::AccentPattern},
        FillIntensity::Explosive,
        {90, 127},
        0.1f, true, false
    };
    profiles_["rage"] = profiles_["anger"];

    // Fear - erratic fills
    profiles_["fear"] = {
        {FillType::Stutter, FillType::Flam, FillType::Syncopated},
        FillIntensity::Moderate,
        {50, 95},
        0.3f, false, false
    };

    // Anxiety - busy, restless fills
    profiles_["anxiety"] = {
        {FillType::SixteenthRush, FillType::Syncopated, FillType::Stutter},
        FillIntensity::Intense,
        {60, 100},
        0.25f, true, false
    };

    // Peace - minimal fills
    profiles_["peace"] = {
        {FillType::GhostNotes, FillType::Flam},
        FillIntensity::Subtle,
        {25, 55},
        0.5f, false, true
    };

    // Love - expressive fills
    profiles_["love"] = {
        {FillType::TomRoll, FillType::Linear, FillType::Triplet},
        FillIntensity::Moderate,
        {45, 85},
        0.3f, false, false
    };

    // Neutral
    profiles_["neutral"] = {
        {FillType::TomRoll, FillType::Linear, FillType::SixteenthRush},
        FillIntensity::Moderate,
        {60, 95},
        0.2f, false, false
    };
}

int FillEngine::lengthToTicks(FillLength length) {
    switch (length) {
        case FillLength::Quarter: return TICKS_PER_BEAT;
        case FillLength::Half: return TICKS_PER_BEAT * static_cast<int>(BEATS_PER_HALF_NOTE);
        case FillLength::Full: return TICKS_PER_BEAT * static_cast<int>(BEATS_PER_BAR);
        case FillLength::Double: return TICKS_PER_BEAT * static_cast<int>(BEATS_PER_BAR * 2);
        default: return TICKS_PER_BEAT * static_cast<int>(BEATS_PER_BAR);
    }
}

int FillEngine::intensityToVelocityBoost(FillIntensity intensity) {
    switch (intensity) {
        case FillIntensity::Subtle: return FILL_VELOCITY_BOOST_SUBTLE;
        case FillIntensity::Moderate: return FILL_VELOCITY_BOOST_MODERATE;
        case FillIntensity::Intense: return FILL_VELOCITY_BOOST_INTENSE;
        case FillIntensity::Explosive: return FILL_VELOCITY_BOOST_EXPLOSIVE;
        default: return FILL_VELOCITY_BOOST_MODERATE;
    }
}

std::vector<DrumHit> FillEngine::generateTomRoll(int startTick, int durationTicks, FillIntensity intensity, unsigned int baseSeed) {
    std::vector<DrumHit> hits;
    // Create unique RNG for this function call from derived seed
    unsigned int seed = baseSeed ? (baseSeed ^ static_cast<unsigned int>(startTick) ^ 0x11111111) :
                        static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> velVar(-VELOCITY_VARIATION_WIDE, VELOCITY_VARIATION_WIDE);

    int baseVel = 80 + intensityToVelocityBoost(intensity);
    int noteDiv = TICKS_PER_BEAT / 4;  // 16th notes

    std::vector<int> toms = {GM::Drum::HIGH_TOM, GM::Drum::MID_TOM, GM::Drum::LOW_TOM};
    int tomIdx = 0;

    for (int tick = 0; tick < durationTicks; tick += noteDiv) {
        int velocity = std::clamp(baseVel + velVar(rng), MIDI_VELOCITY_SOFT - 5, MIDI_VELOCITY_MAX);

        // Descend through toms
        if (tick > durationTicks / 3) tomIdx = 1;
        if (tick > 2 * durationTicks / 3) tomIdx = 2;

        hits.push_back({
            "tom",
            toms[tomIdx],
            startTick + tick,
            noteDiv + NOTE_DURATION_OFFSET_STANDARD,
            velocity,
            false
        });
    }

    return hits;
}

std::vector<DrumHit> FillEngine::generateSnareRoll(int startTick, int durationTicks, FillIntensity intensity, unsigned int baseSeed) {
    std::vector<DrumHit> hits;
    unsigned int seed = baseSeed ? (baseSeed ^ static_cast<unsigned int>(startTick) ^ 0x22222222) :
                        static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> velVar(-VELOCITY_VARIATION_MEDIUM, VELOCITY_VARIATION_MEDIUM);

    int baseVel = 70 + intensityToVelocityBoost(intensity);
    int noteDiv = TICKS_PER_BEAT / 8;  // 32nd notes for roll

    if (intensity == FillIntensity::Subtle) {
        noteDiv = TICKS_PER_BEAT / 4;
    }

    for (int tick = 0; tick < durationTicks; tick += noteDiv) {
        float progress = static_cast<float>(tick) / durationTicks;
        int velocity = baseVel + static_cast<int>(progress * 20) + velVar(rng);
        velocity = std::clamp(velocity, MIDI_VELOCITY_SOFT - 10, MIDI_VELOCITY_MAX);

        hits.push_back({
            "snare",
            GM::Drum::SNARE,
            startTick + tick,
            noteDiv + NOTE_DURATION_OFFSET_SMALL,
            velocity,
            false
        });
    }

    return hits;
}

std::vector<DrumHit> FillEngine::generateLinear(int startTick, int durationTicks, FillIntensity intensity, unsigned int baseSeed) {
    std::vector<DrumHit> hits;
    unsigned int seed = baseSeed ? (baseSeed ^ static_cast<unsigned int>(startTick) ^ 0x33333333) :
                        static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> velVar(-VELOCITY_VARIATION_NARROW, VELOCITY_VARIATION_NARROW);

    int baseVel = 85 + intensityToVelocityBoost(intensity);
    int noteDiv = TICKS_PER_BEAT / 4;

    // Linear pattern: K-S-T-H alternating
    std::vector<int> pattern = {
        GM::Drum::KICK, GM::Drum::SNARE, GM::Drum::HIGH_TOM, GM::Drum::CLOSED_HAT,
        GM::Drum::KICK, GM::Drum::MID_TOM, GM::Drum::SNARE, GM::Drum::LOW_TOM
    };

    int patternIdx = 0;
    for (int tick = 0; tick < durationTicks; tick += noteDiv) {
        int velocity = std::clamp(baseVel + velVar(rng), MIDI_VELOCITY_DEFAULT - 10, MIDI_VELOCITY_MAX);

        hits.push_back({
            "linear",
            pattern[patternIdx % pattern.size()],
            startTick + tick,
            noteDiv + NOTE_DURATION_OFFSET_STANDARD,
            velocity,
            false
        });

        patternIdx++;
    }

    return hits;
}

std::vector<DrumHit> FillEngine::generateBuildup(int startTick, int durationTicks, FillIntensity intensity, unsigned int baseSeed) {
    std::vector<DrumHit> hits;
    unsigned int seed = baseSeed ? (baseSeed ^ static_cast<unsigned int>(startTick) ^ 0x44444444) :
                        static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(seed);

    int baseVel = 60 + intensityToVelocityBoost(intensity);

    // Start slow, accelerate
    int divisions[] = {2, 2, 4, 4, 8, 8, 16, 16};
    int currentTick = 0;
    int sectionIdx = 0;

    while (currentTick < durationTicks && sectionIdx < 8) {
        int sectionLength = durationTicks / 8;
        int noteDiv = TICKS_PER_BEAT / divisions[sectionIdx];

        for (int tick = 0; tick < sectionLength && currentTick + tick < durationTicks; tick += noteDiv) {
            float progress = static_cast<float>(currentTick + tick) / durationTicks;
            int velocity = baseVel + static_cast<int>(progress * 50);
            velocity = std::clamp(velocity, MIDI_VELOCITY_SOFT - 5, MIDI_VELOCITY_MAX);

            hits.push_back({
                "buildup",
                GM::Drum::SNARE,
                startTick + currentTick + tick,
                noteDiv + NOTE_DURATION_OFFSET_SMALL,
                velocity,
                false
            });
        }

        currentTick += sectionLength;
        sectionIdx++;
    }

    return hits;
}

std::vector<DrumHit> FillEngine::generateTriplet(int startTick, int durationTicks, FillIntensity intensity, unsigned int baseSeed) {
    std::vector<DrumHit> hits;
    unsigned int seed = baseSeed ? (baseSeed ^ static_cast<unsigned int>(startTick) ^ 0x55555555) :
                        static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> velVar(-VELOCITY_VARIATION_MEDIUM, VELOCITY_VARIATION_MEDIUM);

    int baseVel = 80 + intensityToVelocityBoost(intensity);
    int tripletDiv = TICKS_PER_BEAT / 3;  // Triplet 8ths

    std::vector<int> drums = {GM::Drum::HIGH_TOM, GM::Drum::MID_TOM, GM::Drum::LOW_TOM};

    for (int tick = 0; tick < durationTicks; tick += tripletDiv) {
        int drumIdx = (tick / tripletDiv) % 3;
        int velocity = std::clamp(baseVel + velVar(rng), MIDI_VELOCITY_DEFAULT - 10, MIDI_VELOCITY_MAX);

        hits.push_back({
            "triplet",
            drums[drumIdx],
            startTick + tick,
            tripletDiv + NOTE_DURATION_OFFSET_STANDARD,
            velocity,
            false
        });
    }

    return hits;
}

std::vector<DrumHit> FillEngine::generateSixteenthRush(int startTick, int durationTicks, FillIntensity intensity, unsigned int baseSeed) {
    std::vector<DrumHit> hits;
    unsigned int seed = baseSeed ? (baseSeed ^ static_cast<unsigned int>(startTick) ^ 0x66666666) :
                        static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> velVar(-VELOCITY_VARIATION_NARROW, VELOCITY_VARIATION_NARROW);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int baseVel = 90 + intensityToVelocityBoost(intensity);
    int noteDiv = TICKS_PER_BEAT / 4;

    for (int tick = 0; tick < durationTicks; tick += noteDiv) {
        int velocity = std::clamp(baseVel + velVar(rng), MIDI_VELOCITY_DEFAULT, MIDI_VELOCITY_MAX);
        float progress = static_cast<float>(tick) / durationTicks;

        // Alternate snare with accented kicks
        int instrument = ((tick / noteDiv) % 2 == 0) ? GM::Drum::SNARE : GM::Drum::KICK;

        // Add toms in last quarter
        if (progress > 0.75f && dist(rng) > 0.5f) {
            instrument = GM::Drum::MID_TOM;
        }

        hits.push_back({
            "rush",
            instrument,
            startTick + tick,
            noteDiv + NOTE_DURATION_OFFSET_SMALL,
            velocity,
            false
        });
    }

    return hits;
}

std::vector<DrumHit> FillEngine::generateFlam(int startTick, int durationTicks, FillIntensity intensity, unsigned int baseSeed) {
    std::vector<DrumHit> hits;
    unsigned int seed = baseSeed ? (baseSeed ^ static_cast<unsigned int>(startTick) ^ 0x77777777) :
                        static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> velVar(-VELOCITY_VARIATION_NARROW, VELOCITY_VARIATION_NARROW);

    int baseVel = 80 + intensityToVelocityBoost(intensity);
    int noteDiv = TICKS_PER_BEAT / 2;  // 8th notes
    int flamGap = FLAM_GRACE_NOTE_GAP_TICKS;  // Ticks between grace note and main

    for (int tick = 0; tick < durationTicks; tick += noteDiv) {
        int velocity = std::clamp(baseVel + velVar(rng), MIDI_VELOCITY_DEFAULT - 10, MIDI_VELOCITY_MAX);

        // Grace note (quieter)
        hits.push_back({
            "flam_grace",
            GM::Drum::SNARE,
            startTick + tick,
            flamGap + NOTE_DURATION_OFFSET_TINY,
            velocity - FLAM_GRACE_NOTE_VELOCITY_REDUCTION,
            true
        });

        // Main note
        hits.push_back({
            "flam_main",
            GM::Drum::SNARE,
            startTick + tick + flamGap,
            noteDiv - flamGap - 5,
            velocity,
            false
        });
    }

    return hits;
}

FillOutput FillEngine::generate(
    const std::string& emotion,
    FillLength length,
    int startTick,
    int tempoBpm
) {
    FillConfig config;
    config.emotion = emotion;
    config.length = length;
    config.startTick = startTick;
    config.tempoBpm = tempoBpm;
    return generate(config);
}

FillOutput FillEngine::generate(const FillConfig& config) {
    // Create RNG from seed - each generation gets its own RNG
    unsigned int baseSeed = config.seed >= 0 ? static_cast<unsigned int>(config.seed) :
                           static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(baseSeed);

    std::string emotionLower = config.emotion;
    std::transform(emotionLower.begin(), emotionLower.end(), emotionLower.begin(), ::tolower);

    auto it = profiles_.find(emotionLower);
    const auto& profile = (it != profiles_.end()) ? it->second : profiles_["neutral"];

    FillType type = config.typeOverride.value_or(
        profile.types[rng() % profile.types.size()]
    );
    FillIntensity intensity = config.intensityOverride.value_or(profile.intensity);

    int durationTicks = lengthToTicks(config.length);

    std::vector<DrumHit> hits;
    std::string description;

    switch (type) {
        case FillType::TomRoll:
            hits = generateTomRoll(config.startTick, durationTicks, intensity, baseSeed);
            description = "Descending tom roll";
            break;

        case FillType::SnareRoll:
        case FillType::ThirtySecondRoll:
            hits = generateSnareRoll(config.startTick, durationTicks, intensity, baseSeed);
            description = "Snare roll";
            break;

        case FillType::Linear:
            hits = generateLinear(config.startTick, durationTicks, intensity, baseSeed);
            description = "Linear fill pattern";
            break;

        case FillType::Buildup:
            hits = generateBuildup(config.startTick, durationTicks, intensity, baseSeed);
            description = "Accelerating buildup";
            break;

        case FillType::Triplet:
            hits = generateTriplet(config.startTick, durationTicks, intensity, baseSeed);
            description = "Triplet tom fill";
            break;

        case FillType::SixteenthRush:
            hits = generateSixteenthRush(config.startTick, durationTicks, intensity, baseSeed);
            description = "16th note rush";
            break;

        case FillType::Flam:
            hits = generateFlam(config.startTick, durationTicks, intensity, baseSeed);
            description = "Flam pattern";
            break;

        case FillType::GhostNotes:
            hits = generateSnareRoll(config.startTick, durationTicks, FillIntensity::Subtle, baseSeed);
            for (auto& h : hits) h.isGhost = true;
            description = "Ghost note fill";
            break;

        default:
            hits = generateTomRoll(config.startTick, durationTicks, intensity, baseSeed);
            description = "Standard fill";
    }

    // Add crash at end if requested
    if (config.endWithCrash && !hits.empty()) {
        int crashTick = config.startTick + durationTicks;
        int crashVel = MIDI_VELOCITY_LOUD + intensityToVelocityBoost(intensity);
        hits.push_back({
            "crash",
            GM::Drum::CRASH_1,
            crashTick,
            TICKS_PER_BEAT,
            std::clamp(crashVel, MIDI_VELOCITY_MEDIUM + 5, MIDI_VELOCITY_MAX),
            false
        });
    }

    return {
        hits,
        type,
        intensity,
        config.length,
        durationTicks,
        description
    };
}

} // namespace kelly
