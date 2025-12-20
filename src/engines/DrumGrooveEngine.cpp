#include "DrumGrooveEngine.h"
#include "../common/MusicConstants.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace kelly {
using namespace MusicConstants;

namespace {
    // Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition
}

DrumGrooveEngine::DrumGrooveEngine() {
    initializeProfiles();
    initializeGenrePatterns();
}

void DrumGrooveEngine::initializeProfiles() {
    profiles_["grief"] = {
        GrooveStyle::Straight,  // Was Laid
        0.3f, 0.25f, 0.4f,  // Low density
        0.0f,               // No swing
        {30, 65},
        0.2f, 0.15f
    };

    profiles_["sadness"] = {
        GrooveStyle::Halftime,  // Fixed casing
        0.25f, 0.25f, 0.35f,
        0.0f,
        {25, 55},
        0.15f, 0.1f
    };

    profiles_["hope"] = {
        GrooveStyle::Straight,
        0.5f, 0.5f, 0.6f,
        0.1f,
        {50, 85},
        0.25f, 0.1f
    };

    profiles_["anger"] = {
        GrooveStyle::Syncopated,  // Was Pushed
        0.8f, 0.7f, 0.9f,
        0.0f,
        {85, MIDI_VELOCITY_MAX},
        0.1f, 0.05f
    };

    profiles_["fear"] = {
        GrooveStyle::Broken,
        0.4f, 0.3f, 0.5f,
        0.0f,
        {40, 80},
        0.3f, 0.2f
    };

    profiles_["joy"] = {
        GrooveStyle::Swing,
        0.6f, 0.6f, 0.8f,
        0.3f,
        {70, 110},
        0.2f, 0.1f
    };

    profiles_["anxiety"] = {
        GrooveStyle::DoubleTime,
        0.5f, 0.4f, 0.85f,
        0.0f,
        {50, 90},
        0.35f, 0.15f
    };

    profiles_["tension"] = {
        GrooveStyle::Syncopated,  // Was Pushed
        0.6f, 0.5f, 0.7f,
        0.0f,
        {60, 100},
        0.2f, 0.1f
    };

    profiles_["neutral"] = {
        GrooveStyle::Straight,
        0.5f, 0.5f, 0.6f,
        0.0f,
        {60, 85},
        0.15f, 0.1f
    };
}

void DrumGrooveEngine::initializeGenrePatterns() {
    // 16-step patterns (one bar of 16th notes)
    // Kick patterns
    genreKickPatterns_["rock"] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0};
    genreKickPatterns_["pop"] = {1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0};
    genreKickPatterns_["hip-hop"] = {1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0};
    genreKickPatterns_["lo-fi"] = {1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0};
    genreKickPatterns_["edm"] = {1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0};
    genreKickPatterns_["jazz"] = {1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0};

    // Snare patterns
    genreSnarePatterns_["rock"] = {0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0};
    genreSnarePatterns_["pop"] = {0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0};
    genreSnarePatterns_["hip-hop"] = {0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1};
    genreSnarePatterns_["lo-fi"] = {0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0};
    genreSnarePatterns_["edm"] = {0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0};
    genreSnarePatterns_["jazz"] = {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0};
}

GrooveOutput DrumGrooveEngine::generate(
    const std::string& emotion,
    const std::string& genre,
    int bars,
    int tempoBpm
) {
    GrooveConfig config;
    config.emotion = emotion;
    config.genre = genre;
    config.bars = bars;
    config.tempoBpm = tempoBpm;
    return generate(config);
}

GrooveOutput DrumGrooveEngine::generate(const GrooveConfig& config) {
    GrooveOutput output;
    output.emotion = config.emotion;

    auto profileIt = profiles_.find(config.emotion);
    const auto& profile = profileIt != profiles_.end() ? profileIt->second : profiles_["neutral"];

    output.styleUsed = config.styleOverride.value_or(profile.preferredStyle);
    output.swingAmount = (swingOverride_ >= 0) ? swingOverride_ : profile.swingAmount;

    std::mt19937 rng(config.seed >= 0 ? static_cast<unsigned int>(config.seed) : std::random_device{}());

    int ticksPerBar = TICKS_PER_BEAT * config.beatsPerBar;
    int ticksPer16th = TICKS_PER_BEAT / 4;
    int stepsPerBar = 16;

    // Get genre patterns or use defaults
    std::string genre = config.genre.empty() ? "pop" : config.genre;
    auto kickIt = genreKickPatterns_.find(genre);
    auto snareIt = genreSnarePatterns_.find(genre);

    std::vector<bool> kickPattern = kickIt != genreKickPatterns_.end() ?
        kickIt->second : genreKickPatterns_["pop"];
    std::vector<bool> snarePattern = snareIt != genreSnarePatterns_.end() ?
        snareIt->second : genreSnarePatterns_["pop"];

    for (int bar = 0; bar < config.bars; ++bar) {
        int barStart = bar * ticksPerBar;

        for (int step = 0; step < stepsPerBar; ++step) {
            int baseTick = barStart + step * ticksPer16th;
            int tick = applySwing(baseTick, output.swingAmount, ticksPer16th);
            tick = applyHumanization(tick, config.humanization, rng);

            // Kick
            if (kickPattern[static_cast<size_t>(step)] && (rng() % 100) < profile.kickDensity * 100) {
                GrooveHit hit;
                hit.element = GrooveElement::Kick;
                hit.tick = tick;
                int velRange = profile.velocityRange.second - profile.velocityRange.first;
                hit.velocity = profile.velocityRange.first +
                    static_cast<int>(rng() % static_cast<unsigned int>(std::max(1, velRange)));
                hit.pitch = getElementPitch(GrooveElement::Kick);
                hit.durationTicks = ticksPer16th;
                output.hits.push_back(hit);
            }

            // Snare
            if (snarePattern[static_cast<size_t>(step)] && (rng() % 100) < profile.snareDensity * 100) {
                GrooveHit hit;
                hit.element = GrooveElement::Snare;
                hit.tick = tick;
                int velRange = profile.velocityRange.second - profile.velocityRange.first;
                hit.velocity = profile.velocityRange.first +
                    static_cast<int>(rng() % static_cast<unsigned int>(std::max(1, velRange)));
                hit.pitch = getElementPitch(GrooveElement::Snare);
                hit.durationTicks = ticksPer16th;
                output.hits.push_back(hit);
            }

            // Hi-hat (based on density)
            if ((rng() % 100) < profile.hatDensity * 100) {
                bool isOffbeat = (step % 2 == 1);
                int velRange = profile.velocityRange.second - profile.velocityRange.first;
                int hatVelocity = profile.velocityRange.first +
                    static_cast<int>(rng() % static_cast<unsigned int>(std::max(1, velRange)));
                if (isOffbeat) hatVelocity = static_cast<int>(hatVelocity * VELOCITY_OFFBEAT_MULTIPLIER);

                GrooveHit hit;
                hit.element = GrooveElement::HiHat;
                hit.tick = tick;
                hit.velocity = hatVelocity;
                hit.pitch = getElementPitch(GrooveElement::HiHat);
                hit.durationTicks = ticksPer16th / 2;
                output.hits.push_back(hit);
            }

            // Ghost notes
            if ((rng() % 100) < profile.ghostNoteProbability * 100) {
                GrooveHit hit;
                hit.element = GrooveElement::Snare;
                hit.tick = tick + ticksPer16th / 2;
                hit.velocity = profile.velocityRange.first / 2;
                hit.pitch = getElementPitch(GrooveElement::Snare);
                hit.durationTicks = ticksPer16th / 4;
                output.hits.push_back(hit);
            }
        }
    }

    output.totalTicks = config.bars * ticksPerBar;
    return output;
}

int DrumGrooveEngine::applySwing(int tick, float swingAmount, int stepSize) {
    if (swingAmount <= 0) return tick;

    int stepInBeat = (tick % (TICKS_PER_BEAT)) / stepSize;
    if (stepInBeat % 2 == 1) {
        return tick + static_cast<int>(stepSize * swingAmount * 0.5f);
    }
    return tick;
}

int DrumGrooveEngine::applyHumanization(int tick, float amount, std::mt19937& rng) {
    if (amount <= 0) return tick;

    int maxOffset = static_cast<int>(TICKS_PER_BEAT / 8 * amount);
    int offset = static_cast<int>((static_cast<unsigned int>(rng()) % static_cast<unsigned int>(maxOffset * 2 + 1))) - maxOffset;
    return std::max(0, tick + offset);
}

int DrumGrooveEngine::getElementPitch(GrooveElement element) {
    switch (element) {
        case GrooveElement::Kick: return 36;
        case GrooveElement::Snare: return 38;
        case GrooveElement::HiHat: return 42;
        case GrooveElement::Ride: return 51;
        case GrooveElement::Crash: return 49;
        case GrooveElement::Tom: return 45;
        case GrooveElement::Percussion: return 56;
    }
    return 36;
}

} // namespace kelly
