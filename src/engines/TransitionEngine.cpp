#include "TransitionEngine.h"
#include "../common/MusicConstants.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace kelly {
using namespace MusicConstants;

namespace {
    // Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition
}

TransitionEngine::TransitionEngine() {
    initializeProfiles();
}

void TransitionEngine::initializeProfiles() {
    profiles_["grief"] = {
        {TransitionType::Crossfade, TransitionType::Silence, TransitionType::Breakdown},
        -0.2f, {30, 60}, false, false, 95  // Sweep Pad
    };
    profiles_["sadness"] = {
        {TransitionType::Crossfade, TransitionType::Filter, TransitionType::Breakdown},
        -0.1f, {25, 55}, false, false, 95
    };
    profiles_["hope"] = {
        {TransitionType::Build, TransitionType::Crossfade, TransitionType::Riser},
        0.3f, {50, 85}, true, true, 95
    };
    profiles_["anger"] = {
        {TransitionType::Cut, TransitionType::Drop, TransitionType::Stutter},
        0.4f, {80, 120}, true, true, 119  // Reverse Cymbal
    };
    profiles_["fear"] = {
        {TransitionType::Build, TransitionType::Riser, TransitionType::Silence},
        0.2f, {40, 80}, true, false, 95
    };
    profiles_["joy"] = {
        {TransitionType::Build, TransitionType::Drop, TransitionType::Crossfade},
        0.3f, {60, 100}, true, true, 119
    };
    profiles_["tension"] = {
        {TransitionType::Build, TransitionType::Riser, TransitionType::Stutter},
        0.4f, {50, 95}, true, true, 119
    };
    profiles_["neutral"] = {
        {TransitionType::Crossfade, TransitionType::Cut},
        0.0f, {50, 75}, false, false, 95
    };
}

TransitionOutput TransitionEngine::generate(
    const std::string& emotion,
    TransitionType type,
    int durationBars,
    int tempoBpm
) {
    TransitionConfig config;
    config.emotion = emotion;
    config.type = type;
    config.durationBars = durationBars;
    config.tempoBpm = tempoBpm;
    return generate(config);
}

TransitionOutput TransitionEngine::generate(const TransitionConfig& config) {
    TransitionOutput output;

    auto profileIt = profiles_.find(config.emotion);
    const auto& profile = profileIt != profiles_.end() ? profileIt->second : profiles_["neutral"];

    output.typeUsed = config.type;
    output.gmInstrument = profile.gmInstrument;

    std::mt19937 rng(config.seed >= 0 ? static_cast<unsigned int>(config.seed) : std::random_device{}());

    int ticksPerBar = TICKS_PER_BEAT * static_cast<int>(BEATS_PER_BAR);
    output.durationTicks = ticksPerBar * config.durationBars;

    output.energyCurve = generateEnergyCurve(config.type, config.durationBars * static_cast<int>(BEATS_PER_BAR));

    int velRange = profile.velocityRange.second - profile.velocityRange.first;
    int baseVelocity = profile.velocityRange.first +
        static_cast<int>(rng() % static_cast<unsigned int>(std::max(1, velRange)));

    switch (config.type) {
        case TransitionType::Build:
        case TransitionType::Riser:
            if (profile.useRisers) {
                auto riser = generateRiser(0, output.durationTicks, baseVelocity);
                output.notes.insert(output.notes.end(), riser.begin(), riser.end());
            }
            break;

        case TransitionType::Drop:
            if (profile.useImpacts) {
                auto impact = generateImpact(output.durationTicks - TICKS_PER_BEAT, baseVelocity);
                output.notes.insert(output.notes.end(), impact.begin(), impact.end());
            }
            break;

        case TransitionType::Breakdown:
            // Sparse notes fading out
            for (int i = 0; i < config.durationBars; ++i) {
                TransitionNote note;
                note.pitch = 60 - i * 2;
                note.startTick = i * ticksPerBar;
                note.durationTicks = ticksPerBar;
                note.velocity = baseVelocity - i * 10;
                note.type = "pad";
                output.notes.push_back(note);
            }
            break;

        case TransitionType::Stutter:
            // Rhythmic stuttering effect
            for (int i = 0; i < config.durationBars * 8; ++i) {
                if (rng() % 3 == 0) {
                    TransitionNote note;
                    note.pitch = 36 + (rng() % 24);
                    note.startTick = i * (ticksPerBar / 8);
                    note.durationTicks = TICKS_PER_BEAT / 4;
                    note.velocity = baseVelocity + (rng() % 20 - 10);
                    note.type = "stutter";
                    output.notes.push_back(note);
                }
            }
            break;

        case TransitionType::Silence:
            // Just a rest with maybe one note at the end
            {
                TransitionNote note;
                note.pitch = 48;
                note.startTick = output.durationTicks - TICKS_PER_BEAT;
                note.durationTicks = TICKS_PER_BEAT;
                note.velocity = baseVelocity / 2;
                note.type = "breath";
                output.notes.push_back(note);
            }
            break;

        default:
            // Crossfade/Cut - simple sustained note
            {
                TransitionNote note;
                note.pitch = 48;
                note.startTick = 0;
                note.durationTicks = output.durationTicks;
                note.velocity = baseVelocity;
                note.type = "sustain";
                output.notes.push_back(note);
            }
            break;
    }

    return output;
}

TransitionOutput TransitionEngine::createBuild(const std::string& emotion, int bars, int tempoBpm) {
    return generate(emotion, TransitionType::Build, bars, tempoBpm);
}

TransitionOutput TransitionEngine::createBreakdown(const std::string& emotion, int bars, int tempoBpm) {
    return generate(emotion, TransitionType::Breakdown, bars, tempoBpm);
}

TransitionOutput TransitionEngine::createDrop(const std::string& emotion, int bars, int tempoBpm) {
    return generate(emotion, TransitionType::Drop, bars, tempoBpm);
}

std::vector<TransitionNote> TransitionEngine::generateRiser(int startTick, int durationTicks, int velocity) {
    std::vector<TransitionNote> notes;

    // Rising pitch sweep
    int numNotes = 8;
    int ticksPerNote = durationTicks / numNotes;

    for (int i = 0; i < numNotes; ++i) {
        TransitionNote note;
        note.pitch = 36 + i * 3;  // Rising pitch
        note.startTick = startTick + i * ticksPerNote;
        note.durationTicks = ticksPerNote + TICKS_PER_BEAT / 2;  // Overlap
        note.velocity = velocity + i * 5;  // Crescendo
        note.type = "riser";
        notes.push_back(note);
    }

    return notes;
}

std::vector<TransitionNote> TransitionEngine::generateImpact(int tick, int velocity) {
    std::vector<TransitionNote> notes;

    // Low impact hit
    TransitionNote impact;
    impact.pitch = 36;  // Low C
    impact.startTick = tick;
    impact.durationTicks = TICKS_PER_BEAT * 2;
    impact.velocity = std::min(127, velocity + 30);
    impact.type = "impact";
    notes.push_back(impact);

    // Cymbal crash
    TransitionNote crash;
    crash.pitch = 49;  // Crash cymbal (drum map)
    crash.startTick = tick;
    crash.durationTicks = TICKS_PER_BEAT * 4;
    crash.velocity = velocity;
    crash.type = "crash";
    notes.push_back(crash);

    return notes;
}

std::vector<float> TransitionEngine::generateEnergyCurve(TransitionType type, int numPoints) {
    std::vector<float> curve(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        float t = static_cast<float>(i) / (numPoints - 1);

        switch (type) {
            case TransitionType::Build:
            case TransitionType::Riser:
                curve[i] = t * t;  // Exponential rise
                break;
            case TransitionType::Breakdown:
                curve[i] = 1.0f - t;  // Linear decay
                break;
            case TransitionType::Drop:
                curve[i] = (t < 0.9f) ? t : 0.2f;  // Build then drop
                break;
            case TransitionType::Filter:
                curve[i] = 0.5f + 0.3f * std::sin(t * 3.14159f);
                break;
            case TransitionType::Silence:
                curve[i] = (t < 0.9f) ? 0.0f : 0.3f;
                break;
            default:
                curve[i] = 0.5f;
                break;
        }
    }

    return curve;
}

} // namespace kelly
