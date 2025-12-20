#include "DynamicsEngine.h"
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

DynamicsEngine::DynamicsEngine() {
    initializeProfiles();
    initializeSectionPatterns();
}

void DynamicsEngine::initializeProfiles() {
    // Grief - quiet, minimal variation
    profiles_["grief"] = {
        DynamicMarking::Piano,
        {30, 70},
        {DynamicShape::Constant, DynamicShape::Decrescendo, DynamicShape::Swell},
        0.2f, 0.15f, 0.3f
    };

    // Sadness - soft with gentle swells
    profiles_["sadness"] = {
        DynamicMarking::MezzoPiano,
        {35, 80},
        {DynamicShape::Swell, DynamicShape::Hairpin, DynamicShape::Decrescendo},
        0.3f, 0.2f, 0.5f
    };

    // Hope - building dynamics
    profiles_["hope"] = {
        DynamicMarking::MezzoPiano,
        {45, 95},
        {DynamicShape::Crescendo, DynamicShape::Swell, DynamicShape::Hairpin},
        0.4f, 0.3f, 0.6f
    };

    // Joy - bright, energetic
    profiles_["joy"] = {
        DynamicMarking::MezzoForte,
        {60, 110},
        {DynamicShape::Constant, DynamicShape::Crescendo, DynamicShape::Terraced},
        0.5f, 0.35f, 0.4f
    };

    // Anger - loud with accents
    profiles_["anger"] = {
        DynamicMarking::Forte,
        {80, 127},
        {DynamicShape::Sforzando, DynamicShape::Terraced, DynamicShape::Subito},
        0.8f, 0.4f, 0.3f
    };
    profiles_["rage"] = profiles_["anger"];

    // Fear - quiet with sudden changes
    profiles_["fear"] = {
        DynamicMarking::Piano,
        {25, 90},
        {DynamicShape::Subito, DynamicShape::Sforzando, DynamicShape::Hairpin},
        0.6f, 0.5f, 0.4f
    };

    // Anxiety - restless dynamics
    profiles_["anxiety"] = {
        DynamicMarking::MezzoForte,
        {50, 100},
        {DynamicShape::Hairpin, DynamicShape::Swell, DynamicShape::Terraced},
        0.5f, 0.45f, 0.5f
    };

    // Peace - very quiet, stable
    profiles_["peace"] = {
        DynamicMarking::Pianissimo,
        {20, 55},
        {DynamicShape::Constant, DynamicShape::Swell},
        0.15f, 0.1f, 0.2f
    };

    // Love - warm, expressive swells
    profiles_["love"] = {
        DynamicMarking::MezzoPiano,
        {40, 85},
        {DynamicShape::Swell, DynamicShape::Hairpin, DynamicShape::Crescendo},
        0.35f, 0.25f, 0.6f
    };

    // Neutral
    profiles_["neutral"] = {
        DynamicMarking::MezzoForte,
        {55, 95},
        {DynamicShape::Constant, DynamicShape::Swell},
        0.4f, 0.25f, 0.4f
    };
}

void DynamicsEngine::initializeSectionPatterns() {
    // Beat emphasis patterns (0 = no accent, 1 = accent, 2 = strong accent)
    sectionAccentPatterns_["4/4_standard"] = {2, 0, 1, 0};
    sectionAccentPatterns_["4/4_backbeat"] = {0, 0, 2, 0};
    sectionAccentPatterns_["3/4_waltz"] = {2, 0, 0};
    sectionAccentPatterns_["6/8_compound"] = {2, 0, 0, 1, 0, 0};
    sectionAccentPatterns_["verse"] = {1, 0, 1, 0};
    sectionAccentPatterns_["chorus"] = {2, 0, 2, 0};
    sectionAccentPatterns_["bridge"] = {1, 0, 0, 1};
}

int DynamicsEngine::markingToVelocity(DynamicMarking marking) {
    switch (marking) {
        case DynamicMarking::Pianississimo: return 16;
        case DynamicMarking::Pianissimo: return 33;
        case DynamicMarking::Piano: return 49;
        case DynamicMarking::MezzoPiano: return 64;
        case DynamicMarking::MezzoForte: return 80;
        case DynamicMarking::Forte: return 96;
        case DynamicMarking::Fortissimo: return 112;
        case DynamicMarking::Fortississimo: return 127;
        default: return 80;
    }
}

DynamicMarking DynamicsEngine::velocityToMarking(int velocity) {
    if (velocity < 25) return DynamicMarking::Pianississimo;
    if (velocity < 41) return DynamicMarking::Pianissimo;
    if (velocity < 57) return DynamicMarking::Piano;
    if (velocity < 72) return DynamicMarking::MezzoPiano;
    if (velocity < 88) return DynamicMarking::MezzoForte;
    if (velocity < 104) return DynamicMarking::Forte;
    if (velocity < 120) return DynamicMarking::Fortissimo;
    return DynamicMarking::Fortississimo;
}

DynamicCurve DynamicsEngine::generateCurve(
    int totalTicks,
    DynamicShape shape,
    DynamicMarking startMarking,
    DynamicMarking endMarking
) {
    DynamicCurve curve;
    curve.shape = shape;

    int startVel = markingToVelocity(startMarking);
    int endVel = markingToVelocity(endMarking);
    int steps = 16;
    int tickStep = totalTicks / steps;

    for (int i = 0; i <= steps; ++i) {
        float progress = static_cast<float>(i) / steps;
        int tick = i * tickStep;
        int velocity;
        DynamicMarking marking;
        std::string annotation;

        switch (shape) {
            case DynamicShape::Constant:
                velocity = startVel;
                break;

            case DynamicShape::Crescendo:
                velocity = startVel + static_cast<int>(progress * (endVel - startVel));
                if (i == 0) annotation = "cresc.";
                break;

            case DynamicShape::Decrescendo:
                velocity = startVel - static_cast<int>(progress * (startVel - endVel));
                if (i == 0) annotation = "dim.";
                break;

            case DynamicShape::Swell:
                if (progress < 0.5f) {
                    velocity = startVel + static_cast<int>((progress * 2) * (endVel - startVel));
                } else {
                    velocity = endVel - static_cast<int>(((progress - 0.5f) * 2) * (endVel - startVel));
                }
                break;

            case DynamicShape::Hairpin:
                velocity = startVel + static_cast<int>(std::sin(progress * M_PI) * (endVel - startVel));
                break;

            case DynamicShape::Sforzando:
                if (progress < 0.1f) {
                    velocity = 127;
                    if (i == 0) annotation = "sfz";
                } else {
                    velocity = startVel;
                }
                break;

            case DynamicShape::FortePiano:
                if (progress < 0.1f) {
                    velocity = markingToVelocity(DynamicMarking::Forte);
                    if (i == 0) annotation = "fp";
                } else {
                    velocity = markingToVelocity(DynamicMarking::Piano);
                }
                break;

            case DynamicShape::Terraced:
                {
                    int level = static_cast<int>(progress * 4);
                    velocity = startVel + level * (endVel - startVel) / 4;
                }
                break;

            case DynamicShape::Subito:
                velocity = (progress < 0.5f) ? startVel : endVel;
                if (progress >= 0.5f && progress < 0.55f) annotation = "subito";
                break;

            default:
                velocity = startVel;
        }

        velocity = std::clamp(velocity, MIDI_VELOCITY_MIN + 1, MIDI_VELOCITY_MAX);
        marking = velocityToMarking(velocity);

        curve.points.push_back({tick, velocity, marking, annotation});
    }

    return curve;
}

int DynamicsEngine::interpolateVelocity(const DynamicCurve& curve, int tick) {
    if (curve.points.empty()) return MIDI_VELOCITY_MEDIUM + 5;

    // Find surrounding points
    for (size_t i = 0; i < curve.points.size() - 1; ++i) {
        if (tick >= curve.points[i].tick && tick < curve.points[i + 1].tick) {
            float t = static_cast<float>(tick - curve.points[i].tick) /
                     (curve.points[i + 1].tick - curve.points[i].tick);
            return curve.points[i].velocity +
                   static_cast<int>(t * (curve.points[i + 1].velocity - curve.points[i].velocity));
        }
    }

    return curve.points.back().velocity;
}

std::vector<MidiNote> DynamicsEngine::applyCurve(
    const std::vector<MidiNote>& notes,
    const DynamicCurve& curve
) {
    std::vector<MidiNote> result;

    for (auto note : notes) {
        int curveVel = interpolateVelocity(curve, static_cast<int>(note.startBeat * TICKS_PER_BEAT));
        // Blend original velocity with curve
        note.velocity = (note.velocity + curveVel) / 2;
        note.velocity = std::clamp(note.velocity, MIDI_VELOCITY_MIN + 1, MIDI_VELOCITY_MAX);
        result.push_back(note);
    }

    return result;
}

std::vector<MidiNote> DynamicsEngine::applyAccents(
    const std::vector<MidiNote>& notes,
    const std::string& emotion,
    float strength
) {
    (void)emotion;  // Mark as intentionally unused
    std::vector<MidiNote> result;
    // Create RNG from notes data for variation
    unsigned int seed = notes.empty() ? 0 :
                       static_cast<unsigned int>(notes[0].pitch + notes[0].velocity + static_cast<int>(strength * 1000));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto note : notes) {
        int noteTick = static_cast<int>(note.startBeat * TICKS_PER_BEAT);
        bool isDownbeat = (noteTick % TICKS_PER_BEAT) < TIMING_OFFSET_MAX;
        bool isBarStart = (noteTick % (TICKS_PER_BEAT * static_cast<int>(BEATS_PER_BAR))) < TIMING_OFFSET_MAX;

        int accentBoost = 0;
        if (isBarStart) {
            accentBoost = static_cast<int>(VELOCITY_ACCENT_BOOST + 5 * strength);
        } else if (isDownbeat) {
            accentBoost = static_cast<int>((VELOCITY_ACCENT_BOOST - 5) * strength);
        }

        // Random micro-variations
        int variation = static_cast<int>((dist(rng) - 0.5f) * TIMING_OFFSET_MAX * strength);

        note.velocity = std::clamp(note.velocity + accentBoost + variation, MIDI_VELOCITY_MIN + 1, MIDI_VELOCITY_MAX);
        result.push_back(note);
    }

    return result;
}

DynamicsOutput DynamicsEngine::apply(
    const std::vector<MidiNote>& notes,
    const std::string& emotion,
    float expressiveness
) {
    DynamicsConfig config;
    config.notes = notes;
    config.emotion = emotion;
    config.expressiveness = expressiveness;
    if (!notes.empty()) {
        config.totalTicks = static_cast<int>((notes.back().startBeat + notes.back().duration) * TICKS_PER_BEAT);
    }
    return apply(config);
}

DynamicsOutput DynamicsEngine::apply(const DynamicsConfig& config) {
    // Create RNG from seed - each generation gets its own RNG
    unsigned int seed = config.seed >= 0 ? static_cast<unsigned int>(config.seed) :
                       static_cast<unsigned int>(std::random_device{}());
    std::mt19937 rng(seed);

    std::string emotionLower = config.emotion;
    std::transform(emotionLower.begin(), emotionLower.end(), emotionLower.begin(), ::tolower);

    auto it = profiles_.find(emotionLower);
    const auto& profile = (it != profiles_.end()) ? it->second : profiles_["neutral"];

    DynamicMarking baseMarking = config.baseMarking.value_or(profile.baseMarking);
    DynamicShape shape = config.shapeOverride.value_or(
        profile.shapes[rng() % profile.shapes.size()]
    );

    // Determine start/end markings based on shape
    DynamicMarking startMarking = baseMarking;
    DynamicMarking endMarking = baseMarking;

    int baseIdx = static_cast<int>(baseMarking);
    switch (shape) {
        case DynamicShape::Crescendo:
            startMarking = static_cast<DynamicMarking>(std::max(0, baseIdx - 2));
            endMarking = static_cast<DynamicMarking>(std::min(7, baseIdx + 1));
            break;
        case DynamicShape::Decrescendo:
            startMarking = static_cast<DynamicMarking>(std::min(7, baseIdx + 1));
            endMarking = static_cast<DynamicMarking>(std::max(0, baseIdx - 2));
            break;
        case DynamicShape::Swell:
        case DynamicShape::Hairpin:
            endMarking = static_cast<DynamicMarking>(std::min(7, baseIdx + 2));
            break;
        default:
            break;
    }

    int totalTicks = config.totalTicks;
    if (totalTicks == 0 && !config.notes.empty()) {
        totalTicks = static_cast<int>((config.notes.back().startBeat + config.notes.back().duration) * TICKS_PER_BEAT);
    }

    auto curve = generateCurve(totalTicks, shape, startMarking, endMarking);

    std::vector<MidiNote> result = applyCurve(config.notes, curve);

    if (config.applyAccents) {
        result = applyAccents(result, emotionLower, profile.accentStrength * config.expressiveness);
    }

    return {
        result,
        curve,
        emotionLower,
        baseMarking,
        shape,
        profile.velocityRange
    };
}

} // namespace kelly
