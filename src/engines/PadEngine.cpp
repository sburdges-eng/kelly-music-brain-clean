#include "PadEngine.h"
#include "../common/MusicConstants.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace kelly {
using namespace MusicConstants;

namespace {
    // Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition
    const std::vector<std::string> CHROMATIC = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

    const std::map<std::string, std::vector<int>> CHORD_INTERVALS = {
        {"maj", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH}},
        {"min", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FIFTH}},
        {"m", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FIFTH}},
        {"dim", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_TRITONE}},
        {"aug", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_MINOR_SIXTH}},
        {"7", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SEVENTH}},
        {"maj7", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH, INTERVAL_MAJOR_SEVENTH}},
        {"min7", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SEVENTH}},
        {"m7", {INTERVAL_UNISON, INTERVAL_MINOR_THIRD, INTERVAL_PERFECT_FIFTH, INTERVAL_MINOR_SEVENTH}},
        {"sus2", {INTERVAL_UNISON, INTERVAL_MAJOR_SECOND, INTERVAL_PERFECT_FIFTH}},
        {"sus4", {INTERVAL_UNISON, INTERVAL_PERFECT_FOURTH, INTERVAL_PERFECT_FIFTH}},
        {"add9", {INTERVAL_UNISON, INTERVAL_MAJOR_THIRD, INTERVAL_PERFECT_FIFTH, INTERVAL_OCTAVE + INTERVAL_MAJOR_SECOND}},
    };
}

PadEngine::PadEngine() {
    initializeProfiles();
}

void PadEngine::initializeProfiles() {
    profiles_["grief"] = {
        PadTexture::Warm,
        PadMovement::Breathing,
        PadVoicing::Open,
        PadRegister::Mid,
        {40, 70},
        0.7f,
        0.95f,
        0.5f,
        true,
        0.4f,
        89  // GM Warm Pad
    };

    profiles_["sadness"] = {
        PadTexture::Hollow,
        PadMovement::Fading,
        PadVoicing::Sparse,
        PadRegister::Mid,
        {35, 60},
        0.5f,
        0.85f,
        0.3f,
        true,
        0.5f,
        91  // GM Choir Pad
    };

    profiles_["hope"] = {
        PadTexture::Bright,
        PadMovement::Swelling,
        PadVoicing::Open,
        PadRegister::High,
        {50, 85},
        0.6f,
        0.9f,
        0.6f,
        true,
        0.3f,
        94  // GM Halo Pad
    };

    profiles_["anger"] = {
        PadTexture::Gritty,
        PadMovement::Pulsing,
        PadVoicing::Close,
        PadRegister::Low,
        {70, 100},
        0.8f,
        0.8f,
        0.8f,
        false,
        0.2f,
        93  // GM Metallic Pad
    };

    profiles_["fear"] = {
        PadTexture::Dark,
        PadMovement::Unstable,
        PadVoicing::Cluster,
        PadRegister::Low,
        {45, 75},
        0.6f,
        0.7f,
        0.7f,
        true,
        0.6f,
        95  // GM Sweep Pad
    };

    profiles_["joy"] = {
        PadTexture::Shimmering,
        PadMovement::Breathing,
        PadVoicing::Full,
        PadRegister::Wide,
        {60, 90},
        0.8f,
        0.9f,
        0.5f,
        true,
        0.4f,
        88  // GM New Age Pad
    };

    profiles_["anxiety"] = {
        PadTexture::Glassy,
        PadMovement::Tremolo,
        PadVoicing::Close,
        PadRegister::High,
        {50, 80},
        0.5f,
        0.6f,
        0.9f,
        true,
        0.5f,
        98  // GM Crystal
    };

    profiles_["peace"] = {
        PadTexture::Airy,
        PadMovement::Static,
        PadVoicing::Open,
        PadRegister::Mid,
        {35, 55},
        0.4f,
        0.98f,
        0.2f,
        true,
        0.3f,
        89  // GM Warm Pad
    };

    profiles_["neutral"] = {
        PadTexture::Warm,
        PadMovement::Static,
        PadVoicing::Open,
        PadRegister::Mid,
        {50, 70},
        0.5f,
        0.9f,
        0.4f,
        false,
        0.2f,
        89  // GM Warm Pad
    };
}

PadOutput PadEngine::generate(
    const std::string& emotion,
    const std::vector<std::string>& chordProgression,
    const std::string& key,
    int bars,
    int tempoBpm
) {
    PadConfig config;
    config.emotion = emotion;
    config.chordProgression = chordProgression;
    config.key = key;
    config.bars = bars;
    config.tempoBpm = tempoBpm;
    return generate(config);
}

PadOutput PadEngine::generate(const PadConfig& config) {
    PadOutput output;
    output.emotion = config.emotion;

    auto profileIt = profiles_.find(config.emotion);
    const auto& profile = profileIt != profiles_.end() ? profileIt->second : profiles_["neutral"];

    output.textureUsed = config.textureOverride.value_or(profile.texture);
    output.movementUsed = config.movementOverride.value_or(profile.movement);
    output.voicingUsed = config.voicingOverride.value_or(profile.voicing);
    output.gmInstrument = profile.gmInstrument;

    std::mt19937 rng(config.seed >= 0 ? static_cast<unsigned int>(config.seed) : std::random_device{}());

    int ticksPerBar = TICKS_PER_BEAT * config.beatsPerBar;
    int ticksPerChord = ticksPerBar;

    if (!config.chordProgression.empty()) {
        ticksPerChord = (ticksPerBar * config.bars) / config.chordProgression.size();
    }

    int currentTick = 0;
    int chordIndex = 0;

    for (int bar = 0; bar < config.bars; ++bar) {
        std::string chord = config.chordProgression.empty() ?
            config.key : config.chordProgression[static_cast<size_t>(chordIndex % static_cast<int>(config.chordProgression.size()))];

        auto pitches = parseChord(chord, config.key);

        int basePitch = MIDI_C3;
        switch (profile.padRegister) {
            case PadRegister::Sub: basePitch = MIDI_C2; break;
            case PadRegister::Low: basePitch = MIDI_Fs2; break;
            case PadRegister::Mid: basePitch = MIDI_C3; break;
            case PadRegister::High: basePitch = MIDI_C4; break;
            case PadRegister::Wide: basePitch = MIDI_Fs2; break;
        }

        auto voicedPitches = applyVoicing(pitches, output.voicingUsed, basePitch);

        int duration = static_cast<int>(ticksPerChord * profile.sustainRatio);
        int baseVelocity = profile.velocityRange.first +
            rng() % (profile.velocityRange.second - profile.velocityRange.first);

        auto velocities = applyVelocityCurve(baseVelocity, output.movementUsed, static_cast<int>(voicedPitches.size()));

        for (size_t i = 0; i < voicedPitches.size(); ++i) {
            PadNote note;
            note.pitch = voicedPitches[i];
            note.startTick = currentTick;
            note.durationTicks = duration;
            note.velocity = velocities[i];
            output.notes.push_back(note);
        }

        currentTick += ticksPerChord;
        chordIndex++;
    }

    output.totalTicks = currentTick;
    return output;
}

std::vector<int> PadEngine::parseChord(const std::string& chord, const std::string& /* key */) {
    std::vector<int> pitches;

    std::string root = chord.substr(0, 1);
    if (chord.size() > 1 && (chord[1] == '#' || chord[1] == 'b')) {
        root = chord.substr(0, 2);
    }

    std::string quality = chord.substr(root.size());
    if (quality.empty()) quality = "maj";

    int rootIndex = 0;
    for (size_t i = 0; i < CHROMATIC.size(); ++i) {
        if (CHROMATIC[i] == root) {
            rootIndex = static_cast<int>(i);
            break;
        }
    }

    auto it = CHORD_INTERVALS.find(quality);
    const auto& intervals = it != CHORD_INTERVALS.end() ? it->second : CHORD_INTERVALS.at("maj");

    for (int interval : intervals) {
        pitches.push_back(rootIndex + interval);
    }

    return pitches;
}

std::vector<int> PadEngine::applyVoicing(const std::vector<int>& pitches, PadVoicing voicing, int basePitch) {
    std::vector<int> voiced;

    switch (voicing) {
        case PadVoicing::Close:
            for (int p : pitches) {
                voiced.push_back(basePitch + p);
            }
            break;

        case PadVoicing::Open:
            if (!pitches.empty()) voiced.push_back(basePitch + pitches[0]);
            if (pitches.size() > 2) voiced.push_back(basePitch + pitches[2] + INTERVAL_OCTAVE);
            if (pitches.size() > 1) voiced.push_back(basePitch + pitches[1] + INTERVAL_OCTAVE);
            break;

        case PadVoicing::Octaves:
            if (!pitches.empty()) {
                voiced.push_back(basePitch + pitches[0]);
                voiced.push_back(basePitch + pitches[0] + INTERVAL_OCTAVE);
            }
            break;

        case PadVoicing::Fifths:
            if (!pitches.empty()) voiced.push_back(basePitch + pitches[0]);
            if (pitches.size() > 2) voiced.push_back(basePitch + pitches[2]);
            break;

        case PadVoicing::Full:
            for (int p : pitches) {
                voiced.push_back(basePitch + p);
            }
            if (!pitches.empty()) voiced.push_back(basePitch + pitches[0] + INTERVAL_OCTAVE);
            break;

        case PadVoicing::Sparse:
            if (!pitches.empty()) voiced.push_back(basePitch + pitches[0]);
            if (pitches.size() > 2) voiced.push_back(basePitch + pitches[2] + INTERVAL_OCTAVE);
            break;

        case PadVoicing::Cluster:
            for (int p : pitches) {
                voiced.push_back(basePitch + p);
                voiced.push_back(basePitch + p + INTERVAL_MINOR_SECOND);
            }
            break;

        case PadVoicing::Shell:
            if (!pitches.empty()) voiced.push_back(basePitch + pitches[0]);
            if (pitches.size() > 1) voiced.push_back(basePitch + pitches[1] + INTERVAL_OCTAVE);
            if (pitches.size() > 3) voiced.push_back(basePitch + pitches[3] + INTERVAL_OCTAVE);
            break;
    }

    return voiced;
}

std::vector<int> PadEngine::applyVelocityCurve(int baseVelocity, PadMovement movement, int numNotes) {
    std::vector<int> velocities(numNotes, baseVelocity);

    switch (movement) {
        case PadMovement::Static:
            // No change, velocities already set to baseVelocity
            break;

        case PadMovement::Breathing:
            for (int i = 0; i < numNotes; ++i) {
                float phase = static_cast<float>(i) / numNotes;
                velocities[static_cast<size_t>(i)] = baseVelocity + static_cast<int>(10 * std::sin(phase * 3.14159f));
            }
            break;

        case PadMovement::Swelling:
            for (int i = 0; i < numNotes; ++i) {
                float factor = static_cast<float>(i) / numNotes;
                velocities[static_cast<size_t>(i)] = baseVelocity + static_cast<int>(20 * factor);
            }
            break;

        case PadMovement::Fading:
            for (int i = 0; i < numNotes; ++i) {
                float factor = 1.0f - static_cast<float>(i) / numNotes;
                velocities[static_cast<size_t>(i)] = static_cast<int>(baseVelocity * factor);
            }
            break;

        case PadMovement::Pulsing:
            for (int i = 0; i < numNotes; ++i) {
                float phase = static_cast<float>(i) / numNotes;
                velocities[static_cast<size_t>(i)] = baseVelocity + static_cast<int>(15 * std::sin(phase * 6.28318f));
            }
            break;

        case PadMovement::Drifting:
            for (int i = 0; i < numNotes; ++i) {
                float phase = static_cast<float>(i) / numNotes;
                velocities[static_cast<size_t>(i)] = baseVelocity + static_cast<int>(8 * std::sin(phase * 1.5708f));
            }
            break;

        case PadMovement::Tremolo:
            for (int i = 0; i < numNotes; ++i) {
                float phase = static_cast<float>(i) / numNotes;
                velocities[static_cast<size_t>(i)] = baseVelocity + static_cast<int>(12 * std::sin(phase * 12.5664f));
            }
            break;

        case PadMovement::Evolving:
            for (int i = 0; i < numNotes; ++i) {
                float factor = static_cast<float>(i) / numNotes;
                velocities[static_cast<size_t>(i)] = baseVelocity + static_cast<int>(25 * factor * factor);
            }
            break;

        case PadMovement::Unstable:
            for (int i = 0; i < numNotes; ++i) {
                float phase = static_cast<float>(i) / numNotes;
                velocities[static_cast<size_t>(i)] = baseVelocity + static_cast<int>(20 * std::sin(phase * 9.4248f));
            }
            break;

        default:
            break;
    }

    for (auto& v : velocities) {
        v = std::clamp(v, MIDI_VELOCITY_MIN + 1, MIDI_VELOCITY_MAX);
    }

    return velocities;
}

} // namespace kelly
