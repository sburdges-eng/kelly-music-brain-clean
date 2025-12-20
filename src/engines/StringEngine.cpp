#include "StringEngine.h"
#include "../common/MusicConstants.h"
#include <random>
#include <algorithm>

namespace kelly {
using namespace MusicConstants;

namespace {
    // Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition
}

StringEngine::StringEngine() {
    initializeProfiles();
}

void StringEngine::initializeProfiles() {
    profiles_["grief"] = {
        StringArticulation::Sustain,
        StringSection::FullEnsemble,
        StringDynamicCurve::Swell,
        {40, 75},
        0.95f,
        0.6f,
        true,
        {48, 49}  // String Ensemble 1 & 2
    };

    profiles_["sadness"] = {
        StringArticulation::Legato,
        StringSection::Cellos,
        StringDynamicCurve::Decrescendo,
        {35, 65},
        0.9f,
        0.5f,
        false,
        {42}  // Cello
    };

    profiles_["hope"] = {
        StringArticulation::Sustain,
        StringSection::Violins1,
        StringDynamicCurve::Crescendo,
        {50, 85},
        0.85f,
        0.4f,
        false,
        {40}  // Violin
    };

    profiles_["anger"] = {
        StringArticulation::Marcato,
        StringSection::FullEnsemble,
        StringDynamicCurve::Sforzando,
        {80, 110},
        0.6f,
        0.2f,
        false,
        {44}  // Tremolo Strings
    };

    profiles_["fear"] = {
        StringArticulation::Tremolo,
        StringSection::Violas,
        StringDynamicCurve::Crescendo,
        {45, 80},
        0.7f,
        0.8f,
        true,
        {44}  // Tremolo Strings
    };

    profiles_["joy"] = {
        StringArticulation::Spiccato,
        StringSection::Violins1,
        StringDynamicCurve::Accent,
        {70, 100},
        0.5f,
        0.3f,
        false,
        {45}  // Pizzicato Strings
    };

    profiles_["tension"] = {
        StringArticulation::Tremolo,
        StringSection::FullEnsemble,
        StringDynamicCurve::Crescendo,
        {50, 95},
        0.8f,
        0.7f,
        true,
        {44}  // Tremolo Strings
    };

    profiles_["neutral"] = {
        StringArticulation::Sustain,
        StringSection::FullEnsemble,
        StringDynamicCurve::Flat,
        {55, 75},
        0.85f,
        0.3f,
        false,
        {48}  // String Ensemble 1
    };
}

StringOutput StringEngine::generate(
    const std::string& emotion,
    const std::vector<std::string>& chordProgression,
    const std::string& key,
    int bars,
    int tempoBpm
) {
    StringConfig config;
    config.emotion = emotion;
    config.chordProgression = chordProgression;
    config.key = key;
    config.bars = bars;
    config.tempoBpm = tempoBpm;
    return generate(config);
}

StringOutput StringEngine::generate(const StringConfig& config) {
    StringOutput output;
    output.emotion = config.emotion;

    auto profileIt = profiles_.find(config.emotion);
    const auto& profile = profileIt != profiles_.end() ? profileIt->second : profiles_["neutral"];

    output.articulationUsed = config.articulationOverride.value_or(profile.articulation);
    output.sectionUsed = config.sectionOverride.value_or(profile.section);
    output.gmInstruments = profile.gmInstruments;

    std::mt19937 rng(config.seed >= 0 ? static_cast<unsigned int>(config.seed) : std::random_device{}());

    int ticksPerBar = TICKS_PER_BEAT * static_cast<int>(BEATS_PER_BAR);
    int ticksPerChord = ticksPerBar;

    if (!config.chordProgression.empty()) {
        ticksPerChord = (ticksPerBar * config.bars) / static_cast<int>(config.chordProgression.size());
    }

    int currentTick = 0;

    for (int bar = 0; bar < config.bars; ++bar) {
        int velRange = profile.velocityRange.second - profile.velocityRange.first;
        int baseVelocity = profile.velocityRange.first +
            static_cast<int>(rng() % static_cast<unsigned int>(std::max(1, velRange)));

        // Apply dynamic curve
        switch (profile.dynamicCurve) {
            case StringDynamicCurve::Flat:
                // No change
                break;
            case StringDynamicCurve::Crescendo:
                baseVelocity += (bar * 10);
                break;
            case StringDynamicCurve::Decrescendo:
                baseVelocity -= (bar * 8);
                break;
            case StringDynamicCurve::Swell:
                baseVelocity += static_cast<int>(15 * std::sin(bar * 0.5f));
                break;
            case StringDynamicCurve::Accent:
                baseVelocity += (bar % 2 == 0 ? 10 : -5);
                break;
            case StringDynamicCurve::Sforzando:
                baseVelocity += (bar == 0 ? 20 : 0);
                break;
        }
        baseVelocity = std::clamp(baseVelocity, MIDI_VELOCITY_SOFT - 15, MIDI_VELOCITY_LOUD + 20);

        int duration = static_cast<int>(ticksPerChord * profile.sustainRatio);

        // Generate notes for section
        std::vector<int> pitches;
        switch (output.sectionUsed) {
            case StringSection::Violins1:
                pitches = {67, 71, 74};  // G4, B4, D5
                break;
            case StringSection::Violins2:
                pitches = {62, 66, 69};  // D4, F#4, A4
                break;
            case StringSection::Violas:
                pitches = {57, 60, 64};  // A3, C4, E4
                break;
            case StringSection::Cellos:
                pitches = {48, 52, 55};  // C3, E3, G3
                break;
            case StringSection::Basses:
                pitches = {36, 40, 43};  // C2, E2, G2
                break;
            case StringSection::FullEnsemble:
                pitches = {36, 48, 55, 60, 67};  // Full voicing
                break;
        }

        for (int pitch : pitches) {
            StringNote note;
            note.pitch = pitch;
            note.startTick = currentTick;
            note.durationTicks = duration;
            note.velocity = baseVelocity;
            note.articulation = output.articulationUsed;
            note.section = output.sectionUsed;
            output.notes.push_back(note);
        }

        currentTick += ticksPerChord;
    }

    output.totalTicks = currentTick;
    return output;
}

int StringEngine::getGmInstrument(StringSection section) const {
    switch (section) {
        case StringSection::Violins1:
        case StringSection::Violins2:
            return 40;  // Violin
        case StringSection::Violas:
            return 41;  // Viola
        case StringSection::Cellos:
            return 42;  // Cello
        case StringSection::Basses:
            return 43;  // Contrabass
        case StringSection::FullEnsemble:
            return 48;  // String Ensemble 1
    }
    return 48;
}

std::vector<int> StringEngine::assignToSections(const std::vector<int>& pitches, StringSection /* section */) {
    return pitches;  // Simplified - returns as-is
}

} // namespace kelly
