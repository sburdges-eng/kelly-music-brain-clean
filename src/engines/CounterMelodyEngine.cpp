#include "CounterMelodyEngine.h"
#include "../common/MusicConstants.h"
#include <random>
#include <algorithm>

namespace kelly {
using namespace MusicConstants;

namespace {
    // Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition
}

CounterMelodyEngine::CounterMelodyEngine() {
    initializeProfiles();
}

void CounterMelodyEngine::initializeProfiles() {
    profiles_["grief"] = {CounterMelodyType::Contrary, CounterMelodyRelation::Sixth, 0.4f, {35, 60}, 73};
    profiles_["sadness"] = {CounterMelodyType::Oblique, CounterMelodyRelation::Third, 0.3f, {30, 55}, 71};
    profiles_["hope"] = {CounterMelodyType::Parallel, CounterMelodyRelation::Third, 0.6f, {50, 80}, 73};
    profiles_["anger"] = {CounterMelodyType::Contrary, CounterMelodyRelation::Fourth, 0.7f, {70, 100}, 56};
    profiles_["fear"] = {CounterMelodyType::Ostinato, CounterMelodyRelation::Fifth, 0.5f, {40, 70}, 71};
    profiles_["joy"] = {CounterMelodyType::Imitation, CounterMelodyRelation::Third, 0.7f, {60, 90}, 73};
    profiles_["tension"] = {CounterMelodyType::Contrary, CounterMelodyRelation::Fourth, 0.6f, {50, 85}, 68};
    profiles_["neutral"] = {CounterMelodyType::Parallel, CounterMelodyRelation::Third, 0.5f, {50, 70}, 73};
}

CounterMelodyOutput CounterMelodyEngine::generate(
    const std::string& emotion,
    const std::vector<MidiNote>& primaryMelody,
    const std::string& key,
    const std::string& mode
) {
    CounterMelodyConfig config;
    config.emotion = emotion;
    config.primaryMelody = primaryMelody;
    config.key = key;
    config.mode = mode;
    return generate(config);
}

CounterMelodyOutput CounterMelodyEngine::generate(const CounterMelodyConfig& config) {
    CounterMelodyOutput output;

    auto profileIt = profiles_.find(config.emotion);
    const auto& profile = profileIt != profiles_.end() ? profileIt->second : profiles_["neutral"];

    output.typeUsed = profile.preferredType;
    output.relationUsed = profile.preferredRelation;
    output.gmInstrument = profile.gmInstrument;

    std::mt19937 rng(config.seed >= 0 ? static_cast<unsigned int>(config.seed) : std::random_device{}());
    auto scale = getScalePitches(config.key, config.mode);

    int maxTick = 0;

    for (const auto& note : config.primaryMelody) {
        if (static_cast<float>(rng() % 100) / 100.0f > profile.density) continue;

        CounterMelodyNote counterNote;

        switch (output.typeUsed) {
            case CounterMelodyType::Parallel:
                counterNote.pitch = transposeByInterval(note.pitch, output.relationUsed, false);
                break;
            case CounterMelodyType::Contrary:
                counterNote.pitch = transposeByInterval(note.pitch, output.relationUsed, true);
                break;
            case CounterMelodyType::Oblique:
                counterNote.pitch = config.primaryMelody[0].pitch - 12;
                break;
            case CounterMelodyType::Imitation:
                counterNote.pitch = note.pitch - 12;
                counterNote.startTick = static_cast<int>(note.startBeat * TICKS_PER_BEAT) + TICKS_PER_BEAT;
                break;
            case CounterMelodyType::Pedal:
                counterNote.pitch = scale[0] + 36;
                break;
            case CounterMelodyType::Ostinato:
                counterNote.pitch = scale[static_cast<size_t>(rng() % static_cast<unsigned int>(scale.size()))] + 48;
                break;
            case CounterMelodyType::Independent:
                counterNote.pitch = transposeByInterval(note.pitch, output.relationUsed, rng() % 2 == 0);
                break;
            default:
                counterNote.pitch = transposeByInterval(note.pitch, output.relationUsed, rng() % 2 == 0);
                break;
        }

        counterNote.pitch = snapToScale(counterNote.pitch, scale);
        if (output.typeUsed != CounterMelodyType::Imitation) {
            counterNote.startTick = static_cast<int>(note.startBeat * TICKS_PER_BEAT);
        }
        counterNote.durationTicks = static_cast<int>(note.duration * TICKS_PER_BEAT);
        int velRange = profile.velocityRange.second - profile.velocityRange.first;
        counterNote.velocity = profile.velocityRange.first +
            static_cast<int>(rng() % static_cast<unsigned int>(std::max(1, velRange)));

        output.notes.push_back(counterNote);
        maxTick = std::max(maxTick, counterNote.startTick + counterNote.durationTicks);
    }

    output.totalTicks = maxTick;
    return output;
}

int CounterMelodyEngine::transposeByInterval(int pitch, CounterMelodyRelation relation, bool above) {
    int interval = 0;
    switch (relation) {
        case CounterMelodyRelation::Third: interval = 3; break;
        case CounterMelodyRelation::Fourth: interval = 5; break;
        case CounterMelodyRelation::Fifth: interval = 7; break;
        case CounterMelodyRelation::Sixth: interval = 9; break;
        case CounterMelodyRelation::Octave: interval = 12; break;
        case CounterMelodyRelation::Tenth: interval = 15; break;
    }
    return above ? pitch + interval : pitch - interval;
}

std::vector<int> CounterMelodyEngine::getScalePitches(const std::string& key, const std::string& mode) {
    static const std::vector<std::string> chromatic = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    int rootIndex = 0;
    for (size_t i = 0; i < chromatic.size(); ++i) {
        if (chromatic[i] == key) { rootIndex = static_cast<int>(i); break; }
    }

    std::vector<int> intervals = (mode == "minor") ?
        std::vector<int>{0, 2, 3, 5, 7, 8, 10} : std::vector<int>{0, 2, 4, 5, 7, 9, 11};

    std::vector<int> pitches;
    for (int i : intervals) pitches.push_back((rootIndex + i) % INTERVAL_OCTAVE);
    return pitches;
}

int CounterMelodyEngine::snapToScale(int pitch, const std::vector<int>& scale) {
    int pc = pitch % INTERVAL_OCTAVE;
    int octave = pitch / INTERVAL_OCTAVE;

    for (int s : scale) {
        if (s == pc) return pitch;
    }

    int closest = scale[0];
    int minDist = INTERVAL_OCTAVE;
    for (int s : scale) {
        int dist = std::min(std::abs(pc - s), INTERVAL_OCTAVE - std::abs(pc - s));
        if (dist < minDist) { minDist = dist; closest = s; }
    }
    return octave * INTERVAL_OCTAVE + closest;
}

} // namespace kelly
