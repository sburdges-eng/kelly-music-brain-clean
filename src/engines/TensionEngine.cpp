#include "TensionEngine.h"
#include "../common/MusicConstants.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace kelly {
using namespace MusicConstants;

namespace {
    // Use global TICKS_PER_BEAT from KellyTypes.h instead of local definition
}

TensionEngine::TensionEngine() {
    initializeProfiles();
}

void TensionEngine::initializeProfiles() {
    profiles_["grief"] = {
        {TensionTechnique::Suspension, TensionTechnique::Appoggiatura, TensionTechnique::Pedal},
        TensionCurve::Wave,
        0.4f, 0.2f
    };
    profiles_["sadness"] = {
        {TensionTechnique::Suspension, TensionTechnique::Chromatic},
        TensionCurve::Releasing,
        0.3f, 0.15f
    };
    profiles_["hope"] = {
        {TensionTechnique::Dominant, TensionTechnique::Suspension},
        TensionCurve::Building,
        0.3f, 0.3f
    };
    profiles_["anger"] = {
        {TensionTechnique::Tritone, TensionTechnique::Diminished, TensionTechnique::Cluster},
        TensionCurve::Spike,
        0.7f, 0.3f
    };
    profiles_["fear"] = {
        {TensionTechnique::Diminished, TensionTechnique::Chromatic, TensionTechnique::Cluster},
        TensionCurve::Building,
        0.6f, 0.25f
    };
    profiles_["joy"] = {
        {TensionTechnique::Dominant, TensionTechnique::Suspension},
        TensionCurve::Releasing,
        0.2f, 0.2f
    };
    profiles_["anxiety"] = {
        {TensionTechnique::Diminished, TensionTechnique::Chromatic, TensionTechnique::Tritone},
        TensionCurve::Wave,
        0.65f, 0.2f
    };
    profiles_["neutral"] = {
        {TensionTechnique::Dominant, TensionTechnique::Suspension},
        TensionCurve::Plateau,
        0.3f, 0.15f
    };
}

TensionOutput TensionEngine::generate(
    const std::string& emotion,
    const std::vector<std::string>& chordProgression,
    int bars,
    int tempoBpm
) {
    TensionConfig config;
    config.emotion = emotion;
    config.chordProgression = chordProgression;
    config.bars = bars;
    config.tempoBpm = tempoBpm;
    return generate(config);
}

TensionOutput TensionEngine::generate(const TensionConfig& config) {
    TensionOutput output;

    auto profileIt = profiles_.find(config.emotion);
    const auto& profile = profileIt != profiles_.end() ? profileIt->second : profiles_["neutral"];

    output.curveUsed = profile.preferredCurve;

    std::mt19937 rng(config.seed >= 0 ? static_cast<unsigned int>(config.seed) : std::random_device{}());

    int ticksPerBar = TICKS_PER_BEAT * static_cast<int>(BEATS_PER_BAR);
    int totalTicks = ticksPerBar * config.bars;
    int numPoints = config.bars * 2;  // Two tension points per bar

    output.tensionCurve = generateCurve(output.curveUsed, numPoints, config.maxTension);

    output.peakTension = 0;
    output.peakTick = 0;

    for (int i = 0; i < numPoints; ++i) {
        TensionPoint point;
        point.tick = (i * totalTicks) / numPoints;
        point.tensionLevel = profile.baseTension +
            output.tensionCurve[i] * profile.tensionVariance;
        point.tensionLevel = std::clamp(point.tensionLevel, 0.0f, 1.0f);

        // Select technique based on tension level
        if (!profile.preferredTechniques.empty()) {
            size_t techIndex = static_cast<size_t>(
                point.tensionLevel * (profile.preferredTechniques.size() - 1)
            );
            point.technique = profile.preferredTechniques[techIndex];
        } else {
            point.technique = TensionTechnique::Dominant;
        }

        output.tensionPoints.push_back(point);

        if (point.tensionLevel > output.peakTension) {
            output.peakTension = point.tensionLevel;
            output.peakTick = point.tick;
        }
    }

    return output;
}

float TensionEngine::calculateTension(const std::vector<int>& pitches, const std::string& /* context */) {
    if (pitches.size() < 2) return 0.0f;

    float tension = 0.0f;

    // Check for dissonant intervals
    for (size_t i = 0; i < pitches.size(); ++i) {
        for (size_t j = i + 1; j < pitches.size(); ++j) {
            int interval = std::abs(pitches[i] - pitches[j]) % INTERVAL_OCTAVE;

            // Tritone
            if (interval == INTERVAL_TRITONE) tension += 0.3f;
            // Minor second
            if (interval == INTERVAL_MINOR_SECOND) tension += 0.25f;
            // Major seventh
            if (interval == INTERVAL_MAJOR_SEVENTH) tension += 0.2f;
            // Minor ninth (over octave)
            if (interval == INTERVAL_MINOR_SECOND || interval == INTERVAL_OCTAVE + INTERVAL_MINOR_SECOND) tension += 0.2f;
        }
    }

    return std::clamp(tension, 0.0f, 1.0f);
}

std::vector<int> TensionEngine::addTensionNotes(
    const std::vector<int>& chordPitches,
    TensionTechnique technique,
    float amount
) {
    (void)amount;  // Mark as intentionally unused
    std::vector<int> result = chordPitches;
    if (chordPitches.empty()) return result;

    int root = chordPitches[0];

    switch (technique) {
        case TensionTechnique::Dominant:
            result.push_back(root + INTERVAL_MINOR_SEVENTH);  // Add b7
            break;
        case TensionTechnique::Diminished:
            result.push_back(root + INTERVAL_TRITONE);   // Tritone
            break;
        case TensionTechnique::Chromatic:
            result.push_back(root + INTERVAL_MINOR_SECOND);   // Minor 2nd
            break;
        case TensionTechnique::Suspension:
            if (chordPitches.size() > 1) {
                result[1] = root + INTERVAL_PERFECT_FOURTH;     // Replace 3rd with 4th
            }
            break;
        case TensionTechnique::Appoggiatura:
            result.push_back(root + INTERVAL_OCTAVE + INTERVAL_MINOR_SECOND);  // Minor 9th
            break;
        case TensionTechnique::Tritone:
            result.push_back(root + INTERVAL_TRITONE);
            result.push_back(root + INTERVAL_MINOR_SEVENTH);
            break;
        case TensionTechnique::Cluster:
            result.push_back(root + INTERVAL_MINOR_SECOND);
            result.push_back(root + INTERVAL_MAJOR_SECOND);
            break;
        case TensionTechnique::Pedal:
            // Pedal point - keep root, add tension above
            result.push_back(root + INTERVAL_PERFECT_FIFTH);  // Add 5th above
            break;
        case TensionTechnique::Polytonality:
            // Add pitches from different key
            result.push_back(root + INTERVAL_MINOR_THIRD);  // Minor 3rd
            result.push_back(root + INTERVAL_MINOR_SIXTH);  // Minor 6th
            break;
    }

    return result;
}

std::vector<float> TensionEngine::generateCurve(TensionCurve curve, int numPoints, float maxTension) {
    std::vector<float> result(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(std::max(1, numPoints - 1));

        switch (curve) {
            case TensionCurve::Building:
                result[i] = t * maxTension;
                break;
            case TensionCurve::Releasing:
                result[i] = (1.0f - t) * maxTension;
                break;
            case TensionCurve::Plateau:
                result[i] = maxTension * 0.7f;
                break;
            case TensionCurve::Spike:
                result[i] = (t < 0.8f) ? t * maxTension : maxTension;
                break;
            case TensionCurve::Wave:
                result[i] = (std::sin(t * 3.14159f * 2) + 1.0f) * 0.5f * maxTension;
                break;
            case TensionCurve::Ramp:
                result[i] = t * t * maxTension;
                break;
        }
    }

    return result;
}

} // namespace kelly
