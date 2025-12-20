#include "voice/LyriSync.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace kelly {

LyriSync::LyriSync()
    : bpm_(120.0f)
{
}

LyriSync::SyncResult LyriSync::synchronize(
    const LyricStructure& lyrics,
    const std::vector<VoiceSynthesizer::VocalNote>& vocalNotes,
    float bpm)
{
    SyncResult result;
    bpm_ = bpm;

    if (vocalNotes.empty()) {
        return result;
    }

    // Calculate total duration
    double totalDuration = 0.0;
    for (const auto& note : vocalNotes) {
        totalDuration = std::max(totalDuration, note.startBeat + note.duration);
    }
    result.totalDuration = totalDuration;

    // Synchronize each line with vocal notes
    size_t noteIndex = 0;
    double currentBeat = 0.0;

    for (const auto& section : lyrics.sections) {
        for (const auto& line : section.lines) {
            // Find vocal notes for this line
            std::vector<VoiceSynthesizer::VocalNote> lineNotes;
            while (noteIndex < vocalNotes.size() &&
                   vocalNotes[noteIndex].startBeat < currentBeat + line.targetSyllables * 0.5) {
                lineNotes.push_back(vocalNotes[noteIndex]);
                noteIndex++;
            }

            // Calculate durations for syllables in this line
            double lineDuration = 0.0;
            for (const auto& note : lineNotes) {
                lineDuration += note.duration;
            }

            std::vector<double> syllableDurations;
            if (!line.syllables.empty()) {
                syllableDurations = calculateSyllableDurations(line.syllables, lineDuration);
            } else {
                // Estimate: divide duration evenly among words
                std::istringstream iss(line.text);
                std::vector<std::string> words;
                std::string word;
                while (iss >> word) {
                    words.push_back(word);
                }

                if (!words.empty()) {
                    double durationPerWord = lineDuration / words.size();
                    syllableDurations.resize(words.size(), durationPerWord);
                }
            }

            // Create sync items for this line
            double lineStartBeat = currentBeat;
            for (size_t i = 0; i < line.syllables.size() && i < syllableDurations.size(); ++i) {
                SyncItem item;
                item.text = line.syllables[i].text;
                item.startBeat = lineStartBeat;
                item.duration = syllableDurations[i];

                result.items.push_back(item);
                lineStartBeat += syllableDurations[i];
            }

            // If no syllables, create item for entire line
            if (line.syllables.empty() && !line.text.empty()) {
                SyncItem item;
                item.text = line.text;
                item.startBeat = currentBeat;
                item.duration = lineDuration > 0.0 ? lineDuration : 1.0; // Default 1 beat
                result.items.push_back(item);
                lineStartBeat += item.duration;
            }

            currentBeat = lineStartBeat;
        }
    }

    return result;
}

int LyriSync::getCurrentItem(const SyncResult& syncResult, double currentBeat) const {
    for (size_t i = 0; i < syncResult.items.size(); ++i) {
        const auto& item = syncResult.items[i];
        if (currentBeat >= item.startBeat &&
            currentBeat < item.startBeat + item.duration) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::vector<double> LyriSync::calculateSyllableDurations(
    const std::vector<Syllable>& syllables,
    double totalDuration) const
{
    std::vector<double> durations;

    if (syllables.empty()) {
        return durations;
    }

    if (syllables.size() == 1) {
        durations.push_back(totalDuration);
        return durations;
    }

    // Apply stress timing
    durations = applyStressTiming(syllables, totalDuration / syllables.size());

    return durations;
}

std::vector<double> LyriSync::applyStressTiming(
    const std::vector<Syllable>& syllables,
    double baseDuration) const
{
    std::vector<double> durations;

    if (syllables.empty()) {
        return durations;
    }

    // Calculate total weight (stressed syllables get more time)
    double totalWeight = 0.0;
    std::vector<double> weights;

    for (const auto& syllable : syllables) {
        double weight = 1.0;
        if (syllable.stress == 2) {
            weight = 1.5;  // Primary stress: 50% longer
        } else if (syllable.stress == 1) {
            weight = 1.2;  // Secondary stress: 20% longer
        }
        weights.push_back(weight);
        totalWeight += weight;
    }

    // Distribute duration proportionally
    double totalDuration = baseDuration * syllables.size();

    for (size_t i = 0; i < syllables.size(); ++i) {
        double duration = totalDuration * (weights[i] / totalWeight);
        durations.push_back(duration);
    }

    return durations;
}

double LyriSync::applyRubato(double baseDuration, float rubatoAmount, float position) const {
    // Rubato: slight tempo variation (typically slowing at phrase end)
    // Simple implementation: slow down towards the end
    float rubatoFactor = 1.0f + (rubatoAmount * position * 0.2f); // Up to 20% slower
    return baseDuration * rubatoFactor;
}

} // namespace kelly
