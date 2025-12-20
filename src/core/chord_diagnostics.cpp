#include "chord_diagnostics.h"
#include <algorithm>
#include <cmath>

namespace kelly {

float ChordDiagnostics::intervalDissonance(int interval) const {
    // Simplified dissonance ratings for intervals (semitones mod 12)
    static const float dissonanceMap[12] = {
        0.0f,  // unison
        0.8f,  // minor 2nd
        0.4f,  // major 2nd
        0.5f,  // minor 3rd
        0.3f,  // major 3rd
        0.2f,  // perfect 4th
        0.9f,  // tritone
        0.1f,  // perfect 5th
        0.4f,  // minor 6th
        0.3f,  // major 6th
        0.5f,  // minor 7th
        0.4f   // major 7th
    };
    
    int normInterval = std::abs(interval) % 12;
    return dissonanceMap[normInterval];
}

float ChordDiagnostics::calculateDissonance(const Chord& chord) const {
    if (chord.notes.size() < 2) {
        return 0.0f;
    }
    
    float totalDissonance = 0.0f;
    int pairCount = 0;
    
    for (size_t i = 0; i < chord.notes.size(); ++i) {
        for (size_t j = i + 1; j < chord.notes.size(); ++j) {
            int interval = chord.notes[j] - chord.notes[i];
            totalDissonance += intervalDissonance(interval);
            pairCount++;
        }
    }
    
    return pairCount > 0 ? totalDissonance / pairCount : 0.0f;
}

std::string ChordDiagnostics::identifyChord(const std::vector<uint8_t>& notes) const {
    if (notes.size() < 3) {
        return "incomplete";
    }
    
    // Simple chord identification (would be more sophisticated in full implementation)
    std::vector<uint8_t> sortedNotes = notes;
    std::sort(sortedNotes.begin(), sortedNotes.end());
    
    int root = sortedNotes[0] % 12;
    std::vector<int> intervals;
    for (size_t i = 1; i < sortedNotes.size(); ++i) {
        intervals.push_back((sortedNotes[i] - sortedNotes[0]) % 12);
    }
    
    // Check for common triads
    if (intervals.size() >= 2) {
        if (intervals[0] == 4 && intervals[1] == 7) {
            return "major";
        } else if (intervals[0] == 3 && intervals[1] == 7) {
            return "minor";
        } else if (intervals[0] == 3 && intervals[1] == 6) {
            return "diminished";
        } else if (intervals[0] == 4 && intervals[1] == 8) {
            return "augmented";
        }
    }
    
    return "unknown";
}

bool ChordDiagnostics::isConsonant(const Chord& chord, float threshold) const {
    return calculateDissonance(chord) < threshold;
}

} // namespace kelly
