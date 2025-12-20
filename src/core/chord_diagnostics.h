#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace kelly {

struct Chord {
    std::vector<uint8_t> notes;
    std::string name;
    float dissonanceLevel = 0.0f;
};

class ChordDiagnostics {
public:
    ChordDiagnostics() = default;
    ~ChordDiagnostics() = default;

    float calculateDissonance(const Chord& chord) const;
    std::string identifyChord(const std::vector<uint8_t>& notes) const;
    bool isConsonant(const Chord& chord, float threshold = 0.3f) const;

private:
    float intervalDissonance(int interval) const;
};

} // namespace kelly
