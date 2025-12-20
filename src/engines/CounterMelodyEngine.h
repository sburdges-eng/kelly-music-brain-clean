#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>

namespace kelly {

enum class CounterMelodyType {
    Parallel,       // Move in same direction
    Contrary,       // Move in opposite direction
    Oblique,        // One voice static
    Imitation,      // Echo/canon
    Independent,    // Free counterpoint
    Pedal,          // Sustained note
    Ostinato        // Repeating pattern
};

enum class CounterMelodyRelation {
    Third,
    Sixth,
    Tenth,
    Fourth,
    Fifth,
    Octave
};

struct CounterMelodyNote {
    int pitch;
    int startTick;
    int durationTicks;
    int velocity;
};

struct CounterMelodyConfig {
    std::string emotion = "neutral";
    std::vector<MidiNote> primaryMelody;
    std::string key = "C";
    std::string mode = "major";
    CounterMelodyType type = CounterMelodyType::Contrary;
    CounterMelodyRelation relation = CounterMelodyRelation::Third;
    float density = 0.5f;
    int seed = -1;
};

struct CounterMelodyOutput {
    std::vector<CounterMelodyNote> notes;
    CounterMelodyType typeUsed;
    CounterMelodyRelation relationUsed;
    int gmInstrument;
    int totalTicks;
};

struct CounterMelodyEmotionProfile {
    CounterMelodyType preferredType;
    CounterMelodyRelation preferredRelation;
    float density;
    std::pair<int, int> velocityRange;
    int gmInstrument;
};

class CounterMelodyEngine {
public:
    CounterMelodyEngine();
    
    CounterMelodyOutput generate(
        const std::string& emotion,
        const std::vector<MidiNote>& primaryMelody,
        const std::string& key = "C",
        const std::string& mode = "major"
    );
    
    CounterMelodyOutput generate(const CounterMelodyConfig& config);

private:
    std::map<std::string, CounterMelodyEmotionProfile> profiles_;
    
    void initializeProfiles();
    int transposeByInterval(int pitch, CounterMelodyRelation relation, bool above);
    std::vector<int> getScalePitches(const std::string& key, const std::string& mode);
    int snapToScale(int pitch, const std::vector<int>& scale);
};

} // namespace kelly
