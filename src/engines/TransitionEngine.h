#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>

namespace kelly {

enum class TransitionType {
    Cut,            // Immediate change
    Crossfade,      // Gradual blend
    Build,          // Energy ramp up
    Breakdown,      // Strip down
    Riser,          // Ascending tension
    Drop,           // Sudden release
    Filter,         // Filter sweep
    Reverse,        // Reverse effect
    Stutter,        // Rhythmic glitch
    Silence         // Dramatic pause
};

struct TransitionNote {
    int pitch;
    int startTick;
    int durationTicks;
    int velocity;
    std::string type;  // "riser", "impact", "sweep", etc.
};

struct TransitionConfig {
    std::string emotion = "neutral";
    TransitionType type = TransitionType::Crossfade;
    int durationBars = 2;
    int tempoBpm = 120;
    float intensity = 0.5f;
    std::string fromSection;
    std::string toSection;
    int seed = -1;
};

struct TransitionOutput {
    std::vector<TransitionNote> notes;
    TransitionType typeUsed;
    int durationTicks;
    std::vector<float> energyCurve;
    int gmInstrument;
};

struct TransitionEmotionProfile {
    std::vector<TransitionType> preferredTypes;
    float energyChange;
    std::pair<int, int> velocityRange;
    bool useRisers;
    bool useImpacts;
    int gmInstrument;
};

class TransitionEngine {
public:
    TransitionEngine();
    
    TransitionOutput generate(
        const std::string& emotion,
        TransitionType type,
        int durationBars = 2,
        int tempoBpm = 120
    );
    
    TransitionOutput generate(const TransitionConfig& config);
    
    TransitionOutput createBuild(const std::string& emotion, int bars, int tempoBpm);
    TransitionOutput createBreakdown(const std::string& emotion, int bars, int tempoBpm);
    TransitionOutput createDrop(const std::string& emotion, int bars, int tempoBpm);

private:
    std::map<std::string, TransitionEmotionProfile> profiles_;
    
    void initializeProfiles();
    std::vector<TransitionNote> generateRiser(int startTick, int durationTicks, int velocity);
    std::vector<TransitionNote> generateImpact(int tick, int velocity);
    std::vector<float> generateEnergyCurve(TransitionType type, int numPoints);
};

} // namespace kelly
