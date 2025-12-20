#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>

namespace kelly {

enum class TensionTechnique {
    Dominant,           // V chord tension
    Diminished,         // Diminished chord
    Chromatic,          // Chromatic approach
    Suspension,         // Suspended notes
    Appoggiatura,       // Non-chord tone approach
    Pedal,              // Pedal point
    Tritone,            // Tritone substitution
    Cluster,            // Tone cluster
    Polytonality        // Multiple keys
};

enum class TensionCurve {
    Building,           // Increasing tension
    Releasing,          // Decreasing tension
    Plateau,            // Sustained tension
    Spike,              // Sudden peak
    Wave,               // Oscillating
    Ramp                // Linear change
};

struct TensionPoint {
    int tick;
    float tensionLevel;     // 0-1
    TensionTechnique technique;
    std::vector<int> additionalPitches;
};

struct TensionConfig {
    std::string emotion = "neutral";
    std::vector<std::string> chordProgression;
    int bars = 4;
    int tempoBpm = 120;
    TensionCurve curve = TensionCurve::Building;
    float maxTension = 0.8f;
    int seed = -1;
};

struct TensionOutput {
    std::vector<TensionPoint> tensionPoints;
    TensionCurve curveUsed;
    std::vector<float> tensionCurve;
    float peakTension;
    int peakTick;
};

struct TensionEmotionProfile {
    std::vector<TensionTechnique> preferredTechniques;
    TensionCurve preferredCurve;
    float baseTension;
    float tensionVariance;
};

class TensionEngine {
public:
    TensionEngine();
    
    TensionOutput generate(
        const std::string& emotion,
        const std::vector<std::string>& chordProgression,
        int bars = 4,
        int tempoBpm = 120
    );
    
    TensionOutput generate(const TensionConfig& config);
    
    float calculateTension(const std::vector<int>& pitches, const std::string& context);
    std::vector<int> addTensionNotes(
        const std::vector<int>& chordPitches, 
        TensionTechnique technique,
        float amount
    );

private:
    std::map<std::string, TensionEmotionProfile> profiles_;
    
    void initializeProfiles();
    std::vector<float> generateCurve(TensionCurve curve, int numPoints, float maxTension);
};

} // namespace kelly
