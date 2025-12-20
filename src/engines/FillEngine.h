#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace kelly {

//=============================================================================
// ENUMS
//=============================================================================

enum class FillType {
    TomRoll,
    SnareRoll,
    Flam,
    Buildup,
    Breakdown,
    Stutter,
    Crash,
    Linear,
    Syncopated,
    Triplet,
    SixteenthRush,
    ThirtySecondRoll,
    GhostNotes,
    AccentPattern,
    Polyrhythmic
};

enum class FillLength {
    Quarter,      // 1 beat
    Half,         // 2 beats
    Full,         // 4 beats (1 bar)
    Double        // 8 beats (2 bars)
};

enum class FillIntensity {
    Subtle,
    Moderate,
    Intense,
    Explosive
};

//=============================================================================
// DATA STRUCTURES
//=============================================================================

// DrumHit is now defined in common/Types.h

struct FillConfig {
    std::string emotion = "neutral";
    FillLength length = FillLength::Full;
    int startTick = 0;
    int tempoBpm = 120;
    std::pair<int, int> timeSignature = {4, 4};
    std::optional<FillType> typeOverride;
    std::optional<FillIntensity> intensityOverride;
    bool endWithCrash = true;
    int seed = -1;
};

struct FillOutput {
    std::vector<DrumHit> hits;
    FillType typeUsed;
    FillIntensity intensityUsed;
    FillLength length;
    int totalTicks;
    std::string description;
};

struct FillEmotionProfile {
    std::vector<FillType> types;
    FillIntensity intensity;
    std::pair<int, int> velocityRange;
    float ghostProbability;
    bool preferBusyFills;
    bool preferSpareFills;
};

//=============================================================================
// FILL ENGINE
//=============================================================================

class FillEngine {
public:
    FillEngine();
    
    FillOutput generate(
        const std::string& emotion = "neutral",
        FillLength length = FillLength::Full,
        int startTick = 0,
        int tempoBpm = 120
    );
    
    FillOutput generate(const FillConfig& config);
    
    // Specific fill types
    std::vector<DrumHit> generateTomRoll(int startTick, int durationTicks, FillIntensity intensity, unsigned int seed = 0);
    std::vector<DrumHit> generateSnareRoll(int startTick, int durationTicks, FillIntensity intensity, unsigned int seed = 0);
    std::vector<DrumHit> generateLinear(int startTick, int durationTicks, FillIntensity intensity, unsigned int seed = 0);
    std::vector<DrumHit> generateBuildup(int startTick, int durationTicks, FillIntensity intensity, unsigned int seed = 0);
    std::vector<DrumHit> generateTriplet(int startTick, int durationTicks, FillIntensity intensity, unsigned int seed = 0);
    std::vector<DrumHit> generateSixteenthRush(int startTick, int durationTicks, FillIntensity intensity, unsigned int seed = 0);
    std::vector<DrumHit> generateFlam(int startTick, int durationTicks, FillIntensity intensity, unsigned int seed = 0);

private:
    std::map<std::string, FillEmotionProfile> profiles_;
    
    void initializeProfiles();
    
    int lengthToTicks(FillLength length);
    int intensityToVelocityBoost(FillIntensity intensity);
};

} // namespace kelly
