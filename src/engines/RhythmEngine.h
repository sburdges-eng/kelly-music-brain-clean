#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <optional>

namespace kelly {

//=============================================================================
// GM DRUM MAP (Channel 10)
//=============================================================================

namespace GMDrum {
    constexpr int KICK = 36;
    constexpr int SNARE = 38;
    constexpr int SIDESTICK = 37;
    constexpr int CLAP = 39;
    constexpr int CLOSED_HAT = 42;
    constexpr int OPEN_HAT = 46;
    constexpr int PEDAL_HAT = 44;
    constexpr int CRASH = 49;
    constexpr int RIDE = 51;
    constexpr int RIDE_BELL = 53;
    constexpr int LOW_TOM = 45;
    constexpr int MID_TOM = 47;
    constexpr int HIGH_TOM = 50;
    constexpr int TAMBOURINE = 54;
    constexpr int COWBELL = 56;
    constexpr int SHAKER = 70;
}

//=============================================================================
// ENUMS
//=============================================================================

// GrooveType is now defined in common/Types.h

enum class PatternDensity {
    Minimal,
    Sparse,
    Moderate,
    Busy,
    Chaotic
};

enum class AccentPattern {
    Backbeat,
    Downbeat,
    Offbeat,
    AllBeats,
    None
};

//=============================================================================
// DATA STRUCTURES
//=============================================================================

// DrumHit is now defined in common/Types.h

struct RhythmConfig {
    std::string emotion = "neutral";
    int bars = 4;
    int tempoBpm = 120;
    std::pair<int, int> timeSignature = {4, 4};
    std::string genre;
    std::optional<GrooveType> grooveOverride;
    std::optional<PatternDensity> densityOverride;
    bool includeFills = true;
    std::vector<int> fillOnBar;
    int seed = -1;
};

struct RhythmOutput {
    std::vector<DrumHit> hits;
    RhythmConfig config;
    GrooveType grooveUsed;
    PatternDensity densityUsed;
    int totalTicks;
    std::set<std::string> instrumentsUsed;
};

struct RhythmEmotionProfile {
    std::vector<GrooveType> grooves;
    PatternDensity density;
    AccentPattern accent;
    std::vector<int> kickPattern;   // 16 steps
    std::vector<int> snarePattern;  // 16 steps
    float hatProbability;
    float ghostNoteProbability;
    float fillProbability;
    float swingAmount;
    std::pair<int, int> velocityRange;
    float timingHumanize;
    float dropProbability;
};

//=============================================================================
// RHYTHM ENGINE
//=============================================================================

class RhythmEngine {
public:
    RhythmEngine();
    
    RhythmOutput generate(
        const std::string& emotion = "neutral",
        int bars = 4,
        int tempoBpm = 120,
        const std::string& genre = ""
    );
    
    RhythmOutput generate(const RhythmConfig& config);
    
    RhythmOutput generateIntro(const std::string& emotion, int bars, int tempo, const std::string& genre = "");
    RhythmOutput generateBuildup(const std::string& emotion, int bars, int tempo, const std::string& genre = "");
    
private:
    std::map<std::string, RhythmEmotionProfile> profiles_;
    std::map<std::string, std::map<std::string, float>> genreModifiers_;
    
    void initializeProfiles();
    void initializeGenreModifiers();
    
    int applySwing(int tick, int beatTicks, float swingAmount);
    std::vector<std::tuple<int, std::string, int>> generateHatPattern(int barTicks, const RhythmEmotionProfile& profile, GrooveType groove);
    std::vector<std::tuple<int, std::string, int>> generateTrapHats(int barTicks, const RhythmEmotionProfile& profile);
    std::vector<std::tuple<int, std::string, int>> generateFill(int startTick, int durationTicks);
};

} // namespace kelly
