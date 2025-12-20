#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <random>

namespace kelly {

// GrooveStyle is now GrooveType, defined in common/Types.h
using GrooveStyle = GrooveType;  // Type alias for compatibility

enum class GrooveElement {
    Kick,
    Snare,
    HiHat,
    Ride,
    Crash,
    Tom,
    Percussion
};

struct GrooveHit {
    GrooveElement element;
    int tick;
    int velocity;
    int pitch;  // For drum map
    int durationTicks;
};

struct GrooveConfig {
    std::string emotion = "neutral";
    std::string genre;
    int bars = 4;
    int tempoBpm = 120;
    int beatsPerBar = 4;
    std::optional<GrooveStyle> styleOverride;
    float humanization = 0.1f;
    int seed = -1;
};

struct GrooveOutput {
    std::vector<GrooveHit> hits;
    std::string emotion;
    GrooveStyle styleUsed;
    float swingAmount;
    int totalTicks;
    std::map<GrooveElement, std::vector<int>> patterns;
};

struct GrooveEmotionProfile {
    GrooveStyle preferredStyle;
    float kickDensity;
    float snareDensity;
    float hatDensity;
    float swingAmount;
    std::pair<int, int> velocityRange;
    float ghostNoteProbability;
    float humanization;
};

// Renamed to avoid conflict with src/midi/GrooveEngine
class GroovePatternEngine {
public:
    GroovePatternEngine();
    
    GrooveOutput generate(
        const std::string& emotion,
        const std::string& genre = "",
        int bars = 4,
        int tempoBpm = 120
    );
    
    GrooveOutput generate(const GrooveConfig& config);
    
    void setHumanization(float amount) { humanization_ = amount; }
    void setSwing(float amount) { swingOverride_ = amount; }

private:
    std::map<std::string, GrooveEmotionProfile> profiles_;
    std::map<std::string, std::vector<bool>> genreKickPatterns_;
    std::map<std::string, std::vector<bool>> genreSnarePatterns_;
    float humanization_ = 0.1f;
    float swingOverride_ = -1.0f;
    
    void initializeProfiles();
    void initializeGenrePatterns();
    int applySwing(int tick, float swingAmount, int stepSize);
    int applyHumanization(int tick, float amount, std::mt19937& rng);
    int getElementPitch(GrooveElement element);
};

} // namespace kelly
