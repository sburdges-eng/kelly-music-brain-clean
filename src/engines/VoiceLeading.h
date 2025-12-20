#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>

namespace kelly {

enum class VoiceLeadingStyle {
    Smooth,         // Minimal movement between voices
    Common,         // Keep common tones
    Contrary,       // Opposite motion
    Parallel,       // Same direction
    Oblique,        // One voice static
    Free            // No constraints
};

struct VoiceMovement {
    int fromPitch;
    int toPitch;
    int interval;
    std::string direction;  // "up", "down", "static"
};

struct VoiceLeadingResult {
    std::vector<int> fromVoicing;
    std::vector<int> toVoicing;
    std::vector<VoiceMovement> movements;
    float smoothnessScore;
    bool hasParallelFifths;
    bool hasParallelOctaves;
};

struct VoiceLeadingConfig {
    VoiceLeadingStyle style = VoiceLeadingStyle::Smooth;
    bool avoidParallelFifths = true;
    bool avoidParallelOctaves = true;
    int maxVoiceMovement = 4;  // semitones
    bool preferStepwiseMotion = true;
};

class VoiceLeadingEngine {
public:
    VoiceLeadingEngine();
    
    std::vector<int> voice(
        const std::vector<int>& chordTones,
        const std::vector<int>& previousVoicing,
        int bassPitch = -1
    );
    
    VoiceLeadingResult analyze(
        const std::vector<int>& fromVoicing,
        const std::vector<int>& toVoicing
    );
    
    std::vector<std::vector<int>> voiceProgression(
        const std::vector<std::vector<int>>& chordTones,
        int startingBass = 36
    );
    
    void setConfig(const VoiceLeadingConfig& config) { config_ = config; }

private:
    VoiceLeadingConfig config_;
    
    int findClosestPitch(int target, const std::vector<int>& candidates);
    bool wouldCreateParallel(int v1From, int v1To, int v2From, int v2To, int interval);
    float calculateSmoothness(const std::vector<VoiceMovement>& movements);
    std::vector<int> invertVoicing(const std::vector<int>& voicing, int inversion);
};

} // namespace kelly
