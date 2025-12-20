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

enum class PadTexture {
    Warm,
    Bright,
    Dark,
    Glassy,
    Gritty,
    Airy,
    Thick,
    Hollow,
    Shimmering,
    Frozen
};

enum class PadMovement {
    Static,
    Breathing,
    Swelling,
    Fading,
    Pulsing,
    Drifting,
    Tremolo,
    Evolving,
    Unstable
};

enum class PadVoicing {
    Close,
    Open,
    Octaves,
    Fifths,
    Full,
    Sparse,
    Cluster,
    Shell
};

enum class PadRegister {
    Sub,
    Low,
    Mid,
    High,
    Wide
};

//=============================================================================
// DATA STRUCTURES
//=============================================================================

struct PadNote {
    int pitch;
    int startTick;
    int durationTicks;
    int velocity;
    int channel = 0;
};

struct PadConfig {
    std::string emotion = "neutral";
    std::vector<std::string> chordProgression;
    std::string key = "C";
    int bars = 4;
    int tempoBpm = 120;
    int beatsPerBar = 4;
    std::optional<PadTexture> textureOverride;
    std::optional<PadMovement> movementOverride;
    std::optional<PadVoicing> voicingOverride;
    int seed = -1;
};

struct PadOutput {
    std::vector<PadNote> notes;
    std::string emotion;
    PadTexture textureUsed;
    PadMovement movementUsed;
    PadVoicing voicingUsed;
    int gmInstrument;
    int totalTicks;
};

struct PadEmotionProfile {
    PadTexture texture;
    PadMovement movement;
    PadVoicing voicing;
    PadRegister padRegister;
    std::pair<int, int> velocityRange;
    float density;
    float sustainRatio;
    float breathingRate;
    bool addExtensions;
    float extensionProbability;
    int gmInstrument;
};

//=============================================================================
// PAD ENGINE
//=============================================================================

class PadEngine {
public:
    PadEngine();
    
    PadOutput generate(
        const std::string& emotion,
        const std::vector<std::string>& chordProgression,
        const std::string& key = "C",
        int bars = 4,
        int tempoBpm = 120
    );
    
    PadOutput generate(const PadConfig& config);

private:
    std::map<std::string, PadEmotionProfile> profiles_;
    
    void initializeProfiles();
    std::vector<int> parseChord(const std::string& chord, const std::string& key);
    std::vector<int> applyVoicing(const std::vector<int>& pitches, PadVoicing voicing, int basePitch);
    std::vector<int> applyVelocityCurve(int baseVelocity, PadMovement movement, int numNotes);
};

} // namespace kelly
