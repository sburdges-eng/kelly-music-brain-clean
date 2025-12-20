#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <random>

namespace kelly {

//=============================================================================
// ENUMS
//=============================================================================

enum class BassPattern {
    RootOnly,
    RootFifth,
    Walking,
    Pedal,
    Arpeggiated,
    Syncopated,
    Driving,
    Pulsing,
    Breathing,
    Descending,
    Climbing,
    Ghost
};

enum class BassArticulation {
    Sustained,
    Staccato,
    Muted,
    Slide,
    Dead
};

enum class BassRegister {
    Sub,    // Octave 1
    Low,    // Octave 2
    Mid,    // Octave 3
    Flexible
};

//=============================================================================
// DATA STRUCTURES
//=============================================================================

struct BassNote {
    int pitch;
    int startTick;
    int durationTicks;
    int velocity;
    BassArticulation articulation = BassArticulation::Sustained;
    std::string function; // "root", "fifth", "chromatic", etc.
};

struct BassConfig {
    std::string emotion = "neutral";
    std::vector<std::string> chordProgression;
    std::string key = "C";
    int bars = 4;
    int tempoBpm = 120;
    std::pair<int, int> timeSignature = {4, 4};
    std::optional<BassPattern> patternOverride;
    std::optional<BassRegister> registerOverride;
    int seed = -1;
};

struct BassOutput {
    std::vector<BassNote> notes;
    std::string emotion;
    BassPattern patternUsed;
    BassRegister registerUsed;
    int gmInstrument;
    int totalTicks;
};

struct BassEmotionProfile {
    std::vector<BassPattern> patterns;
    BassRegister preferredRegister;
    std::pair<int, int> velocityRange;
    float noteLengthRatio;
    float syncoProbability;
    int gmInstrument;
};

//=============================================================================
// BASS ENGINE
//=============================================================================

class BassEngine {
public:
    BassEngine();
    
    BassOutput generate(
        const std::string& emotion,
        const std::vector<std::string>& chordProgression,
        const std::string& key = "C",
        int bars = 4,
        int tempoBpm = 120
    );
    
    BassOutput generate(const BassConfig& config);
    
    BassOutput generateForSection(
        const std::string& emotion,
        const std::vector<std::string>& chordProgression,
        const std::string& sectionType,
        const std::string& key,
        int bars,
        int tempoBpm
    );

private:
    std::map<std::string, BassEmotionProfile> profiles_;
    
    void initializeProfiles();
    
    struct ChordTones {
        int root;
        int third;
        int fifth;
        int seventh;
    };
    
    ChordTones parseChord(const std::string& chord, int octave);
    int getRegisterOctave(BassRegister reg);
    
    std::vector<BassNote> generateRootOnly(const ChordTones& tones, int barTicks, const BassEmotionProfile& profile, std::mt19937& rng);
    std::vector<BassNote> generateRootFifth(const ChordTones& tones, int barTicks, const BassEmotionProfile& profile, std::mt19937& rng);
    std::vector<BassNote> generateWalking(const ChordTones& tones, int barTicks, const BassEmotionProfile& profile, std::mt19937& rng);
    std::vector<BassNote> generatePedal(const ChordTones& tones, int barTicks, const BassEmotionProfile& profile, std::mt19937& rng);
    std::vector<BassNote> generateArpeggiated(const ChordTones& tones, int barTicks, const BassEmotionProfile& profile, std::mt19937& rng);
    std::vector<BassNote> generateSyncopated(const ChordTones& tones, int barTicks, const BassEmotionProfile& profile, std::mt19937& rng);
    std::vector<BassNote> generateDriving(const ChordTones& tones, int barTicks, const BassEmotionProfile& profile, std::mt19937& rng);
    std::vector<BassNote> generatePulsing(const ChordTones& tones, int barTicks, const BassEmotionProfile& profile, std::mt19937& rng);
    std::vector<BassNote> generateBreathing(const ChordTones& tones, int barTicks, const BassEmotionProfile& profile, std::mt19937& rng);
};

} // namespace kelly
