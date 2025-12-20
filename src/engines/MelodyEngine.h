#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <optional>
#include <random>

namespace kelly {

//=============================================================================
// ENUMS
//=============================================================================

enum class ContourType {
    Ascending,
    Descending,
    Arch,
    InverseArch,
    Static,
    Wave,
    SpiralDown,
    SpiralUp,
    Jagged,
    Collapse
};

enum class RhythmDensity {
    Sparse,      // 2-4 notes/bar
    Moderate,    // 4-8 notes/bar
    Dense,       // 8-16 notes/bar
    Frantic      // 16+ notes/bar
};

enum class MelodyArticulation {
    Legato,
    Staccato,
    Tenuto,
    Accent,
    Marcato,
    Portato
};

//=============================================================================
// DATA STRUCTURES
//=============================================================================

struct MelodyNote {
    int pitch;
    int startTick;
    int durationTicks;
    int velocity;
    MelodyArticulation articulation = MelodyArticulation::Legato;
    int timingOffset = 0;
};

struct MelodyConfig {
    std::string emotion = "neutral";
    std::string key = "C";
    std::string mode = "major";
    int bars = 4;
    int tempoBpm = 120;
    int beatsPerBar = 4;
    std::optional<ContourType> contourOverride;
    std::optional<RhythmDensity> densityOverride;
    std::vector<Chord> chordProgression;  // Optional: if provided, melody will align with chords
    float chordTonePreference = 0.7f;    // 0.0-1.0, how much to prefer chord tones vs scale tones
    int seed = -1;
};

struct MelodyOutput {
    std::vector<MelodyNote> notes;
    std::string emotion;
    ContourType contourUsed;
    RhythmDensity densityUsed;
    int gmInstrument;
    int totalTicks;
};

//=============================================================================
// EMOTION PROFILE
//=============================================================================

struct MelodyEmotionProfile {
    std::vector<ContourType> preferredContours;
    RhythmDensity preferredDensity;
    std::pair<int, int> velocityRange;
    float legatoRatio;
    float restProbability;
    int gmInstrument;
    std::pair<int, int> registerRange;
    float intervalVariance;
};

//=============================================================================
// MELODY ENGINE
//=============================================================================

class MelodyEngine {
public:
    MelodyEngine();
    
    MelodyOutput generate(
        const std::string& emotion,
        const std::string& key = "C",
        const std::string& mode = "major",
        int bars = 4,
        int tempoBpm = 120
    );
    
    MelodyOutput generate(const MelodyConfig& config);
    
    /**
     * Generate melody aligned with chord progression
     * @param emotion Emotion name
     * @param chordProgression Chord progression to align with
     * @param key Key signature
     * @param mode Mode (major/minor/etc)
     * @param tempoBpm Tempo in BPM
     * @param chordTonePreference How much to prefer chord tones (0.0-1.0)
     */
    MelodyOutput generateWithChords(
        const std::string& emotion,
        const std::vector<Chord>& chordProgression,
        const std::string& key = "C",
        const std::string& mode = "major",
        int tempoBpm = 120,
        float chordTonePreference = 0.7f
    );
    
    MelodyOutput generateForSection(
        const std::string& emotion,
        const std::string& sectionType,
        const std::string& key,
        int bars,
        int tempoBpm
    );

private:
    std::map<std::string, MelodyEmotionProfile> profiles_;
    
    void initializeProfiles();
    std::vector<int> getScalePitches(const std::string& key, const std::string& mode, int octave);
    std::vector<int> generateContour(ContourType contour, int numNotes, int startPitch, int range, std::mt19937& rng);
    int snapToScale(int pitch, const std::vector<int>& scale) const;
    
    // Chord-aware melody generation
    int getChordAtTick(int tick, int totalTicks, const std::vector<Chord>& chords) const;
    std::vector<int> getChordTones(const Chord& chord) const;
    int snapToChordOrScale(int pitch, const std::vector<int>& chordTones, const std::vector<int>& scaleTones, float chordPreference, std::mt19937& rng) const;
};

} // namespace kelly
