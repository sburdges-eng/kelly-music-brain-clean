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

enum class VariationType {
    Ornamentation,
    Simplification,
    Rhythmic,
    Melodic,
    Harmonic,
    Transposition,
    Inversion,
    Retrograde,
    Augmentation,
    Diminution,
    Fragmentation,
    Extension,
    Displacement,
    Embellishment,
    Reduction
};

enum class OrnamentType {
    Trill,
    Mordent,
    Turn,
    Appoggiatura,
    Acciaccatura,
    Slide,
    Bend,
    Vibrato,
    GraceNote
};

//=============================================================================
// DATA STRUCTURES
//=============================================================================

struct VariationConfig {
    std::string emotion = "neutral";
    std::vector<MidiNote> source;
    std::string key = "C";
    std::string mode = "major";
    std::optional<VariationType> typeOverride;
    float intensity = 0.5f;
    bool preserveContour = true;
    bool preserveRhythm = false;
    int seed = -1;
};

struct VariationOutput {
    std::vector<MidiNote> notes;
    VariationType typeUsed;
    std::string emotion;
    float similarityScore;
    std::string description;
};

struct VariationEmotionProfile {
    std::vector<VariationType> preferredTypes;
    float ornamentProbability;
    float simplifyProbability;
    float intensityMultiplier;
    std::vector<OrnamentType> ornaments;
};

//=============================================================================
// VARIATION ENGINE
//=============================================================================

class VariationEngine {
public:
    VariationEngine();
    
    VariationOutput generate(
        const std::vector<MidiNote>& source,
        const std::string& emotion = "neutral",
        float intensity = 0.5f
    );
    
    VariationOutput generate(const VariationConfig& config);
    
    // Specific variation types
    std::vector<MidiNote> ornament(const std::vector<MidiNote>& source, OrnamentType type, std::mt19937& rng);
    std::vector<MidiNote> simplify(const std::vector<MidiNote>& source, float amount, std::mt19937& rng);
    std::vector<MidiNote> rhythmicVariation(const std::vector<MidiNote>& source, float amount, std::mt19937& rng);
    std::vector<MidiNote> transpose(const std::vector<MidiNote>& source, int semitones);
    std::vector<MidiNote> invert(const std::vector<MidiNote>& source, int axis);
    std::vector<MidiNote> retrograde(const std::vector<MidiNote>& source);
    std::vector<MidiNote> augment(const std::vector<MidiNote>& source, float factor);
    std::vector<MidiNote> diminish(const std::vector<MidiNote>& source, float factor);
    std::vector<MidiNote> fragment(const std::vector<MidiNote>& source, int fragments, std::mt19937& rng);
    std::vector<MidiNote> extend(const std::vector<MidiNote>& source, int bars);
    std::vector<MidiNote> displace(const std::vector<MidiNote>& source, int ticks);

private:
    std::map<std::string, VariationEmotionProfile> profiles_;
    
    void initializeProfiles();
    
    std::vector<int> getScalePitches(const std::string& key, const std::string& mode);
    int snapToScale(int pitch, const std::vector<int>& scale);
    float calculateSimilarity(const std::vector<MidiNote>& a, const std::vector<MidiNote>& b);
};

} // namespace kelly
