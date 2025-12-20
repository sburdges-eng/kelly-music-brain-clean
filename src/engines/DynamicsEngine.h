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

enum class DynamicMarking {
    Pianississimo,  // ppp
    Pianissimo,     // pp
    Piano,          // p
    MezzoPiano,     // mp
    MezzoForte,     // mf
    Forte,          // f
    Fortissimo,     // ff
    Fortississimo   // fff
};

enum class DynamicShape {
    Constant,
    Crescendo,
    Decrescendo,
    Sforzando,
    FortePiano,
    Swell,
    Hairpin,
    Terraced,
    Subito
};

enum class AccentType {
    None,
    Normal,
    Strong,
    Tenuto,
    Staccato,
    Marcato
};

//=============================================================================
// DATA STRUCTURES
//=============================================================================

struct DynamicPoint {
    int tick;
    int velocity;
    DynamicMarking marking;
    std::string annotation;
};

struct DynamicCurve {
    std::vector<DynamicPoint> points;
    DynamicShape shape;
};

struct DynamicsConfig {
    std::string emotion = "neutral";
    std::vector<MidiNote> notes;
    int totalTicks = 0;
    std::optional<DynamicMarking> baseMarking;
    std::optional<DynamicShape> shapeOverride;
    float expressiveness = 0.5f;
    bool applyAccents = true;
    std::string section;
    int seed = -1;
};

struct DynamicsOutput {
    std::vector<MidiNote> notes;
    DynamicCurve curve;
    std::string emotion;
    DynamicMarking baseMarking;
    DynamicShape shapeUsed;
    std::pair<int, int> velocityRange;
};

struct DynamicsEmotionProfile {
    DynamicMarking baseMarking;
    std::pair<int, int> velocityRange;
    std::vector<DynamicShape> shapes;
    float accentStrength;
    float variability;
    float swellAmount;
};

//=============================================================================
// DYNAMICS ENGINE
//=============================================================================

class DynamicsEngine {
public:
    DynamicsEngine();
    
    DynamicsOutput apply(
        const std::vector<MidiNote>& notes,
        const std::string& emotion = "neutral",
        float expressiveness = 0.5f
    );
    
    DynamicsOutput apply(const DynamicsConfig& config);
    
    // Generate dynamics curve
    DynamicCurve generateCurve(
        int totalTicks,
        DynamicShape shape,
        DynamicMarking startMarking,
        DynamicMarking endMarking
    );
    
    // Apply curve to notes
    std::vector<MidiNote> applyCurve(
        const std::vector<MidiNote>& notes,
        const DynamicCurve& curve
    );
    
    // Utility
    int markingToVelocity(DynamicMarking marking);
    DynamicMarking velocityToMarking(int velocity);
    
    // Accent application
    std::vector<MidiNote> applyAccents(
        const std::vector<MidiNote>& notes,
        const std::string& emotion,
        float strength
    );

private:
    std::map<std::string, DynamicsEmotionProfile> profiles_;
    std::map<std::string, std::vector<int>> sectionAccentPatterns_;
    
    void initializeProfiles();
    void initializeSectionPatterns();
    
    int interpolateVelocity(const DynamicCurve& curve, int tick);
};

} // namespace kelly
