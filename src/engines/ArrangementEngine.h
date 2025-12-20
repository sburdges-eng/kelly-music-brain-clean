#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>

namespace kelly {

enum class SectionType {
    Intro,
    Verse,
    PreChorus,
    Chorus,
    Bridge,
    Breakdown,
    Build,
    Drop,
    Outro,
    Interlude
};

struct Section {
    SectionType type;
    std::string name;
    int bars;
    float energy;          // 0-1
    float intensity;       // 0-1
    std::vector<std::string> activeInstruments;
    std::vector<std::string> chordProgression;
};

struct ArrangementConfig {
    std::string emotion = "neutral";
    std::string genre;
    int targetBars = 32;
    float emotionalArc = 0.5f;  // 0 = descending, 1 = ascending
    bool includeIntro = true;
    bool includeOutro = true;
    int seed = -1;
};

struct ArrangementOutput {
    std::vector<Section> sections;
    std::vector<float> energyCurve;
    int totalBars;
    std::string narrativeArc;
};

struct ArrangementTemplate {
    std::string name;
    std::vector<SectionType> structure;
    std::vector<int> sectionBars;
    std::vector<float> energyProfile;
};

class ArrangementEngine {
public:
    ArrangementEngine();
    
    ArrangementOutput generate(
        const std::string& emotion,
        const std::string& genre = "",
        int targetBars = 32
    );
    
    ArrangementOutput generate(const ArrangementConfig& config);
    
    Section createSection(SectionType type, const std::string& emotion, int bars);
    std::vector<std::string> getInstrumentsForSection(SectionType type, float energy);

private:
    std::map<std::string, ArrangementTemplate> templates_;
    std::map<std::string, std::vector<std::string>> emotionToInstruments_;
    
    void initializeTemplates();
    std::string sectionTypeToString(SectionType type) const;
    float getDefaultEnergy(SectionType type) const;
};

} // namespace kelly
