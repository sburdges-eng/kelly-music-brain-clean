#pragma once

#include "../common/Types.h"
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace kelly {

enum class StringArticulation {
    Sustain,
    Tremolo,
    Pizzicato,
    Spiccato,
    Legato,
    Marcato,
    Staccato
};

enum class StringSection {
    Violins1,
    Violins2,
    Violas,
    Cellos,
    Basses,
    FullEnsemble
};

enum class StringDynamicCurve {
    Flat,
    Crescendo,
    Decrescendo,
    Swell,
    Accent,
    Sforzando
};

struct StringNote {
    int pitch;
    int startTick;
    int durationTicks;
    int velocity;
    StringArticulation articulation;
    StringSection section;
};

struct StringConfig {
    std::string emotion = "neutral";
    std::vector<std::string> chordProgression;
    std::string key = "C";
    int bars = 4;
    int tempoBpm = 120;
    std::optional<StringArticulation> articulationOverride;
    std::optional<StringSection> sectionOverride;
    bool divisi = false;
    int seed = -1;
};

struct StringOutput {
    std::vector<StringNote> notes;
    std::string emotion;
    StringArticulation articulationUsed;
    StringSection sectionUsed;
    std::vector<int> gmInstruments;
    int totalTicks;
};

struct StringEmotionProfile {
    StringArticulation articulation;
    StringSection section;
    StringDynamicCurve dynamicCurve;
    std::pair<int, int> velocityRange;
    float sustainRatio;
    float vibratoAmount;
    bool useDivisi;
    std::vector<int> gmInstruments;
};

class StringEngine {
public:
    StringEngine();
    
    StringOutput generate(
        const std::string& emotion,
        const std::vector<std::string>& chordProgression,
        const std::string& key = "C",
        int bars = 4,
        int tempoBpm = 120
    );
    
    StringOutput generate(const StringConfig& config);

private:
    std::map<std::string, StringEmotionProfile> profiles_;
    
    void initializeProfiles();
    std::vector<int> assignToSections(const std::vector<int>& pitches, StringSection section);
    int getGmInstrument(StringSection section) const;
};

} // namespace kelly
