#include "ArrangementEngine.h"
#include <random>
#include <algorithm>

namespace kelly {

ArrangementEngine::ArrangementEngine() {
    initializeTemplates();
}

void ArrangementEngine::initializeTemplates() {
    templates_["pop"] = {
        "Pop Standard",
        {SectionType::Intro, SectionType::Verse, SectionType::PreChorus, SectionType::Chorus,
         SectionType::Verse, SectionType::PreChorus, SectionType::Chorus, SectionType::Bridge,
         SectionType::Chorus, SectionType::Outro},
        {4, 8, 4, 8, 8, 4, 8, 8, 8, 4},
        {0.3f, 0.5f, 0.7f, 1.0f, 0.5f, 0.7f, 1.0f, 0.6f, 1.0f, 0.4f}
    };
    
    templates_["ballad"] = {
        "Emotional Ballad",
        {SectionType::Intro, SectionType::Verse, SectionType::Chorus, SectionType::Verse,
         SectionType::Chorus, SectionType::Bridge, SectionType::Chorus, SectionType::Outro},
        {4, 8, 8, 8, 8, 8, 8, 4},
        {0.2f, 0.4f, 0.7f, 0.5f, 0.8f, 0.6f, 1.0f, 0.3f}
    };
    
    templates_["edm"] = {
        "EDM Build-Drop",
        {SectionType::Intro, SectionType::Build, SectionType::Drop, SectionType::Breakdown,
         SectionType::Build, SectionType::Drop, SectionType::Outro},
        {8, 8, 16, 8, 8, 16, 8},
        {0.4f, 0.8f, 1.0f, 0.3f, 0.9f, 1.0f, 0.5f}
    };
    
    templates_["ambient"] = {
        "Ambient Flow",
        {SectionType::Intro, SectionType::Verse, SectionType::Interlude, SectionType::Verse,
         SectionType::Bridge, SectionType::Outro},
        {8, 16, 8, 16, 8, 8},
        {0.2f, 0.4f, 0.3f, 0.5f, 0.4f, 0.2f}
    };
    
    emotionToInstruments_["grief"] = {"piano", "strings", "pad", "voice"};
    emotionToInstruments_["sadness"] = {"piano", "cello", "pad"};
    emotionToInstruments_["hope"] = {"piano", "strings", "bells", "pad"};
    emotionToInstruments_["anger"] = {"drums", "bass", "distortion_guitar", "brass"};
    emotionToInstruments_["fear"] = {"strings", "bass", "synth", "percussion"};
    emotionToInstruments_["joy"] = {"piano", "strings", "brass", "drums"};
}

ArrangementOutput ArrangementEngine::generate(
    const std::string& emotion,
    const std::string& genre,
    int targetBars
) {
    ArrangementConfig config;
    config.emotion = emotion;
    config.genre = genre;
    config.targetBars = targetBars;
    return generate(config);
}

ArrangementOutput ArrangementEngine::generate(const ArrangementConfig& config) {
    ArrangementOutput output;
    
    std::mt19937 rng(config.seed >= 0 ? static_cast<unsigned int>(config.seed) : std::random_device{}());
    
    // Select template
    std::string templateKey = config.genre.empty() ? "ballad" : config.genre;
    auto it = templates_.find(templateKey);
    const auto& templ = it != templates_.end() ? it->second : templates_["ballad"];
    
    // Generate sections
    float scaleFactor = static_cast<float>(config.targetBars) / 
        std::accumulate(templ.sectionBars.begin(), templ.sectionBars.end(), 0);
    
    for (size_t i = 0; i < templ.structure.size(); ++i) {
        int bars = static_cast<int>(templ.sectionBars[i] * scaleFactor);
        bars = std::max(2, bars);
        
        Section section = createSection(templ.structure[i], config.emotion, bars);
        section.energy = templ.energyProfile[i];
        
        // Adjust energy based on emotional arc
        if (config.emotionalArc > 0.5f) {
            section.energy *= (0.7f + 0.3f * (static_cast<float>(i) / templ.structure.size()));
        } else {
            section.energy *= (1.0f - 0.3f * (static_cast<float>(i) / templ.structure.size()));
        }
        
        output.sections.push_back(section);
        output.energyCurve.push_back(section.energy);
    }
    
    // Calculate totals
    output.totalBars = 0;
    for (const auto& s : output.sections) {
        output.totalBars += s.bars;
    }
    
    // Determine narrative arc
    float startEnergy = output.energyCurve.empty() ? 0.5f : output.energyCurve.front();
    float endEnergy = output.energyCurve.empty() ? 0.5f : output.energyCurve.back();
    
    if (endEnergy > startEnergy + 0.2f) output.narrativeArc = "ascending";
    else if (endEnergy < startEnergy - 0.2f) output.narrativeArc = "descending";
    else output.narrativeArc = "cyclical";
    
    return output;
}

Section ArrangementEngine::createSection(SectionType type, const std::string& /* emotion */, int bars) {
    Section section;
    section.type = type;
    section.name = sectionTypeToString(type);
    section.bars = bars;
    section.energy = getDefaultEnergy(type);
    section.intensity = section.energy;
    section.activeInstruments = getInstrumentsForSection(type, section.energy);
    
    return section;
}

std::vector<std::string> ArrangementEngine::getInstrumentsForSection(SectionType type, float /* energy */) {
    std::vector<std::string> instruments;
    
    switch (type) {
        case SectionType::Intro:
            instruments = {"piano", "pad"};
            break;
        case SectionType::Verse:
            instruments = {"piano", "bass", "drums_light"};
            break;
        case SectionType::PreChorus:
            instruments = {"piano", "bass", "drums", "strings"};
            break;
        case SectionType::Chorus:
            instruments = {"piano", "bass", "drums", "strings", "pad"};
            break;
        case SectionType::Bridge:
            instruments = {"piano", "strings"};
            break;
        case SectionType::Breakdown:
            instruments = {"pad", "bass"};
            break;
        case SectionType::Build:
            instruments = {"drums", "bass", "synth", "riser"};
            break;
        case SectionType::Drop:
            instruments = {"drums", "bass", "synth", "lead"};
            break;
        case SectionType::Outro:
            instruments = {"piano", "pad"};
            break;
        case SectionType::Interlude:
            instruments = {"pad", "ambient"};
            break;
    }
    
    return instruments;
}

std::string ArrangementEngine::sectionTypeToString(SectionType type) const {
    switch (type) {
        case SectionType::Intro: return "Intro";
        case SectionType::Verse: return "Verse";
        case SectionType::PreChorus: return "Pre-Chorus";
        case SectionType::Chorus: return "Chorus";
        case SectionType::Bridge: return "Bridge";
        case SectionType::Breakdown: return "Breakdown";
        case SectionType::Build: return "Build";
        case SectionType::Drop: return "Drop";
        case SectionType::Outro: return "Outro";
        case SectionType::Interlude: return "Interlude";
    }
    return "Unknown";
}

float ArrangementEngine::getDefaultEnergy(SectionType type) const {
    switch (type) {
        case SectionType::Intro: return 0.3f;
        case SectionType::Verse: return 0.5f;
        case SectionType::PreChorus: return 0.7f;
        case SectionType::Chorus: return 1.0f;
        case SectionType::Bridge: return 0.6f;
        case SectionType::Breakdown: return 0.3f;
        case SectionType::Build: return 0.8f;
        case SectionType::Drop: return 1.0f;
        case SectionType::Outro: return 0.4f;
        case SectionType::Interlude: return 0.4f;
    }
    return 0.5f;
}

} // namespace kelly
