#include "groove_templates.h"

namespace kelly {

GrooveTemplates::GrooveTemplates() {
    initializeTemplates();
}

void GrooveTemplates::initializeTemplates() {
    templates_["straight"] = GrooveTemplate{
        "Straight", 4, 4,
        {{0.0f, 100}, {0.25f, 80}, {0.5f, 100}, {0.75f, 80}},
        0.0f
    };
    
    templates_["swing"] = GrooveTemplate{
        "Swing", 4, 4,
        {{0.0f, 100}, {0.33f, 80}, {0.5f, 100}, {0.83f, 80}},
        0.66f
    };
    
    templates_["syncopated"] = GrooveTemplate{
        "Syncopated", 4, 4,
        {{0.0f, 100}, {0.125f, 60}, {0.375f, 90}, {0.625f, 85}, {0.875f, 70}},
        0.0f
    };
}

const GrooveTemplate* GrooveTemplates::getTemplate(const std::string& name) const {
    auto it = templates_.find(name);
    return (it != templates_.end()) ? &it->second : nullptr;
}

std::vector<std::string> GrooveTemplates::getTemplateNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : templates_) {
        names.push_back(name);
    }
    return names;
}

} // namespace kelly
