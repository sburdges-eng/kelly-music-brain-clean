#pragma once

#include <string>
#include <vector>
#include <map>

namespace kelly {

struct GrooveTemplate {
    std::string name;
    int numerator;
    int denominator;
    std::vector<std::pair<float, int>> pattern;  // time, velocity
    float swing = 0.0f;
};

class GrooveTemplates {
public:
    GrooveTemplates();
    ~GrooveTemplates() = default;

    const GrooveTemplate* getTemplate(const std::string& name) const;
    std::vector<std::string> getTemplateNames() const;

private:
    void initializeTemplates();
    std::map<std::string, GrooveTemplate> templates_;
};

} // namespace kelly
