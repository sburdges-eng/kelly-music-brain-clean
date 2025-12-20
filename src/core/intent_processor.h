#pragma once

#include <string>

namespace kelly {

enum class IntentPhase {
    Wound,
    Emotion,
    RuleBreak
};

struct Wound {
    std::string description;
    float intensity;
    std::string source;
};

class IntentProcessor {
public:
    IntentProcessor() = default;
    ~IntentProcessor() = default;

    int processWound(const Wound& wound);
    // Additional methods would be implemented

private:
    // Implementation details
};

} // namespace kelly
