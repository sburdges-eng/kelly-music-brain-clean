#pragma once

#include <string>

namespace kelly {

struct EmotionThesaurusNode {
    int id;
    std::string name;
    // Additional fields would be here
};

class EmotionThesaurus {
public:
    EmotionThesaurus();
    ~EmotionThesaurus() = default;

    const EmotionThesaurusNode* getNode(int id) const;
    // Additional methods for the 216-node thesaurus

private:
    void initializeThesaurus();
    // Storage for 216 nodes
};

} // namespace kelly
