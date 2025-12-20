#include "intent_processor.h"

namespace kelly {

int IntentProcessor::processWound(const Wound& wound) {
    // Simplified wound processing - returns emotion ID
    // Full implementation would use ML/pattern matching
    
    if (wound.description.find("loss") != std::string::npos ||
        wound.description.find("grief") != std::string::npos) {
        return 2;  // grief
    } else if (wound.description.find("anger") != std::string::npos ||
               wound.description.find("rage") != std::string::npos) {
        return 4;  // rage
    } else if (wound.description.find("fear") != std::string::npos ||
               wound.description.find("anxiety") != std::string::npos) {
        return 7;  // anxiety
    }
    
    return 3;  // default to melancholy
}

} // namespace kelly
