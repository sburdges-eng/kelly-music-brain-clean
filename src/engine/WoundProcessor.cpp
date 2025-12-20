#include "engine/WoundProcessor.h"
#include <cmath>

namespace kelly {

std::vector<WoundProcessor::EmotionClue> WoundProcessor::buildEmotionClues() {
    return {
        // Grief/Loss cluster - phrases first, then keywords
        {{"loss", "lost", "death", "died", "gone", "grief", "mourn", "bereavement"}, 
         {"loss of", "passed away", "no longer", "can't believe they're"}, "Grief", 0.9f, 0.15f},
        {{"melancholy", "melancholic", "blue", "gloomy", "somber"}, {}, "Melancholy", 0.85f, 0.1f},
        {{"despair", "hopeless", "desperate", "helpless"}, {"give up", "no hope", "can't go on"}, "Despair", 0.9f, 0.2f},
        {{"sorrow", "sorrowful", "mournful"}, {}, "Sorrow", 0.8f, 0.1f},
        {{"wistful", "wistfulness"}, {}, "Wistfulness", 0.7f, 0.05f},
        {{"longing", "yearn", "yearning", "miss", "missing"}, {"longing for", "yearning for", "miss you"}, "Longing", 0.8f, 0.1f},
        {{"heartache", "heartbreak", "heartbroken"}, {"broken heart", "heart breaking"}, "Heartache", 0.85f, 0.15f},
        {{"empty", "emptiness", "void", "hollow", "numb"}, {"feeling empty", "empty inside"}, "Emptiness", 0.75f, 0.1f},
        {{"numb", "numbness", "detached"}, {"feeling numb"}, "Numbness", 0.7f, 0.05f},
        {{"resigned", "resignation"}, {"given up", "accept defeat"}, "Resignation", 0.65f, -0.1f},
        
        // Anger cluster
        {{"rage", "raging", "enraged", "furious"}, {"blinding rage", "see red", "boiling with"}, "Rage", 0.95f, 0.2f},
        {{"fury", "furious", "incensed"}, {}, "Fury", 0.9f, 0.15f},
        {{"frustrated", "frustration", "frustrating"}, {"so frustrated", "frustrated with"}, "Frustration", 0.8f, 0.1f},
        {{"irritated", "irritation", "irritating"}, {}, "Irritation", 0.7f, 0.05f},
        {{"annoyed", "annoyance", "annoying"}, {}, "Annoyance", 0.65f, 0.0f},
        {{"resentment", "resentful", "resent"}, {"deep resentment"}, "Resentment", 0.8f, 0.1f},
        {{"bitter", "bitterness"}, {"bitter about", "bitter towards"}, "Bitterness", 0.75f, 0.1f},
        {{"hostile", "hostility", "hostile"}, {}, "Hostility", 0.85f, 0.15f},
        {{"betrayed", "betrayal"}, {"betrayed by", "feel betrayed"}, "Rage", 0.9f, 0.2f},
        
        // Fear/Anxiety cluster
        {{"terror", "terrified", "terrifying"}, {"absolutely terrified", "terrified of"}, "Terror", 0.95f, 0.2f},
        {{"panic", "panicked", "panicking"}, {"panic attack", "in panic", "panicking about"}, "Panic", 0.9f, 0.2f},
        {{"anxious", "anxiety"}, {"anxious about", "full of anxiety", "overwhelming anxiety"}, "Anxiety", 0.85f, 0.15f},
        {{"worried", "worry", "worrying"}, {"worried about", "can't stop worrying"}, "Worry", 0.75f, 0.1f},
        {{"uneasy", "unease", "uncomfortable"}, {}, "Unease", 0.7f, 0.05f},
        {{"dread", "dreading"}, {"dread the", "dreading what"}, "Dread", 0.85f, 0.15f},
        {{"apprehensive", "apprehension"}, {}, "Apprehension", 0.75f, 0.1f},
        {{"vulnerable", "vulnerability"}, {"feeling vulnerable"}, "Vulnerability", 0.7f, 0.05f},
        {{"scared", "afraid", "fear"}, {"scared of", "afraid of", "fear of", "so scared"}, "Terror", 0.85f, 0.15f},
        
        // Joy cluster
        {{"ecstatic", "ecstasy"}, {"absolutely ecstatic"}, "Ecstasy", 0.95f, 0.2f},
        {{"elated", "elation"}, {}, "Elation", 0.9f, 0.15f},
        {{"happy", "happiness"}, {"so happy", "really happy", "happiest"}, "Happiness", 0.85f, 0.1f},
        {{"content", "contentment"}, {}, "Contentment", 0.75f, 0.05f},
        {{"serene", "serenity"}, {}, "Serenity", 0.7f, 0.0f},
        {{"peaceful", "peace"}, {"at peace", "inner peace"}, "Peace", 0.7f, 0.0f},
        {{"calm", "calming"}, {"feeling calm"}, "Calm", 0.65f, -0.05f},
        {{"relieved", "relief"}, {"so relieved", "relief that"}, "Relief", 0.75f, 0.1f},
        {{"grateful", "gratitude", "thankful"}, {"so grateful", "full of gratitude"}, "Gratitude", 0.8f, 0.1f},
        {{"hopeful", "hope", "hoping"}, {"full of hope", "hoping for"}, "Hope", 0.8f, 0.1f},
        
        // Trust/Love cluster
        {{"love", "loving"}, {"in love", "love you", "love this", "deeply love"}, "Love", 0.9f, 0.15f},
        {{"adore", "adoration", "adoring"}, {}, "Adoration", 0.85f, 0.1f},
        {{"affection", "affectionate"}, {}, "Affection", 0.8f, 0.1f},
        {{"tender", "tenderness"}, {}, "Tenderness", 0.75f, 0.05f},
        {{"warm", "warmth"}, {"warm feeling"}, "Warmth", 0.7f, 0.05f},
        {{"accepted", "acceptance"}, {"feel accepted"}, "Acceptance", 0.7f, 0.05f},
        {{"belonging", "belong"}, {"sense of belonging", "belong here"}, "Belonging", 0.75f, 0.1f},
        
        // Anticipation cluster
        {{"excited", "excitement"}, {"so excited", "really excited", "excited about"}, "Excitement", 0.85f, 0.15f},
        {{"eager", "eagerness"}, {"eager to", "eager for"}, "Eagerness", 0.8f, 0.1f},
        {{"curious", "curiosity"}, {}, "Curiosity", 0.75f, 0.05f},
        {{"interested", "interest"}, {}, "Interest", 0.7f, 0.05f},
        {{"vigilant", "vigilance"}, {}, "Vigilance", 0.7f, 0.05f},
        
        // Surprise cluster
        {{"amazed", "amazement"}, {"absolutely amazed"}, "Amazement", 0.85f, 0.15f},
        {{"astonished", "astonishment"}, {}, "Astonishment", 0.85f, 0.15f},
        {{"shocked", "shock"}, {"completely shocked", "shocked by"}, "Shock", 0.8f, 0.15f},
        {{"startled", "startle"}, {}, "Startle", 0.75f, 0.1f},
        {{"wonder", "wondrous"}, {}, "Wonder", 0.8f, 0.1f},
        
        // Disgust cluster
        {{"revulsion", "revolted"}, {}, "Revulsion", 0.85f, 0.15f},
        {{"loathing", "loathe"}, {}, "Loathing", 0.85f, 0.15f},
        {{"contempt", "contemptuous"}, {}, "Contempt", 0.8f, 0.1f},
        {{"distaste"}, {}, "Distaste", 0.7f, 0.05f},
        {{"disapproval", "disapprove"}, {}, "Disapproval", 0.7f, 0.05f},
        
        // Complex emotions - phrases prioritized
        {{"bittersweet"}, {"bittersweet feeling"}, "Bittersweetness", 0.85f, 0.1f},
        {{"nostalgic", "nostalgia"}, {"feeling nostalgic", "nostalgia for"}, "Nostalgia", 0.85f, 0.1f},
        {{"catharsis", "cathartic", "release"}, {"cathartic release", "feeling cathartic"}, "Catharsis", 0.8f, 0.15f},
        {{"yearning"}, {"yearning for"}, "Yearning", 0.8f, 0.1f},
        {{"ambivalent", "ambivalence", "torn", "conflicted"}, {"torn between", "conflicted about"}, "Ambivalence", 0.75f, 0.05f},
        {{"melancholic hope"}, {}, "Melancholic Hope", 0.8f, 0.1f},
        {{"tender grief"}, {}, "Tender Grief", 0.85f, 0.1f},
        {{"defiant", "defiance"}, {"defiant joy"}, "Defiant Joy", 0.8f, 0.15f},
        {{"quiet rage", "simmering"}, {"simmering rage", "quiet rage"}, "Quiet Rage", 0.8f, 0.1f},
        {{"anxious hope"}, {}, "Anxious Hope", 0.8f, 0.1f},
        
        // Rejection/Loneliness
        {{"rejected", "rejection"}, {"feel rejected", "rejected by"}, "Sorrow", 0.85f, 0.15f},
        {{"lonely", "loneliness", "alone", "isolated"}, {"feeling lonely", "so alone", "completely alone"}, "Emptiness", 0.8f, 0.15f},
        {{"abandoned", "abandonment"}, {"feel abandoned", "abandoned by"}, "Despair", 0.9f, 0.2f},
        
        // Shame/Guilt
        {{"shame", "ashamed", "shameful"}, {"feel ashamed", "ashamed of", "full of shame"}, "Sorrow", 0.85f, 0.15f},
        {{"guilt", "guilty"}, {"feel guilty", "guilty about"}, "Sorrow", 0.8f, 0.1f},
        {{"embarrassed", "embarrassment"}, {}, "Sorrow", 0.75f, 0.1f},
        {{"humiliated", "humiliation"}, {}, "Sorrow", 0.9f, 0.2f},
        
        // Failure/Regret
        {{"failed", "failure"}, {"feel like a failure", "failed at"}, "Despair", 0.85f, 0.15f},
        {{"regret", "regretful"}, {"regret that", "full of regret"}, "Sorrow", 0.8f, 0.1f},
        {{"disappointed", "disappointment"}, {"so disappointed", "disappointed in"}, "Sorrow", 0.75f, 0.1f},
        
        // Generic fallbacks (lower confidence, checked last)
        {{"sad", "sadness", "unhappy", "depressed", "down"}, {"feeling sad", "so sad"}, "Sorrow", 0.7f, 0.1f},
        {{"angry", "anger", "mad", "pissed"}, {"so angry", "really angry"}, "Rage", 0.8f, 0.15f},
        {{"afraid", "fear"}, {"afraid of"}, "Terror", 0.8f, 0.15f},
        {{"joy", "joyful", "glad"}, {"full of joy"}, "Happiness", 0.8f, 0.1f}
    };
}

std::string WoundProcessor::normalizeText(const std::string& text) const {
    std::string normalized = text;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
    // Remove common punctuation that might interfere with matching
    std::string result;
    for (char c : normalized) {
        if (std::isalnum(c) || c == ' ' || c == '\'') {
            result += c;
        } else {
            result += ' ';  // Replace punctuation with space
        }
    }
    
    // Collapse multiple spaces
    std::string final;
    bool lastWasSpace = false;
    for (char c : result) {
        if (c == ' ') {
            if (!lastWasSpace) {
                final += ' ';
                lastWasSpace = true;
            }
        } else {
            final += c;
            lastWasSpace = false;
        }
    }
    
    return final;
}

bool WoundProcessor::matchesKeyword(const std::string& text, const std::string& keyword) const {
    // Check for exact phrase match first
    size_t pos = text.find(keyword);
    if (pos == std::string::npos) {
        return false;
    }
    
    // Check word boundaries for single words
    if (keyword.find(' ') == std::string::npos) {
        // Single word - check boundaries
        if (pos > 0 && std::isalnum(text[pos - 1])) {
            return false;  // Not at word boundary
        }
        if (pos + keyword.length() < text.length() && std::isalnum(text[pos + keyword.length()])) {
            return false;  // Not at word boundary
        }
    }
    
    return true;
}

std::vector<WoundProcessor::MatchResult> WoundProcessor::findMatches(const std::string& text) {
    std::vector<MatchResult> matches;
    auto clues = buildEmotionClues();
    std::string normalized = normalizeText(text);
    
    for (const auto& clue : clues) {
        float bestConfidence = 0.0f;
        bool found = false;
        bool isPhrase = false;
        
        // Check phrases first (higher priority)
        for (const auto& phrase : clue.phrases) {
            if (matchesKeyword(normalized, phrase)) {
                bestConfidence = clue.confidence + 0.1f;  // Boost for phrase match
                found = true;
                isPhrase = true;
                break;
            }
        }
        
        // Then check keywords
        if (!found) {
            for (const auto& keyword : clue.keywords) {
                if (matchesKeyword(normalized, keyword)) {
                    bestConfidence = clue.confidence;
                    found = true;
                    break;
                }
            }
        }
        
        if (found) {
            matches.push_back({clue.emotionName, bestConfidence, clue.intensityBoost, isPhrase});
        }
    }
    
    // Sort by confidence (highest first)
    std::sort(matches.begin(), matches.end(), 
              [](const MatchResult& a, const MatchResult& b) {
                  return a.confidence > b.confidence;
              });
    
    return matches;
}

float WoundProcessor::extractIntensifierModifier(const std::string& text) const {
    std::string lower = normalizeText(text);
    float modifier = 0.0f;
    
    // Strong intensifiers (increase intensity)
    std::vector<std::pair<std::string, float>> intensifiers = {
        {"extremely", 0.3f}, {"incredibly", 0.3f}, {"absolutely", 0.3f},
        {"completely", 0.25f}, {"totally", 0.25f}, {"utterly", 0.25f},
        {"overwhelming", 0.3f}, {"overwhelmingly", 0.3f},
        {"intensely", 0.25f}, {"deeply", 0.2f}, {"profoundly", 0.25f},
        {"so", 0.15f}, {"really", 0.15f}, {"very", 0.1f},
        {"terribly", 0.2f}, {"awfully", 0.2f}, {"horribly", 0.2f},
        {"devastating", 0.3f}, {"devastated", 0.3f},
        {"unbearable", 0.3f}, {"unbearably", 0.3f},
        {"excruciating", 0.3f}, {"agonizing", 0.3f},
        {"blinding", 0.25f}, {"searing", 0.25f},
        {"all-consuming", 0.3f}, {"all consuming", 0.3f},
    };
    
    // Weak intensifiers (decrease intensity)
    std::vector<std::pair<std::string, float>> weakeners = {
        {"slightly", -0.15f}, {"a bit", -0.15f}, {"a little", -0.15f},
        {"somewhat", -0.1f}, {"kind of", -0.1f}, {"sort of", -0.1f},
        {"mildly", -0.15f}, {"moderately", -0.1f},
        {"barely", -0.2f}, {"hardly", -0.2f},
    };
    
    // Check for strong intensifiers
    for (const auto& [word, boost] : intensifiers) {
        if (lower.find(word) != std::string::npos) {
            modifier = std::max(modifier, boost);
        }
    }
    
    // Check for weakeners (they override if found)
    for (const auto& [word, reduction] : weakeners) {
        if (lower.find(word) != std::string::npos) {
            modifier = std::min(modifier, reduction);
        }
    }
    
    // Check for negation patterns that might reduce intensity
    if (lower.find("not ") != std::string::npos || 
        lower.find("don't ") != std::string::npos ||
        lower.find("doesn't ") != std::string::npos) {
        modifier -= 0.1f;
    }
    
    return modifier;
}

float WoundProcessor::calculateIntensity(const std::string& description, float baseIntensity) {
    // Start with base intensity
    float intensity = baseIntensity;
    
    // Extract intensifier modifier
    float intensifierMod = extractIntensifierModifier(description);
    intensity += intensifierMod;
    
    // Check for multiple emotion indicators (suggests higher intensity)
    auto matches = findMatches(description);
    if (matches.size() > 1) {
        intensity += 0.1f;  // Multiple emotions detected = more complex/intense
    }
    
    // Check for exclamation marks or caps (emotional emphasis)
    int exclamationCount = 0;
    int capsCount = 0;
    for (char c : description) {
        if (c == '!') exclamationCount++;
        if (std::isupper(c)) capsCount++;
    }
    if (exclamationCount > 0) {
        intensity += std::min(0.15f, exclamationCount * 0.05f);
    }
    if (capsCount > description.length() * 0.3f) {  // More than 30% caps
        intensity += 0.1f;
    }
    
    // Apply intensity boost from matched emotion
    if (!matches.empty()) {
        intensity += matches[0].intensityModifier;
    }
    
    // Clamp to valid range
    intensity = std::max(0.0f, std::min(1.0f, intensity));
    
    return intensity;
}

EmotionNode WoundProcessor::processWound(const Wound& wound) {
    // Calculate intensity from description
    float calculatedIntensity = calculateIntensity(wound.description, wound.intensity);
    
    // Find emotion matches
    return findEmotionByKeywords(wound.description, calculatedIntensity);
}

EmotionNode WoundProcessor::findEmotionByKeywords(const std::string& text, float intensity) {
    std::string normalized = normalizeText(text);
    
    // Find all matches
    auto matches = findMatches(normalized);
    
    if (!matches.empty()) {
        // Use the best match
        const auto& bestMatch = matches[0];
        auto emotion = thesaurus_.findByName(bestMatch.emotionName);
        
        if (emotion) {
            auto result = *emotion;
            result.intensity = intensity;
            return result;
        }
    }
    
    // Fallback: estimate VAD coordinates from common words
    float estimatedValence = 0.0f;
    float estimatedArousal = 0.5f;
    
    if (normalized.find("sad") != std::string::npos || 
        normalized.find("grief") != std::string::npos ||
        normalized.find("loss") != std::string::npos ||
        normalized.find("depressed") != std::string::npos) {
        estimatedValence = -0.7f;
        estimatedArousal = 0.3f;
    } else if (normalized.find("angry") != std::string::npos ||
               normalized.find("rage") != std::string::npos ||
               normalized.find("fury") != std::string::npos ||
               normalized.find("mad") != std::string::npos) {
        estimatedValence = -0.8f;
        estimatedArousal = 0.9f;
    } else if (normalized.find("happy") != std::string::npos ||
               normalized.find("joy") != std::string::npos ||
               normalized.find("love") != std::string::npos ||
               normalized.find("glad") != std::string::npos) {
        estimatedValence = 0.7f;
        estimatedArousal = 0.6f;
    } else if (normalized.find("fear") != std::string::npos ||
               normalized.find("anxious") != std::string::npos ||
               normalized.find("scared") != std::string::npos ||
               normalized.find("afraid") != std::string::npos) {
        estimatedValence = -0.6f;
        estimatedArousal = 0.8f;
    } else if (normalized.find("excited") != std::string::npos ||
               normalized.find("excitement") != std::string::npos) {
        estimatedValence = 0.6f;
        estimatedArousal = 0.85f;
    } else {
        // Default: mild negative (melancholy)
        estimatedValence = -0.4f;
        estimatedArousal = 0.4f;
    }
    
    return thesaurus_.findNearest(estimatedValence, estimatedArousal, intensity);
}

} // namespace kelly
