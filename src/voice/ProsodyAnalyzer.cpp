#include "voice/ProsodyAnalyzer.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace kelly {

ProsodyAnalyzer::ProsodyAnalyzer() {
    // Initialize prosody analyzer
}

std::vector<int> ProsodyAnalyzer::detectStress(const std::string& word) {
    std::string normalized = normalizeWord(word);
    if (normalized.empty()) return {};

    // Simplified stress detection based on word patterns
    // For a full implementation, this would use a dictionary with stress markings

    std::vector<int> stress;

    // Count syllables first
    int numSyllables = countSyllables(normalized);
    stress.resize(numSyllables, 0);

    if (numSyllables == 1) {
        // Single syllable words are typically stressed
        stress[0] = 2;
    } else if (numSyllables == 2) {
        // Two-syllable words: typically stress first syllable
        stress[0] = 2;
        stress[1] = 0;
    } else {
        // Multi-syllable words: complex rules
        // Generally: stress often falls on the first or second syllable

        // Check for common suffixes that affect stress
        if (normalized.length() >= 3) {
            std::string suffix3 = normalized.substr(normalized.length() - 3);
            if (suffix3 == "ing" || suffix3 == "ion" || suffix3 == "ial") {
                // Stress syllable before suffix
                if (numSyllables >= 2) {
                    stress[numSyllables - 2] = 2;
                }
                return stress;
            }
        }

        if (normalized.length() >= 2) {
            std::string suffix2 = normalized.substr(normalized.length() - 2);
            if (suffix2 == "ed" || suffix2 == "er" || suffix2 == "ly") {
                // Suffix is unstressed, stress earlier syllable
                if (numSyllables >= 2) {
                    stress[numSyllables - 2] = 2;
                }
                return stress;
            }
        }

        // Default: stress first syllable
        stress[0] = 2;

        // Optional secondary stress on third syllable if word is long enough
        if (numSyllables >= 4) {
            stress[2] = 1;
        }
    }

    return stress;
}

std::vector<int> ProsodyAnalyzer::detectStressPattern(const std::vector<std::string>& words) {
    std::vector<int> pattern;

    for (const auto& word : words) {
        std::vector<int> wordStress = detectStress(word);
        pattern.insert(pattern.end(), wordStress.begin(), wordStress.end());
    }

    return pattern;
}

float ProsodyAnalyzer::matchMeter(const std::vector<int>& stressPattern, MeterType meterType) const {
    if (stressPattern.empty()) return 0.0f;

    MeterPattern targetPattern = getMeterPattern(meterType, static_cast<int>(stressPattern.size()));
    if (targetPattern.pattern.empty()) return 0.0f;

    // Normalize stress pattern to binary (0 or 1) for comparison
    std::vector<int> normalizedStress;
    std::vector<int> normalizedTarget;

    for (int s : stressPattern) {
        normalizedStress.push_back(s >= 1 ? 1 : 0);
    }

    for (int t : targetPattern.pattern) {
        normalizedTarget.push_back(t >= 1 ? 1 : 0);
    }

    // Pad or truncate to same length
    size_t minLen = std::min(normalizedStress.size(), normalizedTarget.size());
    normalizedStress.resize(minLen);
    normalizedTarget.resize(minLen);

    // Calculate match score
    int matches = 0;
    for (size_t i = 0; i < minLen; ++i) {
        if (normalizedStress[i] == normalizedTarget[i]) {
            matches++;
        }
    }

    return minLen > 0 ? static_cast<float>(matches) / static_cast<float>(minLen) : 0.0f;
}

ProsodyAnalyzer::MeterType ProsodyAnalyzer::detectMeter(const std::vector<int>& stressPattern) const {
    if (stressPattern.empty()) return MeterType::None;

    // Try each meter type and find best match
    float bestScore = 0.0f;
    MeterType bestMeter = MeterType::Mixed;

    std::vector<MeterType> meters = {
        MeterType::Iambic,
        MeterType::Trochaic,
        MeterType::Anapestic,
        MeterType::Dactylic
    };

    for (MeterType meter : meters) {
        float score = matchMeter(stressPattern, meter);
        if (score > bestScore) {
            bestScore = score;
            bestMeter = meter;
        }
    }

    // If best score is too low, return Mixed
    if (bestScore < 0.6f) {
        return MeterType::Mixed;
    }

    return bestMeter;
}

ProsodyAnalyzer::MeterPattern ProsodyAnalyzer::getMeterPattern(MeterType meterType, int numSyllables) const {
    MeterPattern pattern;
    pattern.type = meterType;
    pattern.pattern.reserve(numSyllables);

    switch (meterType) {
        case MeterType::Iambic:
            pattern.name = "iambic";
            // unstressed-stressed pattern
            for (int i = 0; i < numSyllables; ++i) {
                pattern.pattern.push_back(i % 2 == 0 ? 0 : 2);
            }
            break;

        case MeterType::Trochaic:
            pattern.name = "trochaic";
            // stressed-unstressed pattern
            for (int i = 0; i < numSyllables; ++i) {
                pattern.pattern.push_back(i % 2 == 0 ? 2 : 0);
            }
            break;

        case MeterType::Anapestic:
            pattern.name = "anapestic";
            // unstressed-unstressed-stressed pattern
            for (int i = 0; i < numSyllables; ++i) {
                pattern.pattern.push_back((i % 3 == 2) ? 2 : 0);
            }
            break;

        case MeterType::Dactylic:
            pattern.name = "dactylic";
            // stressed-unstressed-unstressed pattern
            for (int i = 0; i < numSyllables; ++i) {
                pattern.pattern.push_back((i % 3 == 0) ? 2 : 0);
            }
            break;

        default:
            pattern.name = "mixed";
            // Mixed: no specific pattern
            pattern.pattern.resize(numSyllables, 1);
            break;
    }

    return pattern;
}

bool ProsodyAnalyzer::validateLineLength(const LyricLine& line, int targetSyllables) const {
    // Count syllables in the line
    int actualSyllables = 0;
    for (const auto& syllable : line.syllables) {
        if (!syllable.text.empty()) {
            actualSyllables++;
        }
    }

    // If syllables not populated, count from text
    if (actualSyllables == 0 && !line.text.empty()) {
        // Simple word count estimate (rough approximation)
        std::istringstream iss(line.text);
        std::string word;
        while (iss >> word) {
            actualSyllables += countSyllables(word);
        }
    }

    // Allow tolerance of ±1 syllable
    int tolerance = 1;
    return std::abs(actualSyllables - targetSyllables) <= tolerance;
}

int ProsodyAnalyzer::countSyllables(const std::string& word) const {
    std::string normalized = normalizeWord(word);
    if (normalized.empty()) return 0;

    int syllables = 0;
    bool inVowelCluster = false;

    for (size_t i = 0; i < normalized.length(); ++i) {
        char c = normalized[i];
        bool isVowel = isVowelLetter(c);

        if (isVowel) {
            if (!inVowelCluster) {
                syllables++;
                inVowelCluster = true;
            }
        } else {
            inVowelCluster = false;
        }
    }

    // Handle silent 'e' at end
    if (normalized.length() > 1 &&
        normalized[normalized.length() - 1] == 'e' &&
        !isVowelLetter(normalized[normalized.length() - 2])) {
        // Silent e doesn't add a syllable, but might indicate previous vowel is long
        // For simplicity, we don't subtract here
    }

    // Ensure at least one syllable
    return syllables > 0 ? syllables : 1;
}

int ProsodyAnalyzer::countSyllables(const std::vector<std::string>& words) const {
    int total = 0;
    for (const auto& word : words) {
        total += countSyllables(word);
    }
    return total;
}

std::vector<std::string> ProsodyAnalyzer::selectWordsForMeter(
    const std::vector<std::string>& words,
    const MeterPattern& targetMeter) const
{
    // Simplified implementation: return words as-is
    // Full implementation would score each word against target meter and select best matches
    return words;
}

float ProsodyAnalyzer::calculateRhythmScore(const std::vector<int>& stressPattern) const {
    if (stressPattern.empty()) return 0.0f;

    // Calculate rhythm score based on alternation and variety
    float score = 0.0f;
    int variations = 0;

    // Check for good alternation (not all same, not too repetitive)
    for (size_t i = 1; i < stressPattern.size(); ++i) {
        if (stressPattern[i] != stressPattern[i-1]) {
            variations++;
        }
    }

    // Score based on variation (more variation = more natural)
    score = static_cast<float>(variations) / static_cast<float>(stressPattern.size());

    return score;
}

bool ProsodyAnalyzer::isVowelLetter(char c) const {
    c = std::tolower(c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

std::string ProsodyAnalyzer::normalizeWord(const std::string& word) const {
    std::string normalized = word;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    normalized.erase(std::remove_if(normalized.begin(), normalized.end(), ::ispunct), normalized.end());
    return normalized;
}

int ProsodyAnalyzer::getWordStressLevel(const std::string& word, int syllableIndex) const {
    // Simplified: return default stress based on position
    // Full implementation would use dictionary lookup
    if (syllableIndex == 0) {
        return 2; // Primary stress on first syllable
    }
    return 0; // Unstressed
}

} // namespace kelly
