#include "voice/PhonemeConverter.h"
#include "common/PathResolver.h"
#include <juce_core/juce_core.h>
#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>

namespace kelly {

PhonemeConverter::PhonemeConverter() {
    initializeDefaultPhonemes();
    initializeWordDictionary();

    // Initialize CMU Dictionary
    cmuDictionary_ = std::make_unique<CMUDictionary>();
    useCMUDictionary_ = true;

    // Attempt to load phoneme database from JSON file
    // Falls back to default phonemes if file is not found
    loadPhonemeDatabase("");  // Empty path will use PathResolver to find file
}

void PhonemeConverter::initializeDefaultPhonemes() {
    // Initialize phoneme database with default formant values
    // Note: In a full implementation, these would be loaded from phonemes.json

    auto addPhoneme = [this](const std::string& ipa, const std::string& type,
                             const std::array<float, 4>& formants,
                             const std::array<float, 4>& bandwidths,
                             float duration_ms, bool voiced) {
        Phoneme p;
        p.ipa = ipa;
        p.type = type;
        p.formants = formants;
        p.bandwidths = bandwidths;
        p.duration_ms = duration_ms;
        p.voiced = voiced;
        phonemeDatabase_[ipa] = p;
    };

    // Vowels (using typical formant values)
    addPhoneme("/i/", "vowel", {270, 2290, 3010, 3600}, {30, 100, 200, 250}, 120, true);    // EE
    addPhoneme("/ɪ/", "vowel", {390, 1990, 2550, 3600}, {40, 100, 200, 250}, 120, true);    // IH
    addPhoneme("/e/", "vowel", {530, 1840, 2480, 3600}, {50, 100, 200, 250}, 120, true);    // EH
    addPhoneme("/ɛ/", "vowel", {610, 1900, 2600, 3600}, {60, 100, 200, 250}, 130, true);    // EH
    addPhoneme("/æ/", "vowel", {860, 1720, 2410, 3600}, {70, 100, 200, 250}, 140, true);    // AA
    addPhoneme("/ɑ/", "vowel", {730, 1090, 2440, 3600}, {70, 120, 200, 250}, 150, true);    // AH
    addPhoneme("/ɔ/", "vowel", {570, 840, 2410, 3600}, {60, 120, 200, 250}, 140, true);     // AW
    addPhoneme("/o/", "vowel", {570, 840, 2410, 3600}, {60, 120, 200, 250}, 140, true);     // OH
    addPhoneme("/ʊ/", "vowel", {440, 1020, 2240, 3600}, {50, 120, 200, 250}, 130, true);    // UH
    addPhoneme("/u/", "vowel", {300, 870, 2240, 3600}, {40, 120, 200, 250}, 120, true);     // OO
    addPhoneme("/ʌ/", "vowel", {640, 1190, 2390, 3600}, {65, 110, 200, 250}, 130, true);    // UH
    addPhoneme("/ə/", "vowel", {500, 1500, 2500, 3600}, {60, 110, 200, 250}, 100, true);    // Schwa

    // Diphthongs
    addPhoneme("/aɪ/", "diphthong", {500, 1700, 2500, 3600}, {55, 105, 200, 250}, 180, true);  // AY
    addPhoneme("/aʊ/", "diphthong", {650, 1200, 2400, 3600}, {65, 115, 200, 250}, 180, true);  // OW
    addPhoneme("/ɔɪ/", "diphthong", {560, 950, 2410, 3600}, {60, 120, 200, 250}, 180, true);   // OY
    addPhoneme("/eɪ/", "diphthong", {520, 1800, 2480, 3600}, {55, 105, 200, 250}, 180, true);  // EY
    addPhoneme("/oʊ/", "diphthong", {570, 840, 2410, 3600}, {60, 120, 200, 250}, 180, true);   // OH

    // Common consonants (simplified formants - many are unvoiced)
    addPhoneme("/p/", "consonant", {0, 0, 0, 0}, {0, 0, 0, 0}, 80, false);
    addPhoneme("/b/", "consonant", {300, 700, 2100, 3600}, {200, 300, 400, 500}, 80, true);
    addPhoneme("/t/", "consonant", {0, 0, 0, 0}, {0, 0, 0, 0}, 70, false);
    addPhoneme("/d/", "consonant", {300, 1800, 2600, 3600}, {200, 300, 400, 500}, 70, true);
    addPhoneme("/k/", "consonant", {0, 0, 0, 0}, {0, 0, 0, 0}, 90, false);
    addPhoneme("/g/", "consonant", {300, 1200, 2400, 3600}, {200, 300, 400, 500}, 90, true);
    addPhoneme("/f/", "consonant", {0, 0, 0, 0}, {0, 0, 0, 0}, 100, false);
    addPhoneme("/v/", "consonant", {200, 1200, 2300, 3600}, {300, 400, 500, 600}, 100, true);
    addPhoneme("/s/", "consonant", {0, 0, 0, 0}, {0, 0, 0, 0}, 120, false);
    addPhoneme("/z/", "consonant", {200, 1400, 2800, 3600}, {300, 400, 500, 600}, 120, true);
    addPhoneme("/m/", "consonant", {300, 1200, 2400, 3600}, {200, 300, 400, 500}, 100, true);
    addPhoneme("/n/", "consonant", {350, 1800, 2600, 3600}, {200, 300, 400, 500}, 100, true);
    addPhoneme("/l/", "consonant", {350, 1000, 2400, 3600}, {200, 300, 400, 500}, 90, true);
    addPhoneme("/r/", "consonant", {400, 1200, 1600, 3600}, {200, 300, 400, 500}, 80, true);
    addPhoneme("/w/", "consonant", {300, 600, 2200, 3600}, {200, 300, 400, 500}, 90, true);
    addPhoneme("/j/", "consonant", {300, 2100, 3000, 3600}, {200, 300, 400, 500}, 80, true);
    addPhoneme("/h/", "consonant", {0, 0, 0, 0}, {0, 0, 0, 0}, 50, false);
}

void PhonemeConverter::initializeWordDictionary() {
    // Initialize common word dictionary with known pronunciations
    // In a full implementation, this would load from CMU Pronouncing Dictionary

    wordDictionary_["the"] = {"/ð/", "/ə/"};
    wordDictionary_["be"] = {"/b/", "/i/"};
    wordDictionary_["to"] = {"/t/", "/u/"};
    wordDictionary_["of"] = {"/ʌ/", "/v/"};
    wordDictionary_["and"] = {"/æ/", "/n/", "/d/"};
    wordDictionary_["a"] = {"/ə/"};
    wordDictionary_["in"] = {"/ɪ/", "/n/"};
    wordDictionary_["that"] = {"/ð/", "/æ/", "/t/"};
    wordDictionary_["have"] = {"/h/", "/æ/", "/v/"};
    wordDictionary_["i"] = {"/aɪ/"};
    wordDictionary_["it"] = {"/ɪ/", "/t/"};
    wordDictionary_["for"] = {"/f/", "/ɔ/", "/r/"};
    wordDictionary_["not"] = {"/n/", "/ɑ/", "/t/"};
    wordDictionary_["on"] = {"/ɑ/", "/n/"};
    wordDictionary_["with"] = {"/w/", "/ɪ/", "/ð/"};
    wordDictionary_["he"] = {"/h/", "/i/"};
    wordDictionary_["as"] = {"/æ/", "/z/"};
    wordDictionary_["you"] = {"/j/", "/u/"};
    wordDictionary_["do"] = {"/d/", "/u/"};
    wordDictionary_["at"] = {"/æ/", "/t/"};
    wordDictionary_["this"] = {"/ð/", "/ɪ/", "/s/"};
    wordDictionary_["but"] = {"/b/", "/ʌ/", "/t/"};
    wordDictionary_["his"] = {"/h/", "/ɪ/", "/z/"};
    wordDictionary_["by"] = {"/b/", "/aɪ/"};
    wordDictionary_["from"] = {"/f/", "/r/", "/ʌ/", "/m/"};
    wordDictionary_["they"] = {"/ð/", "/eɪ/"};
    wordDictionary_["we"] = {"/w/", "/i/"};
    wordDictionary_["say"] = {"/s/", "/eɪ/"};
    wordDictionary_["her"] = {"/h/", "/ɝ/"};
    wordDictionary_["she"] = {"/ʃ/", "/i/"};
    wordDictionary_["or"] = {"/ɔ/", "/r/"};
    wordDictionary_["an"] = {"/æ/", "/n/"};
    wordDictionary_["will"] = {"/w/", "/ɪ/", "/l/"};
    wordDictionary_["my"] = {"/m/", "/aɪ/"};
    wordDictionary_["one"] = {"/w/", "/ʌ/", "/n/"};
    wordDictionary_["all"] = {"/ɔ/", "/l/"};
    wordDictionary_["would"] = {"/w/", "/ʊ/", "/d/"};
    wordDictionary_["there"] = {"/ð/", "/ɛ/", "/r/"};
    wordDictionary_["their"] = {"/ð/", "/ɛ/", "/r/"};

    // Common emotional/lyric words
    wordDictionary_["love"] = {"/l/", "/ʌ/", "/v/"};
    wordDictionary_["heart"] = {"/h/", "/ɑ/", "/r/", "/t/"};
    wordDictionary_["soul"] = {"/s/", "/oʊ/", "/l/"};
    wordDictionary_["light"] = {"/l/", "/aɪ/", "/t/"};
    wordDictionary_["dark"] = {"/d/", "/ɑ/", "/r/", "/k/"};
    wordDictionary_["night"] = {"/n/", "/aɪ/", "/t/"};
    wordDictionary_["day"] = {"/d/", "/eɪ/"};
    wordDictionary_["time"] = {"/t/", "/aɪ/", "/m/"};
    wordDictionary_["dream"] = {"/d/", "/r/", "/i/", "/m/"};
    wordDictionary_["feel"] = {"/f/", "/i/", "/l/"};
    wordDictionary_["know"] = {"/n/", "/oʊ/"};
    wordDictionary_["see"] = {"/s/", "/i/"};
    wordDictionary_["come"] = {"/k/", "/ʌ/", "/m/"};
    wordDictionary_["go"] = {"/g/", "/oʊ/"};
    wordDictionary_["take"] = {"/t/", "/eɪ/", "/k/"};
    wordDictionary_["make"] = {"/m/", "/eɪ/", "/k/"};
    wordDictionary_["give"] = {"/g/", "/ɪ/", "/v/"};
    wordDictionary_["live"] = {"/l/", "/ɪ/", "/v/"};
    wordDictionary_["want"] = {"/w/", "/ɑ/", "/n/", "/t/"};
    wordDictionary_["need"] = {"/n/", "/i/", "/d/"};
}

std::vector<Phoneme> PhonemeConverter::textToPhonemes(const std::string& text) {
    std::vector<Phoneme> phonemes;

    // Split text into words
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        std::string normalized = normalizeWord(word);
        if (normalized.empty()) continue;

        // Get phonemes for this word
        std::vector<std::string> ipaSymbols = wordToPhonemes(normalized);

        // Convert IPA symbols to Phoneme structs
        for (const auto& ipa : ipaSymbols) {
            Phoneme p = getPhonemeFromIPA(ipa);
            if (!p.ipa.empty()) {
                phonemes.push_back(p);
            }
        }
    }

    return phonemes;
}

std::vector<std::string> PhonemeConverter::wordToPhonemes(const std::string& word) {
    std::string normalized = normalizeWord(word);

    // Try CMU Dictionary first if enabled
    if (useCMUDictionary_ && cmuDictionary_) {
        auto arpabetPhonemes = cmuDictionary_->lookup(normalized);
        if (!arpabetPhonemes.empty()) {
            // Convert ARPABET to IPA
            return CMUDictionary::arpabetToIPA(arpabetPhonemes);
        }
    }

    // Check built-in dictionary
    auto dictIt = wordDictionary_.find(normalized);
    if (dictIt != wordDictionary_.end()) {
        return dictIt->second;
    }

    // Fall back to rule-based G2P
    return graphemeToPhoneme(normalized);
}

std::vector<std::string> PhonemeConverter::graphemeToPhoneme(const std::string& word) {
    std::vector<std::string> phonemes;

    // Simplified rule-based G2P
    // This is a basic implementation - a full system would use more sophisticated rules

    for (size_t i = 0; i < word.length(); ++i) {
        char c = std::tolower(word[i]);

        // Handle common patterns
        if (i < word.length() - 1) {
            // Check digraphs
            std::string digraph = word.substr(i, 2);
            std::transform(digraph.begin(), digraph.end(), digraph.begin(), ::tolower);

            if (digraph == "th") {
                phonemes.push_back("/ð/");  // Voiced TH
                i++;
                continue;
            } else if (digraph == "sh") {
                phonemes.push_back("/ʃ/");
                i++;
                continue;
            } else if (digraph == "ch") {
                phonemes.push_back("/tʃ/");
                i++;
                continue;
            } else if (digraph == "ng") {
                phonemes.push_back("/ŋ/");
                i++;
                continue;
            } else if (digraph == "ai" || digraph == "ay") {
                phonemes.push_back("/eɪ/");
                i++;
                continue;
            } else if (digraph == "ei" || digraph == "ey") {
                phonemes.push_back("/eɪ/");
                i++;
                continue;
            } else if (digraph == "oi" || digraph == "oy") {
                phonemes.push_back("/ɔɪ/");
                i++;
                continue;
            } else if (digraph == "ou" || digraph == "ow") {
                phonemes.push_back("/aʊ/");
                i++;
                continue;
            } else if (digraph == "oo") {
                phonemes.push_back("/u/");
                i++;
                continue;
            }
        }

        // Single character mapping
        switch (c) {
            case 'a':
                phonemes.push_back("/æ/");
                break;
            case 'e':
                phonemes.push_back("/ɛ/");
                break;
            case 'i':
                phonemes.push_back("/ɪ/");
                break;
            case 'o':
                phonemes.push_back("/ɑ/");
                break;
            case 'u':
                phonemes.push_back("/ʌ/");
                break;
            case 'y':
                if (isVowelLetter(c) || i == 0) {
                    phonemes.push_back("/j/");
                } else {
                    phonemes.push_back("/ɪ/");
                }
                break;
            case 'p':
                phonemes.push_back("/p/");
                break;
            case 'b':
                phonemes.push_back("/b/");
                break;
            case 't':
                phonemes.push_back("/t/");
                break;
            case 'd':
                phonemes.push_back("/d/");
                break;
            case 'k':
            case 'c':
                phonemes.push_back("/k/");
                break;
            case 'g':
                phonemes.push_back("/g/");
                break;
            case 'f':
                phonemes.push_back("/f/");
                break;
            case 'v':
                phonemes.push_back("/v/");
                break;
            case 's':
                phonemes.push_back("/s/");
                break;
            case 'z':
                phonemes.push_back("/z/");
                break;
            case 'm':
                phonemes.push_back("/m/");
                break;
            case 'n':
                phonemes.push_back("/n/");
                break;
            case 'l':
                phonemes.push_back("/l/");
                break;
            case 'r':
                phonemes.push_back("/r/");
                break;
            case 'w':
                phonemes.push_back("/w/");
                break;
            case 'h':
                phonemes.push_back("/h/");
                break;
            default:
                // Skip unknown characters
                break;
        }
    }

    return phonemes;
}

std::vector<std::string> PhonemeConverter::splitIntoSyllables(const std::string& word) {
    std::vector<std::string> syllables;

    // Simplified syllable splitting based on vowel patterns
    std::string normalized = normalizeWord(word);
    if (normalized.empty()) return syllables;

    std::string currentSyllable;
    bool inVowelCluster = false;

    for (size_t i = 0; i < normalized.length(); ++i) {
        char c = normalized[i];
        bool isVowel = isVowelLetter(c);

        if (isVowel) {
            if (!inVowelCluster && !currentSyllable.empty() &&
                i > 0 && !isVowelLetter(normalized[i-1])) {
                // Start new syllable before vowel (unless previous was also vowel)
                if (!currentSyllable.empty()) {
                    syllables.push_back(currentSyllable);
                    currentSyllable.clear();
                }
            }
            currentSyllable += c;
            inVowelCluster = true;
        } else {
            if (inVowelCluster) {
                // Vowel cluster ended, add consonant to current syllable
                currentSyllable += c;
                inVowelCluster = false;

                // Check if we should split after this consonant
                if (i < normalized.length() - 1 && isVowelLetter(normalized[i+1])) {
                    // Next char is vowel, start new syllable
                    syllables.push_back(currentSyllable);
                    currentSyllable.clear();
                }
            } else {
                // Consonant after consonant
                currentSyllable += c;
            }
        }
    }

    if (!currentSyllable.empty()) {
        syllables.push_back(currentSyllable);
    }

    // Ensure at least one syllable
    if (syllables.empty()) {
        syllables.push_back(normalized);
    }

    return syllables;
}

std::vector<int> PhonemeConverter::detectStress(const std::string& word) {
    std::vector<int> stress;

    // Get syllables first
    std::vector<std::string> syllables = splitIntoSyllables(word);
    stress.resize(syllables.size(), 0);

    // Simplified stress detection:
    // - For 1-2 syllable words, stress first syllable
    // - For longer words, stress patterns vary, but typically stress is on first or second syllable
    if (syllables.size() == 1) {
        stress[0] = 2; // Primary stress
    } else if (syllables.size() == 2) {
        stress[0] = 2; // Primary stress on first syllable
        stress[1] = 0; // Unstressed second syllable
    } else {
        // Longer words: stress first syllable (simplified)
        stress[0] = 2;
        // Optional: stress third syllable as secondary
        if (syllables.size() >= 3) {
            stress[2] = 1;
        }
    }

    return stress;
}

Phoneme PhonemeConverter::getPhonemeFromIPA(const std::string& ipa) {
    auto it = phonemeDatabase_.find(ipa);
    if (it != phonemeDatabase_.end()) {
        return it->second;
    }

    // Return empty phoneme if not found
    return Phoneme();
}

int PhonemeConverter::countSyllables(const std::string& word) {
    return static_cast<int>(splitIntoSyllables(word).size());
}

bool PhonemeConverter::loadPhonemeDatabase(const std::string& filePath) {
    juce::File jsonFile;

    // If filePath is provided, use it; otherwise try to find phonemes.json
    if (!filePath.empty()) {
        jsonFile = juce::File(filePath);
    } else {
        jsonFile = PathResolver::findDataFile("phonemes.json");
    }

    if (!jsonFile.existsAsFile()) {
        juce::Logger::writeToLog("PhonemeConverter: phonemes.json not found, using default phonemes");
        return false;
    }

    juce::String jsonText = jsonFile.loadFileAsString();
    if (jsonText.isEmpty()) {
        juce::Logger::writeToLog("PhonemeConverter: Empty or unreadable file: " + jsonFile.getFullPathName());
        return false;
    }

    juce::var parsedJson = juce::JSON::parse(jsonText);
    if (!parsedJson.isObject()) {
        juce::Logger::writeToLog("PhonemeConverter: Invalid JSON in file: " + jsonFile.getFullPathName());
        return false;
    }

    auto* root = parsedJson.getDynamicObject();
    if (!root) return false;

    // Get phonemes array
    auto phonemesArray = root->getProperty("phonemes");
    if (!phonemesArray.isArray()) {
        juce::Logger::writeToLog("PhonemeConverter: No 'phonemes' array found in JSON");
        return false;  // Return early before clearing database
    }

    int loadedCount = 0;

    // Store existing database as backup (in case loading fails)
    std::map<std::string, Phoneme> backupDatabase = phonemeDatabase_;

    // Clear existing database and reload from JSON
    phonemeDatabase_.clear();

    // Process each phoneme in the array
    for (const auto& phonemeVar : *phonemesArray.getArray()) {
        if (!phonemeVar.isObject()) continue;

        auto* phonemeObj = phonemeVar.getDynamicObject();
        if (!phonemeObj) continue;

        Phoneme p;

        // Get IPA symbol
        p.ipa = phonemeObj->getProperty("ipa").toString().toStdString();
        if (p.ipa.empty()) continue;

        // Get type
        p.type = phonemeObj->getProperty("type").toString().toStdString();

        // Get formants
        auto formantsObj = phonemeObj->getProperty("formants");
        if (formantsObj.isObject()) {
            auto* formants = formantsObj.getDynamicObject();
            p.formants[0] = static_cast<float>(static_cast<double>(formants->getProperty("F1")));
            p.formants[1] = static_cast<float>(static_cast<double>(formants->getProperty("F2")));
            p.formants[2] = static_cast<float>(static_cast<double>(formants->getProperty("F3")));
            p.formants[3] = static_cast<float>(static_cast<double>(formants->getProperty("F4")));
        }

        // Get bandwidths
        auto bandwidthsObj = phonemeObj->getProperty("bandwidths");
        if (bandwidthsObj.isObject()) {
            auto* bandwidths = bandwidthsObj.getDynamicObject();
            p.bandwidths[0] = static_cast<float>(static_cast<double>(bandwidths->getProperty("B1")));
            p.bandwidths[1] = static_cast<float>(static_cast<double>(bandwidths->getProperty("B2")));
            p.bandwidths[2] = static_cast<float>(static_cast<double>(bandwidths->getProperty("B3")));
            p.bandwidths[3] = static_cast<float>(static_cast<double>(bandwidths->getProperty("B4")));
        }

        // Get duration
        p.duration_ms = static_cast<float>(static_cast<double>(phonemeObj->getProperty("duration_ms")));
        if (p.duration_ms <= 0.0f) p.duration_ms = 100.0f; // Default

        // Get voiced flag
        auto voicedVar = phonemeObj->getProperty("voiced");
        p.voiced = voicedVar.isBool() ? static_cast<bool>(voicedVar) : false;

        // Store in database
        phonemeDatabase_[p.ipa] = p;
        loadedCount++;
    }

    juce::Logger::writeToLog("PhonemeConverter: Loaded " + juce::String(loadedCount) + " phonemes from " + jsonFile.getFileName());

    // If no phonemes loaded, restore defaults
    if (loadedCount == 0) {
        juce::Logger::writeToLog("PhonemeConverter: No phonemes loaded, restoring defaults");
        phonemeDatabase_ = backupDatabase;
        return false;
    }

    return true;
}

std::pair<std::array<float, 4>, std::array<float, 4>> PhonemeConverter::getFormants(const Phoneme& phoneme) const {
    return {phoneme.formants, phoneme.bandwidths};
}

std::pair<std::array<float, 4>, std::array<float, 4>> PhonemeConverter::interpolatePhonemes(
    const Phoneme& p1,
    const Phoneme& p2,
    float t) const
{
    t = std::clamp(t, 0.0f, 1.0f);

    std::array<float, 4> interpolatedFormants;
    std::array<float, 4> interpolatedBandwidths;

    for (size_t i = 0; i < 4; ++i) {
        interpolatedFormants[i] = p1.formants[i] * (1.0f - t) + p2.formants[i] * t;
        interpolatedBandwidths[i] = p1.bandwidths[i] * (1.0f - t) + p2.bandwidths[i] * t;
    }

    return {interpolatedFormants, interpolatedBandwidths};
}

std::string PhonemeConverter::normalizeWord(const std::string& word) const {
    std::string normalized = word;

    // Convert to lowercase
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

    // Remove punctuation
    normalized.erase(std::remove_if(normalized.begin(), normalized.end(), ::ispunct), normalized.end());

    return normalized;
}

bool PhonemeConverter::isVowelLetter(char c) const {
    c = std::tolower(c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

bool PhonemeConverter::isConsonantLetter(char c) const {
    c = std::tolower(c);
    return (c >= 'b' && c <= 'z') && !isVowelLetter(c);
}

} // namespace kelly
