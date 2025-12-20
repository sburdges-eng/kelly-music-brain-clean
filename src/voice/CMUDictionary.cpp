#include "voice/CMUDictionary.h"
#include "common/PathResolver.h"
#include <juce_core/juce_core.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace kelly {

CMUDictionary::CMUDictionary() {
    // Try to load from file first, fall back to embedded dictionary
    juce::File dataDir = PathResolver::getDataDirectory();
    juce::File cmuDictFile = dataDir.getChildFile("cmudict-0.7b.txt");

    if (cmuDictFile.existsAsFile()) {
        loadFromFile(cmuDictFile.getFullPathName().toStdString());
    } else {
        // Fall back to embedded dictionary (subset of common words)
        loadEmbeddedDictionary();
    }
}

bool CMUDictionary::loadFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    size_t loaded = 0;

    while (std::getline(file, line)) {
        // Skip comments and blank lines
        if (line.empty() || line[0] == ';' || line[0] == '#') {
            continue;
        }

        // Parse line: WORD  PHONEME1 PHONEME2 ...
        std::istringstream iss(line);
        std::string word;
        iss >> word;

        if (word.empty()) {
            continue;
        }

        // Remove alternative pronunciation markers (e.g., "WORD(2)")
        size_t parenPos = word.find('(');
        if (parenPos != std::string::npos) {
            word = word.substr(0, parenPos);
        }

        // Normalize word (already uppercase in CMU dict, but ensure it)
        std::transform(word.begin(), word.end(), word.begin(), ::toupper);
        word = normalizeWord(word);

        // Read phonemes
        std::vector<std::string> phonemes;
        std::string phoneme;
        while (iss >> phoneme) {
            phonemes.push_back(phoneme);
        }

        if (!phonemes.empty()) {
            // Store primary pronunciation (first occurrence)
            if (dictionary_.find(word) == dictionary_.end()) {
                dictionary_[word] = phonemes;
                loaded++;
            }
        }
    }

    return loaded > 0;
}

void CMUDictionary::loadEmbeddedDictionary() {
    // Load a subset of common words from embedded data
    // This is a fallback when the full dictionary file is not available

    // Common words with their ARPABET pronunciations
    dictionary_["THE"] = {"DH", "AH0"};
    dictionary_["BE"] = {"B", "IY1"};
    dictionary_["TO"] = {"T", "UW1"};
    dictionary_["OF"] = {"AH1", "V"};
    dictionary_["AND"] = {"AE1", "N", "D"};
    dictionary_["A"] = {"AH0"};
    dictionary_["IN"] = {"IH1", "N"};
    dictionary_["THAT"] = {"DH", "AE1", "T"};
    dictionary_["HAVE"] = {"HH", "AE1", "V"};
    dictionary_["I"] = {"AY1"};
    dictionary_["IT"] = {"IH1", "T"};
    dictionary_["FOR"] = {"F", "AO1", "R"};
    dictionary_["NOT"] = {"N", "AA1", "T"};
    dictionary_["ON"] = {"AA1", "N"};
    dictionary_["WITH"] = {"W", "IH1", "DH"};
    dictionary_["HE"] = {"HH", "IY1"};
    dictionary_["AS"] = {"AE1", "Z"};
    dictionary_["YOU"] = {"Y", "UW1"};
    dictionary_["DO"] = {"D", "UW1"};
    dictionary_["AT"] = {"AE1", "T"};
    dictionary_["THIS"] = {"DH", "IH1", "S"};
    dictionary_["BUT"] = {"B", "AH1", "T"};
    dictionary_["HIS"] = {"HH", "IH1", "Z"};
    dictionary_["BY"] = {"B", "AY1"};
    dictionary_["FROM"] = {"F", "R", "AH1", "M"};
    dictionary_["THEY"] = {"DH", "EY1"};
    dictionary_["WE"] = {"W", "IY1"};
    dictionary_["SAY"] = {"S", "EY1"};
    dictionary_["HER"] = {"HH", "ER1"};
    dictionary_["SHE"] = {"SH", "IY1"};
    dictionary_["OR"] = {"AO1", "R"};
    dictionary_["AN"] = {"AE1", "N"};
    dictionary_["WILL"] = {"W", "IH1", "L"};
    dictionary_["MY"] = {"M", "AY1"};
    dictionary_["ONE"] = {"W", "AH1", "N"};
    dictionary_["ALL"] = {"AO1", "L"};
    dictionary_["WOULD"] = {"W", "UH1", "D"};
    dictionary_["THERE"] = {"DH", "EH1", "R"};
    dictionary_["THEIR"] = {"DH", "EH1", "R"};

    // Common emotional/lyric words
    dictionary_["LOVE"] = {"L", "AH1", "V"};
    dictionary_["HEART"] = {"HH", "AA1", "R", "T"};
    dictionary_["SOUL"] = {"S", "OW1", "L"};
    dictionary_["LIGHT"] = {"L", "AY1", "T"};
    dictionary_["DARK"] = {"D", "AA1", "R", "K"};
    dictionary_["NIGHT"] = {"N", "AY1", "T"};
    dictionary_["DAY"] = {"D", "EY1"};
    dictionary_["TIME"] = {"T", "AY1", "M"};
    dictionary_["DREAM"] = {"D", "R", "IY1", "M"};
    dictionary_["FEEL"] = {"F", "IY1", "L"};
    dictionary_["KNOW"] = {"N", "OW1"};
    dictionary_["SEE"] = {"S", "IY1"};
    dictionary_["COME"] = {"K", "AH1", "M"};
    dictionary_["GO"] = {"G", "OW1"};
    dictionary_["TAKE"] = {"T", "EY1", "K"};
    dictionary_["MAKE"] = {"M", "EY1", "K"};
    dictionary_["GIVE"] = {"G", "IH1", "V"};
    dictionary_["LIVE"] = {"L", "IH1", "V"};
    dictionary_["WANT"] = {"W", "AA1", "N", "T"};
    dictionary_["NEED"] = {"N", "IY1", "D"};
}

std::vector<std::string> CMUDictionary::lookup(const std::string& word) const {
    std::string normalized = normalizeWord(word);
    auto it = dictionary_.find(normalized);
    if (it != dictionary_.end()) {
        return it->second;
    }
    return {};
}

std::vector<std::string> CMUDictionary::lookupWithoutStress(const std::string& word) const {
    auto phonemes = lookup(word);
    std::vector<std::string> result;

    for (const auto& phoneme : phonemes) {
        auto [phonemeOnly, stress] = extractStress(phoneme);
        result.push_back(phonemeOnly);
    }

    return result;
}

std::vector<std::pair<std::string, int>> CMUDictionary::lookupWithStress(const std::string& word) const {
    auto phonemes = lookup(word);
    std::vector<std::pair<std::string, int>> result;

    for (const auto& phoneme : phonemes) {
        result.push_back(extractStress(phoneme));
    }

    return result;
}

bool CMUDictionary::contains(const std::string& word) const {
    std::string normalized = normalizeWord(word);
    return dictionary_.find(normalized) != dictionary_.end();
}

std::string CMUDictionary::normalizeWord(const std::string& word) {
    std::string normalized = word;

    // Convert to uppercase
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::toupper);

    // Remove punctuation (keep apostrophes for contractions, but remove for simplicity)
    normalized.erase(std::remove_if(normalized.begin(), normalized.end(),
        [](char c) { return !std::isalnum(c) && c != '\''; }), normalized.end());

    return normalized;
}

std::pair<std::string, int> CMUDictionary::extractStress(const std::string& arpabet) {
    if (arpabet.empty()) {
        return {"", 0};
    }

    char lastChar = arpabet.back();
    if (std::isdigit(lastChar)) {
        int stress = lastChar - '0';
        std::string phoneme = arpabet.substr(0, arpabet.length() - 1);
        return {phoneme, stress};
    }

    return {arpabet, 0};
}

std::map<std::string, std::string> CMUDictionary::createARPABETToIPAMap() {
    // ARPABET to IPA mapping
    // Based on standard ARPABET phoneme set
    return {
        // Vowels
        {"AA", "/ɑ/"},   // father
        {"AE", "/æ/"},   // cat
        {"AH", "/ʌ/"},   // cut
        {"AH0", "/ə/"},  // about (schwa, unstressed)
        {"AO", "/ɔ/"},   // caught
        {"AW", "/aʊ/"},  // cow
        {"AY", "/aɪ/"},  // bite
        {"EH", "/ɛ/"},   // bet
        {"ER", "/ɝ/"},   // bird
        {"ER0", "/ɚ/"},  // better (r-colored schwa)
        {"EY", "/eɪ/"},  // bait
        {"IH", "/ɪ/"},   // bit
        {"IY", "/i/"},   // beat
        {"OW", "/oʊ/"},  // boat
        {"OY", "/ɔɪ/"},  // boy
        {"UH", "/ʊ/"},   // book
        {"UW", "/u/"},   // boot

        // Consonants
        {"B", "/b/"},
        {"CH", "/tʃ/"},
        {"D", "/d/"},
        {"DH", "/ð/"},   // this
        {"F", "/f/"},
        {"G", "/g/"},
        {"HH", "/h/"},
        {"JH", "/dʒ/"},  // judge
        {"K", "/k/"},
        {"L", "/l/"},
        {"M", "/m/"},
        {"N", "/n/"},
        {"NG", "/ŋ/"},   // sing
        {"P", "/p/"},
        {"R", "/r/"},
        {"S", "/s/"},
        {"SH", "/ʃ/"},   // ship
        {"T", "/t/"},
        {"TH", "/θ/"},   // thin
        {"V", "/v/"},
        {"W", "/w/"},
        {"Y", "/j/"},    // yes
        {"Z", "/z/"},
        {"ZH", "/ʒ/"},   // measure
    };
}

std::string CMUDictionary::arpabetToIPA(const std::string& arpabet) {
    static auto mapping = createARPABETToIPAMap();

    // Remove stress marker if present
    std::string phoneme = arpabet;
    if (!phoneme.empty() && std::isdigit(phoneme.back())) {
        phoneme = phoneme.substr(0, phoneme.length() - 1);
    }

    auto it = mapping.find(phoneme);
    if (it != mapping.end()) {
        return it->second;
    }

    // Return original if not found (shouldn't happen with valid ARPABET)
    return phoneme;
}

std::vector<std::string> CMUDictionary::arpabetToIPA(const std::vector<std::string>& arpabetPhonemes) {
    std::vector<std::string> ipaPhonemes;
    for (const auto& arpabet : arpabetPhonemes) {
        ipaPhonemes.push_back(arpabetToIPA(arpabet));
    }
    return ipaPhonemes;
}

} // namespace kelly
