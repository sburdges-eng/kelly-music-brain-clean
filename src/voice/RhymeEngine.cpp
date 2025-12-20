#include "voice/RhymeEngine.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <cctype>

namespace kelly {

RhymeEngine::RhymeEngine() {
    // Initialize rhyme engine
}

RhymeEngine::RhymeMatch RhymeEngine::checkRhyme(
    const std::string& word1,
    const std::string& word2)
{
    RhymeMatch match;
    match.word1 = word1;
    match.word2 = word2;

    if (word1 == word2) {
        // Same word = perfect rhyme (though not typically desired)
        match.type = RhymeType::Perfect;
        match.score = 1.0f;
        return match;
    }

    // Get phonemes for both words
    std::vector<std::string> phonemes1 = getPhonemes(word1);
    std::vector<std::string> phonemes2 = getPhonemes(word2);

    if (phonemes1.empty() || phonemes2.empty()) {
        match.type = RhymeType::None;
        match.score = 0.0f;
        return match;
    }

    // Extract end phonemes (last 2-3 phonemes typically determine rhyme)
    std::vector<std::string> end1 = extractEndPhonemes(word1, 3);
    std::vector<std::string> end2 = extractEndPhonemes(word2, 3);

    // Compare end phonemes
    float score = comparePhonemeSequences(end1, end2);

    // Determine rhyme type based on score
    if (score >= 0.95f) {
        match.type = RhymeType::Perfect;
    } else if (score >= 0.7f) {
        match.type = RhymeType::Slant;
    } else {
        match.type = RhymeType::None;
    }

    match.score = score;
    return match;
}

std::vector<RhymeEngine::RhymeMatch> RhymeEngine::findRhymes(
    const std::string& targetWord,
    const std::vector<std::string>& vocabulary,
    int maxResults)
{
    std::vector<RhymeMatch> matches;

    for (const auto& word : vocabulary) {
        if (word == targetWord) continue; // Skip same word

        RhymeMatch match = checkRhyme(targetWord, word);
        if (match.type != RhymeType::None) {
            matches.push_back(match);
        }
    }

    // Sort by score (highest first)
    std::sort(matches.begin(), matches.end(),
              [](const RhymeMatch& a, const RhymeMatch& b) {
                  return a.score > b.score;
              });

    // Limit results
    if (maxResults > 0 && matches.size() > static_cast<size_t>(maxResults)) {
        matches.resize(maxResults);
    }

    return matches;
}

std::map<int, std::vector<std::string>> RhymeEngine::generateRhymeWords(
    const std::vector<int>& scheme,
    const std::vector<std::string>& vocabulary,
    const std::map<int, std::string>& existingWords)
{
    std::map<int, std::vector<std::string>> result;

    // Group positions by rhyme scheme index
    std::map<int, std::vector<size_t>> schemeGroups;
    for (size_t i = 0; i < scheme.size(); ++i) {
        int groupIndex = scheme[i];
        schemeGroups[groupIndex].push_back(i);
    }

    // For each rhyme group, find words that rhyme with existing words in that group
    for (const auto& group : schemeGroups) {
        int groupIndex = group.first;
        std::vector<std::string> groupWords;

        // Check if we have an existing word for this group
        auto existingIt = existingWords.find(groupIndex);
        if (existingIt != existingWords.end()) {
            // Find rhymes for the existing word
            std::vector<RhymeMatch> rhymes = findRhymes(existingIt->second, vocabulary, 20);
            for (const auto& rhyme : rhymes) {
                groupWords.push_back(rhyme.word2);
            }
            // Also include the existing word itself
            groupWords.push_back(existingIt->second);
        } else {
            // No existing word, use vocabulary as candidates
            groupWords = vocabulary;
        }

        result[groupIndex] = groupWords;
    }

    return result;
}

std::vector<std::string> RhymeEngine::extractEndPhonemes(
    const std::string& word,
    int numPhonemes)
{
    std::vector<std::string> allPhonemes = getPhonemes(word);

    if (allPhonemes.size() <= static_cast<size_t>(numPhonemes)) {
        return allPhonemes;
    }

    // Extract last N phonemes
    std::vector<std::string> endPhonemes;
    size_t start = allPhonemes.size() - numPhonemes;
    for (size_t i = start; i < allPhonemes.size(); ++i) {
        endPhonemes.push_back(allPhonemes[i]);
    }

    return endPhonemes;
}

float RhymeEngine::comparePhonemeSequences(
    const std::vector<std::string>& phonemes1,
    const std::vector<std::string>& phonemes2) const
{
    if (phonemes1.empty() || phonemes2.empty()) {
        return 0.0f;
    }

    // Compare from the end (rhyming typically happens at word endings)
    size_t len1 = phonemes1.size();
    size_t len2 = phonemes2.size();
    size_t compareLen = std::min(len1, len2);

    int matches = 0;
    int totalComparisons = 0;

    for (size_t i = 0; i < compareLen; ++i) {
        size_t idx1 = len1 - 1 - i;
        size_t idx2 = len2 - 1 - i;

        std::string p1 = phonemes1[idx1];
        std::string p2 = phonemes2[idx2];

        if (p1 == p2) {
            matches++;
        } else {
            // Check similarity (for slant rhymes)
            float similarity = phonemeSimilarity(p1, p2);
            if (similarity > 0.8f) {
                matches++;
            }
        }
        totalComparisons++;
    }

    return totalComparisons > 0 ? static_cast<float>(matches) / static_cast<float>(totalComparisons) : 0.0f;
}

std::vector<RhymeEngine::RhymeMatch> RhymeEngine::detectInternalRhymes(
    const std::vector<std::string>& words)
{
    std::vector<RhymeMatch> matches;

    // Check all pairs of words in the line
    for (size_t i = 0; i < words.size(); ++i) {
        for (size_t j = i + 1; j < words.size(); ++j) {
            RhymeMatch match = checkRhyme(words[i], words[j]);
            if (match.type != RhymeType::None) {
                match.type = RhymeType::Internal;
                matches.push_back(match);
            }
        }
    }

    return matches;
}

void RhymeEngine::buildRhymeDatabase(const std::vector<std::string>& vocabulary) {
    rhymeDatabase_.clear();

    for (const auto& word : vocabulary) {
        std::vector<std::string> endPhonemes = extractEndPhonemes(word, 3);

        // Create key from end phonemes
        std::string key;
        for (const auto& phoneme : endPhonemes) {
            key += phoneme;
        }

        rhymeDatabase_[key].push_back(word);
    }
}

std::vector<std::string> RhymeEngine::getPhonemes(const std::string& word) {
    return phonemeConverter_.wordToPhonemes(word);
}

std::vector<std::string> RhymeEngine::normalizeForRhyme(const std::vector<std::string>& phonemes) const {
    std::vector<std::string> normalized;

    // Keep vowels and final consonants (important for rhyme)
    // Skip initial consonants and some unstressed vowels
    for (const auto& phoneme : phonemes) {
        if (isVowelPhoneme(phoneme)) {
            normalized.push_back(phoneme);
        }
    }

    // Add final consonants if any
    if (!phonemes.empty()) {
        std::string last = phonemes.back();
        if (!isVowelPhoneme(last)) {
            normalized.push_back(last);
        }
    }

    return normalized;
}

bool RhymeEngine::isVowelPhoneme(const std::string& ipa) const {
    // Check if IPA symbol represents a vowel or diphthong
    // Vowels: /i/, /ɪ/, /e/, /ɛ/, /æ/, /ɑ/, /ɔ/, /o/, /ʊ/, /u/, /ʌ/, /ə/
    // Diphthongs: /aɪ/, /aʊ/, /ɔɪ/, /eɪ/, /oʊ/, /ɪə/, /eə/

    std::string normalized = ipa;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

    // Remove slashes
    normalized.erase(std::remove(normalized.begin(), normalized.end(), '/'), normalized.end());

    // Vowel phonemes
    std::vector<std::string> vowels = {
        "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɝ",
        "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ", "ɪə", "eə"
    };

    return std::find(vowels.begin(), vowels.end(), normalized) != vowels.end();
}

float RhymeEngine::phonemeSimilarity(const std::string& ipa1, const std::string& ipa2) const {
    if (ipa1 == ipa2) {
        return 1.0f;
    }

    // Similar vowel groups (for slant rhymes)
    std::map<std::string, std::vector<std::string>> vowelGroups = {
        {"high_front", {"i", "ɪ", "e"}},
        {"mid_front", {"e", "ɛ", "æ"}},
        {"low", {"æ", "ɑ", "ʌ"}},
        {"mid_back", {"ɔ", "o", "ʊ"}},
        {"high_back", {"u", "ʊ", "o"}},
        {"central", {"ə", "ʌ", "ɝ"}}
    };

    // Check if phonemes are in the same vowel group
    for (const auto& group : vowelGroups) {
        bool found1 = false, found2 = false;
        for (const auto& vowel : group.second) {
            if (ipa1.find(vowel) != std::string::npos) found1 = true;
            if (ipa2.find(vowel) != std::string::npos) found2 = true;
        }
        if (found1 && found2) {
            return 0.85f; // High similarity for same vowel group
        }
    }

    // Check consonant similarity
    std::map<std::string, std::vector<std::string>> consonantGroups = {
        {"stops", {"p", "b", "t", "d", "k", "g"}},
        {"fricatives", {"f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h"}},
        {"nasals", {"m", "n", "ŋ"}},
        {"liquids", {"l", "r"}},
        {"glides", {"w", "j"}}
    };

    for (const auto& group : consonantGroups) {
        bool found1 = false, found2 = false;
        for (const auto& consonant : group.second) {
            if (ipa1.find(consonant) != std::string::npos) found1 = true;
            if (ipa2.find(consonant) != std::string::npos) found2 = true;
        }
        if (found1 && found2) {
            return 0.7f; // Moderate similarity for same consonant group
        }
    }

    return 0.0f; // No similarity
}

} // namespace kelly
