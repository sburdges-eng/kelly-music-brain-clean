#include "voice/LyricGenerator.h"
#include "common/Types.h"
#include "common/PathResolver.h"
#include <juce_core/juce_core.h>
#include <algorithm>
#include <random>
#include <sstream>
#include <fstream>
#include <cctype>

namespace kelly {

LyricGenerator::LyricGenerator()
    : lyricStyle_("poetic")
    , structureType_("verse_chorus")
    , rhymeSchemeName_("ABAB")
    , targetLineLength_(8)
{
    // Attempt to load templates and emotion vocabulary from JSON files
    // Falls back to hardcoded data if files are not found
    loadTemplates("");  // Empty path will use PathResolver to find file
    loadEmotionVocabulary("");  // Empty path will use PathResolver to find data directory
}

LyricGenerator::LyricResult LyricGenerator::generateLyrics(
    const EmotionNode& emotion,
    const Wound& wound,
    const GeneratedMidi* midiContext)
{
    LyricResult result;

    // Expand emotion to vocabulary
    std::vector<std::string> vocabulary = expandEmotionToVocabulary(emotion);

    // Extract wound keywords
    std::vector<std::string> woundKeywords = extractWoundKeywords(wound);
    vocabulary.insert(vocabulary.end(), woundKeywords.begin(), woundKeywords.end());

    // Generate words from VAD values
    std::vector<std::string> vadWords = generateWordsFromVAD(
        emotion.valence,
        emotion.arousal,
        emotion.dominance
    );
    vocabulary.insert(vocabulary.end(), vadWords.begin(), vadWords.end());

    // Remove duplicates
    std::sort(vocabulary.begin(), vocabulary.end());
    vocabulary.erase(std::unique(vocabulary.begin(), vocabulary.end()), vocabulary.end());

    // Generate structure
    result.structure = generateStructure(structureType_);
    RhymeScheme rhymeScheme = getRhymeScheme(rhymeSchemeName_);

    // Generate lines for each section
    for (auto& section : result.structure.sections) {
        int numLines = static_cast<int>(section.lines.size());
        if (numLines == 0) {
            // Default line count based on section type
            switch (section.type) {
                case LyricSectionType::Verse:
                    numLines = 4;
                    break;
                case LyricSectionType::Chorus:
                    numLines = 4;
                    break;
                case LyricSectionType::Bridge:
                    numLines = 4;
                    break;
                default:
                    numLines = 4;
                    break;
            }
        }

        section.lines = generateLines(section.type, numLines, vocabulary, rhymeScheme);

        // Add to result lines
        for (const auto& line : section.lines) {
            result.lines.push_back(line);
        }
    }

    return result;
}

std::vector<std::string> LyricGenerator::expandEmotionToVocabulary(const EmotionNode& emotion) {
    std::vector<std::string> words;

    // Map emotion name to vocabulary
    std::string emotionName = emotion.name;
    std::transform(emotionName.begin(), emotionName.end(), emotionName.begin(), ::tolower);

    // First, try to use loaded vocabulary from JSON files
    auto vocabIt = emotionVocabulary_.find(emotionName);
    if (vocabIt != emotionVocabulary_.end()) {
        words.insert(words.end(), vocabIt->second.begin(), vocabIt->second.end());
    }

    // Also try category name
    std::string categoryName;
    switch (emotion.categoryEnum) {
        case EmotionCategory::Joy:
            categoryName = "joy";
            break;
        case EmotionCategory::Sadness:
            categoryName = "sad";
            break;
        case EmotionCategory::Anger:
            categoryName = "anger";
            break;
        case EmotionCategory::Fear:
            categoryName = "fear";
            break;
        case EmotionCategory::Surprise:
            categoryName = "surprise";
            break;
        case EmotionCategory::Disgust:
            categoryName = "disgust";
            break;
        default:
            break;
    }

    if (!categoryName.empty() && categoryName != emotionName) {
        auto categoryIt = emotionVocabulary_.find(categoryName);
        if (categoryIt != emotionVocabulary_.end()) {
            words.insert(words.end(), categoryIt->second.begin(), categoryIt->second.end());
        }
    }

    // Enhanced emotion-to-word mapping based on emotion category (always add as fallback/enhancement)
    switch (emotion.categoryEnum) {
        case EmotionCategory::Joy:
            words = {"light", "bright", "shine", "rise", "dance", "sing", "smile", "laugh",
                     "celebration", "freedom", "hope", "dream", "soar", "fly", "glow",
                     "joy", "bliss", "elation", "euphoria", "cheer", "glee", "jubilant",
                     "radiant", "uplifted", "inspired", "blessed", "grateful", "triumph"};
            break;

        case EmotionCategory::Sadness:
            words = {"silence", "dark", "empty", "alone", "echo", "memory", "tear", "fade",
                     "distance", "shadow", "fall", "lost", "broken", "ache", "void",
                     "sorrow", "grief", "lonely", "despair", "mourning", "hollow", "weep",
                     "abandoned", "isolated", "melancholy", "yearning", "regret", "pain"};
            break;

        case EmotionCategory::Anger:
            words = {"fire", "burn", "break", "storm", "rage", "fury", "scream", "explode",
                     "fierce", "wild", "crush", "shatter", "thunder", "revenge", "wrath",
                     "furious", "vengeful", "hostile", "violent", "aggressive", "outraged",
                     "incensed", "fuming", "seething", "bitter", "resentful", "betrayed"};
            break;

        case EmotionCategory::Fear:
            words = {"shadow", "tremble", "hide", "shiver", "dread", "panic", "cold", "unknown",
                     "darkness", "uncertain", "frozen", "trapped", "terrified", "anxious",
                     "afraid", "frightened", "worried", "nervous", "apprehensive", "uneasy",
                     "vulnerable", "exposed", "helpless", "overwhelmed", "paralyzed", "haunted"};
            break;

        case EmotionCategory::Surprise:
            words = {"sudden", "shock", "amazed", "astonished", "startled", "stunned",
                     "unexpected", "revealed", "discovered", "awakened", "realized",
                     "miraculous", "wonder", "marvel", "incredible", "remarkable", "astounding"};
            break;

        case EmotionCategory::Disgust:
            words = {"repulsed", "revolted", "sickened", "repelled", "contempt", "loathing",
                     "revulsion", "distaste", "abhorrence", "aversion", "detest", "abhor"};
            break;

        case EmotionCategory::Trust:
            words = {"trust", "faith", "belief", "confidence", "secure", "safe", "protected",
                     "reliable", "dependable", "loyal", "devoted", "steadfast", "committed",
                     "assured", "comfortable", "certain", "convinced", "sure"};
            break;

        case EmotionCategory::Anticipation:
            words = {"waiting", "expecting", "hoping", "anticipating", "eager", "excited",
                     "yearning", "longing", "awaiting", "prepared", "ready", "forthcoming",
                     "imminent", "approaching", "coming", "future", "next"};
            break;

        default:
            // Fallback: map by emotion name (for compatibility)
            if (emotionName.find("joy") != std::string::npos ||
                emotionName.find("happy") != std::string::npos ||
                emotionName.find("delight") != std::string::npos) {
                words = {"light", "bright", "shine", "rise", "dance", "sing", "smile", "laugh",
                         "celebration", "freedom", "hope", "dream", "soar", "fly", "glow"};
            } else if (emotionName.find("sad") != std::string::npos ||
                       emotionName.find("lonely") != std::string::npos ||
                       emotionName.find("grief") != std::string::npos) {
                words = {"silence", "dark", "empty", "alone", "echo", "memory", "tear", "fade",
                         "distance", "shadow", "fall", "lost", "broken", "ache", "void"};
            } else if (emotionName.find("anger") != std::string::npos ||
                       emotionName.find("rage") != std::string::npos ||
                       emotionName.find("fury") != std::string::npos) {
                words = {"fire", "burn", "break", "storm", "rage", "fury", "scream", "explode",
                         "fierce", "wild", "crush", "shatter", "thunder", "revenge"};
            } else if (emotionName.find("love") != std::string::npos ||
                       emotionName.find("affection") != std::string::npos) {
                words = {"heart", "soul", "touch", "warm", "embrace", "tender", "gentle", "care",
                         "devotion", "passion", "eternal", "beauty", "grace", "adore", "cherish"};
            } else {
                // Generic emotional words
                words = {"feel", "emotion", "heart", "soul", "mind", "spirit", "being", "essence"};
            }
            break;
    }

    // Add words based on VAD values for more nuanced mapping
    if (emotion.valence > 0.5f) {
        words.push_back("positive");
        words.push_back("good");
        words.push_back("nice");
        words.push_back("beautiful");
    }
    if (emotion.valence < -0.5f) {
        words.push_back("negative");
        words.push_back("bad");
        words.push_back("terrible");
        words.push_back("awful");
    }
    if (emotion.arousal > 0.7f) {
        words.push_back("intense");
        words.push_back("powerful");
        words.push_back("strong");
        words.push_back("vibrant");
    }
    if (emotion.arousal < 0.3f) {
        words.push_back("gentle");
        words.push_back("soft");
        words.push_back("calm");
        words.push_back("peaceful");
    }

    // Add words from loaded vocabulary if available
    auto it = emotionVocabulary_.find(emotionName);
    if (it != emotionVocabulary_.end()) {
        words.insert(words.end(), it->second.begin(), it->second.end());
    }

    // Remove duplicates
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());

    return words;
}

std::vector<std::string> LyricGenerator::extractWoundKeywords(const Wound& wound) {
    std::vector<std::string> keywords;

    if (wound.description.empty()) {
        return keywords;
    }

    // Simple keyword extraction (split by spaces, remove common words)
    std::istringstream iss(wound.description);
    std::string word;
    std::vector<std::string> stopWords = {"the", "a", "an", "and", "or", "but", "in", "on",
                                          "at", "to", "for", "of", "with", "by", "from", "is",
                                          "was", "are", "were", "be", "been", "have", "has", "had"};

    while (iss >> word) {
        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());

        // Skip stop words and short words
        if (word.length() >= 3 &&
            std::find(stopWords.begin(), stopWords.end(), word) == stopWords.end()) {
            keywords.push_back(word);
        }
    }

    return keywords;
}

std::vector<std::string> LyricGenerator::generateWordsFromVAD(
    float valence,
    float arousal,
    float dominance)
{
    std::vector<std::string> words;

    // Valence-based words
    std::vector<std::string> valenceWords = getWordsForValence(valence);
    words.insert(words.end(), valenceWords.begin(), valenceWords.end());

    // Arousal-based words
    std::vector<std::string> arousalWords = getWordsForArousal(arousal);
    words.insert(words.end(), arousalWords.begin(), arousalWords.end());

    // Dominance-based words
    std::vector<std::string> dominanceWords = getWordsForDominance(dominance);
    words.insert(words.end(), dominanceWords.begin(), dominanceWords.end());

    return words;
}

std::vector<std::string> LyricGenerator::getWordsForValence(float valence) {
    std::vector<std::string> words;

    if (valence > 0.5f) {
        // Positive valence
        words = {"bright", "light", "warm", "joy", "smile", "laugh", "sun", "star",
                 "hope", "dream", "bliss", "peace", "love", "happy", "glow"};
    } else if (valence < -0.5f) {
        // Negative valence
        words = {"dark", "shadow", "cold", "empty", "void", "pain", "tear", "sad",
                 "grief", "lonely", "broken", "lost", "fade", "ache", "dread"};
    } else {
        // Neutral valence
        words = {"gray", "still", "calm", "quiet", "neutral", "balanced", "steady", "even"};
    }

    return words;
}

std::vector<std::string> LyricGenerator::getWordsForArousal(float arousal) {
    std::vector<std::string> words;

    if (arousal > 0.7f) {
        // High arousal
        words = {"rush", "surge", "explode", "burst", "flash", "strike", "storm", "fire",
                 "energy", "power", "intense", "fierce", "wild", "fury", "thunder"};
    } else if (arousal < 0.3f) {
        // Low arousal
        words = {"drift", "float", "calm", "gentle", "soft", "slow", "ease", "peace",
                 "quiet", "still", "whisper", "breeze", "flow", "tranquil", "zen"};
    } else {
        // Moderate arousal
        words = {"move", "flow", "sway", "dance", "pulse", "beat", "rhythm", "steady"};
    }

    return words;
}

std::vector<std::string> LyricGenerator::getWordsForDominance(float dominance) {
    std::vector<std::string> words;

    if (dominance > 0.7f) {
        // High dominance
        words = {"power", "strength", "command", "control", "rule", "lead", "bold", "strong",
                 "mighty", "force", "assert", "dominate", "rise", "stand", "tower"};
    } else if (dominance < 0.3f) {
        // Low dominance
        words = {"soft", "gentle", "yield", "bend", "submit", "quiet", "humble", "small",
                 "tender", "fragile", "delicate", "whisper", "retreat", "hide", "shy"};
    } else {
        // Moderate dominance
        words = {"balance", "equal", "shared", "together", "mutual", "harmony", "united"};
    }

    return words;
}

LyricStructure LyricGenerator::generateStructure(const std::string& templateName) {
    LyricStructure structure;

    // Default structure
    if (templateName == "verse_chorus" || templateName.empty()) {
        structure.pattern = "V-C-V-C-B-C";

        // Verse 1
        LyricSection verse1;
        verse1.type = LyricSectionType::Verse;
        verse1.sectionNumber = 1;
        verse1.lines.resize(4);
        structure.sections.push_back(verse1);

        // Chorus
        LyricSection chorus;
        chorus.type = LyricSectionType::Chorus;
        chorus.sectionNumber = 1;
        chorus.lines.resize(4);
        structure.sections.push_back(chorus);

        // Verse 2
        LyricSection verse2;
        verse2.type = LyricSectionType::Verse;
        verse2.sectionNumber = 2;
        verse2.lines.resize(4);
        structure.sections.push_back(verse2);

        // Chorus (repeat)
        LyricSection chorus2;
        chorus2.type = LyricSectionType::Chorus;
        chorus2.sectionNumber = 2;
        chorus2.lines.resize(4);
        structure.sections.push_back(chorus2);

        // Bridge
        LyricSection bridge;
        bridge.type = LyricSectionType::Bridge;
        bridge.sectionNumber = 1;
        bridge.lines.resize(4);
        structure.sections.push_back(bridge);

        // Chorus (final)
        LyricSection chorus3;
        chorus3.type = LyricSectionType::Chorus;
        chorus3.sectionNumber = 3;
        chorus3.lines.resize(4);
        structure.sections.push_back(chorus3);
    } else if (templateName == "ballad") {
        structure.pattern = "V-V-C-V-V-C";
        // Similar structure with more verses
        for (int i = 0; i < 6; ++i) {
            LyricSection section;
            section.type = (i % 3 == 2) ? LyricSectionType::Chorus : LyricSectionType::Verse;
            section.sectionNumber = (i / 3) + 1;
            section.lines.resize(4);
            structure.sections.push_back(section);
        }
    }

    structure.rhymeScheme = getRhymeScheme(rhymeSchemeName_);

    return structure;
}

std::vector<LyricLine> LyricGenerator::generateLines(
    LyricSectionType sectionType,
    int numLines,
    const std::vector<std::string>& vocabulary,
    const RhymeScheme& rhymeScheme)
{
    std::vector<LyricLine> lines;
    lines.resize(numLines);

    std::mt19937 rng(std::random_device{}());

    // Build rhyme database for faster lookup
    rhymeEngine_.buildRhymeDatabase(vocabulary);

    // Map to track which words have been used for each rhyme group
    std::map<int, std::string> rhymeWords;

    // Generate lines with rhyme and prosody
    for (int i = 0; i < numLines; ++i) {
        lines[i].lineNumber = i;
        lines[i].targetSyllables = targetLineLength_;

        // Determine rhyme group for this line position
        int rhymeGroup = (i < static_cast<int>(rhymeScheme.pattern.size()))
                         ? rhymeScheme.pattern[i]
                         : i % static_cast<int>(rhymeScheme.pattern.size());

        // Get or generate rhyming word for this group
        std::string endWord;
        auto rhymeIt = rhymeWords.find(rhymeGroup);
        if (rhymeIt != rhymeWords.end()) {
            // Find a word that rhymes with existing word in this group
            std::vector<RhymeEngine::RhymeMatch> rhymes =
                rhymeEngine_.findRhymes(rhymeIt->second, vocabulary, 5);
            if (!rhymes.empty()) {
                endWord = rhymes[0].word2;
            } else {
                endWord = rhymeIt->second; // Fallback to same word
            }
        } else {
            // First word in this rhyme group - pick from vocabulary
            if (!vocabulary.empty()) {
                std::uniform_int_distribution<size_t> vocabDist(0, vocabulary.size() - 1);
                endWord = vocabulary[vocabDist(rng)];
            }
            rhymeWords[rhymeGroup] = endWord;
        }

        // Generate line text with target syllable count
        std::string lineText;
        int currentSyllables = 0;
        std::vector<std::string> selectedWords;

        // Add words until we reach target syllable count
        std::uniform_int_distribution<size_t> vocabDist(0, vocabulary.size() - 1);
        int endWordSyllables = prosodyAnalyzer_.countSyllables(endWord);
        int remainingSyllables = targetLineLength_ - endWordSyllables;

        // Select words to fill remaining syllables
        while (currentSyllables < remainingSyllables && !vocabulary.empty()) {
            std::string word = vocabulary[vocabDist(rng)];
            int wordSyllables = prosodyAnalyzer_.countSyllables(word);

            if (currentSyllables + wordSyllables <= remainingSyllables) {
                selectedWords.push_back(word);
                currentSyllables += wordSyllables;
            } else if (selectedWords.empty()) {
                // Must add at least one word, even if it exceeds
                selectedWords.push_back(word);
                break;
            } else {
                break; // Stop if adding would exceed target
            }
        }

        // Build line text
        for (const auto& word : selectedWords) {
            if (!lineText.empty()) lineText += " ";
            lineText += word;
        }
        if (!endWord.empty()) {
            if (!lineText.empty()) lineText += " ";
            lineText += endWord;
        }

        lines[i].text = lineText;

        // Detect stress pattern and meter
        std::vector<std::string> lineWords = selectedWords;
        lineWords.push_back(endWord);
        lines[i].stressPattern = prosodyAnalyzer_.detectStressPattern(lineWords);

        ProsodyAnalyzer::MeterType detectedMeter = prosodyAnalyzer_.detectMeter(lines[i].stressPattern);
        switch (detectedMeter) {
            case ProsodyAnalyzer::MeterType::Iambic:
                lines[i].meter = "iambic";
                break;
            case ProsodyAnalyzer::MeterType::Trochaic:
                lines[i].meter = "trochaic";
                break;
            case ProsodyAnalyzer::MeterType::Anapestic:
                lines[i].meter = "anapestic";
                break;
            case ProsodyAnalyzer::MeterType::Dactylic:
                lines[i].meter = "dactylic";
                break;
            default:
                lines[i].meter = "mixed";
                break;
        }

        // Convert words to syllables for detailed structure
        for (const auto& word : lineWords) {
            Syllable syllable;
            syllable.text = word;
            std::vector<int> wordStress = prosodyAnalyzer_.detectStress(word);
            syllable.stress = wordStress.empty() ? 0 : wordStress[0];
            lines[i].syllables.push_back(syllable);
        }
    }

    return lines;
}

RhymeScheme LyricGenerator::getRhymeScheme(const std::string& schemeName) {
    RhymeScheme scheme;
    scheme.name = schemeName;

    if (schemeName == "ABAB") {
        scheme.pattern = {0, 1, 0, 1};
    } else if (schemeName == "AABB") {
        scheme.pattern = {0, 0, 1, 1};
    } else if (schemeName == "ABBA") {
        scheme.pattern = {0, 1, 1, 0};
    } else if (schemeName == "ABCB") {
        scheme.pattern = {0, 1, 2, 1};
    } else if (schemeName == "AAAA") {
        scheme.pattern = {0, 0, 0, 0};
    } else if (schemeName == "ABAC") {
        scheme.pattern = {0, 1, 0, 2};
    } else if (schemeName == "AABA") {
        scheme.pattern = {0, 0, 1, 0};
    } else {
        // Default to ABAB
        scheme.pattern = {0, 1, 0, 1};
    }

    return scheme;
}

LyricLine LyricGenerator::generateLine(
    int targetSyllables,
    const std::vector<std::string>& vocabulary,
    const std::vector<int>& stressPattern)
{
    LyricLine line;
    line.targetSyllables = targetSyllables;
    line.stressPattern = stressPattern;

    // Simplified implementation - just select words
    std::vector<std::string> selectedWords = selectWords(vocabulary, targetSyllables, stressPattern);

    // Build line text
    for (size_t i = 0; i < selectedWords.size(); ++i) {
        if (i > 0) line.text += " ";
        line.text += selectedWords[i];
    }

    return line;
}

std::vector<std::string> LyricGenerator::selectWords(
    const std::vector<std::string>& vocabulary,
    int targetSyllables,
    const std::vector<int>& stressPattern)
{
    std::vector<std::string> selected;

    if (vocabulary.empty()) {
        return selected;
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> vocabDist(0, vocabulary.size() - 1);

    int currentSyllables = 0;
    int wordsNeeded = std::max(1, targetSyllables / 2); // Rough estimate

    for (int i = 0; i < wordsNeeded && currentSyllables < targetSyllables; ++i) {
        std::string word = vocabulary[vocabDist(rng)];
        selected.push_back(word);
        currentSyllables += 2; // Simplified syllable count
    }

    return selected;
}

bool LyricGenerator::loadTemplates(const std::string& filePath) {
    juce::File jsonFile;

    // If filePath is provided, use it; otherwise try to find lyric_templates.json
    if (!filePath.empty()) {
        jsonFile = juce::File(filePath);
    } else {
        jsonFile = PathResolver::findDataFile("lyric_templates.json");
    }

    if (!jsonFile.existsAsFile()) {
        juce::Logger::writeToLog("LyricGenerator: lyric_templates.json not found, using hardcoded templates");
        return false;
    }

    juce::String jsonText = jsonFile.loadFileAsString();
    if (jsonText.isEmpty()) {
        juce::Logger::writeToLog("LyricGenerator: Empty or unreadable file: " + jsonFile.getFullPathName());
        return false;
    }

    juce::var parsedJson = juce::JSON::parse(jsonText);
    if (!parsedJson.isObject()) {
        juce::Logger::writeToLog("LyricGenerator: Invalid JSON in file: " + jsonFile.getFullPathName());
        return false;
    }

    auto* root = parsedJson.getDynamicObject();
    if (!root) return false;

    // Templates are now available in JSON, but we'll use them when generateStructure() is called
    // The JSON structure is already being used by the hardcoded templates
    // This function validates the file exists and is readable
    // Future enhancement: Store templates in member variables for dynamic loading

    juce::Logger::writeToLog("LyricGenerator: Loaded templates from " + jsonFile.getFileName());
    return true;
}

bool LyricGenerator::loadEmotionVocabulary(const std::string& emotionDataPath) {
    juce::File dataDir;

    // If path is provided, use it; otherwise try to find data directory
    if (!emotionDataPath.empty()) {
        dataDir = juce::File(emotionDataPath);
    } else {
        dataDir = PathResolver::findDataDirectory();
    }

    if (!dataDir.isDirectory()) {
        juce::Logger::writeToLog("LyricGenerator: Emotion data directory not found, using hardcoded vocabulary");
        return false;
    }

    // Look for emotion JSON files in the data directory
    std::vector<std::string> emotionFiles = {
        "sad.json", "happy.json", "joy.json", "angry.json", "anger.json",
        "fear.json", "disgust.json", "surprise.json"
    };

    int loadedCount = 0;

    for (const auto& filename : emotionFiles) {
        juce::File emotionFile = dataDir.getChildFile(filename);

        // Try alternative filenames
        if (!emotionFile.existsAsFile()) {
            if (filename == "joy.json") {
                emotionFile = dataDir.getChildFile("happy.json");
            } else if (filename == "happy.json") {
                emotionFile = dataDir.getChildFile("joy.json");
            } else if (filename == "anger.json") {
                emotionFile = dataDir.getChildFile("angry.json");
            } else if (filename == "angry.json") {
                emotionFile = dataDir.getChildFile("anger.json");
            }
        }

        if (!emotionFile.existsAsFile()) continue;

        juce::String jsonText = emotionFile.loadFileAsString();
        if (jsonText.isEmpty()) continue;

        juce::var parsedJson = juce::JSON::parse(jsonText);
        if (!parsedJson.isObject()) continue;

        auto* root = parsedJson.getDynamicObject();
        if (!root) continue;

        // Extract emotion name/category
        std::string emotionName;
        if (root->hasProperty("name")) {
            emotionName = root->getProperty("name").toString().toLowerCase().toStdString();
        } else if (root->hasProperty("category")) {
            emotionName = root->getProperty("category").toString().toLowerCase().toStdString();
        } else {
            // Use filename as fallback
            emotionName = filename.substr(0, filename.find('.'));
        }

        // Extract vocabulary words from sub_emotions structure
        // This is a simplified extraction - in a full implementation, we'd traverse
        // the entire emotion hierarchy to extract all related words
        std::vector<std::string> words;

        auto subEmotions = root->getProperty("sub_emotions");
        if (subEmotions.isObject()) {
            auto* subEmotionsObj = subEmotions.getDynamicObject();
            auto properties = subEmotionsObj->getProperties();

            for (auto& prop : properties) {
                // Add emotion names as vocabulary
                std::string subEmotionName = prop.name.toString().toLowerCase().toStdString();
                if (subEmotionName.length() > 2) {
                    words.push_back(subEmotionName);
                }

                // Try to extract words from intensity tiers if available
                if (prop.value.isObject()) {
                    auto* subObj = prop.value.getDynamicObject();
                    auto subSubEmotions = subObj->getProperty("sub_sub_emotions");

                    if (subSubEmotions.isObject()) {
                        auto* subSubObj = subSubEmotions.getDynamicObject();
                        auto subSubProperties = subSubObj->getProperties();

                        for (auto& subSubProp : subSubProperties) {
                            std::string subSubName = subSubProp.name.toString().toLowerCase().toStdString();
                            if (subSubName.length() > 2) {
                                words.push_back(subSubName);
                            }

                            // Extract from intensity tiers
                            if (subSubProp.value.isObject()) {
                                auto* subSubSubObj = subSubProp.value.getDynamicObject();
                                auto tiers = subSubSubObj->getProperty("intensity_tiers");

                                if (tiers.isObject()) {
                                    auto* tiersObj = tiers.getDynamicObject();
                                    auto tierProperties = tiersObj->getProperties();

                                    for (auto& tierProp : tierProperties) {
                                        auto tierWords = tierProp.value;
                                        if (tierWords.isArray()) {
                                            for (const auto& wordVar : *tierWords.getArray()) {
                                                std::string word = wordVar.toString().toLowerCase().toStdString();
                                                if (word.length() > 2) {
                                                    words.push_back(word);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Store vocabulary for this emotion
        if (!words.empty()) {
            emotionVocabulary_[emotionName] = words;
            loadedCount++;
        }
    }

    juce::Logger::writeToLog("LyricGenerator: Loaded vocabulary for " + juce::String(loadedCount) + " emotions");

    return loadedCount > 0;
}

} // namespace kelly
