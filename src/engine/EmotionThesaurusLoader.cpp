#include "engine/EmotionThesaurusLoader.h"
#include "common/PathResolver.h"
#include "common/KellyTypes.h"  // For categoryToString
#include <algorithm>
#include <cmath>

namespace kelly {

int EmotionThesaurusLoader::loadWithFallbacks(EmotionThesaurus& thesaurus) {
    int nextId = 1;
    int totalLoaded = 0;

    // Try to find data directory using centralized PathResolver
    juce::File dataDir = PathResolver::findDataDirectory();

    if (dataDir.isDirectory()) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: Using data directory: " + dataDir.getFullPathName());
        totalLoaded = loadFromJsonFiles(dataDir, thesaurus);
    }

    // If no files loaded, try loading from embedded defaults
    if (totalLoaded == 0) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: No JSON files found, loading embedded defaults");
        totalLoaded = loadFromEmbeddedDefaults(thesaurus);
    }

    return totalLoaded;
}

// Path resolution methods removed - now using PathResolver

std::vector<std::string> EmotionThesaurusLoader::getAlternativeFilenames(const std::string& baseFilename) {
    std::vector<std::string> alternatives;
    alternatives.push_back(baseFilename);

    // Map common variations
    if (baseFilename == "joy.json") {
        alternatives.push_back("happy.json");
    } else if (baseFilename == "happy.json") {
        alternatives.push_back("joy.json");
    } else if (baseFilename == "anger.json") {
        alternatives.push_back("angry.json");
    } else if (baseFilename == "angry.json") {
        alternatives.push_back("anger.json");
    }

    return alternatives;
}

juce::File EmotionThesaurusLoader::findJsonFile(const std::string& baseFilename) {
    auto alternatives = getAlternativeFilenames(baseFilename);

    // Try each alternative filename using PathResolver
    for (const auto& altFilename : alternatives) {
        juce::File jsonFile = PathResolver::findDataFile(altFilename);
        if (jsonFile.existsAsFile()) {
            return jsonFile;
        }
    }

    return juce::File();
}

std::vector<std::string> EmotionThesaurusLoader::getEmotionFilenames() {
    return {
        "sad.json", "happy.json", "joy.json",
        "angry.json", "anger.json",
        "fear.json",
        "disgust.json",
        "surprise.json"
    };
}

int EmotionThesaurusLoader::loadFromJsonFiles(const juce::File& dataDirectory, EmotionThesaurus& thesaurus) {
    if (!dataDirectory.isDirectory()) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: Data directory does not exist: " + dataDirectory.getFullPathName());
        return 0;
    }

    int nextId = 1;
    int totalLoaded = 0;

    // Use unique filenames (avoid duplicates like joy.json and happy.json)
    std::vector<std::string> emotionFiles = {
        "sad.json", "happy.json", "angry.json", "fear.json",
        "disgust.json", "surprise.json"
    };

    for (const auto& filename : emotionFiles) {
        juce::File jsonFile = dataDirectory.getChildFile(filename);

        // Try alternative filenames if primary doesn't exist
        if (!jsonFile.existsAsFile()) {
            jsonFile = findJsonFile(filename);
        }

        if (jsonFile.existsAsFile()) {
            int loaded = loadFromJsonFile(jsonFile, thesaurus, nextId);
            totalLoaded += loaded;
            juce::Logger::writeToLog("Loaded " + juce::String(loaded) + " emotions from " + jsonFile.getFileName());
        } else {
            juce::Logger::writeToLog("EmotionThesaurusLoader: File not found: " + filename);
        }
    }

    return totalLoaded;
}

std::string EmotionThesaurusLoader::extractCategoryName(const juce::var& root) {
    auto* rootObj = root.getDynamicObject();
    if (!rootObj) return "";

    // Try "name" field first (newer format)
    if (rootObj->hasProperty("name")) {
        juce::String name = rootObj->getProperty("name").toString();
        // Convert "SAD", "HAPPY" to lowercase
        return name.toLowerCase().toStdString();
    }

    // Fall back to "category" field (older format)
    if (rootObj->hasProperty("category")) {
        return rootObj->getProperty("category").toString().toStdString();
    }

    return "";
}

int EmotionThesaurusLoader::loadFromJsonFile(const juce::File& jsonFile, EmotionThesaurus& thesaurus, int& nextId) {
    if (!jsonFile.existsAsFile()) {
        return 0;
    }

    juce::String jsonText = jsonFile.loadFileAsString();
    if (jsonText.isEmpty()) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: Empty or unreadable file: " + jsonFile.getFullPathName());
        return 0;
    }

    juce::var parsedJson = juce::JSON::parse(jsonText);
    if (!parsedJson.isObject()) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: Invalid JSON in file: " + jsonFile.getFullPathName());
        return 0;
    }

    std::string categoryName = extractCategoryName(parsedJson);
    if (categoryName.empty()) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: No category/name field found in: " + jsonFile.getFullPathName());
        return 0;
    }

    auto* root = parsedJson.getDynamicObject();
    if (!root) return 0;

    auto* subEmotions = root->getProperty("sub_emotions").getDynamicObject();
    if (!subEmotions) {
        juce::Logger::writeToLog("EmotionThesaurusLoader: No sub_emotions found in: " + jsonFile.getFullPathName());
        return 0;
    }

    int loaded = 0;

    auto properties = subEmotions->getProperties();
    for (auto& prop : properties) {
        if (prop.value.isObject()) {
            int beforeCount = nextId;
            processSubEmotion(prop.value, categoryName, prop.name.toString().toStdString(), thesaurus, nextId);
            loaded += (nextId - beforeCount);
        }
    }

    return loaded;
}

// Overload for backward compatibility
int EmotionThesaurusLoader::loadFromJsonFile(const juce::File& jsonFile, EmotionThesaurus& thesaurus) {
    int nextId = 1;
    return loadFromJsonFile(jsonFile, thesaurus, nextId);
}

EmotionCategory EmotionThesaurusLoader::categoryFromString(const std::string& categoryStr) {
    std::string lower = categoryStr;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "joy" || lower == "happiness" || lower == "happy") return EmotionCategory::Joy;
    if (lower == "sad" || lower == "sadness") return EmotionCategory::Sadness;
    if (lower == "anger" || lower == "angry") return EmotionCategory::Anger;
    if (lower == "fear") return EmotionCategory::Fear;
    if (lower == "surprise") return EmotionCategory::Surprise;
    if (lower == "disgust") return EmotionCategory::Disgust;
    if (lower == "trust" || lower == "love") return EmotionCategory::Trust;
    if (lower == "anticipation") return EmotionCategory::Anticipation;
    return EmotionCategory::Sadness;
}

float EmotionThesaurusLoader::valenceFromString(const std::string& valenceStr) {
    if (valenceStr == "positive") return 0.7f;
    if (valenceStr == "negative") return -0.7f;
    return 0.0f;
}

float EmotionThesaurusLoader::intensityFromTier(const std::string& tierStr) {
    if (tierStr.find("1_subtle") != std::string::npos) return 0.1f;
    if (tierStr.find("2_mild") != std::string::npos) return 0.3f;
    if (tierStr.find("3_moderate") != std::string::npos) return 0.5f;
    if (tierStr.find("4_strong") != std::string::npos) return 0.7f;
    if (tierStr.find("5_intense") != std::string::npos) return 0.9f;
    if (tierStr.find("6_overwhelming") != std::string::npos) return 1.0f;
    return 0.5f;
}

float EmotionThesaurusLoader::arousalFromIntensity(float intensity, EmotionCategory category) {
    float baseArousal = 0.5f;

    if (category == EmotionCategory::Anger || category == EmotionCategory::Fear) {
        baseArousal = 0.7f + (intensity * 0.3f);
    } else if (category == EmotionCategory::Sadness) {
        baseArousal = 0.2f + (intensity * 0.3f);
    } else {
        baseArousal = 0.4f + (intensity * 0.4f);
    }

    return std::clamp(baseArousal, 0.0f, 1.0f);
}

void EmotionThesaurusLoader::processSubEmotion(
    const juce::var& subData,
    const std::string& categoryName,
    const std::string& subEmotionName,
    EmotionThesaurus& thesaurus,
    int& nextId)
{
    if (!subData.isObject()) return;

    auto* subObj = subData.getDynamicObject();
    if (!subObj) return;

    auto* subSubEmotions = subObj->getProperty("sub_sub_emotions").getDynamicObject();
    if (!subSubEmotions) return;

    auto properties = subSubEmotions->getProperties();
    for (auto& prop : properties) {
        if (prop.value.isObject()) {
            processSubSubEmotion(prop.value, categoryName, subEmotionName,
                                prop.name.toString().toStdString(), thesaurus, nextId);
        }
    }
}

void EmotionThesaurusLoader::processSubSubEmotion(
    const juce::var& subSubData,
    const std::string& categoryName,
    const std::string& subEmotionName,
    const std::string& subSubEmotionName,
    EmotionThesaurus& thesaurus,
    int& nextId)
{
    if (!subSubData.isObject()) return;

    auto* subSubObj = subSubData.getDynamicObject();
    if (!subSubObj) return;

    auto* intensityTiers = subSubObj->getProperty("intensity_tiers").getDynamicObject();
    if (!intensityTiers) {
        // If no intensity tiers, create a single node for this sub-sub-emotion
        // This handles cases where the structure might be different
        EmotionCategory category = categoryFromString(categoryName);
        float valence = valenceFromString(categoryName == "joy" || categoryName == "happiness" ? "positive" : "negative");
        float intensity = 0.5f;
        float arousal = arousalFromIntensity(intensity, category);
        float dominance = 0.5f;

        EmotionNode node;
        node.id = nextId++;
        node.name = subSubEmotionName;
        node.categoryEnum = category;
        node.category = categoryToString(category);
        node.intensity = intensity;
        node.valence = valence;
        node.arousal = arousal;
        node.dominance = dominance;
        node.relatedEmotions = {};

        thesaurus.addNode(node);
        return;
    }

    // Process only the first intensity tier to create ONE node per sub-sub-emotion
    // This ensures we get exactly 216 nodes (6 base × 6 sub × 6 sub-sub)
    auto& properties = intensityTiers->getProperties();
    if (properties.size() > 0) {
        for (int i = 0; i < properties.size(); ++i) {
            auto name = properties.getName(i);
            auto value = properties.getValueAt(i);
            if (value.isArray()) {
                processIntensityTier(value, categoryName, subEmotionName,
                                    subSubEmotionName, name.toString().toStdString(),
                                    thesaurus, nextId);
                break; // Process only first tier
            }
        }
    }
}

void EmotionThesaurusLoader::processIntensityTier(
    const juce::var& tierData,
    const std::string& categoryName,
    const std::string& /* subEmotionName */,
    const std::string& subSubEmotionName,
    const std::string& tierName,
    EmotionThesaurus& thesaurus,
    int& nextId)
{
    if (!tierData.isArray()) return;

    auto* arr = tierData.getArray();
    if (!arr || arr->size() == 0) return;

    EmotionCategory category = categoryFromString(categoryName);

    // Determine valence based on category (handle both lowercase and uppercase)
    std::string lowerCategory = categoryName;
    std::transform(lowerCategory.begin(), lowerCategory.end(), lowerCategory.begin(), ::tolower);

    float valence = 0.0f;
    if (lowerCategory == "joy" || lowerCategory == "happiness" || lowerCategory == "happy") {
        valence = 0.7f;
    } else if (lowerCategory == "sad" || lowerCategory == "sadness" ||
               lowerCategory == "anger" || lowerCategory == "angry" ||
               lowerCategory == "fear" || lowerCategory == "disgust") {
        valence = -0.7f;
    } else if (lowerCategory == "surprise") {
        valence = 0.0f;  // Mixed valence
    }

    float intensity = intensityFromTier(tierName);
    float arousal = arousalFromIntensity(intensity, category);

    // Calculate dominance from VAD model
    // High arousal + positive valence = higher dominance
    // High arousal + negative valence = lower dominance
    float dominance = 0.5f;
    if (valence > 0.0f) {
        dominance = 0.5f + (arousal * 0.3f) + (valence * 0.2f);
    } else {
        dominance = 0.5f - (arousal * 0.2f) + (valence * 0.1f);
    }
    dominance = std::clamp(dominance, 0.0f, 1.0f);

    // Create ONE node per sub-sub-emotion (not per synonym)
    // Use the sub-sub-emotion name as the primary name, first synonym as fallback
    std::string emotionName = subSubEmotionName;
    if (emotionName.empty() && arr->size() > 0) {
        emotionName = arr->getReference(0).toString().toStdString();
    }

    // Capitalize first letter
    if (!emotionName.empty()) {
        emotionName[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(emotionName[0])));
    }

    EmotionNode node;
    node.id = nextId++;
    node.name = emotionName;
    node.categoryEnum = category;
    node.category = categoryToString(category);
    node.intensity = intensity;
    node.valence = valence;
    node.arousal = arousal;
    node.dominance = dominance;
    node.relatedEmotions = {};  // Will be populated later if needed

    thesaurus.addNode(node);

    // Index all synonyms for findByName lookup
    for (int i = 0; i < arr->size(); ++i) {
        std::string synonym = arr->getReference(i).toString().toStdString();
        thesaurus.addSynonym(node.id, synonym);
    }
}

int EmotionThesaurusLoader::loadFromEmbeddedDefaults(EmotionThesaurus& thesaurus) {
    // Embedded minimal JSON defaults as last resort
    // These are simplified versions - full data should come from JSON files

    juce::Logger::writeToLog("EmotionThesaurusLoader: Loading embedded defaults (minimal set)");

    // Note: For a full implementation, you would embed complete JSON strings here
    // For now, we'll return 0 and let EmotionThesaurus::initializeThesaurus() handle it
    // This method provides the hook for future embedded JSON strings

    // Example of how embedded JSON would work:
    // const char* embeddedSadJson = R"({
    //   "name": "SAD",
    //   "sub_emotions": { ... }
    // })";
    // juce::var parsed = juce::JSON::parse(embeddedSadJson);
    // return loadFromJsonString(parsed, thesaurus);

    return 0;  // Signal that embedded defaults weren't used (fallback to hardcoded)
}

} // namespace kelly
