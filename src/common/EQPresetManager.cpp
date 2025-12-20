#include "EQPresetManager.h"
#include "common/PathResolver.h"
#include <juce_core/juce_core.h>

namespace kelly {

EQPresetManager::EQPresetManager() {
    // Use centralized path resolver
    juce::File presetFile = PathResolver::findDataFile("eq_presets.json");
    
    if (presetFile.exists()) {
        loadPresets(presetFile);
    }
    // Note: No embedded fallback for EQ presets - they're optional
}

bool EQPresetManager::loadPresets(const juce::File& jsonFile) {
    if (!jsonFile.existsAsFile()) {
        return false;
    }
    
    juce::var jsonData;
    juce::Result result = juce::JSON::parse(jsonFile.loadFileAsString(), jsonData);
    
    if (result.failed()) {
        return false;
    }
    
    // Load emotion presets
    if (jsonData.hasProperty("emotion_presets")) {
        auto emotionPresets = jsonData["emotion_presets"];
        if (emotionPresets.isObject()) {
            auto* obj = emotionPresets.getDynamicObject();
            if (obj != nullptr) {
                for (auto& prop : obj->getProperties()) {
                    EQPreset preset;
                    if (parsePreset(prop.value, preset)) {
                        emotionPresets_[normalizeEmotionName(prop.name.toString())] = preset;
                    }
                }
            }
        }
    }
    
    // Load genre presets
    if (jsonData.hasProperty("genre_presets")) {
        auto genrePresets = jsonData["genre_presets"];
        if (genrePresets.isObject()) {
            auto* obj = genrePresets.getDynamicObject();
            if (obj != nullptr) {
                for (auto& prop : obj->getProperties()) {
                    EQPreset preset;
                    if (parsePreset(prop.value, preset)) {
                        genrePresets_[normalizeGenreName(prop.name.toString())] = preset;
                    }
                }
            }
        }
    }
    
    return true;
}

bool EQPresetManager::parsePreset(const juce::var& presetData, EQPreset& preset) {
    if (!presetData.isObject()) {
        return false;
    }
    
    auto* obj = presetData.getDynamicObject();
    if (obj == nullptr) {
        return false;
    }
    
    preset.name = obj->getProperty("name").toString();
    preset.description = obj->getProperty("description").toString();
    
    if (obj->hasProperty("bands") && obj->getProperty("bands").isArray()) {
        auto bands = obj->getProperty("bands");
        auto* bandsArray = bands.getArray();
        if (bandsArray != nullptr) {
            for (auto& bandVar : *bandsArray) {
                if (bandVar.isObject()) {
                    auto* bandObj = bandVar.getDynamicObject();
                    if (bandObj != nullptr) {
                        EQBand band;
                        band.frequency = static_cast<float>(bandObj->getProperty("frequency"));
                        band.gain = static_cast<float>(bandObj->getProperty("gain"));
                        band.q = static_cast<float>(bandObj->getProperty("q"));
                        preset.bands.push_back(band);
                    }
                }
            }
        }
    }
    
    return !preset.name.isEmpty() && !preset.bands.empty();
}

std::optional<EQPresetManager::EQPreset> EQPresetManager::getEmotionPreset(const juce::String& emotion) const {
    juce::String normalized = normalizeEmotionName(emotion);
    auto it = emotionPresets_.find(normalized);
    if (it != emotionPresets_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<EQPresetManager::EQPreset> EQPresetManager::getGenrePreset(const juce::String& genre) const {
    juce::String normalized = normalizeGenreName(genre);
    auto it = genrePresets_.find(normalized);
    if (it != genrePresets_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<EQPresetManager::EQPreset> EQPresetManager::getPresetByName(const juce::String& name) const {
    // Try emotion first
    auto emotionPreset = getEmotionPreset(name);
    if (emotionPreset.has_value()) {
        return emotionPreset;
    }
    
    // Try genre
    auto genrePreset = getGenrePreset(name);
    if (genrePreset.has_value()) {
        return genrePreset;
    }
    
    return std::nullopt;
}

std::vector<juce::String> EQPresetManager::getEmotionPresetNames() const {
    std::vector<juce::String> names;
    for (const auto& pair : emotionPresets_) {
        names.push_back(pair.second.name);
    }
    return names;
}

std::vector<juce::String> EQPresetManager::getGenrePresetNames() const {
    std::vector<juce::String> names;
    for (const auto& pair : genrePresets_) {
        names.push_back(pair.second.name);
    }
    return names;
}

std::optional<EQPresetManager::EQPreset> EQPresetManager::getPresetForEmotion(float valence, float arousal, float intensity) const {
    // Map emotional coordinates to preset names
    juce::String presetKey;
    
    // Primary decision: valence
    if (valence < -0.5f) {
        // Negative valence
        if (arousal > 0.7f) {
            presetKey = "anger";
        } else if (intensity > 0.7f) {
            presetKey = "grief";
        } else {
            presetKey = "sadness";
        }
    } else if (valence > 0.5f) {
        // Positive valence
        if (arousal > 0.7f) {
            presetKey = "triumph";
        } else if (arousal > 0.4f) {
            presetKey = "joy";
        } else {
            presetKey = "peace";
        }
    } else {
        // Neutral valence
        if (arousal > 0.7f) {
            presetKey = "anxiety";
        } else if (arousal < 0.3f) {
            presetKey = "peace";
        } else {
            presetKey = "hope";
        }
    }
    
    return getEmotionPreset(presetKey);
}

EQPresetManager::EQPreset EQPresetManager::blendPresets(const EQPreset& preset1, const EQPreset& preset2, float blendFactor) const {
    EQPreset blended;
    blended.name = preset1.name + " / " + preset2.name;
    blended.description = "Blended preset";
    
    // Use the maximum number of bands
    size_t maxBands = std::max(preset1.bands.size(), preset2.bands.size());
    
    for (size_t i = 0; i < maxBands; ++i) {
        EQBand band;
        
        if (i < preset1.bands.size() && i < preset2.bands.size()) {
            // Both presets have this band - blend
            const auto& b1 = preset1.bands[i];
            const auto& b2 = preset2.bands[i];
            
            band.frequency = b1.frequency * (1.0f - blendFactor) + b2.frequency * blendFactor;
            band.gain = b1.gain * (1.0f - blendFactor) + b2.gain * blendFactor;
            band.q = b1.q * (1.0f - blendFactor) + b2.q * blendFactor;
        } else if (i < preset1.bands.size()) {
            // Only preset1 has this band
            band = preset1.bands[i];
            band.gain *= (1.0f - blendFactor);
        } else {
            // Only preset2 has this band
            band = preset2.bands[i];
            band.gain *= blendFactor;
        }
        
        blended.bands.push_back(band);
    }
    
    return blended;
}

juce::String EQPresetManager::normalizeEmotionName(const juce::String& emotion) const {
    return emotion.toLowerCase().removeCharacters(" -_");
}

juce::String EQPresetManager::normalizeGenreName(const juce::String& genre) const {
    juce::String normalized = genre.toLowerCase().removeCharacters(" -_");
    // Handle special cases
    if (normalized == "randb" || normalized == "r&b" || normalized == "rnb") {
        return "r&b";
    }
    return normalized;
}

} // namespace kelly
