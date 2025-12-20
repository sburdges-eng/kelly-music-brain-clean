#pragma once

#include <juce_core/juce_core.h>
#include <vector>
#include <map>
#include <optional>

namespace kelly {

/**
 * EQ Preset Manager
 * 
 * Manages EQ presets based on emotions and genres.
 * Loads presets from JSON file and provides lookup functionality.
 */
class EQPresetManager {
public:
    /**
     * EQ Band definition
     */
    struct EQBand {
        float frequency;  // Hz
        float gain;       // dB
        float q;          // Quality factor
    };
    
    /**
     * EQ Preset definition
     */
    struct EQPreset {
        juce::String name;
        juce::String description;
        std::vector<EQBand> bands;
    };
    
    EQPresetManager();
    ~EQPresetManager() = default;
    
    /**
     * Load presets from JSON file
     */
    bool loadPresets(const juce::File& jsonFile);
    
    /**
     * Get emotion-based preset
     */
    std::optional<EQPreset> getEmotionPreset(const juce::String& emotion) const;
    
    /**
     * Get genre-based preset
     */
    std::optional<EQPreset> getGenrePreset(const juce::String& genre) const;
    
    /**
     * Get preset by name (searches both emotion and genre)
     */
    std::optional<EQPreset> getPresetByName(const juce::String& name) const;
    
    /**
     * Get all emotion preset names
     */
    std::vector<juce::String> getEmotionPresetNames() const;
    
    /**
     * Get all genre preset names
     */
    std::vector<juce::String> getGenrePresetNames() const;
    
    /**
     * Get preset based on valence/arousal/intensity
     * Maps emotional coordinates to appropriate EQ preset
     */
    std::optional<EQPreset> getPresetForEmotion(float valence, float arousal, float intensity) const;
    
    /**
     * Blend two presets
     */
    EQPreset blendPresets(const EQPreset& preset1, const EQPreset& preset2, float blendFactor) const;
    
private:
    std::map<juce::String, EQPreset> emotionPresets_;
    std::map<juce::String, EQPreset> genrePresets_;
    
    bool parsePreset(const juce::var& presetData, EQPreset& preset);
    juce::String normalizeEmotionName(const juce::String& emotion) const;
    juce::String normalizeGenreName(const juce::String& genre) const;
};

} // namespace kelly
