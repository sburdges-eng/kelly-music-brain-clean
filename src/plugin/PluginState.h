#pragma once

#include "common/Types.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_core/juce_core.h>
#include <optional>
#include <vector>
#include <map>

namespace kelly {

/**
 * Plugin State - Manages plugin state persistence and preset management.
 * 
 * Handles:
 * - Saving/loading all plugin settings (parameters, wound description, emotion IDs)
 * - CassetteState (SideA/SideB) persistence
 * - User preset management (save/load/delete/list)
 * - Preset file management in user directory
 */
class PluginState {
public:
    /**
     * Plugin Preset - Complete snapshot of plugin state
     */
    struct Preset {
        juce::String name;
        juce::String description;
        juce::String author;
        juce::Time createdTime;
        juce::Time modifiedTime;
        
        // Parameter values
        float valence = 0.0f;
        float arousal = 0.5f;
        float intensity = 0.5f;
        float complexity = 0.5f;
        float humanize = 0.4f;
        float feel = 0.0f;
        float dynamics = 0.75f;
        int bars = 8;
        bool bypass = false;
        
        // Intent/Emotion state
        juce::String woundDescription;
        std::optional<int> selectedEmotionId;
        
        // Cassette state
        CassetteState cassetteState;
        
        /**
         * Convert preset to JSON ValueTree for serialization
         */
        juce::ValueTree toValueTree() const;
        
        /**
         * Load preset from JSON ValueTree
         */
        static std::optional<Preset> fromValueTree(const juce::ValueTree& tree);
        
        /**
         * Convert preset to JSON var for file storage
         */
        juce::var toJson() const;
        
        /**
         * Load preset from JSON var
         */
        static std::optional<Preset> fromJson(const juce::var& json);
    };
    
    PluginState();
    ~PluginState() = default;
    
    //==============================================================================
    // State Persistence
    //==============================================================================
    
    /**
     * Save current plugin state to ValueTree.
     * Extracts all parameters from APVTS and additional state.
     */
    juce::ValueTree saveState(juce::AudioProcessorValueTreeState& apvts,
                              const juce::String& woundDescription,
                              const std::optional<int>& selectedEmotionId,
                              const CassetteState& cassetteState) const;
    
    /**
     * Load plugin state from ValueTree.
     * Restores all parameters to APVTS and additional state.
     * Returns true if successful.
     */
    bool loadState(juce::AudioProcessorValueTreeState& apvts,
                   juce::String& woundDescription,
                   std::optional<int>& selectedEmotionId,
                   CassetteState& cassetteState,
                   const juce::ValueTree& state) const;
    
    /**
     * Save emotion settings (CassetteState) to state.
     */
    void saveEmotionSettings(juce::ValueTree& state, const CassetteState& cassetteState) const;
    
    /**
     * Load emotion settings (CassetteState) from state.
     */
    CassetteState loadEmotionSettings(const juce::ValueTree& state) const;
    
    //==============================================================================
    // Preset Management
    //==============================================================================
    
    /**
     * Get the user presets directory.
     * Creates directory if it doesn't exist.
     */
    juce::File getPresetsDirectory() const;
    
    /**
     * Get preset file path for a given preset name.
     */
    juce::File getPresetFile(const juce::String& presetName) const;
    
    /**
     * Save current state as a preset.
     * Returns true if successful.
     */
    bool savePreset(const juce::String& presetName,
                    const juce::String& description,
                    const juce::String& author,
                    juce::AudioProcessorValueTreeState& apvts,
                    const juce::String& woundDescription,
                    const std::optional<int>& selectedEmotionId,
                    const CassetteState& cassetteState) const;
    
    /**
     * Load a preset by name.
     * Returns preset if found, nullopt otherwise.
     */
    std::optional<Preset> loadPreset(const juce::String& presetName) const;
    
    /**
     * Delete a preset by name.
     * Returns true if successful.
     */
    bool deletePreset(const juce::String& presetName) const;
    
    /**
     * Get list of all available preset names.
     */
    std::vector<juce::String> getPresetNames() const;
    
    /**
     * Get preset metadata (name, description, author, dates) without loading full state.
     */
    std::optional<Preset> getPresetMetadata(const juce::String& presetName) const;
    
    /**
     * Check if a preset exists.
     */
    bool presetExists(const juce::String& presetName) const;
    
    /**
     * Apply a preset to the plugin.
     * Restores all parameters and state from preset.
     * Returns true if successful.
     */
    bool applyPreset(const Preset& preset,
                     juce::AudioProcessorValueTreeState& apvts,
                     juce::String& woundDescription,
                     std::optional<int>& selectedEmotionId,
                     CassetteState& cassetteState) const;
    
    /**
     * Create preset from current state.
     */
    Preset createPresetFromState(const juce::String& name,
                                  const juce::String& description,
                                  const juce::String& author,
                                  juce::AudioProcessorValueTreeState& apvts,
                                  const juce::String& woundDescription,
                                  const std::optional<int>& selectedEmotionId,
                                  const CassetteState& cassetteState) const;
    
private:
    /**
     * Sanitize preset name for use as filename.
     */
    juce::String sanitizePresetName(const juce::String& name) const;
    
    /**
     * Get parameter value from APVTS.
     */
    float getParameterValue(juce::AudioProcessorValueTreeState& apvts, const char* paramId) const;
    
    /**
     * Set parameter value in APVTS.
     */
    void setParameterValue(juce::AudioProcessorValueTreeState& apvts, const char* paramId, float value) const;
};

} // namespace kelly
