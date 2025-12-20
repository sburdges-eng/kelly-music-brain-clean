#pragma once
/*
 * ProjectManager.h - Full Project Save/Load Management
 * ===================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Plugin Layer: Used by PluginProcessor for project persistence
 * - State Layer: Extends PluginState with full project context
 * - MIDI Layer: Saves/loads GeneratedMidi data
 * - UI Layer: Called from EmotionWorkstation for project menu
 *
 * Purpose: Manages complete project save/load including:
 * - Plugin state (parameters, emotion selections, cassette state)
 * - Generated MIDI data (melody, bass, chords, drums)
 * - Vocal notes and lyrics
 * - 216-node emotion selections
 * - Project metadata (name, created/modified dates, version)
 *
 * Interface Contract (other agents depend on this):
 * - saveProject(): Save complete project to file
 * - loadProject(): Load complete project from file
 * - getLastError(): Get error message if operation failed
 */

#include "common/Types.h"  // GeneratedMidi, EmotionNode, etc.
#include "plugin/PluginState.h"  // PluginState::Preset
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_core/juce_core.h>
#include <optional>
#include <vector>
#include <string>

namespace midikompanion {

// Import kelly types into midikompanion namespace
using kelly::TimeSignature;
using kelly::GeneratedMidi;
using kelly::MidiNote;
using kelly::Chord;

/**
 * Project Manager - Handles complete project save/load functionality.
 *
 * Extends PluginState to include:
 * - Generated MIDI data (all tracks)
 * - Vocal notes and lyrics
 * - 216-node emotion selections
 * - Project metadata
 */
class ProjectManager {
public:
    /**
     * Project data structure - Complete project state
     */
    struct ProjectData {
        // Metadata
        juce::String name = "Untitled Project";
        juce::Time createdTime;
        juce::Time modifiedTime;
        int versionMajor = 1;
        int versionMinor = 0;

        // Project settings
        float tempo = 120.0f;
        TimeSignature timeSignature = {4, 4};

        // Plugin state (uses kelly::PluginState::Preset)
        // Store the full preset to preserve all state including CassetteState
        kelly::PluginState::Preset pluginState;

        // Generated MIDI data
        GeneratedMidi generatedMidi;

        // Vocal synthesis data
        std::vector<MidiNote> vocalNotes;
        std::vector<juce::String> lyrics;  // Lyrics as text lines

        // 216-node emotion selections
        std::vector<int> selectedEmotionIds;  // Multiple emotion selections
        std::optional<int> primaryEmotionId;   // Primary emotion for generation

        /**
         * Convert to JSON ValueTree for serialization
         */
        juce::ValueTree toValueTree() const;

        /**
         * Load from JSON ValueTree
         */
        static std::optional<ProjectData> fromValueTree(const juce::ValueTree& tree);

        /**
         * Convert to JSON var for file storage
         */
        juce::var toJson() const;

        /**
         * Load from JSON var
         */
        static std::optional<ProjectData> fromJson(const juce::var& json);
    };

    ProjectManager();
    ~ProjectManager() = default;

    //==============================================================================
    // Project Save/Load Interface (used by other agents)
    //==============================================================================

    /**
     * Save complete project to file.
     * Includes all plugin state, generated MIDI, vocal notes, lyrics, and emotion selections.
     *
     * @param file Target file to save to
     * @param state PluginStatePreset containing current plugin parameters
     * @param generatedMidi Current generated MIDI data
     * @param vocalNotes Current vocal notes (if any)
     * @param lyrics Current lyrics (if any)
     * @param selectedEmotionIds Current emotion node selections
     * @param primaryEmotionId Primary emotion ID used for generation
     * @return true if successful
     */
    bool saveProject(const juce::File& file,
                     const kelly::PluginState::Preset& state,
                     const GeneratedMidi& generatedMidi,
                     const std::vector<MidiNote>& vocalNotes = {},
                     const std::vector<juce::String>& lyrics = {},
                     const std::vector<int>& selectedEmotionIds = {},
                     const std::optional<int>& primaryEmotionId = std::nullopt);

    /**
     * Load complete project from file.
     * Restores all plugin state, generated MIDI, vocal notes, lyrics, and emotion selections.
     *
     * @param file Source file to load from
     * @param outState Output: PluginStatePreset to restore
     * @param outGeneratedMidi Output: Generated MIDI data
     * @param outVocalNotes Output: Vocal notes
     * @param outLyrics Output: Lyrics
     * @param outSelectedEmotionIds Output: Emotion node selections
     * @param outPrimaryEmotionId Output: Primary emotion ID
     * @return true if successful
     */
    bool loadProject(const juce::File& file,
                     kelly::PluginState::Preset& outState,
                     GeneratedMidi& outGeneratedMidi,
                     std::vector<MidiNote>& outVocalNotes,
                     std::vector<juce::String>& outLyrics,
                     std::vector<int>& outSelectedEmotionIds,
                     std::optional<int>& outPrimaryEmotionId);

    /**
     * Get last error message if save/load failed.
     *
     * @return Error message string, empty if no error
     */
    juce::String getLastError() const { return lastError_; }

    /**
     * Check if a file is a valid project file.
     *
     * @param file File to check
     * @return true if file appears to be a valid project file
     */
    bool isValidProjectFile(const juce::File& file) const;

    /**
     * Get project metadata without loading full project.
     *
     * @param file Project file
     * @return ProjectData with metadata only, or nullopt if failed
     */
    std::optional<ProjectData> getProjectMetadata(const juce::File& file) const;

    /**
     * Get default project file extension.
     */
    static juce::String getProjectFileExtension() { return ".midikompanion"; }

    /**
     * Get default project file pattern for file chooser.
     */
    static juce::String getProjectFilePattern() { return "*.midikompanion"; }

private:
    juce::String lastError_;

    /**
     * Clear last error message.
     */
    void clearError() { lastError_.clear(); }

    /**
     * Set error message.
     */
    void setError(const juce::String& error) { lastError_ = error; }

    /**
     * Serialize GeneratedMidi to JSON.
     */
    juce::var serializeGeneratedMidi(const GeneratedMidi& midi) const;

    /**
     * Deserialize GeneratedMidi from JSON.
     */
    bool deserializeGeneratedMidi(const juce::var& json, GeneratedMidi& outMidi) const;

    /**
     * Serialize MidiNote to JSON.
     */
    juce::var serializeMidiNote(const MidiNote& note) const;

    /**
     * Deserialize MidiNote from JSON.
     */
    bool deserializeMidiNote(const juce::var& json, MidiNote& outNote) const;

    /**
     * Serialize Chord to JSON.
     */
    juce::var serializeChord(const Chord& chord) const;

    /**
     * Deserialize Chord from JSON.
     */
    bool deserializeChord(const juce::var& json, Chord& outChord) const;

    /**
     * Handle version migration for project files.
     */
    bool migrateProjectVersion(juce::ValueTree& tree, int fromVersion, int toVersion) const;
};

} // namespace midikompanion
