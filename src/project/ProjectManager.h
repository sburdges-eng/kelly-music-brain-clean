/**
 * @file ProjectManager.h
 * @brief Project management class for save/load operations
 *
 * Provides complete project persistence including:
 * - 216-node emotion selections
 * - Generated MIDI data
 * - Vocal notes and lyrics
 * - Plugin state and parameters
 * - Version migration support
 */

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_core/juce_core.h>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace midikompanion {

/**
 * @brief Project version information for migration support
 */
struct ProjectVersion {
    int major{1};
    int minor{0};
    int patch{0};

    juce::String toString() const;
    static ProjectVersion fromString(const juce::String& str);
    bool operator<(const ProjectVersion& other) const;
    bool operator==(const ProjectVersion& other) const;
};

/**
 * @brief Time signature representation
 */
struct TimeSignature {
    int numerator{4};
    int denominator{4};
};

/**
 * @brief MIDI note event for project serialization
 */
struct MidiNoteData {
    int pitch{60};
    int velocity{100};
    double startBeat{0.0};
    double durationBeats{1.0};
    int channel{1};
};

/**
 * @brief MIDI track containing notes and metadata
 */
struct TrackData {
    std::string name{"Track"};
    std::string type{"melody"};
    int midiChannel{1};
    float volume{1.0f};
    float pan{0.0f};
    bool muted{false};
    bool soloed{false};
    std::vector<MidiNoteData> notes;
};

/**
 * @brief Vocal note representation
 */
struct VocalNote {
    std::string syllable;
    double startBeat{0.0};
    double durationBeats{1.0};
    int pitch{60};
    float phonemeBlend{0.0f};
};

/**
 * @brief Lyric line with optional vocal notes
 */
struct LyricLine {
    std::string text;
    double startBeat{0.0};
    double endBeat{0.0};
    std::vector<VocalNote> vocalNotes;
};

/**
 * @brief Emotion selection state
 */
struct EmotionState {
    int nodeId{0};                     // 0-215 node ID
    float valence{0.0f};               // -1.0 to 1.0
    float arousal{0.5f};               // 0.0 to 1.0
    float dominance{0.5f};             // 0.0 to 1.0
    float intensity{0.5f};             // 0.0 to 1.0
    std::string emotionTag{"neutral"}; // "happy", "sad", etc.
    std::vector<int> relatedNodeIds;
    std::optional<std::vector<float>> mlEmbedding;
    std::optional<float> mlConfidence;
};

/**
 * @brief Plugin state snapshot
 */
struct PluginState {
    EmotionState emotionState;
    std::map<std::string, float> parameters;
    bool isRecording{false};
    bool isPlaying{false};
    double playheadPosition{0.0};
    std::string woundDescription;
    int bars{4};
    bool enableCloud{false};
    int generationRate{8};
};

/**
 * @brief Project metadata
 */
struct ProjectMetadata {
    std::string name{"Untitled Project"};
    std::string author;
    std::string createdDate;
    std::string modifiedDate;
    ProjectVersion version;
};

/**
 * @brief Complete project state
 */
struct ProjectData {
    ProjectMetadata metadata;
    double tempo{120.0};
    TimeSignature timeSignature{};
    PluginState pluginState;
    std::vector<TrackData> tracks;
    std::vector<VocalNote> vocalNotes;
    std::vector<LyricLine> lyrics;
    double sampleRate{44100.0};
    std::string midiOutputDevice;
    std::map<std::string, std::string> customData;
};

/**
 * @brief Project Manager for save/load operations
 *
 * Thread-safe project persistence with JSON serialization.
 * Supports version migration for backwards compatibility.
 */
class ProjectManager {
public:
    ProjectManager();
    ~ProjectManager() = default;

    bool saveProject(const juce::File& file, const ProjectData& data);
    bool loadProject(const juce::File& file, ProjectData& outData);
    bool saveFromValueTree(const juce::File& file,
                           const juce::AudioProcessorValueTreeState& state,
                           const ProjectData& additionalData = {});
    bool loadToValueTree(const juce::File& file,
                         juce::AudioProcessorValueTreeState& state,
                         ProjectData& outData);

    juce::String getLastError() const { return lastError_; }

    bool isValidProjectFile(const juce::File& file);

    static ProjectVersion getCurrentVersion();
    static juce::String getFileExtension() { return ".mkp"; }
    static juce::String getFileFilter() { return "*.mkp"; }
    static juce::String getFileTypeDescription() { return "miDiKompanion Project"; }

private:
    juce::File createBackup(const juce::File& file);
    bool validateFile(const juce::File& file);
    bool needsMigration(const ProjectVersion& version) const;
    bool migrateProject(ProjectData& data);

    juce::String projectToJson(const ProjectData& data) const;
    bool jsonToProject(const juce::String& json, ProjectData& outData);

    juce::var metadataToVar(const ProjectMetadata& metadata) const;
    bool varToMetadata(const juce::var& var, ProjectMetadata& outMetadata) const;

    juce::var emotionStateToVar(const EmotionState& state) const;
    bool varToEmotionState(const juce::var& var, EmotionState& outState) const;

    juce::var trackDataToVar(const TrackData& track) const;
    bool varToTrackData(const juce::var& var, TrackData& outTrack) const;

    juce::var vocalNoteToVar(const VocalNote& note) const;
    bool varToVocalNote(const juce::var& var, VocalNote& outNote) const;

    juce::var lyricLineToVar(const LyricLine& line) const;
    bool varToLyricLine(const juce::var& var, LyricLine& outLine) const;

    juce::var pluginStateToVar(const PluginState& state) const;
    bool varToPluginState(const juce::var& var, PluginState& outState) const;

    static std::string getCurrentTimestamp();
    void setError(const juce::String& error);

    juce::String lastError_;
    mutable juce::CriticalSection lock_;
};

} // namespace midikompanion
