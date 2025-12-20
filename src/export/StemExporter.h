#pragma once
/*
 * StemExporter.h - Audio Stem Export for miDiKompanion
 * ====================================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - MIDI Layer: Uses GeneratedMidi to render individual tracks
 * - Audio Layer: Uses JUCE audio rendering to generate audio stems
 * - Plugin Layer: Used by PluginEditor for stem export dialog
 * - Voice Layer: Integrates with VoiceSynthesizer for vocal stems
 *
 * Purpose: Export individual MIDI tracks as separate audio stems.
 *          Renders MIDI to audio using internal synthesis and exports as WAV/AIFF/FLAC.
 *
 * Features:
 * - Export individual tracks as audio stems (melody, bass, chords, drums, vocals)
 * - Render MIDI to audio using internal synthesis
 * - Apply track effects and mixing
 * - Support multiple formats (WAV, AIFF, FLAC)
 * - Normalize stems
 * - Export with/without effects
 */

#include "common/Types.h"  // GeneratedMidi, MidiNote
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <vector>
#include <string>
#include <functional>
#include <optional>

namespace midikompanion {

// Import kelly types for use in this namespace
using kelly::Chord;
using kelly::GeneratedMidi;
using kelly::MidiNote;

/**
 * Stem Exporter - Export individual MIDI tracks as separate audio stems.
 *
 * Renders MIDI tracks to audio using internal synthesis and exports
 * as separate audio files for mixing/mastering.
 */
class StemExporter {
public:
    /**
     * Export format options
     */
    enum class Format {
        WAV,
        AIFF,
        FLAC
    };

    /**
     * Export options
     */
    struct ExportOptions {
        Format format;
        double sampleRate;
        int bitDepth;  // 16, 24, or 32
        bool normalizeStems;  // Normalize each stem to -0.1dBFS
        bool includeEffects;  // Apply track effects (if any)
        float durationSeconds;  // 0 = auto-detect from MIDI
        std::string filenameSuffix;  // Optional suffix for filenames

        // Constructor with default values
        ExportOptions()
            : format(Format::WAV)
            , sampleRate(44100.0)
            , bitDepth(24)
            , normalizeStems(true)
            , includeEffects(true)
            , durationSeconds(0.0f)
        {}
    };

    /**
     * Export result for a single stem
     */
    struct StemResult {
        std::string trackName;
        juce::String filepath;
        bool success = false;
        juce::String errorMessage;
        int64_t numSamples = 0;
        double durationSeconds = 0.0;
    };

    /**
     * Progress callback
     * @param currentTrack Current track being exported (0-based)
     * @param totalTracks Total number of tracks
     * @param trackName Name of current track
     */
    using ProgressCallback = std::function<void(int currentTrack, int totalTracks, const juce::String& trackName)>;

    StemExporter();
    ~StemExporter() = default;

    /**
     * Export all tracks from GeneratedMidi as individual stems.
     *
     * @param midi Generated MIDI data
     * @param outputDirectory Directory to save stems
     * @param options Export options
     * @return Vector of export results, one per track
     */
    std::vector<StemResult> exportAllStems(
        const GeneratedMidi& midi,
        const juce::File& outputDirectory,
        const ExportOptions& options = ExportOptions()
    );

    /**
     * Export specific tracks as stems.
     *
     * @param midi Generated MIDI data
     * @param trackNames Names of tracks to export (e.g., "melody", "bass", "chords")
     * @param outputDirectory Directory to save stems
     * @param options Export options
     * @return Vector of export results
     */
    std::vector<StemResult> exportSelectedStems(
        const GeneratedMidi& midi,
        const std::vector<std::string>& trackNames,
        const juce::File& outputDirectory,
        const ExportOptions& options = ExportOptions()
    );

    /**
     * Export a single track as a stem.
     *
     * @param midi Generated MIDI data
     * @param trackName Name of track to export (e.g., "melody", "bass")
     * @param outputFile Output file path
     * @param options Export options
     * @return Export result
     */
    StemResult exportTrack(
        const GeneratedMidi& midi,
        const std::string& trackName,
        const juce::File& outputFile,
        const ExportOptions& options = ExportOptions()
    );

    /**
     * Export vocal track as a stem (if vocals are available).
     *
     * @param vocalNotes Vocal notes to render
     * @param lyrics Lyrics for vocals (optional)
     * @param outputFile Output file path
     * @param options Export options
     * @return Export result
     */
    StemResult exportVocalStem(
        const std::vector<MidiNote>& vocalNotes,
        const std::vector<juce::String>& lyrics,
        const juce::File& outputFile,
        const ExportOptions& options = ExportOptions()
    );

    /**
     * Set progress callback.
     */
    void setProgressCallback(ProgressCallback callback) {
        progressCallback_ = callback;
    }

    /**
     * Get last error message.
     */
    juce::String getLastError() const { return lastError_; }

private:
    juce::String lastError_;
    ProgressCallback progressCallback_;

    /**
     * Clear last error.
     */
    void clearError() { lastError_.clear(); }

    /**
     * Set error message.
     */
    void setError(const juce::String& error) { lastError_ = error; }

    /**
     * Render MIDI notes to audio buffer.
     *
     * @param notes MIDI notes to render
     * @param durationSeconds Duration to render
     * @param sampleRate Sample rate
     * @param channel MIDI channel to use
     * @return Rendered audio buffer
     */
    juce::AudioBuffer<float> renderMidiNotes(
        const std::vector<MidiNote>& notes,
        double durationSeconds,
        double sampleRate,
        int channel = 0
    );

    /**
     * Render chords to audio buffer.
     *
     * @param chords Chords to render
     * @param durationSeconds Duration to render
     * @param sampleRate Sample rate
     * @param channel MIDI channel to use
     * @return Rendered audio buffer
     */
    juce::AudioBuffer<float> renderChords(
        const std::vector<kelly::Chord>& chords,
        double durationSeconds,
        double sampleRate,
        int channel = 0
    );

    /**
     * Render vocal notes to audio buffer (uses simple synthesis).
     *
     * @param vocalNotes Vocal notes to render
     * @param sampleRate Sample rate
     * @return Rendered audio buffer
     */
    juce::AudioBuffer<float> renderVocals(
        const std::vector<MidiNote>& vocalNotes,
        double sampleRate
    );

    /**
     * Normalize audio buffer to target level.
     *
     * @param buffer Audio buffer to normalize (modified in place)
     * @param targetLevel Target peak level (0.0 to 1.0, default: 0.9 = -0.1dBFS)
     */
    void normalizeBuffer(juce::AudioBuffer<float>& buffer, float targetLevel = 0.9f);

    /**
     * Get track notes from GeneratedMidi by name.
     *
     * @param midi Generated MIDI data
     * @param trackName Track name (e.g., "melody", "bass", "chords")
     * @return Vector of MIDI notes, or empty if track not found
     */
    std::vector<MidiNote> getTrackNotes(const GeneratedMidi& midi, const std::string& trackName) const;

    /**
     * Calculate duration from MIDI data.
     *
     * @param midi Generated MIDI data
     * @return Duration in seconds
     */
    double calculateDuration(const GeneratedMidi& midi) const;

    /**
     * Generate stem filename.
     *
     * @param trackName Track name
     * @param outputDirectory Output directory
     * @param format Export format
     * @param suffix Optional suffix
     * @return Full file path
     */
    juce::File generateStemFilename(
        const std::string& trackName,
        const juce::File& outputDirectory,
        Format format,
        const std::string& suffix = ""
    );

    /**
     * Get file extension for format.
     */
    juce::String getFileExtension(Format format) const;

    /**
     * Write audio buffer to file.
     *
     * @param buffer Audio buffer
     * @param file Output file
     * @param options Export options
     * @return true if successful
     */
    bool writeAudioFile(
        const juce::AudioBuffer<float>& buffer,
        const juce::File& file,
        const ExportOptions& options
    );
};

} // namespace midikompanion
