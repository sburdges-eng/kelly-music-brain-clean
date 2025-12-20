#pragma once
/*
 * MidiExporter.h - Comprehensive MIDI File Export
 * ===============================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - MIDI Layer: Uses MidiBuilder for MIDI construction
 * - Plugin Layer: Used by PluginEditor for export dialog
 * - Common Layer: Uses GeneratedMidi, MidiNote, Chord types
 *
 * Purpose: Dedicated MIDI exporter with comprehensive features:
 * - Export all MIDI tracks (melody, bass, chords, drums)
 * - Include tempo and time signature
 * - Export MIDI lyric events (text events 0xFF 05)
 * - Export expression data (CC events)
 * - Support multiple export formats (SMF Type 0/1)
 * - Export with/without vocals option
 */

#include "common/Types.h"  // GeneratedMidi, MidiNote, Chord
#include "midi/MidiBuilder.h"  // MidiBuilder
#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>
#include <string>
#include <optional>

namespace midikompanion {

// Import kelly types for use in this namespace
using kelly::GeneratedMidi;
using kelly::MidiNote;
using kelly::TimeSignature;

/**
 * MIDI Exporter - Comprehensive MIDI file export functionality.
 *
 * Features:
 * - Export all tracks (melody, bass, chords, drums, etc.)
 * - Include tempo and time signature
 * - Export MIDI lyric events (text events 0xFF 05)
 * - Export expression data (CC events)
 * - Support SMF Type 0 (single track) and Type 1 (multi-track)
 * - Export with/without vocals option
 */
class MidiExporter {
public:
    /**
     * Export format options
     */
    enum class Format {
        SMF_Type0,  // Single track format (all tracks merged)
        SMF_Type1   // Multi-track format (separate track per layer)
    };

    /**
     * Export options
     */
    struct ExportOptions {
        Format format;
        bool includeVocals;
        bool includeLyrics;
        bool includeExpression;  // CC events for dynamics, etc.
        int ticksPerQuarterNote;  // Standard MIDI resolution

        // Constructor with default values
        ExportOptions()
            : format(Format::SMF_Type1)
            , includeVocals(true)
            , includeLyrics(true)
            , includeExpression(true)
            , ticksPerQuarterNote(960)
        {}
    };

    MidiExporter();
    ~MidiExporter() = default;

    /**
     * Export GeneratedMidi to MIDI file.
     *
     * @param file Target MIDI file
     * @param midi Generated MIDI data to export
     * @param options Export options
     * @return true if successful
     */
    bool exportToFile(const juce::File& file,
                      const GeneratedMidi& midi,
                      const ExportOptions& options = ExportOptions());

    /**
     * Export GeneratedMidi with lyrics to MIDI file.
     *
     * @param file Target MIDI file
     * @param midi Generated MIDI data to export
     * @param lyrics Lyrics as text lines (will be converted to MIDI lyric events)
     * @param options Export options
     * @return true if successful
     */
    bool exportToFileWithLyrics(const juce::File& file,
                                 const GeneratedMidi& midi,
                                 const std::vector<juce::String>& lyrics,
                                 const ExportOptions& options = ExportOptions());

    /**
     * Export GeneratedMidi with vocal notes to MIDI file.
     *
     * @param file Target MIDI file
     * @param midi Generated MIDI data to export
     * @param vocalNotes Vocal notes to include
     * @param lyrics Lyrics for vocal notes
     * @param options Export options
     * @return true if successful
     */
    bool exportToFileWithVocals(const juce::File& file,
                                const GeneratedMidi& midi,
                                const std::vector<MidiNote>& vocalNotes,
                                const std::vector<juce::String>& lyrics = {},
                                const ExportOptions& options = ExportOptions());

    /**
     * Get last error message if export failed.
     *
     * @return Error message string, empty if no error
     */
    juce::String getLastError() const { return lastError_; }

    /**
     * Validate MIDI data before export.
     *
     * @param midi MIDI data to validate
     * @return true if valid, false if issues found
     */
    bool validateMidiData(const GeneratedMidi& midi) const;

private:
    mutable juce::String lastError_;  // Mutable to allow const methods to set errors
    mutable kelly::MidiBuilder midiBuilder_;  // Mutable to allow const methods to use it

    /**
     * Clear last error message.
     */
    void clearError() { lastError_.clear(); }

    /**
     * Set error message.
     */
    void setError(const juce::String& error) const { lastError_ = error; }

    /**
     * Build MIDI file from GeneratedMidi (Type 1 - multi-track).
     */
    juce::MidiFile buildMultiTrackFile(const GeneratedMidi& midi,
                                        const ExportOptions& options) const;

    /**
     * Build MIDI file from GeneratedMidi (Type 0 - single track).
     */
    juce::MidiFile buildSingleTrackFile(const GeneratedMidi& midi,
                                        const ExportOptions& options) const;

    /**
     * Add tempo and time signature to MIDI sequence.
     */
    void addTempoAndTimeSignature(juce::MidiMessageSequence& sequence,
                                   float tempoBpm,
                                   const TimeSignature& timeSig,
                                   int ticksPerQuarterNote) const;

    /**
     * Add lyric events to MIDI sequence.
     */
    void addLyricEvents(juce::MidiMessageSequence& sequence,
                        const std::vector<juce::String>& lyrics,
                        const GeneratedMidi& midi,
                        int ticksPerQuarterNote) const;

    /**
     * Add expression CC events to MIDI sequence.
     */
    void addExpressionEvents(juce::MidiMessageSequence& sequence,
                             const GeneratedMidi& midi,
                             int ticksPerQuarterNote) const;

    /**
     * Add vocal notes to MIDI sequence.
     */
    void addVocalNotes(juce::MidiMessageSequence& sequence,
                       const std::vector<MidiNote>& vocalNotes,
                       int ticksPerQuarterNote,
                       int channel) const;
};

} // namespace midikompanion
