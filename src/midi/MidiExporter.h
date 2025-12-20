/**
 * @file MidiExporter.h
 * @brief MIDI file export functionality
 * 
 * Implements complete MIDI export with support for:
 * - Standard MIDI File (SMF) Type 0 and Type 1
 * - Multiple tracks (melody, bass, chords, drums)
 * - Tempo and time signature meta events
 * - Lyric events (text events 0xFF 05)
 * - Expression/CC data
 */

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace midikompanion {

/**
 * @brief Export format options
 */
enum class MidiFormat {
    SMF_Type0,  // Single track (all data merged)
    SMF_Type1   // Multi-track (one track per part)
};

/**
 * @brief Export options for customization
 */
struct MidiExportOptions {
    // Format
    MidiFormat format{MidiFormat::SMF_Type1};
    int ticksPerQuarterNote{480};
    
    // Content options
    bool includeTempo{true};
    bool includeTimeSignature{true};
    bool includeLyrics{true};
    bool includeCC{true};
    bool includeVocals{true};
    bool includeProgramChanges{true};
    
    // Quantization
    bool quantize{false};
    int quantizeResolution{16}; // 16th notes
    
    // Humanization
    bool removeHumanization{false};
    
    // Track naming
    bool includeTrackNames{true};
    std::string projectName{"miDiKompanion Export"};
    
    // Velocity scaling
    float velocityScale{1.0f};
    int velocityMin{1};
    int velocityMax{127};
};

/**
 * @brief MIDI note event for export
 */
struct MidiExportNote {
    int pitch{60};
    int velocity{100};
    double startBeat{0.0};
    double durationBeats{1.0};
    int channel{0}; // 0-15
};

/**
 * @brief Control change event for export
 */
struct MidiExportCC {
    int controller{1}; // CC number
    int value{64};
    double beat{0.0};
    int channel{0};
};

/**
 * @brief Program change event for export
 */
struct MidiExportProgramChange {
    int program{0}; // 0-127
    double beat{0.0};
    int channel{0};
};

/**
 * @brief Lyric event for export
 */
struct MidiExportLyric {
    std::string text;
    double beat{0.0};
};

/**
 * @brief Track data for export
 */
struct MidiExportTrack {
    std::string name;
    int midiChannel{0}; // 0-15
    std::vector<MidiExportNote> notes;
    std::vector<MidiExportCC> ccEvents;
    std::vector<MidiExportProgramChange> programChanges;
    std::vector<MidiExportLyric> lyrics;
    
    // Track-specific options
    std::optional<int> programNumber; // GM program number
    float volumeScale{1.0f};
    int channelOffset{0}; // For multi-channel tracks
};

/**
 * @brief Complete MIDI export data
 */
struct MidiExportData {
    // Tempo and timing
    double tempo{120.0};
    int timeSignatureNumerator{4};
    int timeSignatureDenominator{4};
    
    // Tracks
    std::vector<MidiExportTrack> tracks;
    
    // Global lyrics (if not per-track)
    std::vector<MidiExportLyric> lyrics;
    
    // Key signature (optional)
    std::optional<int> keySigSharpsFlats; // -7 to 7 (flats negative)
    std::optional<bool> keySigMajor; // true = major, false = minor
};

/**
 * @brief Export result with statistics
 */
struct MidiExportResult {
    bool success{false};
    juce::String errorMessage;
    
    // Statistics
    int totalNotes{0};
    int totalTracks{0};
    int totalLyrics{0};
    double durationBeats{0.0};
    double durationSeconds{0.0};
    int fileSizeBytes{0};
};

/**
 * @brief MIDI Exporter class for file export
 * 
 * Handles all MIDI export operations including:
 * - SMF Type 0 and Type 1 formats
 * - Multi-track export
 * - Tempo and time signature
 * - Lyric events
 * - CC data and program changes
 */
class MidiExporter {
public:
    MidiExporter() = default;
    ~MidiExporter() = default;
    
    //==============================================================================
    // Export Operations
    //==============================================================================
    
    /**
     * @brief Export MIDI data to file
     * @param file Output file
     * @param data MIDI data to export
     * @param options Export options
     * @return Export result with statistics
     */
    MidiExportResult exportToFile(const juce::File& file,
                                   const MidiExportData& data,
                                   const MidiExportOptions& options = {});
    
    /**
     * @brief Export MIDI data to memory
     * @param data MIDI data to export
     * @param options Export options
     * @return MIDI file as memory block
     */
    juce::MemoryBlock exportToMemory(const MidiExportData& data,
                                      const MidiExportOptions& options = {});
    
    /**
     * @brief Export to JUCE MidiFile object
     * @param data MIDI data to export
     * @param options Export options
     * @return JUCE MidiFile object
     */
    juce::MidiFile exportToMidiFile(const MidiExportData& data,
                                     const MidiExportOptions& options = {});
    
    //==============================================================================
    // Convenience Methods
    //==============================================================================
    
    /**
     * @brief Export a single track to file
     * @param file Output file
     * @param track Track data
     * @param tempo Tempo in BPM
     * @param options Export options
     * @return Export result
     */
    MidiExportResult exportSingleTrack(const juce::File& file,
                                        const MidiExportTrack& track,
                                        double tempo,
                                        const MidiExportOptions& options = {});
    
    /**
     * @brief Export from JUCE MidiMessageSequence
     * @param file Output file
     * @param sequence MIDI sequence
     * @param tempo Tempo in BPM
     * @param options Export options
     * @return Export result
     */
    MidiExportResult exportFromSequence(const juce::File& file,
                                         const juce::MidiMessageSequence& sequence,
                                         double tempo,
                                         const MidiExportOptions& options = {});
    
    //==============================================================================
    // Validation
    //==============================================================================
    
    /**
     * @brief Validate export data before export
     * @param data MIDI data to validate
     * @return Error message if invalid, empty if valid
     */
    juce::String validateData(const MidiExportData& data);
    
    /**
     * @brief Get last error message
     * @return Error message
     */
    juce::String getLastError() const { return lastError_; }
    
    //==============================================================================
    // Utilities
    //==============================================================================
    
    /**
     * @brief Get file extension for MIDI files
     * @return ".mid"
     */
    static juce::String getFileExtension() { return ".mid"; }
    
    /**
     * @brief Get file filter for MIDI files
     * @return "*.mid"
     */
    static juce::String getFileFilter() { return "*.mid;*.midi"; }
    
    /**
     * @brief Convert beats to MIDI ticks
     * @param beats Beat position
     * @param ppq Pulses per quarter note
     * @return Tick position
     */
    static int beatsToTicks(double beats, int ppq);
    
    /**
     * @brief Convert seconds to MIDI ticks
     * @param seconds Time in seconds
     * @param tempo Tempo in BPM
     * @param ppq Pulses per quarter note
     * @return Tick position
     */
    static int secondsToTicks(double seconds, double tempo, int ppq);

private:
    //==============================================================================
    // Internal Methods
    //==============================================================================
    
    void buildMidiFile(juce::MidiFile& midiFile,
                       const MidiExportData& data,
                       const MidiExportOptions& options);
    
    void addTempoTrack(juce::MidiFile& midiFile,
                       const MidiExportData& data,
                       const MidiExportOptions& options);
    
    void addTrack(juce::MidiFile& midiFile,
                  const MidiExportTrack& track,
                  const MidiExportOptions& options,
                  int trackIndex);
    
    void addNotesToSequence(juce::MidiMessageSequence& sequence,
                            const std::vector<MidiExportNote>& notes,
                            const MidiExportOptions& options,
                            float trackVelocityScale = 1.0f,
                            int channelOffset = 0);
    
    void addCCToSequence(juce::MidiMessageSequence& sequence,
                         const std::vector<MidiExportCC>& ccEvents,
                         const MidiExportOptions& options,
                         int channelOffset = 0);
    
    void addLyricsToSequence(juce::MidiMessageSequence& sequence,
                             const std::vector<MidiExportLyric>& lyrics,
                             const MidiExportOptions& options);
    
    void addProgramChangesToSequence(juce::MidiMessageSequence& sequence,
                                     const std::vector<MidiExportProgramChange>& changes,
                                     const MidiExportOptions& options,
                                     int channelOffset = 0);
    
    int scaleVelocity(int velocity, const MidiExportOptions& options);
    
    double quantizeBeat(double beat, int resolution);
    
    void setError(const juce::String& error);
    
    //==============================================================================
    // Member Variables
    //==============================================================================
    
    juce::String lastError_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MidiExporter)
};

} // namespace midikompanion
