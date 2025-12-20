/**
 * @file MidiExporter.cpp
 * @brief MIDI file export implementation
 */

#include "MidiExporter.h"
#include <algorithm>

namespace midikompanion {

//==============================================================================
// Export Operations
//==============================================================================

MidiExportResult MidiExporter::exportToFile(const juce::File& file,
                                             const MidiExportData& data,
                                             const MidiExportOptions& options) {
    MidiExportResult result;
    lastError_.clear();
    
    // Validate data
    auto validationError = validateData(data);
    if (validationError.isNotEmpty()) {
        result.errorMessage = validationError;
        setError(validationError);
        return result;
    }
    
    // Build MIDI file
    juce::MidiFile midiFile = exportToMidiFile(data, options);
    
    // Write to file
    juce::FileOutputStream stream(file);
    if (!stream.openedOk()) {
        result.errorMessage = "Failed to open file for writing: " + file.getFullPathName();
        setError(result.errorMessage);
        return result;
    }
    
    if (!midiFile.writeTo(stream)) {
        result.errorMessage = "Failed to write MIDI data to file";
        setError(result.errorMessage);
        return result;
    }
    
    // Calculate statistics
    result.success = true;
    result.totalTracks = static_cast<int>(data.tracks.size());
    result.totalLyrics = static_cast<int>(data.lyrics.size());
    
    for (const auto& track : data.tracks) {
        result.totalNotes += static_cast<int>(track.notes.size());
        result.totalLyrics += static_cast<int>(track.lyrics.size());
        
        // Calculate duration
        for (const auto& note : track.notes) {
            double endBeat = note.startBeat + note.durationBeats;
            if (endBeat > result.durationBeats) {
                result.durationBeats = endBeat;
            }
        }
    }
    
    result.durationSeconds = result.durationBeats * (60.0 / data.tempo);
    result.fileSizeBytes = static_cast<int>(file.getSize());
    
    return result;
}

juce::MemoryBlock MidiExporter::exportToMemory(const MidiExportData& data,
                                                const MidiExportOptions& options) {
    juce::MemoryBlock memoryBlock;
    
    juce::MidiFile midiFile = exportToMidiFile(data, options);
    
    juce::MemoryOutputStream stream(memoryBlock, false);
    midiFile.writeTo(stream);
    
    return memoryBlock;
}

juce::MidiFile MidiExporter::exportToMidiFile(const MidiExportData& data,
                                               const MidiExportOptions& options) {
    juce::MidiFile midiFile;
    
    // Set time format
    midiFile.setTicksPerQuarterNote(options.ticksPerQuarterNote);
    
    // Build the file
    buildMidiFile(midiFile, data, options);
    
    return midiFile;
}

//==============================================================================
// Convenience Methods
//==============================================================================

MidiExportResult MidiExporter::exportSingleTrack(const juce::File& file,
                                                  const MidiExportTrack& track,
                                                  double tempo,
                                                  const MidiExportOptions& options) {
    MidiExportData data;
    data.tempo = tempo;
    data.tracks.push_back(track);
    
    return exportToFile(file, data, options);
}

MidiExportResult MidiExporter::exportFromSequence(const juce::File& file,
                                                   const juce::MidiMessageSequence& sequence,
                                                   double tempo,
                                                   const MidiExportOptions& options) {
    MidiExportResult result;
    
    // Create a simple MIDI file from the sequence
    juce::MidiFile midiFile;
    midiFile.setTicksPerQuarterNote(options.ticksPerQuarterNote);
    
    // Add tempo track
    juce::MidiMessageSequence tempoTrack;
    auto tempoEvent = juce::MidiMessage::tempoMetaEvent(
        static_cast<int>(60000000.0 / tempo)); // microseconds per beat
    tempoEvent.setTimeStamp(0);
    tempoTrack.addEvent(tempoEvent);
    
    // Add time signature
    auto timeSigEvent = juce::MidiMessage::timeSignatureMetaEvent(4, 4);
    timeSigEvent.setTimeStamp(0);
    tempoTrack.addEvent(timeSigEvent);
    
    // Add track name
    auto trackNameEvent = juce::MidiMessage::textMetaEvent(3, options.projectName);
    trackNameEvent.setTimeStamp(0);
    tempoTrack.addEvent(trackNameEvent);
    
    // End of track
    auto endTrack = juce::MidiMessage::endOfTrack();
    endTrack.setTimeStamp(sequence.getEndTime() + options.ticksPerQuarterNote);
    tempoTrack.addEvent(endTrack);
    
    midiFile.addTrack(tempoTrack);
    
    // Add the main sequence
    juce::MidiMessageSequence mainTrack(sequence);
    auto endMain = juce::MidiMessage::endOfTrack();
    endMain.setTimeStamp(sequence.getEndTime() + options.ticksPerQuarterNote);
    mainTrack.addEvent(endMain);
    
    midiFile.addTrack(mainTrack);
    
    // Write to file
    juce::FileOutputStream stream(file);
    if (!stream.openedOk()) {
        result.errorMessage = "Failed to open file for writing";
        return result;
    }
    
    if (!midiFile.writeTo(stream)) {
        result.errorMessage = "Failed to write MIDI data";
        return result;
    }
    
    result.success = true;
    result.totalTracks = 1;
    result.totalNotes = sequence.getNumEvents();
    result.fileSizeBytes = static_cast<int>(file.getSize());
    
    return result;
}

//==============================================================================
// Validation
//==============================================================================

juce::String MidiExporter::validateData(const MidiExportData& data) {
    // Check tempo
    if (data.tempo <= 0 || data.tempo > 500) {
        return "Invalid tempo: " + juce::String(data.tempo) + " BPM";
    }
    
    // Check time signature
    if (data.timeSignatureNumerator <= 0 || data.timeSignatureNumerator > 32) {
        return "Invalid time signature numerator: " + juce::String(data.timeSignatureNumerator);
    }
    
    if (data.timeSignatureDenominator <= 0 || 
        (data.timeSignatureDenominator & (data.timeSignatureDenominator - 1)) != 0) {
        return "Invalid time signature denominator (must be power of 2): " + 
               juce::String(data.timeSignatureDenominator);
    }
    
    // Check tracks
    for (size_t i = 0; i < data.tracks.size(); ++i) {
        const auto& track = data.tracks[i];
        
        // Check channel
        if (track.midiChannel < 0 || track.midiChannel > 15) {
            return "Track " + juce::String(static_cast<int>(i)) + 
                   " has invalid MIDI channel: " + juce::String(track.midiChannel);
        }
        
        // Check notes
        for (const auto& note : track.notes) {
            if (note.pitch < 0 || note.pitch > 127) {
                return "Track " + juce::String(static_cast<int>(i)) + 
                       " has invalid note pitch: " + juce::String(note.pitch);
            }
            if (note.velocity < 0 || note.velocity > 127) {
                return "Track " + juce::String(static_cast<int>(i)) + 
                       " has invalid note velocity: " + juce::String(note.velocity);
            }
            if (note.startBeat < 0) {
                return "Track " + juce::String(static_cast<int>(i)) + 
                       " has negative note start time";
            }
            if (note.durationBeats <= 0) {
                return "Track " + juce::String(static_cast<int>(i)) + 
                       " has invalid note duration";
            }
        }
    }
    
    return {}; // No errors
}

//==============================================================================
// Utilities
//==============================================================================

int MidiExporter::beatsToTicks(double beats, int ppq) {
    return static_cast<int>(beats * ppq);
}

int MidiExporter::secondsToTicks(double seconds, double tempo, int ppq) {
    double beatsPerSecond = tempo / 60.0;
    double beats = seconds * beatsPerSecond;
    return beatsToTicks(beats, ppq);
}

//==============================================================================
// Internal Methods
//==============================================================================

void MidiExporter::buildMidiFile(juce::MidiFile& midiFile,
                                  const MidiExportData& data,
                                  const MidiExportOptions& options) {
    // For Type 0, merge everything into one track
    if (options.format == MidiFormat::SMF_Type0) {
        juce::MidiMessageSequence mergedTrack;
        
        // Add tempo and time signature
        if (options.includeTempo) {
            auto tempoEvent = juce::MidiMessage::tempoMetaEvent(
                static_cast<int>(60000000.0 / data.tempo));
            tempoEvent.setTimeStamp(0);
            mergedTrack.addEvent(tempoEvent);
        }
        
        if (options.includeTimeSignature) {
            auto timeSigEvent = juce::MidiMessage::timeSignatureMetaEvent(
                data.timeSignatureNumerator, data.timeSignatureDenominator);
            timeSigEvent.setTimeStamp(0);
            mergedTrack.addEvent(timeSigEvent);
        }
        
        // Add track name
        if (options.includeTrackNames) {
            auto trackNameEvent = juce::MidiMessage::textMetaEvent(3, options.projectName);
            trackNameEvent.setTimeStamp(0);
            mergedTrack.addEvent(trackNameEvent);
        }
        
        // Add global lyrics
        if (options.includeLyrics) {
            addLyricsToSequence(mergedTrack, data.lyrics, options);
        }
        
        // Merge all tracks
        double maxEndTime = 0;
        for (const auto& track : data.tracks) {
            const bool isVocalTrack = track.type == "vocal" || track.type == "vocals" ||
                                      juce::String(track.name).toLowerCase().contains("vocal");
            if (!options.includeVocals && isVocalTrack) {
                continue;
            }

            addNotesToSequence(mergedTrack, track.notes, options, track.volumeScale, track.channelOffset);
            
            if (options.includeCC) {
                addCCToSequence(mergedTrack, track.ccEvents, options, track.channelOffset);
            }
            
            if (options.includeLyrics) {
                addLyricsToSequence(mergedTrack, track.lyrics, options);
            }
            
            if (options.includeProgramChanges) {
                addProgramChangesToSequence(mergedTrack, track.programChanges, options, track.channelOffset);
            }
            
            // Track max end time
            for (const auto& note : track.notes) {
                double endBeat = note.startBeat + note.durationBeats;
                if (endBeat > maxEndTime) {
                    maxEndTime = endBeat;
                }
            }
        }
        
        // Add end of track
        auto endTrack = juce::MidiMessage::endOfTrack();
        endTrack.setTimeStamp(beatsToTicks(maxEndTime + 1, options.ticksPerQuarterNote));
        mergedTrack.addEvent(endTrack);
        
        mergedTrack.sort();
        midiFile.addTrack(mergedTrack);
    }
    else {
        // Type 1: Separate tracks
        
        // Track 0: Tempo and time signature
        addTempoTrack(midiFile, data, options);
        
        // Add each track
        int trackIndex = 1;
        for (const auto& track : data.tracks) {
            const bool isVocalTrack = track.type == "vocal" || track.type == "vocals" ||
                                      juce::String(track.name).toLowerCase().contains("vocal");
            if (!options.includeVocals && isVocalTrack) {
                continue;
            }
            addTrack(midiFile, track, options, trackIndex++);
        }
    }
}

void MidiExporter::addTempoTrack(juce::MidiFile& midiFile,
                                  const MidiExportData& data,
                                  const MidiExportOptions& options) {
    juce::MidiMessageSequence tempoTrack;
    
    // Track name
    if (options.includeTrackNames) {
        auto trackNameEvent = juce::MidiMessage::textMetaEvent(3, options.projectName);
        trackNameEvent.setTimeStamp(0);
        tempoTrack.addEvent(trackNameEvent);
    }
    
    // Tempo
    if (options.includeTempo) {
        auto tempoEvent = juce::MidiMessage::tempoMetaEvent(
            static_cast<int>(60000000.0 / data.tempo));
        tempoEvent.setTimeStamp(0);
        tempoTrack.addEvent(tempoEvent);
    }
    
    // Time signature
    if (options.includeTimeSignature) {
        auto timeSigEvent = juce::MidiMessage::timeSignatureMetaEvent(
            data.timeSignatureNumerator, data.timeSignatureDenominator);
        timeSigEvent.setTimeStamp(0);
        tempoTrack.addEvent(timeSigEvent);
    }
    
    // Key signature (if present)
    if (data.keySigSharpsFlats.has_value() && data.keySigMajor.has_value()) {
        auto keySigEvent = juce::MidiMessage::keySignatureMetaEvent(
            data.keySigSharpsFlats.value(), data.keySigMajor.value());
        keySigEvent.setTimeStamp(0);
        tempoTrack.addEvent(keySigEvent);
    }
    
    // Global lyrics
    if (options.includeLyrics && !data.lyrics.empty()) {
        addLyricsToSequence(tempoTrack, data.lyrics, options);
    }
    
    // Calculate max end time from all tracks
    double maxEndTime = 0;
    for (const auto& track : data.tracks) {
        for (const auto& note : track.notes) {
            double endBeat = note.startBeat + note.durationBeats;
            if (endBeat > maxEndTime) {
                maxEndTime = endBeat;
            }
        }
    }
    
    // End of track
    auto endTrack = juce::MidiMessage::endOfTrack();
    endTrack.setTimeStamp(beatsToTicks(maxEndTime + 1, options.ticksPerQuarterNote));
    tempoTrack.addEvent(endTrack);
    
    tempoTrack.sort();
    midiFile.addTrack(tempoTrack);
}

void MidiExporter::addTrack(juce::MidiFile& midiFile,
                             const MidiExportTrack& track,
                             const MidiExportOptions& options,
                             int trackIndex) {
    juce::MidiMessageSequence sequence;
    
    // Track name
    if (options.includeTrackNames && !track.name.empty()) {
        auto trackNameEvent = juce::MidiMessage::textMetaEvent(3, track.name);
        trackNameEvent.setTimeStamp(0);
        sequence.addEvent(trackNameEvent);
    }
    
    // Program change
    if (options.includeProgramChanges && track.programNumber.has_value()) {
        const int programChannel = juce::jlimit(1, 16, track.midiChannel + track.channelOffset + 1);
        auto programChange = juce::MidiMessage::programChange(programChannel, track.programNumber.value());
        programChange.setTimeStamp(0);
        sequence.addEvent(programChange);
    }
    
    // Add notes
    addNotesToSequence(sequence, track.notes, options, track.volumeScale, track.channelOffset);
    
    // Add CC events
    if (options.includeCC) {
        addCCToSequence(sequence, track.ccEvents, options, track.channelOffset);
    }
    
    // Add track-specific lyrics
    if (options.includeLyrics) {
        addLyricsToSequence(sequence, track.lyrics, options);
    }
    
    // Add program changes
    if (options.includeProgramChanges) {
        addProgramChangesToSequence(sequence, track.programChanges, options, track.channelOffset);
    }
    
    // Calculate end time
    double maxEndTime = 0;
    for (const auto& note : track.notes) {
        double endBeat = note.startBeat + note.durationBeats;
        if (endBeat > maxEndTime) {
            maxEndTime = endBeat;
        }
    }
    
    // End of track
    auto endTrack = juce::MidiMessage::endOfTrack();
    endTrack.setTimeStamp(beatsToTicks(maxEndTime + 1, options.ticksPerQuarterNote));
    sequence.addEvent(endTrack);
    
    sequence.sort();
    midiFile.addTrack(sequence);
}

void MidiExporter::addNotesToSequence(juce::MidiMessageSequence& sequence,
                                      const std::vector<MidiExportNote>& notes,
                                      const MidiExportOptions& options,
                                      float trackVelocityScale,
                                      int channelOffset) {
    for (const auto& note : notes) {
        double startBeat = note.startBeat;
        double durationBeats = note.durationBeats;
        
        // Apply quantization
        if (options.quantize) {
            startBeat = quantizeBeat(startBeat, options.quantizeResolution);
            durationBeats = quantizeBeat(durationBeats, options.quantizeResolution);
            if (durationBeats < 1.0 / options.quantizeResolution) {
                durationBeats = 1.0 / options.quantizeResolution;
            }
        }
        
        int baseVelocity = static_cast<int>(note.velocity * trackVelocityScale);
        int velocity = scaleVelocity(baseVelocity, options);
        int channel = juce::jlimit(1, 16, note.channel + channelOffset + 1); // JUCE uses 1-16
        
        int startTick = beatsToTicks(startBeat, options.ticksPerQuarterNote);
        int endTick = beatsToTicks(startBeat + durationBeats, options.ticksPerQuarterNote);
        
        // Note On
        auto noteOn = juce::MidiMessage::noteOn(channel, note.pitch, static_cast<juce::uint8>(velocity));
        noteOn.setTimeStamp(startTick);
        sequence.addEvent(noteOn);
        
        // Note Off
        auto noteOff = juce::MidiMessage::noteOff(channel, note.pitch);
        noteOff.setTimeStamp(endTick);
        sequence.addEvent(noteOff);
    }
}

void MidiExporter::addCCToSequence(juce::MidiMessageSequence& sequence,
                                   const std::vector<MidiExportCC>& ccEvents,
                                   const MidiExportOptions& options,
                                   int channelOffset) {
    for (const auto& cc : ccEvents) {
        double beat = options.quantize ? 
            quantizeBeat(cc.beat, options.quantizeResolution) : cc.beat;
        
        int tick = beatsToTicks(beat, options.ticksPerQuarterNote);
        int channel = juce::jlimit(1, 16, cc.channel + channelOffset + 1);
        
        auto ccMessage = juce::MidiMessage::controllerEvent(channel, cc.controller, cc.value);
        ccMessage.setTimeStamp(tick);
        sequence.addEvent(ccMessage);
    }
}

void MidiExporter::addLyricsToSequence(juce::MidiMessageSequence& sequence,
                                        const std::vector<MidiExportLyric>& lyrics,
                                        const MidiExportOptions& options) {
    for (const auto& lyric : lyrics) {
        double beat = options.quantize ? 
            quantizeBeat(lyric.beat, options.quantizeResolution) : lyric.beat;
        
        int tick = beatsToTicks(beat, options.ticksPerQuarterNote);
        
        // Text meta event type 5 is for lyrics
        auto lyricEvent = juce::MidiMessage::textMetaEvent(5, lyric.text);
        lyricEvent.setTimeStamp(tick);
        sequence.addEvent(lyricEvent);
    }
}

void MidiExporter::addProgramChangesToSequence(juce::MidiMessageSequence& sequence,
                                               const std::vector<MidiExportProgramChange>& changes,
                                               const MidiExportOptions& options,
                                               int channelOffset) {
    for (const auto& change : changes) {
        double beat = options.quantize ? 
            quantizeBeat(change.beat, options.quantizeResolution) : change.beat;
        
        int tick = beatsToTicks(beat, options.ticksPerQuarterNote);
        int channel = juce::jlimit(1, 16, change.channel + channelOffset + 1);
        
        auto programChange = juce::MidiMessage::programChange(channel, change.program);
        programChange.setTimeStamp(tick);
        sequence.addEvent(programChange);
    }
}

int MidiExporter::scaleVelocity(int velocity, const MidiExportOptions& options) {
    float scaled = static_cast<float>(velocity) * options.velocityScale;
    int result = static_cast<int>(scaled);
    return std::clamp(result, options.velocityMin, options.velocityMax);
}

double MidiExporter::quantizeBeat(double beat, int resolution) {
    double grid = 4.0 / resolution; // Quarter note = 4 16ths, etc.
    return std::round(beat / grid) * grid;
}

void MidiExporter::setError(const juce::String& error) {
    lastError_ = error;
    DBG("MidiExporter Error: " + error);
}

} // namespace midikompanion
