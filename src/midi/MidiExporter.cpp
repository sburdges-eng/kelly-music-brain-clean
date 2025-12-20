#include "midi/MidiExporter.h"
#include "midi/MidiBuilder.h"  // Include MidiBuilder implementation
#include "common/MusicConstants.h"
#include "common/Types.h"
#include <juce_audio_basics/juce_audio_basics.h>
#include <algorithm>
#include <cmath>

namespace midikompanion {
using namespace kelly::MusicConstants;

MidiExporter::MidiExporter() {
    clearError();
}

bool MidiExporter::exportToFile(const juce::File& file,
                                  const GeneratedMidi& midi,
                                  const ExportOptions& options) {
    clearError();

    // Validate MIDI data
    if (!validateMidiData(midi)) {
        return false;
    }

    // Build MIDI file based on format
    juce::MidiFile midiFile;
    if (options.format == Format::SMF_Type0) {
        midiFile = buildSingleTrackFile(midi, options);
    } else {
        midiFile = buildMultiTrackFile(midi, options);
    }

    // Write to file
    juce::FileOutputStream outputStream(file);
    if (!outputStream.openedOk()) {
        setError("Could not open file for writing: " + file.getFullPathName());
        return false;
    }

    midiFile.writeTo(outputStream);
    outputStream.flush();

    return true;
}

bool MidiExporter::exportToFileWithLyrics(const juce::File& file,
                                           const GeneratedMidi& midi,
                                           const std::vector<juce::String>& lyrics,
                                           const ExportOptions& options) {
    clearError();

    // Validate MIDI data
    if (!validateMidiData(midi)) {
        return false;
    }

    // Build MIDI file
    juce::MidiFile midiFile;
    if (options.format == Format::SMF_Type0) {
        midiFile = buildSingleTrackFile(midi, options);
    } else {
        midiFile = buildMultiTrackFile(midi, options);
    }

    // Add lyrics to first track
    // Note: Since getTrack() returns const, we rebuild the file with lyrics included
    if (options.includeLyrics && !lyrics.empty() && midiFile.getNumTracks() > 0) {
        // Rebuild with lyrics - create new file and copy tracks
        juce::MidiFile newFile;
        newFile.setTicksPerQuarterNote(midiFile.getTimeFormat());

        // Copy first track and add lyrics
        juce::MidiMessageSequence firstTrack;
        const auto* existingTrack = midiFile.getTrack(0);
        if (existingTrack != nullptr) {
            for (int i = 0; i < existingTrack->getNumEvents(); ++i) {
                auto* event = existingTrack->getEventPointer(i);
                if (event != nullptr) {
                    firstTrack.addEvent(event->message, event->message.getTimeStamp());
                }
            }
        }
        addLyricEvents(firstTrack, lyrics, midi, options.ticksPerQuarterNote);
        firstTrack.updateMatchedPairs();
        newFile.addTrack(firstTrack);

        // Copy remaining tracks
        for (int i = 1; i < midiFile.getNumTracks(); ++i) {
            const auto* track = midiFile.getTrack(i);
            if (track != nullptr) {
                juce::MidiMessageSequence newTrack;
                for (int j = 0; j < track->getNumEvents(); ++j) {
                    auto* event = track->getEventPointer(j);
                    if (event != nullptr) {
                        newTrack.addEvent(event->message, event->message.getTimeStamp());
                    }
                }
                newTrack.updateMatchedPairs();
                newFile.addTrack(newTrack);
            }
        }

        midiFile = newFile;
    }

    // Write to file
    juce::FileOutputStream outputStream(file);
    if (!outputStream.openedOk()) {
        setError("Could not open file for writing: " + file.getFullPathName());
        return false;
    }

    midiFile.writeTo(outputStream);
    outputStream.flush();

    return true;
}

bool MidiExporter::exportToFileWithVocals(const juce::File& file,
                                            const GeneratedMidi& midi,
                                            const std::vector<MidiNote>& vocalNotes,
                                            const std::vector<juce::String>& lyrics,
                                            const ExportOptions& options) {
    clearError();

    // Validate MIDI data
    if (!validateMidiData(midi)) {
        return false;
    }

    // Build MIDI file
    juce::MidiFile midiFile;
    if (options.format == Format::SMF_Type0) {
        midiFile = buildSingleTrackFile(midi, options);
    } else {
        midiFile = buildMultiTrackFile(midi, options);
    }

    // Add vocal track if including vocals
    if (options.includeVocals && !vocalNotes.empty()) {
        juce::MidiMessageSequence vocalTrack;
        addVocalNotes(vocalTrack, vocalNotes, options.ticksPerQuarterNote, MIDI_CHANNEL_MELODY);

        // Add lyrics to vocal track if provided
        if (options.includeLyrics && !lyrics.empty()) {
            addLyricEvents(vocalTrack, lyrics, midi, options.ticksPerQuarterNote);
        }

        vocalTrack.updateMatchedPairs();
        midiFile.addTrack(vocalTrack);
    }

    // Write to file
    juce::FileOutputStream outputStream(file);
    if (!outputStream.openedOk()) {
        setError("Could not open file for writing: " + file.getFullPathName());
        return false;
    }

    midiFile.writeTo(outputStream);
    outputStream.flush();

    return true;
}

bool MidiExporter::validateMidiData(const GeneratedMidi& midi) const {
    // Check if we have any MIDI data
    if (midi.melody.empty() && midi.bass.empty() && midi.chords.empty() &&
        midi.counterMelody.empty() && midi.pad.empty() && midi.strings.empty() &&
        midi.fills.empty() && midi.rhythm.empty() && midi.drumGroove.empty()) {
        setError("No MIDI data to export");
        return false;
    }

    // Validate tempo
    if (midi.bpm <= 0.0f && midi.tempoBpm <= 0) {
        setError("Invalid tempo in MIDI data");
        return false;
    }

    return true;
}

juce::MidiFile MidiExporter::buildMultiTrackFile(const GeneratedMidi& midi,
                                                   const ExportOptions& options) const {
    juce::MidiFile file;
    file.setTicksPerQuarterNote(options.ticksPerQuarterNote);

    // Track 0: Tempo and time signature (meta track)
    juce::MidiMessageSequence metaTrack;
    float tempoBpm = midi.bpm > 0.0f ? midi.bpm : static_cast<float>(midi.tempoBpm);
    TimeSignature timeSig = {4, 4};  // Default, could be extracted from midi if available
    addTempoAndTimeSignature(metaTrack, tempoBpm, timeSig, options.ticksPerQuarterNote);
    file.addTrack(metaTrack);

    // Track 1: Chords
    if (!midi.chords.empty()) {
        juce::MidiMessageSequence chordTrack;
        int channel = midiBuilder_.getChannelForLayer("chords");
        for (const auto& chord : midi.chords) {
            midiBuilder_.addChordToSequence(chordTrack, chord, options.ticksPerQuarterNote, channel, MIDI_VELOCITY_MEDIUM);
        }
        if (options.includeExpression) {
            addExpressionEvents(chordTrack, midi, options.ticksPerQuarterNote);
        }
        chordTrack.updateMatchedPairs();
        file.addTrack(chordTrack);
    }

    // Track 2: Melody
    if (!midi.melody.empty()) {
        juce::MidiMessageSequence melodyTrack;
        int channel = midiBuilder_.getChannelForLayer("melody");
        midiBuilder_.addNotesToSequence(melodyTrack, midi.melody, options.ticksPerQuarterNote, channel);
        if (options.includeExpression) {
            addExpressionEvents(melodyTrack, midi, options.ticksPerQuarterNote);
        }
        melodyTrack.updateMatchedPairs();
        file.addTrack(melodyTrack);
    }

    // Track 3: Bass
    if (!midi.bass.empty()) {
        juce::MidiMessageSequence bassTrack;
        int channel = midiBuilder_.getChannelForLayer("bass");
        midiBuilder_.addNotesToSequence(bassTrack, midi.bass, options.ticksPerQuarterNote, channel);
        if (options.includeExpression) {
            addExpressionEvents(bassTrack, midi, options.ticksPerQuarterNote);
        }
        bassTrack.updateMatchedPairs();
        file.addTrack(bassTrack);
    }

    // Track 4: Counter-melody
    if (!midi.counterMelody.empty()) {
        juce::MidiMessageSequence counterMelodyTrack;
        int channel = midiBuilder_.getChannelForLayer("counterMelody");
        midiBuilder_.addNotesToSequence(counterMelodyTrack, midi.counterMelody, options.ticksPerQuarterNote, channel);
        counterMelodyTrack.updateMatchedPairs();
        file.addTrack(counterMelodyTrack);
    }

    // Track 5: Pad
    if (!midi.pad.empty()) {
        juce::MidiMessageSequence padTrack;
        int channel = midiBuilder_.getChannelForLayer("pad");
        midiBuilder_.addNotesToSequence(padTrack, midi.pad, options.ticksPerQuarterNote, channel);
        padTrack.updateMatchedPairs();
        file.addTrack(padTrack);
    }

    // Track 6: Strings
    if (!midi.strings.empty()) {
        juce::MidiMessageSequence stringsTrack;
        int channel = midiBuilder_.getChannelForLayer("strings");
        midiBuilder_.addNotesToSequence(stringsTrack, midi.strings, options.ticksPerQuarterNote, channel);
        stringsTrack.updateMatchedPairs();
        file.addTrack(stringsTrack);
    }

    // Track 7: Fills
    if (!midi.fills.empty()) {
        juce::MidiMessageSequence fillsTrack;
        int channel = midiBuilder_.getChannelForLayer("fills");
        midiBuilder_.addNotesToSequence(fillsTrack, midi.fills, options.ticksPerQuarterNote, channel);
        fillsTrack.updateMatchedPairs();
        file.addTrack(fillsTrack);
    }

    // Track 8: Rhythm
    if (!midi.rhythm.empty()) {
        juce::MidiMessageSequence rhythmTrack;
        int channel = midiBuilder_.getChannelForLayer("rhythm");
        midiBuilder_.addNotesToSequence(rhythmTrack, midi.rhythm, options.ticksPerQuarterNote, channel);
        rhythmTrack.updateMatchedPairs();
        file.addTrack(rhythmTrack);
    }

    // Track 9: Drum Groove
    if (!midi.drumGroove.empty()) {
        juce::MidiMessageSequence drumTrack;
        int channel = MIDI_CHANNEL_DRUMS;  // Channel 10 (9 in 0-indexed) for drums
        midiBuilder_.addNotesToSequence(drumTrack, midi.drumGroove, options.ticksPerQuarterNote, channel);
        drumTrack.updateMatchedPairs();
        file.addTrack(drumTrack);
    }

    return file;
}

juce::MidiFile MidiExporter::buildSingleTrackFile(const GeneratedMidi& midi,
                                                    const ExportOptions& options) const {
    juce::MidiFile file;
    file.setTicksPerQuarterNote(options.ticksPerQuarterNote);

    // Single track with all data merged
    juce::MidiMessageSequence mergedTrack;

    // Add tempo and time signature
    float tempoBpm = midi.bpm > 0.0f ? midi.bpm : static_cast<float>(midi.tempoBpm);
    TimeSignature timeSig = {4, 4};
    addTempoAndTimeSignature(mergedTrack, tempoBpm, timeSig, options.ticksPerQuarterNote);

    // Merge all tracks into one
    // Chords
    if (!midi.chords.empty()) {
        int channel = midiBuilder_.getChannelForLayer("chords");
        for (const auto& chord : midi.chords) {
            midiBuilder_.addChordToSequence(mergedTrack, chord, options.ticksPerQuarterNote, channel, MIDI_VELOCITY_MEDIUM);
        }
    }

    // Melody
    if (!midi.melody.empty()) {
        int channel = midiBuilder_.getChannelForLayer("melody");
        midiBuilder_.addNotesToSequence(mergedTrack, midi.melody, options.ticksPerQuarterNote, channel);
    }

    // Bass
    if (!midi.bass.empty()) {
        int channel = midiBuilder_.getChannelForLayer("bass");
        midiBuilder_.addNotesToSequence(mergedTrack, midi.bass, options.ticksPerQuarterNote, channel);
    }

    // Add other tracks...
    if (!midi.counterMelody.empty()) {
        int channel = midiBuilder_.getChannelForLayer("counterMelody");
        midiBuilder_.addNotesToSequence(mergedTrack, midi.counterMelody, options.ticksPerQuarterNote, channel);
    }

    if (!midi.pad.empty()) {
        int channel = midiBuilder_.getChannelForLayer("pad");
        midiBuilder_.addNotesToSequence(mergedTrack, midi.pad, options.ticksPerQuarterNote, channel);
    }

    if (!midi.strings.empty()) {
        int channel = midiBuilder_.getChannelForLayer("strings");
        midiBuilder_.addNotesToSequence(mergedTrack, midi.strings, options.ticksPerQuarterNote, channel);
    }

    if (!midi.fills.empty()) {
        int channel = midiBuilder_.getChannelForLayer("fills");
        midiBuilder_.addNotesToSequence(mergedTrack, midi.fills, options.ticksPerQuarterNote, channel);
    }

    if (!midi.rhythm.empty()) {
        int channel = midiBuilder_.getChannelForLayer("rhythm");
        midiBuilder_.addNotesToSequence(mergedTrack, midi.rhythm, options.ticksPerQuarterNote, channel);
    }

    if (!midi.drumGroove.empty()) {
        int channel = MIDI_CHANNEL_DRUMS;
        midiBuilder_.addNotesToSequence(mergedTrack, midi.drumGroove, options.ticksPerQuarterNote, channel);
    }

    // Add expression if requested
    if (options.includeExpression) {
        addExpressionEvents(mergedTrack, midi, options.ticksPerQuarterNote);
    }

    mergedTrack.updateMatchedPairs();
    file.addTrack(mergedTrack);

    return file;
}

void MidiExporter::addTempoAndTimeSignature(juce::MidiMessageSequence& sequence,
                                               float tempoBpm,
                                               const TimeSignature& timeSig,
                                               int ticksPerQuarterNote) const {
    // Time signature: 4/4
    sequence.addEvent(juce::MidiMessage::timeSignatureMetaEvent(timeSig.numerator, timeSig.denominator));

    // Tempo: microseconds per quarter note
    // Formula: 60,000,000 microseconds / BPM = microseconds per beat
    int microsecondsPerBeat = 60000000 / static_cast<int>(tempoBpm);
    sequence.addEvent(juce::MidiMessage::tempoMetaEvent(microsecondsPerBeat));
}

void MidiExporter::addLyricEvents(juce::MidiMessageSequence& sequence,
                                   const std::vector<juce::String>& lyrics,
                                   const GeneratedMidi& midi,
                                   int ticksPerQuarterNote) const {
    // Add lyrics as MIDI text events (0xFF 05)
    // Distribute lyrics across the MIDI timeline
    double totalBeats = midi.lengthInBeats > 0.0 ? midi.lengthInBeats :
                        (midi.bars * 4.0);  // Default: bars * 4 beats per bar

    int lyricCount = static_cast<int>(lyrics.size());
    if (lyricCount == 0) {
        return;
    }

    double beatsPerLyric = totalBeats / static_cast<double>(lyricCount);

    for (size_t i = 0; i < lyrics.size(); ++i) {
        double beatPosition = static_cast<double>(i) * beatsPerLyric;
        int tickPosition = static_cast<int>(beatPosition * ticksPerQuarterNote);

        // Create lyric text event (0xFF 05)
        juce::MidiMessage lyricEvent = juce::MidiMessage::textMetaEvent(5, lyrics[i]);
        sequence.addEvent(lyricEvent, tickPosition);
    }
}

void MidiExporter::addExpressionEvents(juce::MidiMessageSequence& sequence,
                                        const GeneratedMidi& midi,
                                        int ticksPerQuarterNote) const {
    // Add expression CC events (CC 11 = Expression)
    // This is a simplified implementation - in production, you'd extract
    // expression data from the MIDI generation process

    // For now, add a default expression curve
    // In production, this would come from the generation engine
    double totalBeats = midi.lengthInBeats > 0.0 ? midi.lengthInBeats :
                        (midi.bars * 4.0);

    // Add expression curve (simplified - linear increase)
    int numPoints = 8;
    for (int i = 0; i <= numPoints; ++i) {
        double beatPosition = (static_cast<double>(i) / static_cast<double>(numPoints)) * totalBeats;
        int tickPosition = static_cast<int>(beatPosition * ticksPerQuarterNote);

        // Expression value: 0-127, map from 0.0-1.0
        // Simplified: linear curve from 64 to 127
        int expressionValue = 64 + static_cast<int>((static_cast<double>(i) / static_cast<double>(numPoints)) * 63.0);
        expressionValue = std::clamp(expressionValue, 0, 127);

        // Add CC 11 (Expression) to all channels used
        for (int channel = 0; channel < 16; ++channel) {
            if (channel != 9) {  // Skip drum channel
                juce::MidiMessage ccEvent = juce::MidiMessage::controllerEvent(channel + 1, 11, expressionValue);
                sequence.addEvent(ccEvent, tickPosition);
            }
        }
    }
}

void MidiExporter::addVocalNotes(juce::MidiMessageSequence& sequence,
                                  const std::vector<MidiNote>& vocalNotes,
                                  int ticksPerQuarterNote,
                                  int channel) const {
    for (const auto& note : vocalNotes) {
        int pitch = std::clamp(note.pitch, MIDI_PITCH_MIN, MIDI_PITCH_MAX);
        int velocity = std::clamp(note.velocity, 1, 127);

        int startTick = note.startTick;
        int endTick = note.startTick + note.durationTicks;

        sequence.addEvent(juce::MidiMessage::noteOn(channel + 1, pitch, static_cast<juce::uint8>(velocity)), startTick);
        sequence.addEvent(juce::MidiMessage::noteOff(channel + 1, pitch), endTick);
    }
}

} // namespace midikompanion
